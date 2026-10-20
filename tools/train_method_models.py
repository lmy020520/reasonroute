"""Train the independent query classifier or CompactGate; no LLM calls."""
import argparse
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from reasonroute.classification import build_model, metrics
from reasonroute.datasets import disjoint, digest_file, digest_json, read_rows, write_new_json, question_fingerprints
from reasonroute.annotation import verify_labels, policy_sha256


def classifier_model():
    return build_model(cue_repeats=1, word_ngram=(1, 4), char_ngram=(2, 7), min_df=2, c_value=0.75)


def gate_model():
    return Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2,
                                max_features=25000, sublinear_tf=True)),
        ('classifier', LinearSVC(C=4.0, class_weight='balanced', random_state=42, max_iter=15000)),
    ])


def gate_targets(rows):
    labels = []
    for row in rows:
        low, high = row.get('low'), row.get('high')
        if not low or not high or row.get('error'):
            raise ValueError('Gate requires complete paired branch outcomes, not unlabelled candidates')
        for branch in (low, high):
            value = branch.get('scores', {}).get('correctness')
            if type(value) is not int or value not in (0, 1):
                raise ValueError('Invalid paired correctness')
            if not np.isfinite(branch.get('token_cost', np.nan)) or branch['token_cost'] <= 0:
                raise ValueError('Invalid paired token accounting')
        label = int(high['scores']['correctness'] > low['scores']['correctness'])
        if row.get('gate_label') != label:
            raise ValueError('Gate label does not match paired benefit definition')
        labels.append(label)
    return np.asarray(labels)


def choose_threshold(scores, rows, max_average_token):
    choices = []
    for threshold in np.unique(np.r_[np.nextafter(scores.min(), -np.inf),
                                      np.quantile(scores, np.linspace(0, 1, 201)),
                                      np.nextafter(scores.max(), np.inf)]):
        branches = [row['high'] if score >= threshold else row['low'] for score, row in zip(scores, rows)]
        token = float(np.mean([b['token_cost'] for b in branches]))
        acc = float(np.mean([b['scores']['correctness'] for b in branches]))
        if token <= max_average_token:
            choices.append((acc, -token, float(threshold)))
    if not choices:
        raise ValueError('No train-OOF threshold meets the chosen mean token bound')
    acc, negative_token, threshold = max(choices)
    return threshold, {'accuracy': acc, 'average_token': -negative_token,
                       'scope': 'training OOF selection only, not independent test performance'}


def fit_classifier(train, test, folds=5):
    disjoint(train, test)
    x = [r['question'] for r in train]
    y = [r.get('question_type') for r in train]
    truth = [r.get('question_type') for r in test]
    if not set(y + truth) <= {'fact', 'explanation', 'reasoning'}:
        raise ValueError('Missing or invalid query-type labels')
    cv = StratifiedKFold(folds, shuffle=True, random_state=42)
    model = classifier_model()
    oof = cross_val_predict(model, x, y, cv=cv, n_jobs=1)
    model.fit(x, y)
    started = time.perf_counter()
    predictions = model.predict([r['question'] for r in test])
    latency = (time.perf_counter() - started) / len(test)
    return model, {'training_oof': metrics(y, oof), 'independent_test': metrics(truth, predictions),
                   'average_test_latency_s': latency,
                   'latency_measurement': {'batch_size': len(test), 'repetitions': 1, 'warmup': 0,
                                           'scope': 'full-batch prediction time divided by sample count',
                                           'includes_feature_transform': True, 'includes_network': False},
                   'test_predictions': [{'id': r.get('id'), 'question': r['question'], 'gold': gold,
                                         'predicted': pred} for r, gold, pred in zip(test, truth, predictions)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='task', required=True)
    c = sub.add_parser('classifier')
    c.add_argument('--train', required=True, type=Path)
    c.add_argument('--test', required=True, type=Path)
    c.add_argument('--output', required=True, type=Path)
    g = sub.add_parser('gate')
    g.add_argument('--paired', required=True, type=Path)
    g.add_argument('--exclude', nargs='+', required=True, type=Path,
                   help='Must include the 1000-query classifier set and all held-out evaluation sets')
    g.add_argument('--max-average-token', type=float, required=True,
                   help='Reasoning-only training OOF mean token bound, not per-call max_tokens')
    g.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output directory; old runs are preserved')
    if args.task == 'classifier':
        train, test = read_rows(args.train), read_rows(args.test)
        if len(train) != 1000 or len(test) != 1490:
            raise ValueError('Expected independent 1000 train / 1490 test protocol')
        annotation = verify_labels(train)
        verify_labels(test)
        model, report = fit_classifier(train, test)
        artifact = {'kind': 'query_classifier', 'model': model, 'train_n': len(train),
                    'protocol_version': 'independent-v4', 'annotation': annotation,
                    'training_question_sha256': question_fingerprints(train),
                    'train_sha256': digest_file(args.train), 'test_sha256': digest_file(args.test)}
        report.update({'protocol': '1000 train-only OOF, full training fit, independent 1490 prediction',
                       'method': 'LinearSVC', 'annotation': annotation,
                       'training_evaluation_separated': True,
                       'train_n': len(train), 'test_n': len(test), 'train_sha256': artifact['train_sha256'],
                       'test_sha256': artifact['test_sha256'], 'folds': 5, 'split_seed': 42})
    else:
        train = read_rows(args.paired)
        if len(train) != 800:
            raise ValueError('Expected all 800 paired gate development samples')
        exclusions = [read_rows(path) for path in args.exclude]
        if not any(len(rows) == 1000 for rows in exclusions) or not any(len(rows) == 1490 for rows in exclusions):
            raise ValueError('Exclusions must include 1000 classifier and 1490 evaluation samples')
        for excluded in exclusions:
            disjoint(train, excluded)
        labels = gate_targets(train)
        if any(row.get('execution_context', {}).get('implementation_revision') != 'independent-v4' for row in train):
            raise ValueError('Gate pairs must come from the independent-v4 annotation runner')
        if any(row.get('label_policy_sha256') != policy_sha256() or row.get('candidate_question_type') != 'reasoning' for row in train):
            raise ValueError('Gate pairs must record their shared candidate annotation policy')
        contexts = {digest_json(row.get('execution_context')) for row in train}
        if len(contexts) != 1 or any(not row.get('execution_context') for row in train):
            raise ValueError('Paired annotations must share a recorded corpus/profile execution context')
        if len(set(labels)) != 2 or min(np.bincount(labels)) < 5:
            raise ValueError('Five-fold gate training requires at least 5 positives and 5 negatives')
        text = [r['question'] for r in train]
        model = gate_model()
        scores = cross_val_predict(model, text, labels, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                   method='decision_function', n_jobs=1)
        threshold, selection = choose_threshold(scores, train, args.max_average_token)
        model.fit(text, labels)
        artifact = {'kind': 'compact_gate', 'model': model, 'threshold': threshold, 'train_n': len(train),
                    'protocol_version': 'independent-v4',
                    'training_question_sha256': question_fingerprints(train),
                    'excluded_evaluation_sha256': [digest_file(path) for path in args.exclude],
                    'paired_sha256': digest_file(args.paired), 'execution_context': train[0]['execution_context']}
        report = {'protocol': '800 development-only OOF threshold selection; full-800 final fit',
                  'train_n': len(train), 'positive_n': int(labels.sum()), 'threshold': threshold,
                  'max_average_token': args.max_average_token, 'selection': selection,
                  'paired_sha256': artifact['paired_sha256'], 'execution_context': artifact['execution_context'],
                  'exclude_sha256': [digest_file(path) for path in args.exclude], 'folds': 5, 'seed': 42}
    args.output.mkdir(parents=True)
    joblib.dump(artifact, args.output / 'model.joblib')
    write_new_json(args.output / 'report.json', report)
    print(f'Saved {args.task} artifact and new report to {args.output}')
    if args.task == 'classifier':
        print(f"Independent 1490 accuracy: {report['independent_test']['accuracy'] * 100:.2f}%")


if __name__ == '__main__':
    main()
