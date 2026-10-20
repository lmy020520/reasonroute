"""Sequential ReasonRoute evaluation or paired CompactGate benefit annotation."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasonroute.datasets import (answer_text, digest_file, disjoint, read_rows, write_new_json,
                                 load_artifact, verify_model_data)
from reasonroute.annotation import verify_labels
from reasonroute.corpus import verify_corpus


def judge(question, answer, output, strict=False):
    from evaluate_rag import llm_judge_comprehensive
    scores = llm_judge_comprehensive(question, answer, output['final_answer'], output['retrieved_context'])
    if strict and scores.get('judge_fallback'):
        raise ValueError('Gate benefit label requires a successful LLM judge; resume later')
    for name in ('correctness', 'context_recall', 'faithfulness'):
        if type(scores.get(name)) is not int or scores[name] not in (0, 1):
            raise ValueError('Judge did not return three integer 0/1 scores')
    return scores


def record_branch(output, scores):
    token = output.get('metrics', {}).get('token_cost')
    complete = output.get('metrics', {}).get('token_accounting_complete', True)
    if complete and (type(token) is not int or token <= 0):
        raise ValueError('Completed generation requires positive integer Token usage')
    if not complete and token is not None:
        raise ValueError('Unknown total Token usage must be null, never a partial total or zero')
    return {'scores': scores, 'prediction': output['final_answer'],
            'token_cost': token, 'token_accounting_complete': complete, 'metrics': output['metrics']}


def summary(rows):
    table = {}
    for kind in ('overall', 'fact', 'explanation', 'reasoning'):
        subset = rows if kind == 'overall' else [r for r in rows if r.get('question_type') == kind]
        if not subset:
            continue
        acc = sum(r['scores']['correctness'] for r in subset) / len(subset) * 100
        known = [r['token_cost'] for r in subset if r['token_cost'] is not None]
        token = sum(known) / len(subset) if len(known) == len(subset) else None
        table[kind] = {'N': len(subset), 'ACC': acc,
                       'Recall': sum(r['scores']['context_recall'] for r in subset) / len(subset) * 100,
                       'Faith': sum(r['scores']['faithfulness'] for r in subset) / len(subset) * 100,
                       'Token': token, 'ACC_per_1KToken': acc / token * 1000 if token else None,
                       'Token_accounting_complete': len(known) == len(subset),
                       'Token_known_samples': len(known)}
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['evaluate', 'label-gate'])
    parser.add_argument('--qa', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New run directory; --resume verifies its context')
    parser.add_argument('--config', type=Path, default=Path('configs/paper_v3.json'))
    parser.add_argument('--kb', type=Path, required=True)
    parser.add_argument('--corpus-manifest', type=Path, required=True)
    parser.add_argument('--kg', type=Path, required=True)
    parser.add_argument('--embeddings', type=Path, required=True)
    parser.add_argument('--tokenized', type=Path, required=True)
    parser.add_argument('--classifier', type=Path)
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--mode', choices=['quality', 'compact'], default='quality')
    parser.add_argument('--exclude', type=Path, nargs='+', help='Gate annotation: classifier and evaluation datasets')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    rows = read_rows(args.qa)
    ids = [r.get('id') for r in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError('Unique nonempty sample IDs are required')
    for row in rows:
        answer_text(row)
    verify_corpus(args.kb, args.corpus_manifest)
    verify_labels(rows)
    if args.task == 'label-gate':
        args.mode = 'compact'
        if len(rows) != 800 or not args.exclude:
            raise ValueError('Gate annotation requires 800 candidates and explicit exclusion datasets')
        if any(row.get('question_type') != 'reasoning' for row in rows):
            raise ValueError('CompactGate candidates must be reasoning queries under the shared annotation policy')
        exclusions = [read_rows(path) for path in args.exclude]
        if not any(len(r) == 1000 for r in exclusions) or not any(len(r) == 1490 for r in exclusions):
            raise ValueError('Include both classifier-1000 and evaluation-1490 exclusion sets')
        for excluded in exclusions:
            disjoint(rows, excluded)
    elif not args.classifier or (args.mode == 'compact' and not args.gate):
        parser.error('Evaluation requires --classifier and Compact also requires --gate')
    elif len(rows) != 1490 or any(r.get('question_type') not in ('fact', 'explanation', 'reasoning') for r in rows):
        raise ValueError('This independent evaluation protocol requires 1490 labelled held-out questions')
    if args.task == 'evaluate':
        classifier = load_artifact(args.classifier, 'query_classifier')
        verify_model_data(classifier, rows, digest_file(args.qa), 'query_classifier')
        if args.mode == 'compact':
            gate = load_artifact(args.gate, 'compact_gate')
            verify_model_data(gate, rows, digest_file(args.qa), 'compact_gate')
            if set(gate['training_question_sha256']) & set(classifier['training_question_sha256']):
                raise ValueError('Gate and classifier training questions must be disjoint')
    from reasonroute.runtime import load_runtime
    runtime, context = load_runtime(args, need_models=args.task == 'evaluate')
    signature = {'task': args.task, 'mode': args.mode, 'qa_sha256': digest_file(args.qa), 'execution_context': context,
                 'classifier_sha256': digest_file(args.classifier) if args.classifier else None,
                 'gate_sha256': digest_file(args.gate) if args.gate else None}
    if args.output.exists():
        if not args.resume or json.loads((args.output / 'run.json').read_text()) != signature:
            raise ValueError('Existing output requires --resume with exactly the same data, config and models')
    else:
        args.output.mkdir(parents=True)
        write_new_json(args.output / 'run.json', signature)
    journal = args.output / 'samples.jsonl'
    records = read_rows(journal) if journal.exists() else []
    completed = {r['id'] for r in records}
    if not completed <= set(ids):
        raise ValueError('Checkpoint contains foreign sample IDs')
    with journal.open('a', encoding='utf-8') as handle:
        for row in rows:
            if row['id'] in completed:
                continue
            question, reference = row['question'], answer_text(row)
            start = time.perf_counter()
            if args.task == 'label-gate':
                low, _ = runtime.reasoning(question, 'compact', allow_high=False)
                high, branch = runtime.reasoning(question, 'compact', allow_high=True, initial=low)
                low_scores = judge(question, reference, low, strict=True)
                high_scores = low_scores if branch == 'reasoning_low' else judge(question, reference, high, strict=True)
                record = {'id': row['id'], 'question': question,
                          'source_dataset': row.get('source_dataset'), 'execution_context': context,
                          'label_policy': row['label_policy'], 'label_policy_sha256': row['label_policy_sha256'],
                          'candidate_question_type': row['question_type'],
                          'gate_label': int(high_scores['correctness'] > low_scores['correctness']),
                          'low': record_branch(low, low_scores), 'high': record_branch(high, high_scores)}
            else:
                # True labels enter evaluation records only after the routing decision.
                output, route = runtime.run(question, args.mode)
                scores = judge(question, reference, output)
                record = {'id': row['id'], 'question': question, 'question_type': row.get('question_type'),
                          **route, **record_branch(output, scores), 'retrieved_context': output['retrieved_context']}
            record['latency_s'] = time.perf_counter() - start
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()
            records.append(record)
            print(f'{len(records)}/{len(rows)} complete', flush=True)
    report = {'run': signature, 'N': len(records), 'status': 'new execution; does not reproduce historical values by assumption'}
    if args.task == 'evaluate':
        if len(records) != 1490 or any(r.get('route_source') != 'predicted' for r in records):
            raise ValueError('Main results require 1490 real predicted-route records, not gold-branch recombination')
        report.update(method='ReasonRoute-' + args.mode.title(), route_source='predicted',
                      protocol_version='independent-v4', derived_from='actual_sample_journal')
        report['table'] = summary(records)
        report['predicted_route_counts'] = dict(Counter(r['predicted_question_type'] for r in records))
        report['gate_high_n'] = sum(r.get('gate_high', False) for r in records)
    else:
        report['positive_labels'] = sum(r['gate_label'] for r in records)
    final = args.output / 'report.json'
    if not final.exists():
        write_new_json(final, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
