import copy
import json
import os
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from reasonroute.datasets import disjoint
from reasonroute.runtime import MethodRuntime, branch_environment
from tools.train_method_models import fit_classifier, gate_targets, choose_threshold
from tools.train_method_models import main as train_models
from reasonroute.datasets import load_artifact
from reasonroute.annotation import policy_sha256


def output(answer, token=100):
    return {'final_answer': answer, 'retrieved_context': ['evidence'],
            'metrics': {'token_cost': token, 'retrieval_steps': 1, 'context_length': 8}}


class FakeRR:
    def __init__(self):
        self.calls = []

    def execute_rag_flow(self, question, kind, **kwargs):
        self.calls.append((question, kind, kwargs, dict(os.environ)))
        return output('unknown' if kwargs.get('answer_style') == 'reasoning' else 'entity')


class FakeIRCoT:
    def __init__(self):
        self.calls = []

    def run(self, question, enable_answer_verifier=None):
        self.calls.append(enable_answer_verifier)
        return output('repaired', 200)


class FixedModel:
    def __init__(self, label):
        self.label = label

    def predict(self, texts):
        return [self.label] * len(texts)

    def decision_function(self, texts):
        return np.array([float(self.label)] * len(texts))


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path('configs/paper_v3.json').read_text())

    def test_normalized_overlap_fails(self):
        with self.assertRaises(ValueError):
            disjoint([{'question': '  HELLO   World?'}], [{'question': 'hello world? '}])

    def test_tfidf_fits_only_training_folds_and_full_train(self):
        train = [{'question': f'{label} topic example {i}', 'question_type': label}
                 for label in ('fact', 'explanation', 'reasoning') for i in range(6)]
        test = [{'question': 'heldout_unseen private query', 'question_type': 'fact'}]
        fitted = []
        original = TfidfVectorizer.fit_transform
        def track(vectorizer, documents, *args, **kwargs):
            docs = list(documents)
            fitted.append(docs)
            return original(vectorizer, docs, *args, **kwargs)
        with patch.object(TfidfVectorizer, 'fit_transform', track):
            fit_classifier(train, test, folds=3)
        self.assertEqual(sorted(map(len, fitted)), [12] * 6 + [18] * 2)
        self.assertFalse(any('heldout_unseen' in text for docs in fitted for text in docs))

    def test_unlabelled_gate_candidates_are_rejected(self):
        with self.assertRaises(ValueError):
            gate_targets([{'question': 'query', 'gate_label': None}])

    def test_gate_definition_and_token_constraint(self):
        rows = [{'low': {'scores': {'correctness': 0}, 'token_cost': 100},
                 'high': {'scores': {'correctness': 1}, 'token_cost': 300}, 'gate_label': 1},
                {'low': {'scores': {'correctness': 1}, 'token_cost': 100},
                 'high': {'scores': {'correctness': 1}, 'token_cost': 300}, 'gate_label': 0}]
        self.assertEqual(gate_targets(rows).tolist(), [1, 0])
        threshold, selected = choose_threshold(np.array([1., -1.]), rows, 200)
        self.assertLessEqual(selected['average_token'], 200)
        self.assertEqual(selected['accuracy'], 1.)

    def test_full_800_gate_training_and_model_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paired = []
            for i in range(800):
                positive = i % 2
                paired.append({'id': str(i), 'question': f'{"bridge comparison" if positive else "lookup entity"} question {i}',
                               'execution_context': {'config_sha256': 'synthetic-test-only',
                                                     'implementation_revision': 'independent-v4'},
                               'label_policy_sha256': policy_sha256(), 'candidate_question_type': 'reasoning',
                               'gate_label': positive,
                               'low': {'scores': {'correctness': 1 - positive}, 'token_cost': 100},
                               'high': {'scores': {'correctness': 1}, 'token_cost': 300}})
            for filename, rows in [('paired.jsonl', paired),
                                   ('classifier.jsonl', [{'question': f'classifier excluded {i}'} for i in range(1000)]),
                                   ('eval.jsonl', [{'question': f'evaluation excluded {i}'} for i in range(1490)])]:
                (root / filename).write_text(''.join(json.dumps(r) + '\n' for r in rows))
            arguments = ['train', 'gate', '--paired', str(root / 'paired.jsonl'), '--exclude',
                         str(root / 'classifier.jsonl'), str(root / 'eval.jsonl'),
                         '--max-average-token', '200', '--output', str(root / 'trained')]
            with patch.object(sys, 'argv', arguments):
                train_models()
            artifact = load_artifact(root / 'trained/model.joblib', 'compact_gate')
            self.assertEqual(artifact['train_n'], 800)
            self.assertEqual(artifact['model'].decision_function(['bridge comparison question new']).shape, (1,))
            self.assertTrue(np.isfinite(artifact['threshold']))

    def test_compact_low_gate_skips_ircot(self):
        rr, ir = FakeRR(), FakeIRCoT()
        runtime = MethodRuntime(rr, ir, self.config, {'model': FixedModel('reasoning')},
                                {'model': FixedModel(-1), 'threshold': 0})
        value, trace = runtime.run('question', 'compact')
        self.assertEqual(ir.calls, [])
        self.assertEqual(value['metrics']['token_cost'], 100)
        self.assertFalse(trace['gate_high'])

    def test_compact_high_has_no_verifier_and_counts_both_calls(self):
        rr, ir = FakeRR(), FakeIRCoT()
        runtime = MethodRuntime(rr, ir, self.config, {'model': FixedModel('reasoning')},
                                {'model': FixedModel(1), 'threshold': 0})
        value, trace = runtime.run('question', 'compact')
        self.assertEqual(ir.calls, [False])
        self.assertEqual(value['metrics']['token_cost'], 300)
        self.assertEqual(len(rr.calls), 1)
        self.assertTrue(trace['gate_high'])

    def test_quality_enables_verifier_only_on_fallback(self):
        rr, ir = FakeRR(), FakeIRCoT()
        runtime = MethodRuntime(rr, ir, self.config, {'model': FixedModel('reasoning')})
        runtime.run('question', 'quality')
        self.assertEqual(ir.calls, [True])

    def test_predicted_fact_uses_length_profile(self):
        rr, ir = FakeRR(), FakeIRCoT()
        runtime = MethodRuntime(rr, ir, self.config, {'model': FixedModel('fact')})
        _, trace = runtime.run('x' * 60, 'quality')
        self.assertEqual(rr.calls[0][2]['context_top_k_override'], 4)
        self.assertEqual(trace['route_source'], 'predicted')
        self.assertEqual(ir.calls, [])

    def test_bad_classifier_falls_back_to_explanation(self):
        rr, ir = FakeRR(), FakeIRCoT()
        runtime = MethodRuntime(rr, ir, self.config, {'model': FixedModel('not_a_label')})
        _, trace = runtime.run('question', 'quality')
        self.assertEqual(trace['predicted_question_type'], 'explanation')
        self.assertIsNotNone(trace['classifier_error'])

    def test_branch_helper_cannot_silently_use_gold_labels(self):
        from reasonroute.branch_flow import ReasonRouteIRCoTAssisted
        with self.assertRaises(ValueError):
            ReasonRouteIRCoTAssisted(FakeRR(), FakeIRCoT(), route_source='gold')
        flow = ReasonRouteIRCoTAssisted(FakeRR(), FakeIRCoT())
        with self.assertRaises(ValueError):
            flow.route_type('question', 'reasoning')

    def test_environment_is_restored_after_failure(self):
        with patch.dict(os.environ, {'IRCOT_FINAL_CONTEXT_TOP_K': '999'}):
            with self.assertRaises(RuntimeError):
                with branch_environment({'IRCOT_FINAL_CONTEXT_TOP_K': '5'}):
                    self.assertEqual(os.environ['IRCOT_FINAL_CONTEXT_TOP_K'], '5')
                    raise RuntimeError()
            self.assertEqual(os.environ['IRCOT_FINAL_CONTEXT_TOP_K'], '999')


if __name__ == '__main__':
    unittest.main()
