"""Regression tests for independent data, labels and corpus provenance."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from reasonroute.annotation import annotate, verify_labels
from reasonroute.corpus import verify_corpus
from reasonroute.datasets import digest_file, question_fingerprints, verify_model_data
from tools.augment_explanation_from_iirc import select_questions, reference_answers
from tools.build_classifier_training_dataset import load_exclusion_questions, assign_label
from tools.build_fixed_corpus import make_chunks
from tools.prepare_method_data import audit_labels


class DataIntegrityTests(unittest.TestCase):
    def row(self):
        return {'id': 'q', 'question': 'Who wrote the book?', 'answer': 'Author',
                'source_dataset': 'hotpotqa', 'supporting_evidence_count': 2}

    def test_source_name_cannot_change_label(self):
        row = self.row()
        labels = [assign_label({**row, 'source_dataset': source}) for source in
                  ('hotpotqa', 'iirc', 'naturalquestions', 'arbitrary')]
        self.assertEqual(len(set(labels)), 1)

    def test_flat_train_and_evidence_eval_use_same_policy(self):
        train = self.row()
        evaluation = {'question': train['question'], 'answers': [train['answer']],
                      'supporting_evidence': [{'doc_id': 'a'}, {'doc_id': 'b'}]}
        self.assertEqual(annotate(train)['question_type'], annotate(evaluation)['question_type'])

    def test_missing_reference_is_not_a_valid_classification_label(self):
        with self.assertRaises(ValueError):
            annotate({**self.row(), 'answer': ''})

    def test_stale_or_mismatched_labels_are_rejected(self):
        row = annotate(self.row())
        self.assertEqual(verify_labels([row])['source_dataset_used'], False)
        wrong = {**row, 'question_type': 'reasoning'}
        with self.assertRaises(ValueError):
            verify_labels([wrong])
        self.assertEqual(audit_labels([wrong])['changed_labels'], 1)
        self.assertEqual(wrong['question_type'], 'reasoning')

    def test_missing_exclusion_file_is_fatal(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                load_exclusion_questions([Path(folder) / 'missing.jsonl'])

    def test_old_oof_artifacts_are_not_independent_models(self):
        with self.assertRaises(ValueError):
            verify_model_data({'train_n': 1000}, [self.row()], 'qa', 'query_classifier')

    def test_actual_question_overlap_rejected_even_with_forged_counts(self):
        rows = [self.row()]
        artifact = {'protocol_version': 'independent-v4', 'train_n': 1,
                    'training_question_sha256': question_fingerprints(rows), 'test_sha256': 'qa'}
        with self.assertRaises(ValueError):
            verify_model_data(artifact, rows, 'qa', 'query_classifier')

    def test_new_evaluation_snapshot_requires_new_freeze(self):
        artifact = {'protocol_version': 'independent-v4', 'train_n': 1,
                    'training_question_sha256': ['another-question'], 'test_sha256': 'frozen'}
        with self.assertRaises(ValueError):
            verify_model_data(artifact, [self.row()], 'different', 'query_classifier')
        artifact['excluded_evaluation_sha256'] = ['frozen']
        with self.assertRaises(ValueError):
            verify_model_data(artifact, [self.row()], 'different', 'compact_gate')

    def test_builder_rejects_question_conditioned_corpus_input(self):
        with self.assertRaises(ValueError):
            list(make_chunks([{'doc_id': 'a', 'text': 'Gold evidence', 'question': 'Test query'}]))

    def test_complete_source_documents_can_contain_supporting_evidence(self):
        docs = [{'doc_id': 'source-A', 'title': 'Article', 'text': 'A whole independently obtained article.'}]
        chunks = list(make_chunks(docs, 30, 5))
        self.assertTrue(chunks)
        self.assertTrue(all(row['corpus'] == 'full_document_snapshot' for row in chunks))

    def test_iirc_selection_cannot_add_missing_context(self):
        passages = [{'title': 'Article', 'questions': [{'qid': 'a', 'question': 'Why did it change?',
                     'answer': {'answer_value': 'A reason'},
                     'context': [{'passage': 'main', 'text': 'Required evidence.'}]}]}]
        chunks = [{'doc_id': 'doc', 'chunk_id': 'chunk', 'title': 'Article', 'content': 'Unrelated text.'}]
        original = copy.deepcopy(chunks)
        rows, stats = select_questions(passages, chunks, set(), 1)
        self.assertEqual(rows, [])
        self.assertEqual(chunks, original)
        self.assertEqual(stats['support_not_in_fixed_corpus'], 1)
        chunks[0]['content'] = 'Article intro. Required evidence. Article outro.'
        original = copy.deepcopy(chunks)
        rows, _ = select_questions(passages, chunks, set(), 1)
        self.assertEqual(rows[0]['supporting_evidence'][0]['chunk_ids'], ['chunk'])
        self.assertEqual(chunks, original)

    def test_corpus_manifest_detects_legacy_question_chunks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            kb = root / 'kb.jsonl'
            kb.write_text(json.dumps({'doc_id': 'iirc_supp__test', 'content': 'evidence'}) + '\n')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'build_policy': 'full_documents_without_qa_v1',
                                            'kb_sha256': digest_file(kb), 'chunks': 1}))
            with self.assertRaises(ValueError):
                verify_corpus(kb, manifest)

    def test_iirc_value_and_binary_answers_not_dropped(self):
        self.assertEqual(reference_answers({'answer': {'type': 'binary', 'answer_value': 'yes'}}), ['yes'])
        self.assertEqual(reference_answers({'answer': {'type': 'value', 'answer_value': '9 years'}}), ['9 years'])


if __name__ == '__main__':
    unittest.main()
