"""Regression contracts for the corrected equations (2), (4a), and preprocessing."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from reasonroute.preprocessing import SupportMapper, adapt_question, normalized_positions
from reasonroute.runtime import MethodRuntime, branch_environment
from scripts.run_reasonroute import record_branch, summary
from tools.build_fixed_corpus import make_chunks
from tools.build_public_evidence import normalize_source
from tools.query_classifier_features_v3 import QueryStructureFeatures
from tools.rag_resarch_tool import DeepSeekRAGTool
from tools.train_method_models import classifier_model


class ClassifierAndRetrievalContractTests(unittest.TestCase):
    def test_equation_two_has_three_independent_feature_blocks(self):
        model = classifier_model()
        features = model.named_steps['features']
        self.assertEqual([name for name, _ in features.transformer_list], ['word', 'char', 'structure'])
        rows = ['Which film is older Alpha or Beta?', 'Why did Alpha change?', 'Who wrote Beta?'] * 2
        matrix = features.fit_transform(rows)
        blocks = dict(features.transformer_list)
        self.assertFalse(any('__' in w for w in blocks['word'].named_steps['tfidf'].vocabulary_))
        structural = blocks['structure']
        self.assertIn('__compare', structural.vectorizer_.vocabulary_)
        self.assertEqual(matrix.shape[1], sum(block.transform(rows).shape[1] for block in blocks.values()))
        self.assertIsNone(model.named_steps['classifier'].get_params()['random_state'])

    def test_structural_vocabulary_does_not_fit_heldout_queries(self):
        feature = QueryStructureFeatures().fit(['Who wrote the book?'])
        before = dict(feature.vectorizer_.vocabulary_)
        feature.transform(['Why was the film older?'])
        self.assertEqual(feature.vectorizer_.vocabulary_, before)
        self.assertNotIn('__compare', before)

    def test_fact_uses_only_rrf_candidates_before_cross_encoder(self):
        engine = DeepSeekRAGTool.__new__(DeepSeekRAGTool)
        engine.semantic_tool = SimpleNamespace(semantic_search=Mock(side_effect=AssertionError('extra Dense recall')))
        engine.bm25_tool = SimpleNamespace(keyword_search=Mock(return_value=['rrf-first', 'rrf-second']))
        rerank = Mock(side_effect=lambda pool, query, top_k: pool[:top_k])
        engine.rerank_tool = SimpleNamespace(rerank=rerank)
        engine._call_deepseek_llm = Mock(return_value=('entity', 20))
        with branch_environment({}):
            result = engine.execute_rag_flow('question', 'fact', context_top_k_override=3, gen_max_tokens_override=60)
        engine.semantic_tool.semantic_search.assert_not_called()
        self.assertEqual(rerank.call_args.args[0], ['rrf-first', 'rrf-second'])
        self.assertEqual(result['retrieved_context'], ['rrf-first', 'rrf-second'])

    def test_preserved_answer_is_scored_even_when_total_token_is_unknown(self):
        output = {'final_answer': 'candidate', 'metrics': {
            'token_cost': None, 'token_accounting_complete': False, 'known_token_cost': 90,
            'ircot_verifier_error': 'TokenUsageError'}}
        scores = {'correctness': 1, 'context_recall': 1, 'faithfulness': 1}
        record = record_branch(output, scores)
        table = summary([{**record, 'question_type': 'reasoning'}])['overall']
        self.assertEqual(table['ACC'], 100)
        self.assertIsNone(table['Token'])
        self.assertIsNone(table['ACC_per_1KToken'])
        self.assertFalse(table['Token_accounting_complete'])

    def test_reasoning_cost_merge_handles_verifier_failure(self):
        rr = SimpleNamespace(execute_rag_flow=lambda *a, **k: {
            'final_answer': 'unknown', 'metrics': {'token_cost': 30, 'retrieval_steps': 2}})
        ir = SimpleNamespace(run=lambda *a, **k: {'final_answer': 'candidate', 'metrics': {
            'token_cost': None, 'known_token_cost': 50, 'token_accounting_complete': False, 'retrieval_steps': 2}})
        config = json.loads(Path('configs/paper_v3.json').read_text())
        value, _ = MethodRuntime(rr, ir, config).reasoning('question', 'quality')
        self.assertEqual(value['final_answer'], 'candidate')
        self.assertIsNone(value['metrics']['token_cost'])
        self.assertEqual(value['metrics']['known_token_cost'], 80)


class CorpusMappingContractTests(unittest.TestCase):
    def chunks(self):
        docs = [{'doc_id': 'docA', 'title': 'Article A', 'source_dataset': 'fixture',
                 'paragraphs': [{'paragraph_id': 0, 'sentences': ['Intro sentence.', 'Alpha was born in Rome.']},
                                {'paragraph_id': 1, 'text': 'This happened because of flooding.'}]},
                {'doc_id': 'docB', 'title': 'Article B', 'text': 'Beta appeared in 1900.', 'source_dataset': 'fixture'}]
        return list(make_chunks(docs, 1000, 150))

    def test_chunks_include_paper_location_and_source_fields(self):
        rows = self.chunks()
        self.assertEqual(len(rows), 3)
        metadata = rows[0]['metadata']
        for name in ('paragraph_id', 'sentence_start', 'sentence_end', 'sentence_ids',
                     'sentence_spans', 'char_start', 'char_end', 'char_count', 'source_dataset'):
            self.assertIn(name, metadata)
        self.assertEqual(metadata['sentence_ids'], [0, 1])
        self.assertEqual(metadata['source_dataset'], 'fixture')

    def test_sentence_annotation_maps_to_existing_chunks_only(self):
        chunks = self.chunks()
        before = copy.deepcopy(chunks)
        refs = [{'title': 'Article A', 'text': 'Alpha was born in Rome.', 'sentence_ids': [1]}]
        evidence, missing = SupportMapper(chunks).map_all(refs)
        self.assertEqual(missing, [])
        self.assertEqual(evidence[0]['sentence_ids'], [1])
        self.assertEqual(evidence[0]['paragraph_id'], 0)
        self.assertEqual(evidence[0]['chunk_ids'], [chunks[0]['chunk_id']])
        self.assertEqual(chunks, before)

    def test_long_support_spanning_overlapping_chunks_is_mapped(self):
        text = 'Prefix. ' + ('a long supporting sentence ' * 10) + '. Ending.'
        chunks = list(make_chunks([{'doc_id': 'x', 'title': 'Long', 'text': text}], 60, 15))
        evidence, missing = SupportMapper(chunks).map_all([{'title': 'Long', 'text': 'a long supporting sentence ' * 5}])
        self.assertEqual(missing, [])
        self.assertGreater(len(evidence[0]['chunk_ids']), 1)
        self.assertLess(len(evidence[0]['chunk_ids']), len(chunks))

    def test_casefold_and_whitespace_offsets_are_not_raw_string_offsets(self):
        text, positions = normalized_positions('A  Straße\nRome.')
        self.assertEqual(text, 'a strasse rome.')
        i = text.index('strasse')
        self.assertEqual('A  Straße\nRome.'[positions[i]:positions[i + 6] + 1], 'Straße')

    def test_missing_support_is_rejected_without_modifying_corpus(self):
        chunks = self.chunks()
        before = copy.deepcopy(chunks)
        row = {'id': 'q', 'question': 'Which appeared earlier?', 'answer': 'A',
               'context': [['Article A', ['Missing sentence.']]], 'supporting_facts': [['Article A', 0]]}
        records = list(normalize_source([row], 'hotpotqa', 'validation', SupportMapper(chunks)))
        self.assertIsNone(records[0][0])
        self.assertIn('Incomplete', records[0][1]['reason'])
        self.assertEqual(chunks, before)

    def test_hotpot_and_2wiki_supporting_facts_formats(self):
        row = {'id': 'q', 'question': 'Who was born in Rome?', 'answer': 'Alpha',
               'context': {'title': ['Article A'], 'sentences': [['Intro sentence.', 'Alpha was born in Rome.']]},
               'supporting_facts': {'title': ['Article A'], 'sent_id': [1]}}
        for dataset in ('hotpotqa', '2wikimultihopqa'):
            result = list(normalize_source([row], dataset, 'validation', SupportMapper(self.chunks())))
            self.assertIsNone(result[0][1])
            self.assertEqual(result[0][0]['answers'], ['Alpha'])
            self.assertEqual(result[0][0]['supporting_evidence'][0]['sentence_ids'], [1])

    def test_musique_paragraph_support(self):
        row = {'id': 'q', 'question': 'When did Beta appear?', 'answer': '1900',
               'paragraphs': [{'idx': 0, 'title': 'Article B', 'paragraph_text': 'Beta appeared in 1900.', 'is_supporting': True}]}
        result = list(normalize_source([row], 'musique', 'dev', SupportMapper(self.chunks())))
        self.assertIsNone(result[0][1])
        self.assertEqual(result[0][0]['context_doc_ids'], ['docB'])

    def test_iirc_parent_article_and_value_answers(self):
        row = {'title': 'Article A', 'questions': [{'qid': 'q', 'question': 'Why did this happen?',
               'answer': {'type': 'value', 'answer_value': 'flooding'},
               'context': [{'passage': 'main', 'text': 'This happened because of flooding.'}]}]}
        result = list(normalize_source([row], 'iirc', 'dev', SupportMapper(self.chunks())))
        self.assertIsNone(result[0][1])
        self.assertEqual(result[0][0]['supporting_evidence'][0]['paragraph_id'], 1)

    def test_nq_approximate_answer_alignment_is_explicit(self):
        row = {'id': 'q', 'question': {'text': 'Where was Alpha born?'}, 'document': {'title': 'Article A'},
               'annotations': {'short_answers': [{'text': ['Rome']}], 'yes_no_answer': [-1]}}
        result = list(normalize_source([row], 'naturalquestions', 'validation', SupportMapper(self.chunks())))
        self.assertIsNone(result[0][1])
        self.assertEqual(result[0][0]['supporting_evidence'][0]['alignment'], 'approximate_answer_position')

    def test_nq_binary_answers_use_annotated_context_not_a_literal_yes_search(self):
        row = {'id': 'q', 'question': {'text': 'Was Alpha born in Rome?'},
               'document': {'title': 'Article A', 'tokens': {'token': ['Alpha', 'was', 'born', 'in', 'Rome.'], 'is_html': [False] * 5}},
               'annotations': {'short_answers': [], 'yes_no_answer': [1],
                               'long_answer': [{'start_token': 0, 'end_token': 5}]}}
        output = adapt_question(row, 'naturalquestions')
        self.assertEqual(output['answers'], ['yes'])
        self.assertEqual(output['references'][0]['text'], 'Alpha was born in Rome.')
        self.assertEqual(output['references'][0]['alignment'], 'approximate_context_position')

    def test_duplicate_paragraph_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            list(make_chunks([{'doc_id': 'x', 'paragraphs': [{'paragraph_id': 0, 'text': 'A.'},
                                                           {'paragraph_id': 0, 'text': 'B.'}]}]))

    def test_approximate_answer_does_not_match_inside_another_word(self):
        chunks = list(make_chunks([{'doc_id': 'x', 'title': 'Article', 'text': 'This happened because of rain.'}]))
        evidence, missing = SupportMapper(chunks).map_all([
            {'title': 'Article', 'text': 'US', 'alignment': 'approximate_answer_position'}])
        self.assertEqual(evidence, [])
        self.assertEqual(len(missing), 1)

    def test_link_titles_share_the_existing_catalog_normalization(self):
        evidence, missing = SupportMapper(self.chunks()).map_all([
            {'title': 'Article_A', 'text': 'Alpha was born in Rome.'}])
        self.assertEqual(missing, [])
        self.assertEqual(evidence[0]['doc_id'], 'docA')


if __name__ == '__main__':
    unittest.main()
