"""Behavioral checks for equations (4b), (6), (7) and Algorithm 1."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from reasonroute.compression import compress_each
from reasonroute.evidence import EvidenceCatalog
from reasonroute import retrieval
from reasonroute.runtime import MethodRuntime, branch_environment


class Hybrid:
    def __init__(self, docs=None):
        self.calls = []
        self.docs = docs

    def keyword_search(self, query, top_k):
        self.calls.append((query, top_k))
        return self.docs if self.docs is not None else [f'evidence {len(self.calls)}-{i}' for i in range(top_k)]


class Reranker:
    def __init__(self):
        self.calls = []

    def rerank(self, docs, query, top_k):
        self.calls.append((docs, query, top_k))
        return list(reversed(docs))[:top_k]


class PaperAlignmentTests(unittest.TestCase):
    def test_compression_restores_source_order(self):
        text = 'Alpha appears first. Alpha beta appears later.'
        self.assertEqual(compress_each([text], 'alpha beta', 2, 520), [text])

    def test_compression_ties_newlines_and_character_limit(self):
        self.assertEqual(compress_each(['First line\nSecond line\nThird line'], 'absent', 2, 520),
                         ['First line Second line'])
        self.assertEqual(len(compress_each(['a' * 600], 'query', 1, 420)[0]), 420)
        self.assertEqual(compress_each([''], 'query', 1, 420), [])

    def test_verifier_compression_scores_query_only(self):
        with branch_environment({'IRCOT_VERIFIER_SENTENCE_COMPRESSION': '1',
                                 'IRCOT_VERIFIER_SENTENCES_PER_DOC': '1', 'IRCOT_VERIFIER_SENTENCE_MAX_CHARS': '420'}):
            value = retrieval.InterleavedRetrieval.compress_verifier_context(
                'alpha', ['Alpha supports the question. Beta beta beta is an earlier answer.'], 'beta')
        self.assertEqual(value, ['Alpha supports the question.'])

    def test_explanation_reranks_each_search(self):
        hybrid, rerank = Hybrid(), Reranker()
        actions = [('SEARCH("part one")', 10), ('SEARCH("part two")', 10), ('ANSWER("entity")', 10)]
        with branch_environment({'IRCOT_STEP_RERANK': '1', 'IRCOT_STEP_CANDIDATE_TOP_K': '20',
                                 'IRCOT_STEP_TOP_K': '3', 'IRCOT_FINAL_CONTEXT_TOP_K': '9'}):
            with patch.object(retrieval, 'call_deepseek', side_effect=actions):
                output = retrieval.InterleavedRetrieval(hybrid, 3, rerank).run('question', False)
        self.assertEqual([k for _, k in hybrid.calls], [20, 20])
        self.assertEqual(len(rerank.calls), 2)
        self.assertEqual([call[2] for call in rerank.calls], [3, 3])
        self.assertEqual(output['retrieved_context'][0], 'evidence 1-19')
        self.assertEqual(len(output['retrieved_context']), 6)

    def test_reranking_configuration_cannot_silently_skip_model(self):
        with branch_environment({'IRCOT_STEP_RERANK': '1'}):
            with self.assertRaises(ValueError):
                retrieval.InterleavedRetrieval(Hybrid()).run('question', False)

    def test_final_generation_receives_history_and_returns_same_context(self):
        calls = []
        def final(query, context, system, temperature=0.1, history=None):
            calls.append((context, history))
            return 'entity', 20
        with branch_environment({'IRCOT_FINAL_CONTEXT_TOP_K': '5', 'IRCOT_STEP_TOP_K': '3'}):
            with patch.object(retrieval, 'call_deepseek', return_value=('SEARCH("next")', 10)), \
                 patch.object(retrieval, 'generate_answer', side_effect=final):
                output = retrieval.InterleavedRetrieval(Hybrid(), 3).run('question', False)
        self.assertEqual(output['retrieved_context'], calls[0][0])
        self.assertEqual(len(output['retrieved_context']), 5)
        self.assertEqual(sum(m['role'] == 'assistant' for m in calls[0][1]), 3)
        self.assertEqual(output['metrics']['token_cost'], 50)
        self.assertEqual(output['metrics']['context_length'], sum(map(len, output['retrieved_context'])))

    def test_generate_answer_transmits_history_messages(self):
        history = [{'role': 'system', 'content': 'old system'},
                   {'role': 'assistant', 'content': 'SEARCH("bridge")'},
                   {'role': 'user', 'content': 'retrieved first evidence'}]
        with patch.object(retrieval, 'call_deepseek', return_value=('entity', 10)) as call:
            retrieval.generate_answer('question', ['final evidence'], 'new system', history=history)
        messages = call.call_args.args[0]
        self.assertEqual(messages[0], {'role': 'system', 'content': 'new system'})
        self.assertEqual(messages[1:3], history[1:])
        self.assertIn('final evidence', messages[-1]['content'])

    def test_aggregate_dedup_first_occurrence_and_top_five(self):
        hybrid = Hybrid(['duplicate', 'other', 'duplicate'])
        with patch.object(retrieval, 'call_deepseek', return_value=('SEARCH("next")', 1)), \
             patch.object(retrieval, 'generate_answer', return_value=('entity', 2)):
            output = retrieval.InterleavedRetrieval(hybrid, 3).run('question', False)
        self.assertEqual(output['retrieved_context'], ['duplicate', 'other'])

    def test_immediate_answer_has_empty_aggregate_and_no_verifier(self):
        hybrid = Hybrid()
        with patch.object(retrieval, 'call_deepseek', return_value=('ANSWER("entity")', 10)):
            output = retrieval.InterleavedRetrieval(hybrid, 3).run('question', True)
        self.assertEqual(output['retrieved_context'], [])
        self.assertEqual(hybrid.calls, [])
        self.assertFalse(output['metrics']['ircot_answer_verifier'])

    def test_compact_explanation_returns_compressed_top_five(self):
        docs = [f'Alpha {i}. Unrelated filler sentence. More filler.' for i in range(9)]
        with branch_environment({'IRCOT_FINAL_CONTEXT_TOP_K': '5', 'IRCOT_FINAL_SENTENCE_COMPRESSION': '1',
                                 'IRCOT_FINAL_SENTENCES_PER_DOC': '2', 'IRCOT_FINAL_SENTENCE_MAX_CHARS': '25'}):
            with patch.object(retrieval, 'call_deepseek', return_value=('SEARCH("alpha")', 10)), \
                 patch.object(retrieval, 'generate_answer', return_value=('entity', 20)):
                output = retrieval.InterleavedRetrieval(Hybrid(docs), 3).run('alpha', False)
        self.assertEqual(len(output['retrieved_context']), 5)
        self.assertTrue(all(len(text) <= 25 for text in output['retrieved_context']))

    def test_verifier_failure_keeps_candidate(self):
        actions = [('SEARCH("next")', 10)] * 3 + [RuntimeError('mock verifier failure')]
        with branch_environment({'IRCOT_FINAL_CONTEXT_TOP_K': '5', 'IRCOT_ANSWER_VERIFIER_POLICY': 'fallback'}):
            with patch.object(retrieval, 'call_deepseek', side_effect=actions), \
                 patch.object(retrieval, 'generate_answer', return_value=('candidate', 20)):
                output = retrieval.InterleavedRetrieval(Hybrid(), 3).run('question', True)
        self.assertEqual(output['final_answer'], 'candidate')
        self.assertEqual(output['metrics']['ircot_verifier_error'], 'RuntimeError')
        self.assertEqual(output['metrics']['token_cost'], 50)

    def test_empty_verifier_keeps_candidate_and_counts_tokens(self):
        actions = [('SEARCH("next")', 10)] * 3 + [('', 7)]
        with branch_environment({'IRCOT_ANSWER_VERIFIER_POLICY': 'fallback'}):
            with patch.object(retrieval, 'call_deepseek', side_effect=actions), \
                 patch.object(retrieval, 'generate_answer', return_value=('candidate', 20)):
                output = retrieval.InterleavedRetrieval(Hybrid(), 3).run('question', True)
        self.assertEqual(output['final_answer'], 'candidate')
        self.assertEqual(output['metrics']['token_cost'], 57)

    def test_catalog_expands_real_neighbors_and_explicit_links(self):
        rows = [{'doc_id': 'A', 'title': 'Article A', 'content': f'paragraph {i}',
                 'metadata': {'paragraph_id': i}, 'links': ['Article B'] if i == 1 else []}
                for i in range(4)]
        rows.append({'doc_id': 'B', 'title': 'Article B', 'content': 'linked paragraph'})
        expanded = EvidenceCatalog(rows).expand(['paragraph 1'])
        self.assertEqual(expanded, ['paragraph 0', 'paragraph 2', 'linked paragraph'])
        self.assertNotIn('paragraph 3', expanded)

    def test_explanation_fallback_uses_primary_evidence(self):
        from test_method_protocol import FakeRR, FakeIRCoT, FixedModel
        rr, ir = FakeRR(), FakeIRCoT()
        original = ir.run
        def failure(question, enable_answer_verifier=None):
            result = original(question, enable_answer_verifier)
            result['final_answer'] = 'unknown'
            return result
        ir.run = failure
        config = json.loads(Path('configs/paper_v3.json').read_text())
        runtime = MethodRuntime(rr, ir, config, {'model': FixedModel('explanation')})
        output, route = runtime.run('question', 'quality')
        self.assertEqual(rr.calls[0][2]['expansion_seed_context'], ['evidence'])
        self.assertEqual(output['metrics']['token_cost'], 300)
        self.assertEqual(output['metrics']['context_length'], 8)


if __name__ == '__main__':
    unittest.main()
