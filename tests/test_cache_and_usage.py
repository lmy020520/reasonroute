"""Offline coverage of cache identity and strict generation token accounting."""
import json
import os
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from reasonroute.cache import documents_sha256, write_embedding_metadata, validate_embedding_metadata
from reasonroute.llm_response import parse_llm_reply, read_llm_reply, TokenUsageError, LLMResponseError
from reasonroute import retrieval
from reasonroute.runtime import branch_environment
from tools.rag_resarch_tool import DeepSeekRAGTool
from scripts.run_reasonroute import record_branch


def reply(content='entity', tokens=12):
    return {'choices': [{'message': {'content': content}}], 'usage': {'total_tokens': tokens}}


class Response:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'vectors.npz'
        self.docs = ['alpha common sentence', 'beta common sentence']
        self.model = SimpleNamespace(model_name='BAAI/bge-m3')

    def build(self, values=None):
        np.savez_compressed(self.path, embeddings=np.eye(2, dtype='float32') if values is None else values)
        write_embedding_metadata(self.path, self.docs, self.model.model_name, 'kb-hash', '0')

    def test_document_hash_is_ordered_and_boundary_safe(self):
        self.assertNotEqual(documents_sha256(self.docs), documents_sha256(self.docs[::-1]))
        self.assertNotEqual(documents_sha256(['ab', 'c']), documents_sha256(['a', 'bc']))

    def test_embedding_metadata_is_mandatory(self):
        np.savez_compressed(self.path, embeddings=np.eye(2))
        with self.assertRaises(ValueError):
            validate_embedding_metadata(self.path, self.docs, self.model.model_name, '0')

    def test_embedding_same_count_wrong_content_or_order_fails(self):
        self.build()
        for docs in (self.docs[::-1], ['changed common sentence', self.docs[1]]):
            with self.subTest(docs=docs), self.assertRaises(ValueError):
                validate_embedding_metadata(self.path, docs, self.model.model_name, '0')

    def test_embedding_model_title_and_kb_must_match(self):
        self.build()
        for model, title, kb in [('other-model', '0', 'kb-hash'),
                                 (self.model.model_name, '1', 'kb-hash'),
                                 (self.model.model_name, '0', 'other-kb')]:
            with self.subTest(model=model, title=title, kb=kb), self.assertRaises(ValueError):
                validate_embedding_metadata(self.path, self.docs, model, title, kb)

    def test_embedding_byte_changes_detected(self):
        self.build()
        with self.path.open('ab') as handle:
            handle.write(b'changed-bytes')
        with self.assertRaises(ValueError):
            validate_embedding_metadata(self.path, self.docs, self.model.model_name, '0')

    def test_semantic_cache_loader_accepts_valid_cache_without_faiss(self):
        self.build()
        with patch.object(retrieval, 'INDEX_CACHE_FILE', str(self.path)), \
             patch.dict(os.environ, {'DOC_TITLE_CONTEXT': '0', 'USE_FAISS_INDEX': '0'}):
            tool = retrieval.build_semantic_tool(self.docs, self.model, kb_sha256='kb-hash')
        self.assertEqual(tool.embeddings.shape, (2, 2))
        self.assertIsNone(tool.index)

    def test_nonfinite_embedding_values_rejected(self):
        self.build(np.full((2, 2), np.nan))
        with patch.object(retrieval, 'INDEX_CACHE_FILE', str(self.path)), \
             patch.dict(os.environ, {'DOC_TITLE_CONTEXT': '0'}), self.assertRaises(ValueError):
            retrieval.build_semantic_tool(self.docs, self.model)

    def test_bm25_matching_cache_reused_but_wrong_order_fails(self):
        cache = Path(self.temp.name) / 'tokens.pkl'
        with patch.object(retrieval, 'TOKENIZED_CACHE_FILE', str(cache)):
            first = retrieval.build_hybrid_tool(self.docs, self.model, np.eye(2))
            snapshot = cache.read_bytes()
            second = retrieval.build_hybrid_tool(self.docs, self.model, np.eye(2))
            self.assertEqual(first.tokenized_corpus, second.tokenized_corpus)
            self.assertEqual(snapshot, cache.read_bytes())
            with self.assertRaises(ValueError):
                retrieval.build_hybrid_tool(self.docs[::-1], self.model, np.eye(2))
            with self.assertRaises(ValueError):
                retrieval.build_hybrid_tool(['other common text', self.docs[1]], self.model, np.eye(2))

    def test_bm25_legacy_and_changed_tokenizer_rejected(self):
        cache = Path(self.temp.name) / 'tokens.pkl'
        with cache.open('wb') as handle:
            pickle.dump([['alpha'], ['beta']], handle)
        with patch.object(retrieval, 'TOKENIZED_CACHE_FILE', str(cache)), self.assertRaises(ValueError):
            retrieval.build_hybrid_tool(self.docs, self.model, np.eye(2))
        cache.unlink()
        with patch.object(retrieval, 'TOKENIZED_CACHE_FILE', str(cache)):
            retrieval.build_hybrid_tool(self.docs, self.model, np.eye(2))
            with patch.object(retrieval, 'version', return_value='changed'), self.assertRaises(ValueError):
                retrieval.build_hybrid_tool(self.docs, self.model, np.eye(2))

    def test_bm25_corrupted_token_payload_rejected(self):
        cache = Path(self.temp.name) / 'tokens.pkl'
        with patch.object(retrieval, 'TOKENIZED_CACHE_FILE', str(cache)):
            retrieval.build_hybrid_tool(self.docs, self.model, np.eye(2))
            with cache.open('rb') as handle:
                data = pickle.load(handle)
            data['tokenized_corpus'][0] = ['corrupted']
            with cache.open('wb') as handle:
                pickle.dump(data, handle)
            with self.assertRaises(ValueError):
                retrieval.build_hybrid_tool(self.docs, self.model, np.eye(2))


class TokenUsageTests(unittest.TestCase):
    def engine(self):
        engine = DeepSeekRAGTool.__new__(DeepSeekRAGTool)
        engine.api_key = 'test-only'
        engine.api_url = 'https://example.invalid/chat'
        engine.model_name = 'test-model'
        return engine

    def test_usage_missing_or_invalid_never_defaults_to_zero(self):
        cases = [None, {}, {'usage': None}, {'usage': {}}, reply(tokens=0), reply(tokens=-1),
                 reply(tokens=True), reply(tokens='12'), reply(tokens=12.5)]
        for data in cases:
            with self.subTest(data=data), self.assertRaises(TokenUsageError):
                parse_llm_reply(data)
        self.assertEqual(parse_llm_reply(reply(content='')), ('', 12))

    def test_bad_content_is_an_explicit_protocol_error(self):
        with self.assertRaises(LLMResponseError):
            parse_llm_reply({'usage': {'total_tokens': 10}})

    def test_non_json_replies_are_fatal_protocol_errors(self):
        response = SimpleNamespace(json=lambda: json.loads('not JSON'))
        with self.assertRaises(LLMResponseError):
            read_llm_reply(response)

    def test_failed_verifier_is_not_reported_as_complete_cost(self):
        record = record_branch({'final_answer': 'candidate', 'metrics': {
            'token_cost': None, 'token_accounting_complete': False,
            'known_token_cost': 50, 'ircot_verifier_error': 'Timeout'}}, {})
        self.assertEqual(record['prediction'], 'candidate')
        self.assertIsNone(record['token_cost'])

    def test_optional_helpers_do_not_turn_network_failures_into_zero_usage(self):
        engine = self.engine()
        with patch('requests.post', side_effect=RuntimeError('test network failure')):
            for call in (lambda: engine._rescue_reasoning_answer('q', ['evidence'], 'answer'),
                         lambda: engine._plan_ircot_lite_next_query('q', ['evidence'], 1),
                         lambda: engine._decompose_query('q')):
                with self.assertRaises(RuntimeError):
                    call()

    def test_interleaved_missing_usage_does_not_retry(self):
        with patch.object(retrieval, 'DEEPSEEK_API_KEY', 'test-only'), \
             patch.object(retrieval.requests, 'post', return_value=Response({'choices': reply()['choices']})) as post, \
             patch.object(retrieval.time, 'sleep') as sleep:
            with self.assertRaises(TokenUsageError):
                retrieval.call_deepseek([{'role': 'user', 'content': 'question'}])
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    def test_fact_missing_usage_does_not_retry_or_return_zero(self):
        with patch('requests.post', return_value=Response({'choices': reply()['choices']})) as post:
            with self.assertRaises(TokenUsageError):
                self.engine()._call_deepseek_llm('question', ['evidence'], 'instruction')
        self.assertEqual(post.call_count, 1)

    def test_optional_generation_helpers_do_not_swallow_accounting_errors(self):
        engine = self.engine()
        calls = [lambda: engine._rescue_reasoning_answer('question', ['evidence'], 'answer'),
                 lambda: engine._plan_ircot_lite_next_query('question', ['evidence'], 1),
                 lambda: engine._decompose_query('question')]
        with patch('requests.post', return_value=Response({'choices': reply()['choices']})):
            for call in calls:
                with self.subTest(call=call), self.assertRaises(TokenUsageError):
                    call()

    def test_verifier_missing_usage_preserves_answer_without_inventing_cost(self):
        hybrid = SimpleNamespace(keyword_search=lambda *args, **kwargs: ['evidence'])
        actions = [('SEARCH("next")', 10)] * 3 + [TokenUsageError('missing usage')]
        with branch_environment({'IRCOT_ANSWER_VERIFIER_POLICY': 'fallback'}), \
             patch.object(retrieval, 'call_deepseek', side_effect=actions), \
             patch.object(retrieval, 'generate_answer', return_value=('candidate', 20)):
            result = retrieval.InterleavedRetrieval(hybrid, 3).run('question', True)
        self.assertEqual(result['final_answer'], 'candidate')
        self.assertIsNone(result['metrics']['token_cost'])
        self.assertEqual(result['metrics']['known_token_cost'], 50)
        self.assertFalse(result['metrics']['token_accounting_complete'])

    def test_valid_usage_passes_through_both_generation_clients(self):
        with patch('requests.post', return_value=Response(reply(tokens=23))), \
             patch.object(retrieval, 'DEEPSEEK_API_KEY', 'test-only'):
            self.assertEqual(retrieval.call_deepseek([]), ('entity', 23))
            self.assertEqual(self.engine()._call_deepseek_llm('question', ['evidence'], 'instruction'), ('entity', 23))


if __name__ == '__main__':
    unittest.main()
