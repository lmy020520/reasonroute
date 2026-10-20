"""Interleaved retrieval used internally by ReasonRoute.

Extracted from the historical shared module without external baseline runners.
"""
import json
import os
import pickle
import re
import threading
import time
import inspect
from importlib.metadata import version
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import requests
from tqdm import tqdm
from evaluate_rag import API_KEY, LLM_MODEL_NAME, REQUEST_PROXIES, SiliconFlowEmbedding
from tools.rag_resarch_tool import SemanticTool, HybridSearchTool, MultiVectorTool
from reasonroute.compression import compress_each
from reasonroute.llm_response import read_llm_reply, LLMResponseError
from reasonroute.cache import documents_sha256, validate_embedding_metadata
from reasonroute.datasets import digest_json
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", API_KEY).strip()
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
INDEX_CACHE_FILE = os.getenv("INDEX_CACHE_FILE", "data/public_eval/normalized/reclassified_exp500/kb_embeddings_cache_exp500.npz")
TOKENIZED_CACHE_FILE = os.getenv("TOKENIZED_CACHE_FILE", "data/public_eval/normalized/reclassified_exp500/kb_tokenized_corpus_cache_exp500.pkl")


def _pack(context, answer, tokens, steps, extra=None):
        return {
            "retrieved_context": context,
            "final_answer": answer,
            "metrics": {
                "retrieval_steps": steps,
                "context_length": sum(len(str(x)) for x in context),
                "token_cost": tokens,
                **(extra or {}),
            },
        }

def call_deepseek(messages: List[dict], temperature: float = 0.1, max_tokens: int = 260) -> Tuple[str, int]:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is not set.")
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
    retry_max_sleep = float(os.getenv("LLM_RETRY_MAX_SLEEP", "8"))
    timeout = float(os.getenv("LLM_API_TIMEOUT", "60"))
    if max_retries < 1:
        raise ValueError('LLM_MAX_RETRIES must be at least 1')
    for attempt in range(max_retries):
        try:
            response = requests.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
                proxies=REQUEST_PROXIES,
            )
            response.raise_for_status()
            return read_llm_reply(response)
        except LLMResponseError:
            # A response was already received; retrying would add untracked cost.
            raise
        except Exception:
            if attempt + 1 >= max_retries:
                raise
            time.sleep(min(2 ** (attempt + 1), retry_max_sleep))
    return "", 0

def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output

def build_semantic_tool(docs: List[str], model: SiliconFlowEmbedding, kb_sha256=None) -> SemanticTool:
    validate_embedding_metadata(INDEX_CACHE_FILE, docs, model.model_name,
                                os.getenv('DOC_TITLE_CONTEXT', '0'), kb_sha256)
    with np.load(INDEX_CACHE_FILE, allow_pickle=False) as cache_data:
        embeddings = cache_data['embeddings']
    if embeddings.ndim != 2:
        raise ValueError('Embedding cache must contain a two-dimensional matrix')
    if len(embeddings) != len(docs) or embeddings.shape[1] < 1:
        raise ValueError(f"semantic cache size {len(embeddings)} != docs size {len(docs)}")
    if not np.isfinite(embeddings).all():
        raise ValueError('Embedding cache contains non-finite values')

    use_faiss = os.getenv("USE_FAISS_INDEX", "0").strip().lower() not in {"0", "false", "no"}
    index = None
    embeddings = embeddings.astype("float32", copy=False)
    if use_faiss:
        import faiss
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

    tool = SemanticTool.__new__(SemanticTool)
    tool.corpus = docs
    tool.model = model
    tool.device = None
    tool.embeddings = embeddings
    tool.index = index
    return tool

def build_hybrid_tool(docs: List[str], model: SiliconFlowEmbedding, embeddings: np.ndarray) -> HybridSearchTool:
    from rank_bm25 import BM25Okapi

    tool = HybridSearchTool.__new__(HybridSearchTool)
    tool.corpus = docs
    tool.encoder = model
    tool.corpus_embeddings = embeddings
    tool.rrf_k = 60
    tool.stop_words = tool._load_stopwords()
    profile = {'format': 'reasonroute-bm25-v2', 'rows': len(docs),
               'documents_sha256': documents_sha256(docs),
               'tokenizer_sha256': digest_json({'code': inspect.getsource(HybridSearchTool._clean_tokenize),
                                                'stop_words': sorted(tool.stop_words),
                                                'jieba_version': version('jieba')})}
    cache_path = Path(TOKENIZED_CACHE_FILE)
    if cache_path.is_file():
        with cache_path.open('rb') as f:
            cached = pickle.load(f)
        if not isinstance(cached, dict) or any(cached.get(k) != v for k, v in profile.items()):
            raise ValueError('BM25 cache does not match the ordered corpus or tokenizer; choose a new cache path')
        tokenized_corpus = cached.get('tokenized_corpus')
        if not isinstance(tokenized_corpus, list) or len(tokenized_corpus) != len(docs) or any(
                not isinstance(row, list) or any(not isinstance(t, str) for t in row) for row in tokenized_corpus):
            raise ValueError('Invalid tokenized corpus cache')
        if cached.get('tokenized_sha256') != digest_json(tokenized_corpus):
            raise ValueError('BM25 cache token checksum mismatch; choose a new cache path')
    else:
        tokenized_corpus = None
    if tokenized_corpus is None:
        tokenized_corpus = [tool._clean_tokenize(doc) for doc in tqdm(docs, desc="BM25 tokenize", ncols=100)]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open('xb') as f:
            pickle.dump({**profile, 'tokenized_corpus': tokenized_corpus,
                         'tokenized_sha256': digest_json(tokenized_corpus)}, f)
    tool.tokenized_corpus = tokenized_corpus
    tool.bm25 = BM25Okapi(tool.tokenized_corpus)
    tool.index = None
    return tool

def build_mvector_tool(docs: List[str], model: SiliconFlowEmbedding, semantic: SemanticTool) -> MultiVectorTool:
    tool = MultiVectorTool.__new__(MultiVectorTool)
    tool.corpus = docs
    tool.model = model
    tool.device = None
    tool.embeddings = semantic.embeddings
    tool.index = semantic.index
    tool.stop_words = set()
    tool.rrf_k = 60
    tool.tokenized_corpus = []
    tool.bm25 = None
    return tool

def generate_answer(query: str, context: List[str], system_prompt: str, temperature: float = 0.1, history=None) -> Tuple[str, int]:
    context_text = "\n".join([f"Passage {i + 1}: {doc}" for i, doc in enumerate(context)])
    user_prompt = f"Reference passages:\n{context_text}\n\nQuestion:\n{query}\n\nAnswer using only the passages."
    max_tokens = int(os.getenv("GENERATE_ANSWER_MAX_TOKENS", "260"))
    return call_deepseek(
        [{"role": "system", "content": system_prompt}]
        + [dict(message) for message in (history or []) if message.get('role') != 'system']
        + [{"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )

class InterleavedRetrieval:
    name = "ircot"

    def __init__(self, hybrid: HybridSearchTool, max_steps: int = 3, reranker=None):
        self.hybrid = hybrid
        self.max_steps = max_steps
        self.reranker = reranker

    @staticmethod
    def _clean_final_answer(answer: str) -> str:
        answer = (answer or "").strip()
        for marker in ("Final answer:", "Final Answer:", "FINAL ANSWER:", "最终答案：", "最终答案:"):
            if marker in answer:
                answer = answer.split(marker)[-1].strip()
        return answer.replace("\n", "  ").strip()

    @staticmethod
    def _tokenize_for_overlap(text: str) -> set:
        return {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", text or "")
            if len(token) > 2
        }

    @classmethod
    def _compress_context_sentences(
        cls,
        query_text: str,
        context: List[str],
        max_sentences: int,
        max_chars: int,
    ) -> List[str]:
        return compress_each(context, query_text, max_sentences, max_chars)

    @classmethod
    def compress_verifier_context(cls, query: str, context: List[str], answer: str) -> List[str]:
        if os.getenv("IRCOT_VERIFIER_SENTENCE_COMPRESSION", "0").strip().lower() in {"0", "false", "no"}:
            return context
        return cls._compress_context_sentences(
            query,
            context,
            max_sentences=int(os.getenv("IRCOT_VERIFIER_SENTENCES_PER_DOC", "2")),
            max_chars=int(os.getenv("IRCOT_VERIFIER_SENTENCE_MAX_CHARS", "520")),
        )

    @classmethod
    def compress_step_context(cls, query: str, context: List[str]) -> List[str]:
        if os.getenv("IRCOT_STEP_SENTENCE_COMPRESSION", "0").strip().lower() in {"0", "false", "no"}:
            return context
        return cls._compress_context_sentences(
            query,
            context,
            max_sentences=int(os.getenv("IRCOT_STEP_SENTENCES_PER_DOC", "2")),
            max_chars=int(os.getenv("IRCOT_STEP_SENTENCE_MAX_CHARS", "600")),
        )

    @classmethod
    def compress_final_context(cls, query: str, context: List[str]) -> List[str]:
        if os.getenv("IRCOT_FINAL_SENTENCE_COMPRESSION", "0").strip().lower() in {"0", "false", "no"}:
            return context
        return cls._compress_context_sentences(
            query,
            context,
            max_sentences=int(os.getenv("IRCOT_FINAL_SENTENCES_PER_DOC", "2")),
            max_chars=int(os.getenv("IRCOT_FINAL_SENTENCE_MAX_CHARS", "600")),
        )

    def verify_final_answer(self, query: str, context: List[str], answer: str) -> Tuple[str, int]:
        if not context:
            return answer, 0
        context = self.compress_verifier_context(query, context, answer)
        context_text = "\n".join([f"[{i + 1}] {doc}" for i, doc in enumerate(context)])
        prompt = (
            "You are an evidence-only answer verifier for multi-hop QA.\n"
            "Check whether the current answer directly satisfies the question using only the passages.\n"
            "If it is correct, output it unchanged. If it is wrong, incomplete, too broad, or based on outside knowledge, "
            "replace it with the shortest answer supported by the passages.\n"
            "For comparison questions, compare the relevant dates, ranks, quantities, locations, or relations in the passages.\n"
            "For temporal words such as current or present, use the passages rather than today's facts.\n"
            "Output only the final answer span, with no explanation.\n\n"
            f"Question: {query}\n"
            f"Current answer: {answer}\n"
            f"Passages:\n{context_text}"
        )
        max_tokens = int(os.getenv("IRCOT_VERIFIER_MAX_TOKENS", "120"))
        verified, tokens = call_deepseek([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=max_tokens)
        verified = self._clean_final_answer(verified)
        return verified or answer, tokens

    def extract_answer_from_evidence(self, query: str, context: List[str]) -> Tuple[str, int]:
        """Generate a fresh short answer from compact evidence without anchoring on an old answer."""
        if not context:
            return "", 0
        context = self.compress_verifier_context(query, context, "")
        context_text = "\n".join([f"[{i + 1}] {doc}" for i, doc in enumerate(context)])
        prompt = (
            "Answer the question using only the evidence below. The earlier pipeline could not produce "
            "a usable answer, so infer the shortest supported entity, date, number, title, location, or phrase. "
            "For multi-hop questions, connect the minimum necessary evidence and perform any explicit "
            "comparison required by the question. Do not explain your reasoning and do not output SEARCH. "
            "Output only the final answer span; output unknown only when no candidate is present.\n\n"
            f"Question: {query}\nEvidence:\n{context_text}"
        )
        max_tokens = int(os.getenv("IRCOT_FRESH_ANSWER_MAX_TOKENS", "80"))
        answer, tokens = call_deepseek([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=max_tokens)
        return self._clean_final_answer(answer), tokens

    def run(self, query: str, enable_answer_verifier: bool = None) -> Dict:
        step_top_k = int(os.getenv("IRCOT_STEP_TOP_K", "3"))
        final_context_top_k = int(os.getenv("IRCOT_FINAL_CONTEXT_TOP_K", "9"))
        rerank_steps = os.getenv('IRCOT_STEP_RERANK', '0') == '1'
        candidate_k = int(os.getenv('IRCOT_STEP_CANDIDATE_TOP_K', '20'))
        if rerank_steps and (self.reranker is None or candidate_k < step_top_k):
            raise ValueError('Explanation requires a reranker and candidate pool at least as large as step Top-k')
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an interleaved retrieval QA agent. At each step output exactly one action: "
                    "SEARCH(\"short query\") or ANSWER(\"final answer\"). Use concise search planning. "
                    "Use only retrieved passages, never outside knowledge. Temporal words such as current, "
                    "present, older, earlier, or higher must be interpreted from retrieved evidence rather "
                    "than today's real-world facts. When answering, output the shortest supported entity, "
                    "date, number, option, or phrase."
                ),
            },
            {"role": "user", "content": f"Question: {query}"},
        ]
        all_docs = []
        total_tokens = 0
        steps = 0
        final_answer = ""
        last_response = ""
        for _ in range(self.max_steps):
            step_max_tokens = int(os.getenv("IRCOT_STEP_MAX_TOKENS", "180"))
            response, tokens = call_deepseek(messages, temperature=0.1, max_tokens=step_max_tokens)
            total_tokens += tokens
            last_response = response
            messages.append({"role": "assistant", "content": response})
            answer_match = re.search(r'ANSWER\s*\(\s*["\']?(.*?)["\']?\s*\)', response, re.I | re.S)
            if answer_match:
                final_answer = self._clean_final_answer(answer_match.group(1))
                break
            search_match = re.search(r'SEARCH\s*\(\s*["\'](.*?)["\']\s*\)', response, re.I | re.S)
            search_query = search_match.group(1).strip() if search_match else query
            docs = self.hybrid.keyword_search(search_query, top_k=candidate_k if rerank_steps else step_top_k)
            if rerank_steps:
                docs = self.reranker.rerank(docs, search_query, top_k=step_top_k)
            all_docs.extend(docs)
            steps += 2
            prompt_docs = self.compress_step_context(search_query, docs)
            docs_text = "\n".join([f"[{i + 1}] {doc}" for i, doc in enumerate(prompt_docs)])
            messages.append({"role": "user", "content": f"Retrieved passages:\n{docs_text}\n\nContinue with SEARCH or ANSWER."})
        all_docs = dedupe_keep_order(all_docs)
        final_context = self.compress_final_context(query, all_docs[:final_context_top_k])
        final_fallback_used = False
        if not final_answer:
            final_answer, final_tokens = generate_answer(
                query,
                final_context,
                (
                    "You are the final answer stage of an interleaved retrieval QA agent. "
                    "The previous steps may have produced SEARCH actions; do not output SEARCH. "
                    "Use only the retrieved passages and provide only the shortest supported final answer. "
                    "Do not use outside knowledge. For temporal or comparison questions, decide from the "
                    "passages rather than today's real-world facts. If multiple candidates are mentioned, "
                    "choose the one directly satisfying the question. If the passages are truly insufficient, "
                    "answer unknown."
                ),
                temperature=0.0,
                history=messages,
            )
            total_tokens += final_tokens
            final_fallback_used = True
            if not final_answer.strip():
                final_answer = last_response.strip()
        final_answer = self._clean_final_answer(final_answer)
        verifier_used = False
        verifier_error = None
        if enable_answer_verifier is None:
            enable_answer_verifier = os.getenv("IRCOT_ENABLE_ANSWER_VERIFIER", "0").strip().lower() not in {"0", "false", "no"}
        verifier_policy = os.getenv("IRCOT_ANSWER_VERIFIER_POLICY", "fallback").strip().lower()
        uncertain_answer = (
            not final_answer.strip()
            or "search(" in final_answer.lower()
            or "unknown" in final_answer.lower()
            or "insufficient" in final_answer.lower()
            or "证据不足" in final_answer
            or "资料不足" in final_answer
            or len(final_answer) > int(os.getenv("IRCOT_VERIFIER_LONG_ANSWER_CHARS", "220"))
        )
        should_verify = False
        if enable_answer_verifier:
            if verifier_policy == "always":
                should_verify = True
            elif verifier_policy == "fallback":
                should_verify = final_fallback_used
            elif verifier_policy == "uncertain":
                should_verify = uncertain_answer
            elif verifier_policy in {"fallback_bad", "fallback_and_uncertain"}:
                should_verify = final_fallback_used and uncertain_answer
            elif verifier_policy in {"fallback_or_uncertain", "conditional"}:
                should_verify = final_fallback_used or uncertain_answer
            else:
                should_verify = True
        if should_verify:
            verifier_used = True
            try:
                verified_answer, verify_tokens = self.verify_final_answer(query, final_context, final_answer)
                total_tokens += verify_tokens
                if verified_answer:
                    final_answer = verified_answer
            except LLMResponseError:
                raise
            except Exception as exc:
                verifier_error = type(exc).__name__
        return _pack(
            final_context,
            final_answer.strip(),
            total_tokens,
            max(steps, 1),
            {
                "ircot_final_fallback": final_fallback_used,
                "ircot_answer_verifier": verifier_used,
                "ircot_verifier_error": verifier_error,
                "ircot_search_calls": steps // 2,
                "ircot_aggregate_items": len(all_docs),
                "ircot_step_rerank": rerank_steps,
            },
        )
