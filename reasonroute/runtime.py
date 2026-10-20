"""Sequential predicted-type runtime and reusable paired reasoning branches."""
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import threading

from reasonroute.datasets import digest_file, digest_json, load_artifact, read_rows, verify_model_data
from reasonroute.cache import title_setting


ENV_LOCK = threading.RLock()
PREFIXES = ('FACT_', 'EXPLANATION_', 'REASONING_', 'IRCOT_', 'CHEAP_COMPLEX_',
            'SELECTIVE_', 'NO_EVIDENCE_', 'ROUTE_')
EXACT = {'LOW_BUDGET', 'GEN_MAX_TOKENS', 'GENERATE_ANSWER_MAX_TOKENS',
         'DISABLE_EVIDENCE_COMPRESSION', 'ALWAYS_REASONING_DIRECT_IRCOT'}


@contextmanager
def branch_environment(values):
    # Historical helpers read per-call environment knobs; serialize their use.
    with ENV_LOCK:
        keys = {k for k in os.environ if k.startswith(PREFIXES) or k in EXACT} | set(values)
        previous = {key: os.environ.get(key) for key in keys}
        for key in keys:
            os.environ.pop(key, None)
        os.environ.update(values)
        try:
            yield
        finally:
            for key in keys:
                os.environ.pop(key, None)
                if previous[key] is not None:
                    os.environ[key] = previous[key]


class ConstantClassifier:
    def __init__(self, label):
        self.label = label

    def classify(self, question):
        return self.label


class MethodRuntime:
    def __init__(self, rr, interleaved, config, classifier=None, gate=None):
        self.rr, self.interleaved, self.config = rr, interleaved, config
        self.classifier, self.gate = classifier, gate

    def values(self, mode, route):
        return {**self.config['common'], **self.config[mode][route]}

    def reasoning(self, question, mode, allow_high=True, initial=None):
        from reasonroute.branch_flow import ReasonRouteIRCoTAssisted
        with branch_environment(self.values(mode, 'reasoning')):
            first = deepcopy(initial) if initial is not None else self.rr.execute_rag_flow(
                question, 'fact', answer_style='reasoning', context_top_k_override=4, gen_max_tokens_override=80)
            if not allow_high or not ReasonRouteIRCoTAssisted._answer_is_bad(first.get('final_answer', '')):
                return first, 'reasoning_low'
            second = self.interleaved.run(question, enable_answer_verifier=(mode == 'quality'))
            for name in ('token_cost', 'retrieval_steps'):
                second['metrics'][name] = second['metrics'].get(name, 0) + first.get('metrics', {}).get(name, 0)
            return second, 'reasoning_fallback'

    def run(self, question, mode):
        from reasonroute.branch_flow import ReasonRouteIRCoTAssisted
        if self.classifier is None:
            raise ValueError('Predicted routing requires a trained independent query classifier')
        error = None
        try:
            predicted = str(self.classifier['model'].predict([question])[0]).strip().lower()
            if predicted not in ('fact', 'explanation', 'reasoning'):
                raise ValueError('Invalid classifier label')
        except Exception as exc:
            predicted, error = 'explanation', type(exc).__name__
        route = {'route_source': 'predicted', 'predicted_question_type': predicted, 'classifier_error': error}
        if predicted == 'reasoning':
            use_high = True
            if mode == 'compact':
                if self.gate is None:
                    raise ValueError('Compact reasoning requires the frozen independent gate')
                score = float(self.gate['model'].decision_function([question])[0])
                use_high = score >= self.gate['threshold']
                route.update(gate_score=score, gate_threshold=self.gate['threshold'], gate_high=bool(use_high))
            output, branch = self.reasoning(question, mode, use_high)
        else:
            flow = ReasonRouteIRCoTAssisted(self.rr, self.interleaved, route_source='predicted',
                                           classifier=ConstantClassifier(predicted))
            with branch_environment(self.values(mode, predicted)):
                output, branch, _, _ = flow.run(question, '')
        route['branch'] = branch
        return output, route


def load_runtime(args, need_models=True):
    from reasonroute.corpus import verify_corpus
    verify_corpus(args.kb, args.corpus_manifest)
    from evaluate_rag import (API_KEY, SILICONFLOW_API_KEY, SILICONFLOW_RERANK_API_URL,
                              RERANK_MODEL_NAME, DEEPSEEK_API_URL, LLM_MODEL_NAME,
                              SiliconFlowEmbedding, load_json_data)
    from reasonroute import retrieval
    from reasonroute.evidence import EvidenceCatalog
    from tools.rag_resarch_tool import (AgenticRAGTool, DeepSeekRAGTool, GraphTool, RerankTool)
    if not API_KEY or not SILICONFLOW_API_KEY:
        raise ValueError('Set DEEPSEEK_API_KEY and SILICONFLOW_API_KEY in your local environment')
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    if config.get('data_protocol') != 'independent-v4' or config.get('route_source') != 'frozen_classifier_predictions_only':
        raise ValueError('Use the independent predicted-routing profile, not historical gold-routing settings')
    context = {'config_sha256': digest_file(args.config), 'kb_sha256': digest_file(args.kb),
               'kg_sha256': digest_file(args.kg), 'embeddings_sha256': digest_file(args.embeddings),
               'doc_title_context': title_setting(os.getenv('DOC_TITLE_CONTEXT', '0')),
               'generation_model': LLM_MODEL_NAME, 'embedding_model': 'BAAI/bge-m3',
               'reranker_model': RERANK_MODEL_NAME}
    context['implementation_revision'] = 'independent-v4'
    context['corpus_manifest_sha256'] = digest_file(args.corpus_manifest)
    cache_meta = args.embeddings.with_suffix('.metadata.json')
    if not cache_meta.is_file():
        raise ValueError('Embedding cache metadata is required; build a new cache with tools.build_method_index')
    context['embedding_metadata_sha256'] = digest_file(cache_meta)
    classifier = load_artifact(args.classifier, 'query_classifier') if need_models else None
    gate = load_artifact(args.gate, 'compact_gate') if need_models and args.mode == 'compact' else None
    if classifier is not None and classifier.get('train_n') != 1000:
        raise ValueError('Classifier must be trained under the independent 1000-query protocol')
    if classifier is not None:
        verify_model_data(classifier, read_rows(args.qa), digest_file(args.qa), 'query_classifier')
    if gate is not None:
        if gate.get('train_n') != 800:
            raise ValueError('Gate must use the independent 800-query protocol')
        verify_model_data(gate, read_rows(args.qa), digest_file(args.qa), 'compact_gate')
        if gate.get('execution_context') != context:
            raise ValueError('Gate annotations and this run have different corpus, model or branch configuration')
    docs = load_json_data(str(args.kb))
    if not docs:
        raise ValueError('Empty knowledge base')
    retrieval.INDEX_CACHE_FILE = str(args.embeddings)
    retrieval.TOKENIZED_CACHE_FILE = str(args.tokenized)
    args.tokenized.parent.mkdir(parents=True, exist_ok=True)
    embedding = SiliconFlowEmbedding(api_key=SILICONFLOW_API_KEY)
    semantic = retrieval.build_semantic_tool(docs, embedding, kb_sha256=context['kb_sha256'])
    hybrid = retrieval.build_hybrid_tool(docs, embedding, semantic.embeddings)
    mvector = retrieval.build_mvector_tool(docs, embedding, semantic)
    graph = GraphTool(kg_path=str(args.kg), model_obj=embedding)
    rerank = RerankTool(model_name=RERANK_MODEL_NAME, api_key=SILICONFLOW_API_KEY,
                        api_url=SILICONFLOW_RERANK_API_URL)
    rr = DeepSeekRAGTool(hybrid, semantic, graph, mvector, AgenticRAGTool(hybrid, semantic),
                        rerank, API_KEY, model_name=LLM_MODEL_NAME, api_url=DEEPSEEK_API_URL)
    rr.evidence_catalog = EvidenceCatalog.from_jsonl(args.kb, include_titles=context['doc_title_context'] not in ('0', 'false', 'no'))
    return MethodRuntime(rr, retrieval.InterleavedRetrieval(hybrid, max_steps=int(config['common']['IRCOT_MAX_STEPS']), reranker=rerank), config, classifier, gate), context
