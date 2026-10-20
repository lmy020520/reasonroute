import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import time
import requests
import concurrent.futures
from tqdm import tqdm


NO_PROXY_REQUESTS = os.getenv("NO_PROXY_REQUESTS", "1").strip().lower() not in {"0", "false", "no"}
REQUEST_PROXIES = {"http": "", "https": ""} if NO_PROXY_REQUESTS else None

# ================= 配置区域 =================
from tools.rag_resarch_tool import (
    HybridSearchTool, SemanticTool, GraphTool, MultiVectorTool,
    AgenticRAGTool, RerankTool, DeepSeekRAGTool, HybridQuestionClassifier,
    AdaptiveRouteSelector
)

API_KEY = os.getenv("DEEPSEEK_API_KEY", '')
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")

# 模型名称配置（统一从这里读取）
LLM_MODEL_NAME = "deepseek-chat"  # 对话/生成模型

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"  # 嵌入模型
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"  # 重排序模型

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_API_URL = os.getenv("SILICONFLOW_API_URL", "https://api.siliconflow.cn/v1/embeddings")
SILICONFLOW_RERANK_API_URL = os.getenv("SILICONFLOW_RERANK_API_URL", "https://api.siliconflow.cn/v1/rerank")

DOC_JSON_PATH = os.getenv("DOC_JSON_PATH", './data/public_eval/normalized/kb_chunks.jsonl')
KG_PATH = os.getenv("KG_PATH", './data/public_eval/normalized/extracted_kg.json')
BENCHMARK_FILE = os.getenv("BENCHMARK_FILE", './data/public_eval/normalized/qa_dev.jsonl')
REPORT_FILE = os.getenv("REPORT_FILE", "experiments/results/reasonroute_actual_optimized_v6_report.json")
REPORT_TAG = os.getenv("OUTPUT_TAG", "").strip()
ROUTE_MATRIX_FILE = os.getenv(
    "ROUTE_MATRIX_FILE",
    './experiments/results/route_matrix_new_exp500_full_summary.json'
)

INDEX_CACHE_FILE = os.getenv("INDEX_CACHE_FILE", "kb_embeddings_cache.npz")  # 向量缓存文件
HYBRID_CACHE_FILE = os.getenv("HYBRID_CACHE_FILE", "kb_hybrid_embeddings_cache.npz")  # 混合缓存文件
TOKENIZED_CACHE_FILE = os.getenv("TOKENIZED_CACHE_FILE", "kb_tokenized_corpus_cache.pkl")  # 分词缓存文件



# 🌟【并发配置】🌟
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))

# ===========================================
# 1. 辅助函数
# ===========================================


class SiliconFlowEmbedding:
    def __init__(self, model_name: str = None, api_key: str = None, batch_size: int = 64):
        if model_name is None:
            model_name = EMBEDDING_MODEL_NAME
        self.model_name = model_name
        self.api_key = api_key or SILICONFLOW_API_KEY
        self.api_url = SILICONFLOW_API_URL
        self.batch_size = batch_size
        self._embedding_dim = None

    def encode(self, sentences, show_progress_bar: bool = True, convert_to_numpy: bool = True, normalize_embeddings: bool = False):
        if isinstance(sentences, str):
            sentences = [sentences]
        
        all_embeddings = []
        total = len(sentences)
        total_batches = (total + self.batch_size - 1) // self.batch_size
        
        iterator = tqdm(range(0, total, self.batch_size), 
                       desc=f"Encoding {total} docs", 
                       total=total_batches,
                       ncols=100)
        
        for i in iterator:
            batch = sentences[i:i + self.batch_size]
            batch_embeddings = self._encode_batch(batch)
            all_embeddings.extend(batch_embeddings)
            iterator.set_postfix_str(f"Processed: {len(all_embeddings)}/{total}")
        
        import numpy as np
        from sklearn.preprocessing import normalize
        embeddings = np.array(all_embeddings)
        
        if normalize_embeddings:
            embeddings = normalize(embeddings)
        
        return embeddings if convert_to_numpy else embeddings.tolist()

    def _encode_batch(self, batch: list) -> list:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "input": batch,
            "encoding_format": "float"
        }
        
        max_retries = int(os.getenv("EMBED_MAX_RETRIES", "5"))
        for attempt in range(max_retries):
            try:
                response = requests.post(self.api_url, json=payload, headers=headers, timeout=60, proxies=REQUEST_PROXIES)
                response.raise_for_status()
                data = response.json()
                
                embeddings = [item["embedding"] for item in data["data"]]
                return embeddings
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(min(2 ** attempt, 8))
                else:
                    raise RuntimeError(f"SiliconFlow API 调用失败：{e}")


def load_json_data(file_path):
    """加载 JSONL 格式的知识库 (每行一个 JSON 对象)"""
    print(f"Loading knowledge base from {file_path}...")
    if not os.path.exists(file_path):
        return []
    try:
        docs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                content = item.get("content") or item.get("text") or item.get("body") or ""
                title = item.get("title") or item.get("metadata", {}).get("title") or ""
                if content and title and os.getenv("DOC_TITLE_CONTEXT", "0").strip().lower() not in {"0", "false", "no"}:
                    if title.lower() not in content[:160].lower():
                        content = f"Title: {title}\n{content}"
                if content:
                    docs.append(content)
        print(f"✅ Loaded {len(docs)} documents.")
        return docs
    except Exception as e:
        print(f"❌ JSON Load Error: {e}")
        return []

def extract_ground_truth(item):
    """从测试数据项中提取标准答案"""
    if 'ground_truth' in item:
        return item['ground_truth']
    
    if 'answers' in item:
        answers = item['answers']
        all_answers = []
        for answer_list in answers:
            if isinstance(answer_list, list):
                all_answers.extend([a for a in answer_list if a])
            elif answer_list:
                all_answers.append(answer_list)
        
        unique_answers = list(set(all_answers))
        return '; '.join(unique_answers) if unique_answers else ""
    
    return ""


def _normalize_for_match(text):
    import re
    text = str(text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _answer_aliases(ground_truth):
    aliases = []
    for part in str(ground_truth or "").split(";"):
        answer = part.strip()
        if answer:
            aliases.append(answer)
    return aliases


def local_judge_fallback(question, ground_truth, prediction, retrieved_contexts, reason="judge_api_failed"):
    """LLM 裁判不可用时的保守兜底，避免把网络失败误计为方法错误。"""
    prediction_norm = _normalize_for_match(prediction)
    context_norm = _normalize_for_match(" ".join([str(c) for c in retrieved_contexts]))
    aliases = _answer_aliases(ground_truth)

    correctness = 0
    context_recall = 0
    for alias in aliases:
        alias_norm = _normalize_for_match(alias)
        if not alias_norm:
            continue
        if alias_norm in prediction_norm or prediction_norm in alias_norm:
            correctness = 1
        if alias_norm in context_norm:
            context_recall = 1

    faithfulness = 1 if correctness and context_recall else 0
    return {
        "correctness": correctness,
        "context_recall": context_recall,
        "faithfulness": faithfulness,
        "judge_fallback": True,
        "fallback_reason": reason,
    }


def llm_judge_comprehensive(question, ground_truth, prediction, retrieved_contexts):
    """
    全能裁判：调用 LLM 计算 准确率、召回率、忠实度
    带重试机制，确保 API 调用成功后才返回
    """
    context_text = "\n".join([str(c) for c in retrieved_contexts])[:3000]
    
    prompt = f"""你是一个 RAG 评测裁判。请对以下问答进行打分。

【问题】: {question}
【标准答案】: {ground_truth}
【考生回答】: {prediction}

【检索到的上下文】:
{context_text}

请评估以下三个指标（必须输出 JSON）：

1. **correctness (准确性)**: (0或1) 考生回答的核心含义是否与标准答案一致？
2. **context_recall (上下文召回)**: (0或1) 【检索到的上下文】里是否包含了回答问题所需的关键信息？
3. **faithfulness (忠实度)**: (0或1) 考生回答是否**完全依据**【检索到的上下文】生成？(即没有幻觉或外部知识)

请仅输出 JSON:
{{
    "correctness": 1,
    "context_recall": 1,
    "faithfulness": 1
}} """
    
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 50
    }
    
    max_retries = int(os.getenv("JUDGE_MAX_RETRIES", "3"))
    retry_max_sleep = float(os.getenv("JUDGE_RETRY_MAX_SLEEP", "8"))
    api_timeout = float(os.getenv("JUDGE_API_TIMEOUT", "60"))
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            resp = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=api_timeout, proxies=REQUEST_PROXIES)
            
            # 检查 HTTP 状态码
            resp.raise_for_status()
            
            result = resp.json()
            
            # 检查是否有错误
            if 'error' in result:
                print(f"裁判 API 错误：{result['error']}，等待重试...")
                retry_count += 1
                time.sleep(min(2 ** retry_count, retry_max_sleep))
                continue
            
            # 检查是否有 choices
            if 'choices' not in result or len(result['choices']) == 0:
                print(f"裁判 API 响应异常，等待重试...")
                retry_count += 1
                time.sleep(min(2 ** retry_count, retry_max_sleep))
                continue
            
            content = result['choices'][0]['message']['content']
            # 清理 JSON
            content = content.replace("```json", "").replace("```", "").strip()
            import re
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
            
            print(f"裁判返回格式错误，等待重试...")
            retry_count += 1
            time.sleep(min(1, retry_max_sleep))
            continue
                
        except requests.exceptions.Timeout:
            print(f"裁判打分超时，等待重试...")
            retry_count += 1
            time.sleep(min(2 ** retry_count, retry_max_sleep))
        except requests.exceptions.RequestException as e:
            print(f"裁判打分网络异常 ({e})，等待重试...")
            retry_count += 1
            time.sleep(min(2 ** retry_count, retry_max_sleep))
        except Exception as e:
            print(f"裁判打分异常 ({e})，等待重试...")
            retry_count += 1
            time.sleep(min(2 ** retry_count, retry_max_sleep))
    
    # 所有重试都失败，返回默认值
    print(f"⚠️ 裁判打分重试{max_retries}次后仍失败，启用本地保守兜底")
    return local_judge_fallback(question, ground_truth, prediction, retrieved_contexts)

# ===========================================
# 2. 多线程处理的核心函数
# ===========================================
def process_single_item(item, classifier, route_selector, rag_engine):
    """
    处理单个问题的所有逻辑。将独立在子线程中运行。
    """
    q_text = item.get('question', '')
    gt_text = extract_ground_truth(item)
    q_type_gt = item.get('question_type', 'fact')

    fallback_routes = {
        "fact": ("precision-hybrid", "fact"),
        "explanation": ("context-enhanced", "explanation"),
        "reasoning": ("structured-chain", "reasoning"),
    }
    selected_route, route_category = fallback_routes.get(q_type_gt, fallback_routes["fact"])
    route_decision = {
        "selected_route": selected_route,
        "route_category": route_category,
        "default_route": selected_route,
        "matrix_route": None,
        "route_reason": "fallback_before_routing",
        "retrieval_confidence": None,
        "confidence_details": None,
        "explicit_reasoning": None,
    }
    evidence_need_pred = q_type_gt if q_type_gt in fallback_routes else "fact"
    routing_error = None

    try:
        evidence_need_pred = classifier.classify(q_text)
    except Exception as exc:
        routing_error = f"classifier_error: {exc}"
        evidence_need_pred = q_type_gt if q_type_gt in fallback_routes else "fact"

    try:
        route_decision = route_selector.select(q_text, evidence_need_pred)
        selected_route = route_decision["selected_route"]
        route_category = route_decision["route_category"]
    except Exception as exc:
        route_error = f"route_selector_error: {exc}"
        routing_error = f"{routing_error}; {route_error}" if routing_error else route_error
        selected_route, route_category = fallback_routes.get(evidence_need_pred, fallback_routes["fact"])
        route_decision.update({
            "selected_route": selected_route,
            "route_category": route_category,
            "route_reason": "routing_error_fallback",
        })
    
    t0 = time.time()
    error = None
    output = None
    answer_style_source = os.getenv("ROUTE_ANSWER_STYLE_SOURCE", "pred").strip().lower()
    if answer_style_source in {"gold", "label", "type", "question_type"}:
        answer_style_for_generation = q_type_gt
    elif answer_style_source in {"gold_nonreasoning", "gold_non_reasoning", "gold_except_reasoning"}:
        answer_style_for_generation = route_category if q_type_gt == "reasoning" else q_type_gt
    elif answer_style_source in {"route", "category", "route_category"}:
        answer_style_for_generation = route_category
    else:
        answer_style_for_generation = evidence_need_pred

    for attempt in range(int(os.getenv("ITEM_MAX_RETRIES", "2"))):
        try:
            if os.getenv("ROUTE_USE_EVIDENCE_ANSWER_STYLE", "0").strip().lower() not in {"0", "false", "no"}:
                output = rag_engine.execute_rag_flow(q_text, route_category, answer_style=answer_style_for_generation)
            else:
                output = rag_engine.execute_rag_flow(q_text, route_category)
            break
        except Exception as exc:
            error = str(exc)
            if attempt + 1 < int(os.getenv("ITEM_MAX_RETRIES", "2")):
                time.sleep(min(2 ** (attempt + 1), 4))
    latency = time.time() - t0

    if output is None:
        prediction = "PIPELINE_ERROR"
        retrieved_docs = []
        steps = 0
        ctx_len = 0
        tokens = 0
        evidence_compression = None
        context_items = 0
        answer_rescue = False
        scores = {"correctness": 0, "context_recall": 0, "faithfulness": 0}
    else:
        prediction = output['final_answer']
        retrieved_docs = output['retrieved_context']

        flow_metrics = output.get('metrics', {})
        steps = flow_metrics.get('retrieval_steps', 1)
        ctx_len = flow_metrics.get('context_length', 0)
        tokens = flow_metrics.get('token_cost', 0)
        evidence_compression = flow_metrics.get('evidence_compression')
        context_items = flow_metrics.get('context_items')
        answer_rescue = flow_metrics.get('answer_rescue', False)

        scores = llm_judge_comprehensive(q_text, gt_text, prediction, retrieved_docs)
    
    record = {
        "id": item.get("id"),
        "question_type": q_type_gt,
        "type": q_type_gt,
        "question": q_text,
        "evidence_need_pred": evidence_need_pred,
        "answer_style_source": answer_style_source,
        "answer_style": answer_style_for_generation,
        "router_pred": evidence_need_pred,
        "selected_route": selected_route,
        "route_category": route_category,
        "default_route": route_decision.get("default_route"),
        "matrix_route": route_decision.get("matrix_route"),
        "route_reason": route_decision.get("route_reason"),
        "retrieval_confidence": route_decision.get("retrieval_confidence"),
        "confidence_details": route_decision.get("confidence_details"),
        "explicit_reasoning": route_decision.get("explicit_reasoning"),
        "prediction": prediction,
        "scores": scores,
        "latency": round(latency, 2),
        "retrieval_steps": steps,
        "context_length": ctx_len,
        "token_cost": tokens,
        "evidence_compression": evidence_compression,
        "context_items": context_items,
        "answer_rescue": answer_rescue
    }
    if error:
        record["error"] = error
    if routing_error:
        record["routing_error"] = routing_error
    
    return q_type_gt, scores, latency, steps, ctx_len, tokens, selected_route, evidence_need_pred, record

if __name__ == "__main__":
    raise SystemExit('This module supplies judges and retrieval adapters only. Use python -m scripts.run_reasonroute for independent predicted routing.')
    print("\n" + "="*60)
    print("🧪 启动 RAG 全面评测 (ReasonRoute Benchmark)")
    print("="*60)

    docs = load_json_data(DOC_JSON_PATH)
    if not docs:
        print("❌ 未加载到文档，请检查路径。")
        exit()
    
    print(f"Initializing SiliconFlow Embedding API ({EMBEDDING_MODEL_NAME})...")
    model = SiliconFlowEmbedding(model_name=EMBEDDING_MODEL_NAME, api_key=SILICONFLOW_API_KEY)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Initializing Rerank Model ({RERANK_MODEL_NAME} via SiliconFlow)...")
    reranker = RerankTool(model_name=RERANK_MODEL_NAME, api_key=SILICONFLOW_API_KEY, api_url=SILICONFLOW_RERANK_API_URL)

    print("Initializing Search Tools...")
    
    # 尝试加载缓存的索引
    import faiss
    import numpy as np
    
    # 加载 SemanticTool 缓存
    if os.path.exists(INDEX_CACHE_FILE):
        print(f"📂 发现缓存索引文件：{INDEX_CACHE_FILE}")
        print("正在加载缓存索引...")
        try:
            cache_data = np.load(INDEX_CACHE_FILE)
            embeddings = cache_data['embeddings']
            if len(embeddings) != len(docs):
                raise ValueError(f"缓存向量数量 {len(embeddings)} 与文档数量 {len(docs)} 不一致")
            
            if os.getenv("USE_FAISS_INDEX", "1").strip().lower() in {"0", "false", "no"}:
                index = None
                embeddings = embeddings.astype("float32", copy=False)
                print("♻️ USE_FAISS_INDEX=0：跳过 FAISS index 复制，使用单份向量矩阵检索。")
            else:
                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(embeddings.astype('float32'))
            
            semantic_tool = SemanticTool.__new__(SemanticTool)
            semantic_tool.corpus = docs
            semantic_tool.model = model
            semantic_tool.device = device
            semantic_tool.embeddings = embeddings
            semantic_tool.index = index
            
            print(f"✅ 缓存索引加载成功！共 {len(docs)} 个文档")
        except Exception as e:
            print(f"❌ 缓存索引加载失败：{e}")
            if os.getenv("REBUILD_INDEX_ON_CACHE_FAIL", "0").strip().lower() in {"0", "false", "no"}:
                raise RuntimeError(
                    "缓存索引加载失败。为避免重新调用 embedding API 构建全库索引，已停止。"
                    "请释放内存后重试，或显式设置 REBUILD_INDEX_ON_CACHE_FAIL=1。"
                )
            print("将重新构建索引...")
            semantic_tool = SemanticTool(docs, model, device=device)
            try:
                print(f"💾 重建后保存索引到缓存：{INDEX_CACHE_FILE}")
                np.savez(INDEX_CACHE_FILE, embeddings=semantic_tool.embeddings.astype("float32", copy=False))
                print(f"✅ 索引已保存到：{INDEX_CACHE_FILE}")
            except Exception as save_e:
                print(f"⚠️ 保存索引失败：{save_e}")
    else:
        print("未找到缓存索引，正在构建新索引...")
        semantic_tool = SemanticTool(docs, model, device=device)
        try:
            print(f"💾 保存索引到缓存：{INDEX_CACHE_FILE}")
            np.savez(INDEX_CACHE_FILE, embeddings=semantic_tool.embeddings.astype("float32", copy=False))
            print(f"✅ 索引已保存到：{INDEX_CACHE_FILE}")
        except Exception as e:
            print(f"⚠️ 保存索引失败：{e}")
    
    # 加载 HybridSearchTool 缓存
    if os.path.exists(HYBRID_CACHE_FILE):
        print(f"📂 发现混合缓存索引文件：{HYBRID_CACHE_FILE}")
        print("正在加载混合缓存索引...")
        try:
            if getattr(semantic_tool, "embeddings", None) is not None and len(semantic_tool.embeddings) == len(docs):
                print("♻️ 复用 SemanticTool embeddings 作为混合检索向量，避免重复加载大缓存。")
                embeddings = semantic_tool.embeddings
            else:
                cache_data = np.load(HYBRID_CACHE_FILE)
                embeddings = cache_data['embeddings']
            if len(embeddings) != len(docs):
                raise ValueError(f"混合缓存向量数量 {len(embeddings)} 与文档数量 {len(docs)} 不一致")
            
            # 尝试加载分词缓存
            tokenized_corpus = None
            if os.path.exists(TOKENIZED_CACHE_FILE):
                print(f"📂 发现分词缓存文件：{TOKENIZED_CACHE_FILE}")
                try:
                    import pickle
                    with open(TOKENIZED_CACHE_FILE, 'rb') as f:
                        tokenized_corpus = pickle.load(f)
                    if len(tokenized_corpus) != len(docs):
                        raise ValueError(f"分词缓存数量 {len(tokenized_corpus)} 与文档数量 {len(docs)} 不一致")
                    print(f"✅ 分词缓存加载成功！共 {len(tokenized_corpus)} 个文档")
                except Exception as e:
                    print(f"⚠️ 分词缓存加载失败：{e}")
            
            # 提前导入 BM25Okapi 和 jieba，避免后续报错
            from rank_bm25 import BM25Okapi
            import jieba
            
            # 使用 __new__ 创建对象，避免内存溢出
            hybrid = HybridSearchTool.__new__(HybridSearchTool)
            # 只设置必要属性，不转换 corpus 为 numpy 数组
            hybrid.corpus = docs  # 保持为 list，节省内存
            hybrid.encoder = model
            hybrid.corpus_embeddings = embeddings
            hybrid.rrf_k = 60
            # 初始化停用词和分词
            hybrid.stop_words = hybrid._load_stopwords()
            
            if tokenized_corpus:
                # 使用缓存的分词结果
                print("使用缓存的分词结果...")
                hybrid.tokenized_corpus = tokenized_corpus
            else:
                # 重新构建分词
                print("正在构建 BM25 索引...")
                hybrid.tokenized_corpus = []
                for doc in tqdm(docs, desc="📝 BM25 分词处理", unit="doc", ncols=100):
                    hybrid.tokenized_corpus.append(hybrid._clean_tokenize(doc))
                # 保存分词缓存
                try:
                    import pickle
                    with open(TOKENIZED_CACHE_FILE, 'wb') as f:
                        pickle.dump(hybrid.tokenized_corpus, f)
                    print(f"✅ 分词已保存到缓存：{TOKENIZED_CACHE_FILE}")
                except Exception as e:
                    print(f"⚠️ 保存分词缓存失败：{e}")
            
            hybrid.bm25 = BM25Okapi(hybrid.tokenized_corpus)
            # HybridSearchTool.keyword_search 不使用 hybrid.index；保留属性只是为了兼容旧代码。
            hybrid.index = getattr(semantic_tool, "index", None)
            
            print(f"✅ 混合缓存索引加载成功！共 {len(docs)} 个文档")
        except Exception as e:
            print(f"❌ 混合缓存索引加载失败：{e}")
            print("将重新构建索引...")
            hybrid = HybridSearchTool(docs, model, existing_embeddings=getattr(semantic_tool, "embeddings", None))
    else:
        print("未找到混合缓存索引，正在构建新索引...")
        hybrid = HybridSearchTool(docs, model, existing_embeddings=getattr(semantic_tool, "embeddings", None))
        try:
            print(f"💾 保存索引到缓存：{HYBRID_CACHE_FILE}")
            np.savez(HYBRID_CACHE_FILE, embeddings=hybrid.corpus_embeddings.astype("float32", copy=False))
            print(f"✅ 索引已保存到：{HYBRID_CACHE_FILE}")
        except Exception as e:
            print(f"⚠️ 保存索引失败：{e}")
    
    # 初始化 MultiVectorTool（使用缓存的 embeddings，避免重复构建）
    print("Initializing MultiVectorTool...")
    try:
        # 使用 __new__ 创建对象，复用 semantic_tool 的 embeddings
        mvector = MultiVectorTool.__new__(MultiVectorTool)
        mvector.corpus = docs  # 保持为 list
        mvector.model = model
        mvector.device = device
        mvector.embeddings = semantic_tool.embeddings
        mvector.index = semantic_tool.index
        # MultiVectorTool 只需要向量索引，不需要 BM25
        # 设置必要属性即可
        mvector.stop_words = set()  # 空集合，避免报错
        mvector.rrf_k = 60
        mvector.tokenized_corpus = []  # 空列表
        mvector.bm25 = None  # 不需要 BM25
        print(f"✅ MultiVectorTool 初始化成功（复用缓存索引）")
    except Exception as e:
        print(f"⚠️ MultiVectorTool 复用缓存失败：{e}，将重新构建")
        mvector = MultiVectorTool(docs, model, device=device)
    
    if os.path.exists(KG_PATH):
        graph = GraphTool(kg_path=KG_PATH, model_obj=model)
    else:
        print(f"⚠️ 未找到 KG 文件：{KG_PATH}，使用空图谱。")
        graph = GraphTool(model_obj=model)
        
    agentic = AgenticRAGTool(hybrid, semantic_tool)

    classifier = HybridQuestionClassifier(DEEPSEEK_API_URL, API_KEY, model_name=LLM_MODEL_NAME)
    rag_engine = DeepSeekRAGTool(
        hybrid, semantic_tool, graph, mvector, agentic, reranker, API_KEY, 
        model_name=LLM_MODEL_NAME, api_url=DEEPSEEK_API_URL
    )
    route_selector = AdaptiveRouteSelector(
        semantic_tool=semantic_tool,
        hybrid_tool=hybrid,
        matrix_path=ROUTE_MATRIX_FILE,
        confidence_high=float(os.getenv("ROUTE_CONFIDENCE_HIGH", "0.72")),
        confidence_low=float(os.getenv("ROUTE_CONFIDENCE_LOW", "0.45")),
        acceptable_gap=float(os.getenv("ROUTE_ACCEPTABLE_GAP", "8.0")),
    )
    print(f"🧭 自适应路由矩阵：{ROUTE_MATRIX_FILE}")
    print(f"🧭 矩阵最优映射：{route_selector.optimal_mapping}")

    if not os.path.exists(BENCHMARK_FILE):
        print(f"❌ 未找到测试集文件：{BENCHMARK_FILE}")
        exit()
        
    dataset = []
    with open(BENCHMARK_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    if REPORT_TAG:
        report_root, report_ext = os.path.splitext(REPORT_FILE)
        REPORT_FILE = f"{report_root}_{REPORT_TAG}{report_ext}"
        print(f"📝 启用 OUTPUT_TAG={REPORT_TAG}，报告将保存至：{REPORT_FILE}")

    eval_limit = os.getenv("EVAL_LIMIT")
    if eval_limit:
        try:
            eval_limit_int = int(eval_limit)
            if eval_limit_int > 0:
                dataset = dataset[:eval_limit_int]
                report_root, report_ext = os.path.splitext(REPORT_FILE)
                REPORT_FILE = f"{report_root}_limit{eval_limit_int}{report_ext}"
                print(f"🔎 启用 EVAL_LIMIT={eval_limit_int}，本次只评测前 {len(dataset)} 条。")
                print(f"📝 限量评测报告将保存至：{REPORT_FILE}")
        except ValueError:
            print(f"⚠️ 忽略非法 EVAL_LIMIT：{eval_limit}")

    checkpoint_every = int(os.getenv("CHECKPOINT_EVERY", "50"))
    checkpoint_root, checkpoint_ext = os.path.splitext(REPORT_FILE)
    checkpoint_file = f"{checkpoint_root}_partial{checkpoint_ext or '.json'}"

    print(f"📚 开始评测 {len(dataset)} 道题目... (并发数：{MAX_WORKERS})")
    
    results = []
    
    metrics = {k: {
        "acc": 0, "rec": 0, "fai": 0, "total": 0,
        "latency": 0.0, "steps": 0, "ctx_len": 0, "tokens": 0
    } for k in ["total", "fact", "explanation", "reasoning"]}
    route_counts = {
        "precision-hybrid": 0,
        "context-enhanced": 0,
        "structured-chain": 0,
    }
    evidence_need_counts = {
        "fact": 0,
        "explanation": 0,
        "reasoning": 0,
    }
    
    # 使用 ThreadPoolExecutor 进行并发处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        futures = [executor.submit(process_single_item, item, classifier, route_selector, rag_engine) for item in dataset]
        
        # 使用 tqdm 监控进度，设置最小间隔避免闪烁
        pbar = tqdm(concurrent.futures.as_completed(futures), total=len(dataset), ncols=100, mininterval=2.0)
        
        completed = 0
        for future in pbar:
            try:
                # 获取子线程结果
                q_type_gt, scores, latency, steps, ctx_len, tokens, selected_route, evidence_need_pred, record = future.result()
                completed += 1
                if selected_route in route_counts:
                    route_counts[selected_route] += 1
                if evidence_need_pred in evidence_need_counts:
                    evidence_need_counts[evidence_need_pred] += 1
                
                # 主线程安全统计
                for k in ["total", q_type_gt]:
                    if k in metrics:
                        metrics[k]["total"] += 1
                        metrics[k]["acc"] += scores["correctness"]
                        metrics[k]["rec"] += scores["context_recall"]
                        metrics[k]["fai"] += scores["faithfulness"]
                        metrics[k]["latency"] += latency
                        metrics[k]["steps"] += steps
                        metrics[k]["ctx_len"] += ctx_len
                        metrics[k]["tokens"] += tokens
                
                results.append(record)

                if checkpoint_every > 0 and metrics["total"]["total"] % checkpoint_every == 0:
                    try:
                        with open(checkpoint_file, 'w', encoding='utf-8') as f:
                            json.dump({
                                "completed": metrics["total"]["total"],
                                "summary": metrics,
                                "route_summary": {
                                    "evidence_need_counts": evidence_need_counts,
                                    "selected_route_counts": route_counts,
                                    "matrix_file": ROUTE_MATRIX_FILE,
                                    "matrix_optimal_mapping": route_selector.optimal_mapping,
                                    "default_mapping": route_selector.DEFAULT_MAPPING,
                                },
                                "results": results
                            }, f, ensure_ascii=False, indent=2)
                    except Exception as ckpt_e:
                        print(f"⚠️ 保存 checkpoint 失败：{ckpt_e}")
                
                # 实时更新进度条显示（使用已完成的数量）
                if metrics["total"]["total"] > 0:
                    acc_rate = (metrics["total"]["acc"] / metrics["total"]["total"]) * 100
                    # 使用 metrics 中的 total 而不是 completed，因为它是线程安全的
                    current_total = metrics["total"]["total"]
                    pbar.set_description(f"评测进度 [{current_total}/{len(dataset)}]")
                    pbar.set_postfix({"ACC": f"{acc_rate:.1f}%", "完成": f"{current_total*100//len(dataset)}%"})
                    
            except Exception as e:
                print(f"\n⚠️ 某道题目处理异常：{e}")
                # 即使异常也计入完成数
                completed += 1

    print("\n" + "="*95)
    print("📊 RAG 全面评测最终结果")
    print("-" * 95)
    print(f"{'Metric':<14} {'ACC':<7} {'Recall':<8} {'Faith':<7} | {'耗时 (s)':<8} {'检索 (次)':<8} {'上下文 (字)':<10} {'Token 开销':<8}")
    print("-" * 95)
    
    for k in ["total", "fact", "explanation", "reasoning"]:
        tot = metrics[k]["total"]
        if tot == 0: continue
        
        acc = (metrics[k]["acc"] / tot) * 100
        rec = (metrics[k]["rec"] / tot) * 100
        fai = (metrics[k]["fai"] / tot) * 100
        
        avg_lat = metrics[k]["latency"] / tot
        avg_stp = metrics[k]["steps"] / tot
        avg_ctx = metrics[k]["ctx_len"] / tot
        avg_tok = metrics[k]["tokens"] / tot
        
        print(f"{k.capitalize():<14} {acc:<7.1f} {rec:<8.1f} {fai:<7.1f} | {avg_lat:<8.2f} {avg_stp:<8.1f} {avg_ctx:<10.0f} {avg_tok:<8.0f}")
    print("="*95)
    print(f"Evidence need 预测分布：{evidence_need_counts}")
    print(f"Selected route 分布：{route_counts}")
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": metrics,
            "route_summary": {
                "evidence_need_counts": evidence_need_counts,
                "selected_route_counts": route_counts,
                "matrix_file": ROUTE_MATRIX_FILE,
                "matrix_optimal_mapping": route_selector.optimal_mapping,
                "default_mapping": route_selector.DEFAULT_MAPPING,
                "confidence_high": route_selector.confidence_high,
                "confidence_low": route_selector.confidence_low,
                "acceptable_gap": route_selector.acceptable_gap,
                "explicit_chain_max_gap": route_selector.explicit_chain_max_gap,
            },
            "results": results
        }, f, ensure_ascii=False, indent=2)
    print(f"📝 报告已保存至：{REPORT_FILE}")
