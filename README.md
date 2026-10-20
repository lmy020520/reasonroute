# ReasonRoute：独立训练与预测路由修正版

本仓库候选只包含ReasonRoute方法、数据构建、模型训练和运行检查。IRCoT是推理分支内部依赖，代码位于`reasonroute/retrieval.py`。外部对比实验、BERT/Prompt分类对照及消融实验不在本公开候选中。

本版修正独立训练、共享标签规则、预测路由、Gate训练来源和固定知识库协议。它不是历史论文数值的复现包：历史78.12%来自1490条内部CV，56.11%/51.81%来自金标分流组合，323/167来自490条内部OOF。77.38%的历史记录属于LinearSVC而非DeepSeek Prompt。不能将上述数值转移到本版或更换模型名称后继续引用。

生成新的结果前，必须审查标签规则、获得完整独立语料、完成800条双分支标注与Gate训练，再运行1490条真实预测路由。代码修复不代表这些实验已经完成，也不保证原数值或方法排序不变。

## 安装与本地凭据

在仓库根目录使用Python 3.10或以上版本：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m scripts.check_imports
python scripts/audit_public_bundle.py
```

凭据通过环境变量`DEEPSEEK_API_KEY`、`SILICONFLOW_API_KEY`读取。`.env.example`只作模板，程序不自动读取`.env`。Linux使用`export`，PowerShell使用`$env:变量名`设置私密值；不要把实际密钥写进源码或提交记录。

默认API运行只需`requirements.txt`。本地Encoder/Reranker或FAISS索引另用`pip install -r requirements-local.txt`；对应库按需导入，不阻塞API路径。主要CLI支持`python -m 模块名`和直接执行脚本，实际运行仍建议在仓库根目录，以明确相对配置路径。`check_imports`会在禁止联网和导入本地模型库的条件下检查模块导入顺序，并检查两种CLI启动方式。

## 缓存和Token错误处理

- Embedding必须附带`tools.build_method_index`生成的v2元数据，校验有序文档哈希、原始KB哈希、模型名称、标题配置、向量文件哈希和矩阵有效性。旧缓存缺少元数据时拒绝加载，不按文档数量推测兼容。
- BM25缓存为带文档顺序、内容、分词实现、停用词和jieba版本指纹的v2封装，并校验词项校验和。同样文档数量但内容或顺序不同也会报错。旧纯列表缓存需选择新路径重新构建，不自动覆盖旧文件。模型和pickle缓存只应从可信本地来源加载。
- 所有生成调用要求响应提供正整数`usage.total_tokens`。缺失、格式错误或非JSON响应立即报错，不补0，不自动重发已收到的无有效统计响应。答案修复、查询规划、拆解和Verifier不会吞掉此类错误；评测入口不把校验失败的样本写成成本完整的结果。无LLM调用的分支仍允许局部增量为0。
- 发生API错误时保留已完成的逐题记录，修复后按原参数加`--resume`继续。未返回usage的失败请求可能已计费，代码不能据此还原服务端账单；已完成样本的Token不能被解读为失败重试的实际总账单。

## 数据合同

三分类器训练集1000条，独立评测集1490条，CompactGate开发集800条。三者按问题文本去首尾空格、转小写、合并空白后去重；不保证语义改写或证据文档隔离。数据构建参考`tools/build_classifier_training_dataset.py`和`tools/build_compact_gate_training_dataset.py`。原始数据须由使用者依据各数据集条款获取。

训练集与评测集使用同一个`reasonroute/annotation.py`标签入口及规则指纹。它是离线弱标注，使用问题、参考答案和证据结构，不使用数据集名称；在线分类器只读取问题文本。规则标签不能冒充人工标注。旧1490条中208条与共享规则不一致：需先检查差异，再决定人工复核或创建明确的新版本，不能静默改变已报告测试集的标签。

```bash
python -m tools.prepare_method_data audit --input "$OLD_TEST" --output runs/label_audit
# 仅在审查并批准新标签后运行；不覆盖原文件，变更后旧分类型结果不再适用。
python -m tools.prepare_method_data annotate --input "$OLD_TEST" --output data/evaluation_v4 --accept-label-changes
python -m tools.build_classifier_training_dataset --raw-root data/public_eval/raw --exclude "$TEST" "$OTHER_TEST" --output_dir data/classifier_v4
python -m tools.build_compact_gate_training_dataset --raw-root data/public_eval/raw --exclude "$TRAIN" "$TEST" "$OTHER_TEST" --output-dir data/gate_v4
```

上述路径变量需先指向真实文件。构建器遇到排除文件缺失即报错；只写新目录，不覆盖历史数据。NQ读取官方short_answers/yes_no_answer，IIRC读取span/value/binary答案；无答案样本不会作为有效候选。模型记录训练问题指纹，运行时再次校验评测问题不与训练问题重合，并验证评测集版本。

Gate候选每行必须有`id`、`question`和非空`answer`/`answers`。构建时过滤无答案候选，并支持IIRC的`answer_value`。旧800候选集中存在空答案，必须重新构建或有来源地修复后再标注；新构建会拒绝缺失的排除清单，不覆盖旧文件。

检索运行需要本地证据JSONL、图谱JSON以及按证据顺序生成的Embedding缓存。请确保知识库覆盖Gate开发问题的支持证据；缺乏支持文档时不能把双分支都失败误解为门控模型质量。下面以本地路径变量表示文件，所有命令都在仓库根目录运行。

```bash
TRAIN=data/classifier_v4/classifier_train_1000.jsonl
TEST=data/evaluation_v4/questions.jsonl
GATE=data/gate_v4/compact_gate_candidates_800.jsonl
KB=data/corpus_v4/kb_chunks.jsonl
CORPUS_MANIFEST=data/corpus_v4/corpus_manifest.json
KG=data/corpus_v4/extracted_kg.json
EMBED=cache/method_embeddings.npz
TOK=cache/method_tokenized.pkl
```

## 0. 先冻结完整语料，不按测试题补入黄金片段

```bash
python -m tools.build_fixed_corpus --documents data/full_documents.jsonl --output data/corpus_v4
```

输入为独立获取的完整文档快照（每行`doc_id/title/text/links`），不能是QA行或按题筛出的支持片段。默认按1000字符、重叠150字符切块；这是一套明确的新配置，不是历史分块方式的声明。应先审查文档来源、冻结语料，再选题和重建KG/向量缓存。语料可自然包含正确答案和支持文档，这正是检索任务的前提；禁止的是利用测试题自身的标注context定向扩库却不披露。

`tools/augment_explanation_from_iirc.py`现只在既有语料内匹配支持证据，不创建或追加任何块；找不到证据时跳过，不再自动补进KB。运行入口要求`--corpus-manifest`，旧`iirc_supp__`块被拒绝。构建器无法证明上游提供的文档快照绝对独立，其来源仍需人工核实。

## 1. 三分类器：1000训练、1490独立测试

```bash
python -m tools.train_method_models classifier --train "$TRAIN" --test "$TEST" --output runs/classifier_v2
```

词级1–4 gram、字符级char_wb 2–7 gram，`min_df=2`、`sublinear_tf=True`，权重1.25/0.75。结构信号重复一次；`LinearSVC(C=0.75, class_weight=balanced, loss=hinge, max_iter=15000)`。特征转换保留原V3定义。

5折训练内检查使用Pipeline，各折向量器独立拟合，分层划分seed=42；随后在完整1000条上拟合最终模型，再一次性预测1490条。1490条标签仅用于独立评估，测试结果不参与参数或阈值选择。生成`model.joblib`和带数据哈希的`report.json`；报告分别标明训练OOF和独立测试指标。

## 2. Gate：双分支标注与独立800条训练

如需新建向量缓存，运行下列命令会调用Embedding API并产生费用：

```bash
python -m tools.build_method_index --kb "$KB" --output "$EMBED"
```

在同一个语料、配置下，逐题执行Compact推理低成本路径与条件式高成本路径。两者复用同一次初始答案，避免首答随机差异影响收益标签；每条路径分别记录其完整生成Token。高成本路径不启用Quality verifier。

```bash
python -m scripts.run_reasonroute label-gate --qa "$GATE" --exclude "$TRAIN" "$TEST" --kb "$KB" --corpus-manifest "$CORPUS_MANIFEST" --kg "$KG" --embeddings "$EMBED" --tokenized "$TOK" --output runs/gate_pairs_v2
```

该命令会调用生成和裁判API。只有高成本正确且低成本错误时`gate_label=1`。裁判持续失败或非0/1结果时停止，不给失败样本伪造标签；再次执行相同命令加`--resume`可继续。模型、语料、数据及配置指纹变化时拒绝复用检查点。

```bash
python -m tools.train_method_models gate --paired runs/gate_pairs_v2/samples.jsonl --exclude "$TRAIN" "$TEST" --max-average-token 1134.16 --output runs/gate_v2
```

这里1134.16仅为显式选择的推理型平均Token上限示例，可按开发目标另设，不是逐次调用预算，也不是全系统总预算。新Gate不会继承旧阈值或强制产生323/167分流。

Gate采用char_wb 2–5 gram、`min_df=2`、`max_features=25000`、`sublinear_tf=True`和`LinearSVC(C=4, class_weight=balanced, random_state=42, max_iter=15000)`。5折OOF分数及收益、成本只来自800条开发集；在满足开发集平均Token上限的候选阈值中选准确率最高者，再用完整800条拟合冻结模型。该训练内选择成绩不是独立测试成绩。实际正/负样本任一不足5条时，程序拒绝进行5折训练。

## 3. 主实验：只读预测标签路由

```bash
python -m scripts.run_reasonroute evaluate --mode quality --qa "$TEST" --classifier runs/classifier_v2/model.joblib --kb "$KB" --corpus-manifest "$CORPUS_MANIFEST" --kg "$KG" --embeddings "$EMBED" --tokenized "$TOK" --output runs/quality_v2
python -m scripts.run_reasonroute evaluate --mode compact --qa "$TEST" --classifier runs/classifier_v2/model.joblib --gate runs/gate_v2/model.joblib --kb "$KB" --corpus-manifest "$CORPUS_MANIFEST" --kg "$KG" --embeddings "$EMBED" --tokenized "$TOK" --output runs/compact_v2
```

严格单进程逐题执行。路由函数只接收问题文本与模式，不接收真实类型。真实类型在决策后进入分类型统计；模型加载失败直接报错，分类输出无效则回退Explanation并记录异常。Compact对预测为Reasoning的问题使用冻结Gate的`decision_function >= threshold`决定是否允许高成本路径，且仍需首答触发异常规则才进入IRCoT。

新结果只从真实逐样本journal汇总，禁止把金标类型分支报告拼接成预测路由主结果。旧协议模型、缺少训练问题指纹的模型或没有明确排除本次测试集的Gate均被拒绝。训练CV成绩与独立测试成绩分开记录；不能将SVC行重命名为Prompt，也不能在不同评估协议之间直接排序。此版LinearSVC固定`random_state=42`，应重新测量模型和时延，不继承历史成绩。

配置见`configs/paper_v3.json`：Fact的Quality/Compact字符阈值50/85，短Top-3/长Top-4，生成60/80；Explanation每轮RRF召回20条再Cross-Encoder重排至Top-3、最多3轮，Quality最终9条并保留条件回退，Compact最终5条、每块2句/520字符；Reasoning首答Top-4/80，fallback最多3轮、每轮3条、最终5条；Quality启用fallback verifier（1句/420字符、输出60），Compact关闭verifier。各分支按上下文作用域应用参数，执行后恢复环境变量。

最终回退生成读取完整多轮消息历史与所选证据；输出到裁判的`retrieved_context`为实际选择、压缩后的最终证据。E0不并入fallback的聚合证据；提前ANSWER且尚无SEARCH时，最终聚合为空。语境扩展根据本地文档ID、段落顺序、标题和显式链接读取背景；当原始元数据未提供链接时，不假定关联文档存在。压缩后的选中句子恢复原文顺序；Verifier压缩仅按原问题评分，调用异常或空输出时保留原候选答案并记录状态。

评估调用沿用`evaluate_rag.py`的质量指标定义与Token统计，输出新逐样本记录和总体/分类型报告。裁判本地兜底的历史边界问题未在此次补流程中改动。API可变别名与硬件时延可能产生跨时间差异。
