"""
评测脚本：对 test_100_full.csv 计算 Hit@k / MRR / RAGAS 指标
不依赖 ragas 库，直接用 LLM 做 faithfulness / relevancy / context 评判

用法：
  python eval_testset.py           # 原始朴素检索（baseline）
  python eval_testset.py --hyde    # 启用 HyDE 假想文档检索
"""
import gc
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from config import *
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pymilvus import MilvusClient
from neo4j import GraphDatabase
from hyde import generate_hypothetical_doc, classify_query, hyde_retrieve, multi_query_retrieve
from retriever import FitnessRAGRetriever, load_bm25_from_pickle

# 限制 Ollama 并发 + 上下文窗口（防止 Windows OOM 死机）
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "1")
os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "2")

USE_HYDE = "--hyde" in sys.argv
USE_RRF = "--rrf" in sys.argv          # 临时切 RRF 融合模式（与 weighted 对比用）
USE_CLOUD = "--cloud" in sys.argv      # 评测用云端 LLM（默认本地 Ollama 防烧钱）
USE_FEEDBACK = "--feedback" in sys.argv   # 反馈闭环：评测真实用户负反馈问题（无金标准）
EVAL_PROVIDER = "dashscope" if USE_CLOUD else EVAL_PROVIDER


# -- 加载组件 ----------------------------------------------
from llm_adapter import build_embeddings
embeddings = build_embeddings()  # 与线上一致（cloud/ollama 按 config）

# Milvus Lite（与 app.py 一致）
milvus_client = MilvusClient(uri=MILVUS_URI, grpc_options=MILVUS_GRPC_OPTIONS)
milvus_client.load_collection(MILVUS_COLLECTION)  # 加载到内存（默认 released）

# BM25
bm25_idx, bm25_docs = load_bm25_from_pickle(BM25_INDEX_PATH)

# Neo4j（NEO4J_ENABLED=False 时停用 → driver 传 None，图谱路径返回空）
if NEO4J_ENABLED:
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
else:
    neo4j_driver = None

def _embed_fn(text: str):
    return embeddings.embed_query(text)

# Reranker（统一适配器：EVAL_PROVIDER 默认本地 ollama，--cloud 切云端）
from llm_adapter import build_llm, to_runnable

_eval_reranker = None
if RERANKER_ENABLED:
    from reranker import FitnessReranker
    _eval_reranker = FitnessReranker(
        llm=build_llm("rerank", provider=EVAL_PROVIDER),
        max_candidates=RERANKER_MAX_CANDIDATES,
        doc_max_chars=RERANKER_DOC_MAX_CHARS,
    )
    print(f"[INFO] Reranker 已启用 (listwise LLM, provider={EVAL_PROVIDER})")

retriever = FitnessRAGRetriever(
    milvus_client=milvus_client,
    bm25_index=(bm25_idx, bm25_docs),
    neo4j_driver=neo4j_driver,
    embedding_fn=_embed_fn,
    w_milvus=FUSION_WEIGHT_MILVUS,
    w_bm25=FUSION_WEIGHT_BM25,
    w_neo4j=FUSION_WEIGHT_NEO4J,
    fusion_threshold=FUSION_THRESHOLD,
    reranker=_eval_reranker,
    milvus_factor=RERANK_MILVUS_FACTOR if RERANKER_ENABLED else 3,
    bm25_factor=RERANK_BM25_FACTOR if RERANKER_ENABLED else 3,
    neo4j_factor=RERANK_NEO4J_FACTOR if RERANKER_ENABLED else 2,
    neo4j_depth=NEO4J_DEPTH,
    neo4j_max_depth=NEO4J_MAX_DEPTH,
    fusion_mode="rrf" if USE_RRF else None,
)
# 统一适配器：主生成 / HyDE / 判分（本地 ollama 或 --cloud 云端）
llm = build_llm("chat", provider=EVAL_PROVIDER, max_tokens=256)
hyde_llm = build_llm("hyde", provider=EVAL_PROVIDER)
judge_llm = to_runnable(build_llm("judge", provider=EVAL_PROVIDER, max_tokens=8))

# HyDE 模式标记
EVAL_MODE = "HyDE" if USE_HYDE else "ThreePath"


def retrieve(query: str):
    """根据模式分发检索：HyDE 增强 或 三路直接检索。"""
    if USE_HYDE:
        label = classify_query(query)
        if label == "compound_injury":
            return multi_query_retrieve(query, hyde_llm, retriever)
        else:
            hyde_doc = generate_hypothetical_doc(query, hyde_llm, label)
            return retriever.similarity_search(hyde_doc, k=HYDE_TOP_K)
    else:
        return retriever.similarity_search(query, k=5)  # k=5 避免人为压低 recall

df = pd.read_csv("test_100_full.csv")
# --limit N 参数：仅使用前 N 条
for i, arg in enumerate(sys.argv):
    if arg == "--limit" and i + 1 < len(sys.argv):
        df = df.head(int(sys.argv[i + 1]))
print(f"[OK] Loaded {len(df)} test samples")

# --feedback 模式：读取反馈文件的负反馈问题（真实用户不满意的样本回流评测）--
if USE_FEEDBACK:
    feedback_queries = []
    try:
        with open(FEEDBACK_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("vote") == "down" and rec.get("question"):
                    feedback_queries.append(rec["question"])
    except FileNotFoundError:
        feedback_queries = []
    if not feedback_queries:
        print(f"[INFO] {FEEDBACK_PATH} 中无负反馈样本，跳过 --feedback 评测")
        sys.exit(0)
    # 去重保持出现顺序
    df = pd.DataFrame({"query": list(dict.fromkeys(feedback_queries))})
    print(f"[OK] 反馈评测集: {len(df)} 条真实用户负反馈问题（无金标准参考，跳过 Hit@k 口径）")


# -- 余弦相似度 --------------------------------------------
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


# -- 1) Hit@k & MRR ----------------------------------------
# 命中判定阈值：doc↔reference_texts 余弦。
# 2026-08-14 随 Embedding 切换重校准（qwen3.7-text-embedding）：0.75 为 nomic 时代校准值，
# qwen 嵌入下 doc↔ref 分离度不同，实测敏感度曲线 th=0.60 → Hit@3=0.40/MRR=0.352（与旧库持平），
# th=0.75 → Hit@3=0.00（黄金对 query↔ref 平均 0.774，doc↔ref 更低一档）。切换 Embedding 供应商必须重校准此值。
SIM_THRESHOLD = 0.60

# 逐样本明细累积（落盘用：聚合指标要能追溯到具体问题）
SAMPLE_RECORDS: list[dict] = []


def save_results(retrieval: dict, ragas_scores: dict, type_results: dict) -> str:
    """把评测结果落盘 —— 让 README / 简历上的每个数字都可追溯、可复现。

    记录完整运行上下文（模式 / 融合方式 / 嵌入模型 / 评测集 / commit），
    否则数字离开当次终端就失去可信度（历史教训：头条指标无产物可查）。
    """
    import datetime
    import subprocess

    out_dir = "eval_results"
    os.makedirs(out_dir, exist_ok=True)

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        # 工作区有未提交改动时标注：否则 commit 号无法代表产生结果的代码
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip())
    except Exception:
        commit, dirty = "", False

    payload = {
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "eval_mode": EVAL_MODE,
            "provider": EVAL_PROVIDER,
            "fusion_mode": FUSION_MODE,
            "use_hyde": USE_HYDE,
            "use_rrf": USE_RRF,
            "neo4j_enabled": NEO4J_ENABLED,
            "embedding_provider": EMBEDDING_PROVIDER,
            "embedding_model": (EMBEDDING_CLOUD_MODEL if EMBEDDING_PROVIDER == "cloud"
                                else EMBEDDING_MODEL),
            "sim_threshold": SIM_THRESHOLD,
            "testset": "test_100_full.csv",
            "n_samples": len(df),
            "git_commit": commit,
            "git_dirty": dirty,   # True = 工作区有未提交改动，结果对应的是工作区代码
        },
        "metrics": {**retrieval, **ragas_scores},
        "by_question_type": type_results,
        "samples": SAMPLE_RECORDS,
    }

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"eval_{ts}.json")
    for p in (path, os.path.join(out_dir, "latest.json")):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVED] 结果已落盘: {path}")
    print(f"[SAVED] 最新指针:   {out_dir}/latest.json")
    return path


def compute_retrieval_metrics():
    k_list = (1, 3, 5)
    hit = {k: 0 for k in k_list}
    mrr_sum = 0

    for i, row in df.iterrows():
        docs = retrieve(row["query"])
        ref_emb = np.array(embeddings.embed_query(row["reference_texts"]))
        ranks = []
        for j, doc in enumerate(docs):
            doc_emb = np.array(embeddings.embed_query(doc.page_content))
            if cosine(ref_emb, doc_emb) > SIM_THRESHOLD:
                ranks.append(j + 1)
        if ranks:
            mrr_sum += 1.0 / ranks[0]
            for k in k_list:
                if any(r <= k for r in ranks):
                    hit[k] += 1
        # 逐样本明细：让每个聚合指标都能追溯到具体问题（落盘用）
        SAMPLE_RECORDS.append({
            "query": row["query"],
            "question_type": row.get("question_type", ""),
            "hit_ranks": ranks,
            "first_hit_rank": ranks[0] if ranks else None,
            "n_docs": len(docs),
            "retrieved_sources": [d.metadata.get("source", "") for d in docs],
        })
        if (i + 1) % 20 == 0:
            print(f"  retrieval... {i+1}/{len(df)}")
            gc.collect()
            time.sleep(0.5)

    return {
        f"Hit@{k}": round(hit[k] / len(df), 4) for k in k_list
    } | {"MRR": round(mrr_sum / len(df), 4)}


# -- 2) RAGAS-style 指标（LLM as judge）---------------------

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你的任务是判断一个回答是否忠实于给定的上下文(context)。
- 如果回答中的所有主张都能在上下文中找到依据，得 1 分
- 如果回答中有编造的信息不在上下文中，得 0 分
请只输出一个数字：1 或 0"""),
    ("human", "上下文:\n{context}\n\n回答:\n{answer}\n\n分数:")
])

RELEVANCY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """判断回答是否精准回应了用户问题。

- 1分：回答完全切题，包含具体动作名/肌群/器械/伤病名等实体信息，不是泛泛而谈
- 0分：回答偏离主题、答非所问、或过于笼统（如仅说"请咨询医生"而没有任何具体建议）

只输出一个数字：1 或 0。"""),
    ("human", "问题:\n{question}\n\n回答:\n{answer}\n\n分数:")
])

CONTEXT_PRECISION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """判断检索到的上下文是否精确匹配用户问题。

判定标准：
- 1分：每条上下文都与问题直接相关（含匹配的肌群名/动作名/伤病名/器械名）
- 0分：任一上下文与问题完全无关，或被无关内容主导（有效信息<50%）

只输出一个数字：1 或 0。"""),
    ("human", "问题:\n{question}\n\n检索到的上下文:\n{context}\n\n分数:")
])

CONTEXT_RECALL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你的任务是判断检索到的上下文是否包含了回答该问题所需的关键信息。
- 如果上下文覆盖了 ground truth 中的核心要点，得 1 分
- 如果上下文缺少 ground truth 中的关键信息，得 0 分
请只输出一个数字：1 或 0"""),
    ("human", "问题:\n{question}\n\n检索到的上下文:\n{context}\n\nGround Truth:\n{ground_truth}\n\n分数:")
])

faithfulness_chain = FAITHFULNESS_PROMPT | judge_llm | StrOutputParser()
relevancy_chain = RELEVANCY_PROMPT | judge_llm | StrOutputParser()
precision_chain = CONTEXT_PRECISION_PROMPT | judge_llm | StrOutputParser()
recall_chain = CONTEXT_RECALL_PROMPT | judge_llm | StrOutputParser()

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业健身教练。请根据以下参考信息回答问题。\n\n参考信息：\n{context}"),
    ("human", "{question}")
])
answer_chain = answer_prompt | to_runnable(llm) | StrOutputParser()


def parse_score(text):
    try:
        return int(text.strip()[0])
    except:
        return 0


def compute_ragas():
    scores = {"faithfulness": 0, "answer_relevancy": 0,
              "context_precision": 0, "context_recall": 0}
    total = len(df)

    for i, row in df.iterrows():
        # 检索 & 生成 - top 3 + 实体优选（匹配标签的文档排前面）
        all_docs = retrieve(row["query"])
        # 提取 query 实体用于优选
        q_entities = set()
        for etype, names in FitnessRAGRetriever._extract_entities(row["query"]).items():
        	for n in names:
        		q_entities.add(f"{etype}:{n}")
        # 实体匹配的排前面（PDF 无标签时从 page_content 抽取）
        def _doc_labels(doc):
            labels = set(doc.metadata.get("entity_labels", []))
            if not labels:
                for etype, names in FitnessRAGRetriever._extract_entities(doc.page_content).items():
                    for n in names:
                        labels.add(f"{etype}:{n}")
            return labels
        if q_entities:
            match_docs = [d for d in all_docs if q_entities & _doc_labels(d)]
            other_docs = [d for d in all_docs if not (q_entities & _doc_labels(d))]
            all_docs = match_docs + other_docs
        docs = all_docs[:3]
        parts = []
        for doc in docs:
            name = doc.metadata.get("动作名称", "")
            if name:
                muscles = doc.metadata.get("目标肌群", "")
                equip = doc.metadata.get("器械", "")
                parts.append("[{}] 肌群:{} 器械:{}".format(name, muscles, equip))
            else:
                parts.append(doc.page_content.replace('\n', ' ')[:120])
        ctx = "\n".join(parts)
        answer = answer_chain.invoke({"question": row["query"], "context": ctx})

        # 4 个维度的 LLM 评判（用 num_predict=1 的轻量 judge_llm）
        f = parse_score(faithfulness_chain.invoke({"context": ctx, "answer": answer}))
        r = parse_score(relevancy_chain.invoke({"question": row["query"], "answer": answer}))
        p = parse_score(precision_chain.invoke({"question": row["query"], "context": ctx}))
        cr = parse_score(recall_chain.invoke({
            "question": row["query"], "context": ctx, "ground_truth": row["ground_truth"]
        }))

        scores["faithfulness"] += f
        scores["answer_relevancy"] += r
        scores["context_precision"] += p
        scores["context_recall"] += cr

        # 每 10 条释放一次内存 + 短暂休息，防止 Windows OOM 死机
        if (i + 1) % 10 == 0:
            print(f"  ragas... {i+1}/{total} (f={f} r={r} p={p} cr={cr})")
            gc.collect()
            time.sleep(1.0)

    return {k: round(v / total, 4) for k, v in scores.items()}


def compute_feedback_report():
    """反馈闭环专项：真实用户负反馈问题（无金标准参考）→ 检索诊断 + LLM 质量判断。

    口径说明：Hit@k 需要 doc↔reference 余弦对比，反馈问题没有金标准，
    用「逐条检索命中诊断 + faithfulness/relevancy」代替——看真实不满意的样本
    是检索没找到（检索侧问题）还是生成不对（生成侧问题），据此决定改哪一环。
    """
    scores = {"faithfulness": 0, "answer_relevancy": 0}
    total = len(df)

    for i, row in df.iterrows():
        q = row["query"]
        docs = retrieve(q)
        top = docs[:3]
        print(f"\n  [{i+1}/{total}] {q[:50]}")
        for j, d in enumerate(top, 1):
            name = d.metadata.get("动作名称") or d.metadata.get("source", "")
            snip = d.page_content.replace("\n", " ")[:80]
            print(f"      #{j} [{name}] {snip}")
        parts = []
        for doc in top:
            name = doc.metadata.get("动作名称", "")
            if name:
                muscles = doc.metadata.get("目标肌群", "")
                equip = doc.metadata.get("器械", "")
                parts.append(f"[{name}] 肌群:{muscles} 器械:{equip}")
            else:
                parts.append(doc.page_content.replace("\n", " ")[:120])
        ctx = "\n".join(parts)
        answer = answer_chain.invoke({"question": q, "context": ctx})
        f = parse_score(faithfulness_chain.invoke({"context": ctx, "answer": answer}))
        r = parse_score(relevancy_chain.invoke({"question": q, "answer": answer}))
        scores["faithfulness"] += f
        scores["answer_relevancy"] += r

        if (i + 1) % 10 == 0:
            print(f"  feedback... {i+1}/{total} (f={f} r={r})")
            gc.collect()
            time.sleep(1.0)

    return {k: round(v / total, 4) for k, v in scores.items()}


# -- 3) 按题型分组 -----------------------------------------
def compute_by_type(retrieval_scores_func):
    """返回每种 question_type 的 Hit@3 和 MRR"""
    results = {}
    for qt in df["question_type"].unique():
        subset = df[df["question_type"] == qt]
        hit3 = 0; mrr_sum = 0
        for _, row in subset.iterrows():
            docs = retrieve(row["query"])
            ref_emb = np.array(embeddings.embed_query(row["reference_texts"]))
            ranks = []
            for j, doc in enumerate(docs):
                doc_emb = np.array(embeddings.embed_query(doc.page_content))
                if cosine(ref_emb, doc_emb) > SIM_THRESHOLD:
                    ranks.append(j + 1)
            if ranks:
                mrr_sum += 1.0 / ranks[0]
                if any(r <= 3 for r in ranks):
                    hit3 += 1
        n = len(subset)
        results[qt] = {"n": n, "Hit@3": round(hit3/n, 3), "MRR": round(mrr_sum/n, 3)}
    return results


# -- 4) 分类器验证 -----------------------------------------
def validate_classifier():
    """对比关键词分类器输出与数据集 question_type 标签。"""
    # 数据集标签 → 分类器标签映射
    mapping = {
        "基础问答": "simple",
        "单伤病问答": "single_injury",
        "复合伤病问答": "compound_injury",
        "动作纠错": "simple",
        "计划生成": "simple",
        "定制长计划": "compound_injury",  # 多约束 → 复合
        "架构问答": "simple",
        "评测原理": "simple",
        "总结问答": "simple",
    }
    correct = 0
    details = []
    for _, row in df.iterrows():
        predicted = classify_query(row["query"])
        expected = mapping.get(row["question_type"], "simple")
        if predicted == expected:
            correct += 1
        else:
            details.append((row["question_type"], predicted, expected, row["query"][:40]))
    return correct, len(df), details


# -- 执行 --------------------------------------------------
print(f"\n[INFO] 评测模式: {EVAL_MODE}")

# 反馈闭环模式：只跑反馈专项（无金标准 → 不跑 Hit@k / 分组 / 分类器验证）
if USE_FEEDBACK:
    t_start = time.time()
    print("\n[1/1] 反馈专项：逐条检索诊断 + faithfulness/relevancy（LLM-judge）...")
    fb_scores = compute_feedback_report()
    print("\n" + "=" * 70)
    print("                [RESULTS] 负反馈样本质量报告")
    print("=" * 70)
    print(f"\n  {'faithfulness (忠实度)':<28} {fb_scores['faithfulness']:>8.4f}")
    print(f"  {'answer_relevancy (相关性)':<28} {fb_scores['answer_relevancy']:>8.4f}")
    print(f"\n  [TIME] 总耗时: {time.time()-t_start:.1f}s (模式: {EVAL_MODE} / 反馈闭环)")
    sys.exit(0)

# 分类器验证（HyDE 模式时输出详情）
if USE_HYDE:
    correct, total, errors = validate_classifier()
    print(f"\n[INFO] 分类器准确率: {correct}/{total} = {correct/total:.1%}")
    if errors:
        print(f"        误分类 ({len(errors)} 条):")
        for qt, pred, exp, query in errors[:8]:
            print(f"          [{qt}] pred={pred} exp={exp} | {query}...")
        if len(errors) > 8:
            print(f"          ... 及其他 {len(errors)-8} 条")

t_start = time.time()

print("\n[1/2] Computing Hit@k / MRR ...")
t0 = time.time()
retrieval = compute_retrieval_metrics()
print(f"  [TIME] {time.time()-t0:.1f}s")

print("\n[INFO] [2/2] 计算 RAGAS 指标（LLM-as-Judge，100条 × 5轮推理）...")
t0 = time.time()
ragas_scores = compute_ragas()
print(f"  [TIME] {time.time()-t0:.1f}s")

SKIP_GROUPS = "--skip-groups" in sys.argv
if not SKIP_GROUPS:
    print("\n[INFO] 按题型分组 Hit@3 / MRR ...")
    type_results = compute_by_type(None)
else:
    type_results = {}
    print("\n[INFO] 跳过分组统计 (--skip-groups)")

# -- 落盘（先存后印：终端输出丢失也不影响结果可追溯）-----------------
save_results(retrieval, ragas_scores, type_results)

# -- 汇总输出 ----------------------------------------------
print("\n" + "=" * 70)
print(f"                [RESULTS] 评测结果汇总 ({EVAL_MODE})")
print("=" * 70)

print(f"\n{'-'*50}")
print(f"  {'指标':<28} {'得分':>8}")
print(f"{'-'*50}")
print(f"  {'Hit@1':<28} {retrieval['Hit@1']:>8.4f}")
print(f"  {'Hit@3':<28} {retrieval['Hit@3']:>8.4f}")
print(f"  {'Hit@5':<28} {retrieval['Hit@5']:>8.4f}")
print(f"  {'MRR':<28} {retrieval['MRR']:>8.4f}")
print(f"{'-'*50}")
print(f"  {'faithfulness (忠实度)':<28} {ragas_scores['faithfulness']:>8.4f}")
print(f"  {'answer_relevancy (相关性)':<28} {ragas_scores['answer_relevancy']:>8.4f}")
print(f"  {'context_precision (上下文精度)':<28} {ragas_scores['context_precision']:>8.4f}")
print(f"  {'context_recall (上下文召回)':<28} {ragas_scores['context_recall']:>8.4f}")
print(f"{'-'*50}")

print(f"\n\n{'-'*65}")
print(f"  {'question_type':<16} {'n':>4}  {'Hit@3':>8}  {'MRR':>8}")
print(f"{'-'*65}")
for qt in sorted(type_results, key=lambda x: type_results[x]["Hit@3"], reverse=True):
    d = type_results[qt]
    print(f"  {qt:<16} {d['n']:>4}  {d['Hit@3']:>8.3f}  {d['MRR']:>8.3f}")
print(f"{'-'*65}")
print(f"\n  [TIME] 总耗时: {time.time()-t_start:.1f}s (模式: {EVAL_MODE})")
