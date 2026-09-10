"""retriever 单测：实体 boost / 图谱冲突检测 / query 向量复用 / 语义去重开关（无真实 Milvus/Neo4j）。"""
import numpy as np
import pytest

from langchain_core.documents import Document

from retriever import FitnessRAGRetriever


class _FakeBM25:
    def get_scores(self, tokens):
        return np.zeros(1)   # rank-bm25 返回 numpy 数组（.min()/.max()/argsort）


def _make_retriever(embed_fn=None):
    return FitnessRAGRetriever(
        milvus_client=None,
        bm25_index=(_FakeBM25(), [Document(page_content="x")]),
        neo4j_driver=None,
        embedding_fn=embed_fn or (lambda t: [0.0] * 4),
        fusion_threshold=0.0,
    )


# ----------------------------------------------------------------
# 实体匹配 boost（注释与实现一致性）
# ----------------------------------------------------------------

def test_entity_match_boost_weights():
    r = _make_retriever()
    docs = [
        (Document(page_content="深蹲训练", metadata={"entity_labels": ["exercise:深蹲"]}), 1.0),
        (Document(page_content="卧推训练", metadata={"entity_labels": ["exercise:卧推"]}), 1.0),
    ]
    boosted = r._entity_match_boost("深蹲", docs)
    scores = {d.page_content: s for d, s in boosted}
    assert scores["深蹲训练"] == pytest.approx(1.5)   # 实体匹配 → ×1.5
    assert scores["卧推训练"] == pytest.approx(0.85)  # 不匹配 → ×0.85


# ----------------------------------------------------------------
# 图谱路径冲突检测（relation 元数据参与判定）
# ----------------------------------------------------------------

def test_merge_conflicts_marks_contradiction():
    """同一目标动作被标注为禁忌+康复 → 冲突路径应被标注。"""
    r = _make_retriever()
    d1 = Document(page_content="腰突 → 深蹲", metadata={"relation": "禁忌动作"})
    d2 = Document(page_content="腰突 → 深蹲", metadata={"relation": "康复动作"})
    d3 = Document(page_content="腰突 → 臀桥", metadata={"relation": "康复动作"})
    merged = r._merge_conflicts([(d1, 0.9), (d2, 0.8), (d3, 0.7)])

    def _target(doc):
        # 冲突标注追加在路径文本之后，取目标动作需先截断标记行
        return doc.page_content.split(" → ")[-1].split("\n")[0]

    by_target = {_target(d): d for d, _ in merged}
    assert "冲突" in by_target["深蹲"].page_content
    assert "冲突" not in by_target["臀桥"].page_content


def test_merge_conflicts_same_label_no_false_positive():
    """同类别关系（禁忌 vs 禁忌）不同描述不误报冲突。"""
    r = _make_retriever()
    d1 = Document(page_content="腰突 → 深蹲", metadata={"relation": "禁忌动作: 深蹲压迫椎间盘"})
    d2 = Document(page_content="腰突 → 深蹲", metadata={"relation": "禁忌动作: 深蹲挤压椎间盘"})
    merged = r._merge_conflicts([(d1, 0.9), (d2, 0.8)])
    assert "冲突" not in merged[0][0].page_content and "冲突" not in merged[1][0].page_content


# ----------------------------------------------------------------
# query 向量复用（search_with_scores 只嵌入一次）
# ----------------------------------------------------------------

class _FakeMilvus:
    def __init__(self):
        self.seen = []

    def search(self, **kw):
        self.seen.append(kw["data"][0])
        return [[{"entity": {"page_content": "深蹲训练", "metadata_json": "{}"},
                  "distance": 0.8}]]


def test_search_with_scores_reuses_query_embedding(monkeypatch):
    """query 向量只嵌入一次：缓存供 grounding 相关性判定复用。"""
    import config as cfg
    monkeypatch.setattr(cfg, "FUSION_SEMANTIC_DEDUP_ENABLED", False)
    calls = []
    milvus = _FakeMilvus()

    def embed(text):
        calls.append(text)
        return [0.1, 0.2, 0.3]

    r = FitnessRAGRetriever(
        milvus_client=milvus,
        bm25_index=(_FakeBM25(), [Document(page_content="x")]),
        neo4j_driver=None, embedding_fn=embed, fusion_threshold=0.0)
    r.search_with_scores("深蹲", k=1)
    assert calls == ["深蹲"]                                # 只嵌入一次
    assert r._last_query_embedding == [0.1, 0.2, 0.3]       # 缓存可供复用
    assert milvus.seen == [[0.1, 0.2, 0.3]]                 # 检索用的是同一个向量


# ----------------------------------------------------------------
# 语义去重开关（FUSION_SEMANTIC_DEDUP_ENABLED 默认关，避免每请求 N 次 embedding）
# ----------------------------------------------------------------

def test_semantic_dedup_off_by_default(monkeypatch):
    """配置关闭时融合不去重 → 全部候选保留且不做候选 embedding。"""
    import config as cfg
    monkeypatch.setattr(cfg, "FUSION_SEMANTIC_DEDUP_ENABLED", False)
    calls = []
    r = _make_retriever(embed_fn=lambda t: calls.append(t) or [0.5, 0.5])
    docs = [(Document(page_content=f"d{i}", metadata={}), 0.5) for i in range(5)]
    out = r._weighted_fusion([docs, [], []], [1.0, 0.0, 0.0], top_k=None)
    assert len(out) == 5        # 未去重
    assert calls == []          # 未做任何候选 embedding


def test_semantic_dedup_on_embeds_and_dedups(monkeypatch):
    """配置开启时融合才做语义去重（默认关 → 每请求省 N 次 embedding）。"""
    import config as cfg
    monkeypatch.setattr(cfg, "FUSION_SEMANTIC_DEDUP_ENABLED", True)
    r = _make_retriever(embed_fn=lambda t: [0.5, 0.5])
    docs = [(Document(page_content=f"d{i}", metadata={}), 0.5) for i in range(5)]
    out = r._weighted_fusion([docs, [], []], [1.0, 0.0, 0.0], top_k=None)
    assert len(out) == 3        # 5 条两两相似 → 只保留 FUSION_MIN_DOCS 条


# ----------------------------------------------------------------
# 传入预算好的 query 向量：锁外 embedding（延迟关键路径）
# ----------------------------------------------------------------
#
# 背景：云端 embedding 是一次网络往返（实测中位 158ms、最坏 800ms+）。原实现
# 在 search_with_scores 内部现算，而该函数在全局锁内被调用 → 所有并发请求的
# 检索段排队等这一次 HTTP（实测 simple 层 C_retrieve 持有时长的 ~68% 是它）。
# 现在调用方可在锁外算好、传 query_vec 进来。

def _retriever_with(embed, monkeypatch):
    import config as cfg
    monkeypatch.setattr(cfg, "FUSION_SEMANTIC_DEDUP_ENABLED", False)
    return FitnessRAGRetriever(
        milvus_client=_FakeMilvus(),
        bm25_index=(_FakeBM25(), [Document(page_content="x")]),
        neo4j_driver=None, embedding_fn=embed, fusion_threshold=0.0)


def test_query_vec_skips_embedding(monkeypatch):
    """传了 query_vec 就不该再调 embedding——否则「移出锁」等于没做。"""
    calls = []
    r = _retriever_with(lambda t: (calls.append(t), [0.1, 0.2, 0.3])[1], monkeypatch)
    r.search_with_scores("深蹲", k=1, query_vec=[0.5, 0.6, 0.7])
    assert calls == [], f"不应触发 embedding，实际调了 {calls}"
    assert r._last_query_embedding == [0.5, 0.6, 0.7]   # 缓存仍需写入（grounding 复用）
    assert r._milvus.seen == [[0.5, 0.6, 0.7]]          # 检索用的是传入向量


def test_query_vec_none_falls_back_to_embedding(monkeypatch):
    """不传时退回原行为——旧调用方（MCP / eval / demo）不受影响。"""
    calls = []
    r = _retriever_with(lambda t: (calls.append(t), [0.1, 0.2, 0.3])[1], monkeypatch)
    r.search_with_scores("深蹲", k=1)
    assert calls == ["深蹲"]


def test_query_vec_not_used_when_retrieval_degraded(monkeypatch):
    """稀疏降级模式下不走向量检索，也不该把传入向量写进缓存（与原有语义一致）。"""
    calls = []
    r = _retriever_with(lambda t: (calls.append(t), [0.1])[1], monkeypatch)
    r.set_degraded(True)
    r.search_with_scores("深蹲", k=1, query_vec=[0.5, 0.6, 0.7])
    assert calls == []
    assert r._last_query_embedding is None
