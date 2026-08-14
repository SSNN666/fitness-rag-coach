"""
grounding.py —— 知识库依据判定（无依据拒答）
============================================
在生成前判定检索结果是否足以支撑回答（确定性强于生成后让 LLM 自己说不知道）。
复用 app.py 原有 knowledge_gap / ctx_short 逻辑，抽成纯函数。

融合模式语义差异（面试可讲）：
  - weighted 模式：分数是加权累加值，有明确的数值阈值（FUSION_THRESHOLD）
  - rrf 模式：分数是 Σ1/(k+rank) 的小数值，阈值语义不同 → 只看"有无文档通过"，不比数值

用法:
    from grounding import assess_grounding, refusal_message
    verdict = assess_grounding(ctx, scored_docs, has_external=False)
    if not verdict.grounded: return refusal_message(question, verdict)
"""

from dataclasses import dataclass


@dataclass
class GroundingVerdict:
    grounded: bool
    reason: str            # "ok" | "no_docs" | "low_score" | "ctx_too_short" | "external_only"
    max_score: float = 0.0
    n_docs: int = 0
    ctx_chars: int = 0
    external: bool = False


def assess_grounding(ctx: str,
                     scored_docs: list | None,
                     has_external: bool = False,
                     relevance: float | None = None,
                     has_entities: bool = False) -> GroundingVerdict:
    """判定检索依据是否充分（规则短路）：

    1. 有外部资料（CRAG 补充成功）→ grounded("external_only"，引用标注联网来源)
    2. 无文档 → not_grounded("no_docs")
    3. 无实体查询 + 语义相关性低于 GROUNDING_MIN_SIM → not_grounded("low_relevance")
       （小语料下纯余弦有噪声，故该门槛仅对"无实体"查询生效，避免误伤健身问题）
    4. weighted 模式：最高融合分 < 阈值 → not_grounded("low_score")；rrf 模式跳过数值比较
    5. 上下文过短且无外部资料 → not_grounded("ctx_too_short")
    6. 其余 → grounded("ok")
    """
    import config as cfg

    ctx_chars = len(ctx or "")
    docs = scored_docs or []
    n_docs = len(docs)

    # 外部资料优先判定：CRAG 补充成功即视为有依据（引用标注联网来源）
    if has_external and ctx_chars > 0:
        max_score = max((s for _, s in docs), default=0.0)
        return GroundingVerdict(grounded=True, reason="external_only",
                                max_score=max_score, n_docs=n_docs,
                                ctx_chars=ctx_chars, external=True)

    if n_docs == 0:
        return GroundingVerdict(grounded=False, reason="no_docs",
                                n_docs=0, ctx_chars=ctx_chars, external=has_external)

    max_score = max((s for _, s in docs), default=0.0)

    if relevance is not None and not has_entities:
        min_sim = getattr(cfg, "GROUNDING_MIN_SIM", 0.40)  # 与 config 默认一致（宁拒少答；换 Embedding 需重新校准）
        if relevance < min_sim:
            return GroundingVerdict(grounded=False, reason="low_relevance",
                                    max_score=max_score, n_docs=n_docs,
                                    ctx_chars=ctx_chars)

    mode = getattr(cfg, "FUSION_MODE", "weighted")
    if mode == "weighted":
        threshold = getattr(cfg, "GROUNDING_MIN_SCORE", None) or \
            getattr(cfg, "FUSION_THRESHOLD", 0.15)
        if max_score < threshold:
            return GroundingVerdict(grounded=False, reason="low_score",
                                    max_score=max_score, n_docs=n_docs,
                                    ctx_chars=ctx_chars)

    min_chars = getattr(cfg, "GROUNDING_MIN_CTX_CHARS", 80)
    if ctx_chars < min_chars:
        return GroundingVerdict(grounded=False, reason="ctx_too_short",
                                max_score=max_score, n_docs=n_docs,
                                ctx_chars=ctx_chars)

    return GroundingVerdict(grounded=True, reason="ok", max_score=max_score,
                            n_docs=n_docs, ctx_chars=ctx_chars)


def refusal_message(question: str, verdict: GroundingVerdict) -> str:
    """拒答模板。REFUSE_ENABLED=False 时 pipeline 改走"通用知识 + 显著免责声明"路径。"""
    return (
        f"抱歉，知识库中暂无与「{question}」相关的可靠资料，"
        "为避免给出无依据的回答，本次不回答。\n\n"
        "建议：\n"
        "1. 换一种问法，或提供更具体的症状/场景描述；\n"
        "2. 健康问题请咨询专业医师或营养师。"
    )
