"""grounding 单测：拒答判定各分支。"""
from grounding import assess_grounding, refusal_message

LONG_CTX = "深蹲是练腿的王牌动作，注意腰背挺直避免受伤。" * 8  # > 80 字


def test_no_docs():
    v = assess_grounding("", None)
    assert v.grounded is False
    assert v.reason == "no_docs"


def test_external_only_grounded():
    v = assess_grounding("[联网检索资料] 梨状肌综合征的康复建议...", None, has_external=True)
    assert v.grounded is True
    assert v.reason == "external_only"


def test_low_relevance_no_entities_refused():
    # GROUNDING_MIN_SIM=0.40（qwen 嵌入校准）：相关性低于阈值且无实体 → 拒答
    v = assess_grounding(LONG_CTX, [(object(), 0.5)], relevance=0.3, has_entities=False)
    assert v.grounded is False
    assert v.reason == "low_relevance"


def test_relevance_above_threshold_passes():
    v = assess_grounding(LONG_CTX, [(object(), 0.5)], relevance=0.5, has_entities=False)
    assert v.grounded is True


def test_entity_queries_skip_relevance_gate():
    v = assess_grounding(LONG_CTX, [(object(), 0.5)], relevance=0.3, has_entities=True)
    assert v.grounded is True


def test_low_fusion_score_refused():
    v = assess_grounding(LONG_CTX, [(object(), 0.05)])
    assert v.grounded is False
    assert v.reason == "low_score"


def test_ctx_too_short_refused():
    v = assess_grounding("短", [(object(), 0.5)])
    assert v.grounded is False
    assert v.reason == "ctx_too_short"


def test_ok():
    v = assess_grounding(LONG_CTX, [(object(), 0.5)])
    assert v.grounded is True
    assert v.reason == "ok"


def test_refusal_message_mentions_question():
    msg = refusal_message("如何治愈癌症", assess_grounding("", None))
    assert "如何治愈癌症" in msg
    assert "不回答" in msg
