"""hyde 分类器单测：伤病实体计数 / 复合标记独立判定（防单伤病误判复合回归）。"""
from types import SimpleNamespace

from hyde import classify_query, needs_rewrite, rewrite_query


class TestClassifyQuery:
    def test_simple_no_injury(self):
        assert classify_query("深蹲主要锻炼哪些肌群") == "simple"

    # ---- 单伤病（关键词粗计数时代的回归用例）----
    # "肩袖损伤"曾被拆成"肩袖"+"损伤"算 2 个实体 → 误判复合
    def test_single_injury_with_suffix_word(self):
        assert classify_query("肩袖损伤怎么练") == "single_injury"
        assert classify_query("我有半月板损伤，训练怎么安排") == "single_injury"
        assert classify_query("半月板撕裂能深蹲吗") == "single_injury"
        assert classify_query("腰间盘突出怎么康复") == "single_injury"

    # "梨状肌综合征"的"综合征"含"综合"子串 → 曾误命中复合标记词
    def test_single_injury_marker_substring(self):
        assert classify_query("我有梨状肌综合征，训练怎么安排") == "single_injury"
        assert classify_query("腰突怎么康复") == "single_injury"

    # 部位名是伤病名组成部分 → 不算独立部位，仍单伤病
    def test_single_injury_body_part_is_substring(self):
        assert classify_query("骨盆前倾怎么纠正") == "single_injury"
        assert classify_query("膝关节积液能跑步吗") == "single_injury"

    # ---- 复合伤病 ----
    def test_compound_two_injuries(self):
        assert classify_query("腰突合并肩袖损伤怎么练") == "compound_injury"
        assert classify_query("骶髂关节炎和膝内扣能深蹲吗") == "compound_injury"

    def test_compound_injury_plus_independent_body_part(self):
        # 伤病 + 独立部位（"膝盖"不是"腰突"的组成部分）
        assert classify_query("腰突加膝盖疼怎么安排") == "compound_injury"

    # 多字标记词需独立出现（前后非汉字）——"综合征"内的"综合"不触发；
    # "综合训练"这类修饰用法也保守不触发（宁可单伤病 1-hop，不误判复合跑 3-hop）
    def test_marker_standalone_only(self):
        assert classify_query("腰突，综合，训练怎么安排") == "compound_injury"
        assert classify_query("腰突综合症怎么处理") == "single_injury"  # "综合"嵌在词内
        assert classify_query("腰突，综合训练怎么安排") == "single_injury"


# ----------------------------------------------------------------
# 多轮查询改写（needs_rewrite / rewrite_query）
# ----------------------------------------------------------------

_HIST = [{"role": "user", "content": "腰突能深蹲吗"},
         {"role": "assistant", "content": "不建议，深蹲会加重腰椎负担…"}]


class TestNeedsRewrite:
    def test_pronoun_followup_requires_rewrite(self):
        """「那硬拉呢」含指代词且历史非空 → 改写（否则检索缺「腰突」上下文）。"""
        assert needs_rewrite("那硬拉呢", _HIST) is True

    def test_no_history_no_rewrite(self):
        assert needs_rewrite("那硬拉呢", []) is False

    def test_self_contained_with_entity_no_rewrite(self):
        """问题自带伤病/动作实体 → 自包含，无需改写。"""
        assert needs_rewrite("腰突能深蹲吗", _HIST) is False
        assert needs_rewrite("深蹲练什么肌肉", _HIST) is False

    def test_plan_query_no_rewrite(self):
        assert needs_rewrite("帮我制定周训练计划", _HIST) is False

    def test_vague_question_with_history_requires_rewrite(self):
        """无实体无指代词但历史非空（如「给我讲讲」）→ 保守改写。"""
        assert needs_rewrite("给我讲讲", _HIST) is True

    def test_hint_word_inside_entity_not_fooled(self):
        """「踝关节」里的「关」不影响判定——检查的是整词列表而非单字。"""
        assert needs_rewrite("踝关节扭伤怎么处理", _HIST) is False  # 含伤病实体


class _FakeRewriteLLM:
    def __init__(self, content="腰突患者可以做硬拉吗"):
        self.content = content
        self.calls = []

    def invoke(self, messages, **kw):
        self.calls.append((messages, kw))
        return SimpleNamespace(content=self.content)


class TestRewriteQuery:
    def test_rewrite_returns_llm_output(self):
        llm = _FakeRewriteLLM()
        out = rewrite_query("那硬拉呢", _HIST, llm)
        assert out == "腰突患者可以做硬拉吗"
        # 改写 prompt 中应包含历史上下文（供指代消解）
        assert "腰突能深蹲吗" in llm.calls[0][0][0]["content"]
        assert llm.calls[0][1]["max_tokens"] == 64   # 小模型短预算

    def test_rewrite_strips_quotes(self):
        llm = _FakeRewriteLLM('"腰突患者可以做硬拉吗"')
        assert rewrite_query("那硬拉呢", _HIST, llm) == "腰突患者可以做硬拉吗"

    def test_rewrite_empty_output_falls_back(self):
        llm = _FakeRewriteLLM("   ")
        assert rewrite_query("那硬拉呢", _HIST, llm) == "那硬拉呢"

    def test_rewrite_oversized_output_falls_back(self):
        llm = _FakeRewriteLLM("长" * 201)
        assert rewrite_query("那硬拉呢", _HIST, llm) == "那硬拉呢"

    def test_rewrite_exception_falls_back(self):
        class _Boom:
            def invoke(self, messages, **kw):
                raise RuntimeError("LLM 不可用")
        assert rewrite_query("那硬拉呢", _HIST, _Boom()) == "那硬拉呢"
