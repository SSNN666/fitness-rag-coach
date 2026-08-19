"""pipeline 本地禁忌降级单测（仅静态方法，不加载服务）。"""
import threading

from pipeline import PipelineService


def test_local_contraindications_alias_resolution():
    """实体词典提取简称"腰突" → 别名映射到"腰间盘突出" → 命中禁忌动作。"""
    m = PipelineService._local_contraindications(["腰突"])
    assert m, "应命中本地禁忌数据"
    assert "深蹲" in m.get("腰间盘突出", [])
    assert "硬拉" in m.get("腰间盘突出", [])


def test_local_contraindications_exact_name():
    m = PipelineService._local_contraindications(["半月板损伤"])
    assert "深蹲" in m.get("半月板损伤", [])


def test_local_contraindications_unknown_injury_empty():
    assert PipelineService._local_contraindications(["量子力学"]) == {}


def test_local_contraindications_excludes_rehab_actions():
    """仅取「禁忌动作」关系：康复动作（如臀桥）不应出现在黑名单。"""
    m = PipelineService._local_contraindications(["腰突"])
    forbidden = m.get("腰间盘突出", [])
    assert "臀桥" not in forbidden
    assert "平板支撑" not in forbidden


# ----------------------------------------------------------------
# 禁忌拒绝（边界拒绝 + 具体原因）
# ----------------------------------------------------------------

def test_contraindication_reason_lookup():
    """半月板损伤→深蹲 的拒绝原因应来自禁忌数据（而非通用文案）。"""
    r = PipelineService._contraindication_reason(["半月板损伤"], "深蹲")
    assert r and "半月板" in r


def test_contraindication_reason_unknown_action():
    assert PipelineService._contraindication_reason(["半月板损伤"], "瑜伽") is None


def test_reject_message_uses_specific_reason():
    """拒绝文案优先用数据原因，查不到时回退通用文案。"""
    msg = PipelineService._reject_message("深蹲", ["半月板损伤"], "膝盖屈伸负重挤压半月板")
    assert "膝盖屈伸负重挤压半月板" in msg
    assert "明确禁忌动作" in msg
    fallback = PipelineService._reject_message("深蹲", ["半月板损伤"])
    assert "可能加重损伤" in fallback


# ----------------------------------------------------------------
# 阶段进度回调（SSE status 帧数据源）
# ----------------------------------------------------------------

def test_answer_refusal_emits_stage_progress():
    """边界拒绝路径：on_stage 至少收到安全分析阶段播报（不加载服务/Milvus）。"""

    class _DummyRetriever:
        def get_contraindications(self, names):
            return {}   # 空 → 走本地禁忌降级数据源

    stages: list[str] = []
    svc = PipelineService(retriever=_DummyRetriever(), llms={}, gateway=None,
                          fact_engine=None, fact_cache=None, store={})
    result = svc.answer("腰突能深蹲吗", on_stage=stages.append)
    assert result.refusal
    assert any("分析" in s for s in stages), stages   # 拒绝路径至少播报首阶段


# ----------------------------------------------------------------
# 硬性禁忌过滤（含否定语境行保留）
# ----------------------------------------------------------------

def test_hard_filter_removes_violating_lines():
    out, filtered = PipelineService._hard_filter(
        "可以试试平板支撑。\n深蹲可以负重训练。\n臀桥也不错。",
        ["深蹲"])
    assert filtered
    assert "深蹲可以负重训练" not in out
    assert "平板支撑" in out and "臀桥也不错" in out


def test_hard_filter_keeps_negation_lines():
    """「避免深蹲」是安全提醒，应保留而非误删；且未删行时不报「已剔除」警告。"""
    out, filtered = PipelineService._hard_filter(
        "建议避免深蹲。\n可以做臀桥强化臀肌。",
        ["深蹲"])
    assert not filtered
    assert "避免深蹲" in out


def test_hard_filter_mixed_lines():
    """混合场景：违规行删除、否定行保留，警告只列被删的动作。"""
    out, filtered = PipelineService._hard_filter(
        "深蹲训练三组。\n建议避免深蹲。\n可以尝试臀桥。",
        ["深蹲"])
    assert filtered
    assert "深蹲训练三组" not in out
    assert "建议避免深蹲" in out
    assert "深蹲" in out.split("[!]")[1] or "已被自动剔除" in out


def test_hard_filter_equipment_suffix_not_a_hit():
    """2 字动作名后接器械后缀不算命中：「划船」不误删「划船机」行。"""
    out, filtered = PipelineService._hard_filter(
        "可以使用划船机进行有氧训练。\n划船三组。",
        ["划船"])
    assert filtered
    assert "划船机" in out          # 器械行保留
    assert "划船三组" not in out    # 真命中行删除


# ----------------------------------------------------------------
# 会话 store LRU 驱逐
# ----------------------------------------------------------------

def test_session_store_lru_eviction():
    """会话数超过 SESSION_MAX_COUNT 时按最久未访问驱逐。"""
    svc = PipelineService(retriever=None, llms=None, gateway=None, store={})
    for i in range(70):
        svc._append_history(f"s{i}", f"q{i}", f"a{i}")
    assert len(svc._store) <= 64
    # 最早建立的会话应被驱逐
    assert "s0" not in svc._store
    # 最近访问的会话应保留：touch s1 后建立 10 个新会话，s1 仍应存活
    svc2 = PipelineService(retriever=None, llms=None, gateway=None, store={})
    for i in range(64):
        svc2._append_history(f"s{i}", "q", "a")
    svc2._history_messages("s0")    # 读取即访问
    for i in range(10):
        svc2._append_history(f"n{i}", "q", "a")
    assert "s0" in svc2._store      # 最近访问过，LRU 不应驱逐
    assert "s1" not in svc2._store  # 未被访问的最老会话被驱逐


# ----------------------------------------------------------------
# stop_event 取消（SSE 断开保护）
# ----------------------------------------------------------------

def test_answer_cancelled_before_generation_returns_early():
    """stop_event 预置位 → 流水线在检索/生成前及时退出。"""

    class _DummyRetriever:
        def get_contraindications(self, names):
            return {}

    stop = threading.Event()
    stop.set()
    svc = PipelineService(retriever=_DummyRetriever(), llms={}, gateway=None,
                          fact_engine=None, fact_cache=None, store={})
    result = svc.answer("深蹲练什么肌肉", stop_event=stop)
    assert result.answer == ""           # 未进入生成即退出
    assert result.request_id             # 仍返回带 request_id 的占位结果


class _Chunk:
    def __init__(self, text=""):
        self.text = text
        self.usage = None
        self.fallback_switch = False


class _FakeGateway:
    """记录型假网关：默认放行/记录，不触网。"""
    def log_cycle(self, *a, **k): pass
    def log_retrieval(self, *a, **k): pass
    def log_prompt(self, *a, **k): pass
    def log_answer(self, *a, **k): pass
    def log_usage(self, *a, **k): pass
    def guard_token_budget(self, system_prompt, history, session_id, ctx, query):
        return ctx
    def get_active_llm(self, llm): return llm


class _FakeResp:
    content = "深蹲康复训练草稿"
    usage = None
    fallback = False
    error_kind = None


class _FakeChain:
    active_providers = []

    def __init__(self, stop_event):
        self._stop = stop_event

    def invoke(self, messages, **kw):
        return _FakeResp()

    def stream_events(self, messages, **kw):
        yield _Chunk("第一段回答。")
        self._stop.set()          # 模拟客户端在流中途断开
        yield _Chunk("第二段回答。")


def test_answer_stop_event_interrupts_streaming(monkeypatch):
    """流式中 stop_event 置位 → 及时中断，丢弃部分结果且不写历史。"""
    import pipeline as pipeline_mod
    # 注：pipeline 用 `from config import *` 拉取模块级常量，需 patch pipeline 模块本身
    monkeypatch.setattr(pipeline_mod, "REFUSE_ENABLED", False)   # 检索为空时不拒答，走生成路径

    stop = threading.Event()

    class _DummyRetriever:
        def get_contraindications(self, names):
            return {}
        # 无 similarity_search/search_with_scores → 检索异常路径（try/except 兜底）

    store: dict = {}
    svc = PipelineService(
        retriever=_DummyRetriever(),
        llms={"hyde": _FakeChain(stop), "chat_fast": _FakeChain(stop),
              "chat": _FakeChain(stop)},
        gateway=_FakeGateway(), fact_engine=None, fact_cache=None, store=store)

    deltas: list[str] = []
    result = svc.answer("深蹲练什么肌肉", stop_event=stop, on_delta=deltas.append)
    assert result.answer == "第一段回答。"   # 第二段到来前已取消
    assert deltas == ["第一段回答。"]
    assert store == {}                    # 取消时不写会话历史


class _SpyChain(_FakeChain):
    """记录型假链：统计 invoke/stream_events 调用次数。"""

    def __init__(self, stop_event):
        super().__init__(stop_event)
        self.invokes = 0
        self.streams = 0

    def invoke(self, messages, **kw):
        self.invokes += 1
        return _FakeResp()

    def stream_events(self, messages, **kw):
        self.streams += 1
        return super().stream_events(messages, **kw)


class _DummyRetriever:
    def get_contraindications(self, names):
        return {}
    # 无 similarity_search/search_with_scores → 检索异常路径（try/except 兜底）


def test_simple_tier_skips_hyde(monkeypatch):
    """simple 层跳过 HyDE 草稿生成（省 1 次 LLM 调用）：直查结果即引用来源。"""
    import pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "REFUSE_ENABLED", False)

    stop = threading.Event()
    hyde = _SpyChain(stop)
    chat_fast = _SpyChain(stop)
    store: dict = {}
    svc = PipelineService(
        retriever=_DummyRetriever(),
        llms={"hyde": hyde, "chat_fast": chat_fast, "chat": _SpyChain(stop)},
        gateway=_FakeGateway(), fact_engine=None, fact_cache=None, store=store)
    svc.answer("深蹲练什么肌肉")
    assert hyde.invokes == 0       # simple 层不生成 HyDE 草稿
    assert chat_fast.invokes == 1  # 直查后正常生成


def test_injury_tier_still_uses_hyde(monkeypatch):
    """injury 层仍走 HyDE（伤病问题依赖草稿改写召回）。"""
    import pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "CRAG_ENABLED", False)   # 防测试触网

    stop = threading.Event()
    hyde = _SpyChain(stop)
    svc = PipelineService(
        retriever=_DummyRetriever(),
        llms={"hyde": hyde, "chat_fast": _SpyChain(stop), "chat": _SpyChain(stop)},
        gateway=_FakeGateway(), fact_engine=None, fact_cache=None, store={})
    svc.answer("腰突怎么康复")
    assert hyde.invokes == 1       # injury 层保留 HyDE


# ----------------------------------------------------------------
# 引用片段清洗（_clean_snippet）
# ----------------------------------------------------------------

def test_clean_snippet_strips_tail_residue():
    """尾部孤立数字/标点残渣（OCR 常见 "2," 尾巴）应被剔除。"""
    out = PipelineService._clean_snippet("正常的训练建议内容 2,", 120)
    assert out == "正常的训练建议内容"


def test_clean_snippet_whitespace_normalized():
    """OCR 行内换行折叠为空格，多空格合并。"""
    out = PipelineService._clean_snippet("杠\n铃 施加于  肩部的压力", 120)
    assert out == "杠 铃 施加于 肩部的压力"


def test_clean_snippet_sentence_boundary_truncation():
    """超长片段停在完整句子边界，不从词中间切断。"""
    text = "深蹲锻炼股四头肌和臀大肌。深蹲时注意膝盖与脚尖方向一致。后面还有更多内容。"
    out = PipelineService._clean_snippet(text, 30)
    # 30 字符窗口内最后一句到「一致。」为止（后半句被截掉，不留半个词）
    assert out.endswith("一致。")
    assert "后面还有" not in out


def test_clean_snippet_no_boundary_falls_back_to_ellipsis():
    """窗口内无句子标点 → 硬截断 + 省略号。"""
    out = PipelineService._clean_snippet("没有句号结尾的长文本" * 8, 40)
    assert out.endswith("…") and len(out) == 41


def test_clean_snippet_empty_and_residue_only():
    assert PipelineService._clean_snippet("") == ""
    assert PipelineService._clean_snippet("  \n 2, ") == ""


# ----------------------------------------------------------------
# 确定性健康工具注入（Phase E）+ 多轮查询改写（Phase B/C）
# ----------------------------------------------------------------

class _SearchableRetriever:
    """可完成检索的桩：search_with_scores 记录 query 并返回单文档。"""
    def __init__(self):
        self.seen_queries: list[str] = []
        self._last_query_embedding = None

    def get_contraindications(self, names):
        return {}

    def search_with_scores(self, query, k=3, use_rerank=False):
        self.seen_queries.append(query)
        from langchain_core.documents import Document
        doc = Document(
            page_content="深蹲动作要点：膝关节与脚尖方向保持一致，核心收紧，"
                         "下蹲至大腿与地面平行后起身。",
            metadata={"source": "fitness_data.csv", "动作名称": "深蹲"})
        return [(doc, 0.9)]

    def _embed(self, text):
        return [0.1] * 768

    def set_degraded(self, v):
        pass


class _CaptureChain(_FakeChain):
    """记录型假链：捕获最近一次 invoke 的 messages。"""
    def __init__(self, stop_event):
        super().__init__(stop_event)
        self.last_messages = None

    def invoke(self, messages, **kw):
        self.last_messages = messages
        return _FakeResp()


def test_tool_results_injected_into_context(monkeypatch):
    """健康工具触发 → [工具计算结果] 注入 system 上下文 + 结构化引用 kind=tool。"""
    import pipeline as pipeline_mod
    from health_tools import ToolResult

    monkeypatch.setattr(pipeline_mod, "REFUSE_ENABLED", False)   # 检索空路径不拒答
    monkeypatch.setattr(pipeline_mod, "HEALTH_TOOLS_ENABLED", True)
    monkeypatch.setattr(
        pipeline_mod, "run_health_tools",
        lambda q, p: [ToolResult(name="calculate_bmi", title="BMI 计算",
                                 content="身高170cm、体重70kg → BMI=24.2，属超重")])

    stop = threading.Event()
    chat_fast = _CaptureChain(stop)
    svc = PipelineService(
        retriever=_SearchableRetriever(),
        llms={"hyde": _SpyChain(stop), "chat_fast": chat_fast, "chat": _SpyChain(stop)},
        gateway=_FakeGateway(), fact_engine=None, fact_cache=None, store={})
    result = svc.answer("帮我算下BMI", user_profile="身高170cm，体重70kg，目标：减脂")

    sys_content = chat_fast.last_messages[0]["content"]
    assert "[工具计算结果]" in sys_content
    assert "BMI=24.2" in sys_content
    # 工具结果作为引用暴露（前端按 kind 渲染卡片）
    assert any(c["kind"] == "tool" and c["source"] == "BMI 计算" for c in result.citations)


def test_tool_injection_disabled_no_context_marker(monkeypatch):
    """HEALTH_TOOLS_ENABLED=False → 无 [工具计算结果] 标记。"""
    import pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "REFUSE_ENABLED", False)
    monkeypatch.setattr(pipeline_mod, "HEALTH_TOOLS_ENABLED", False)

    stop = threading.Event()
    chat_fast = _CaptureChain(stop)
    svc = PipelineService(
        retriever=_SearchableRetriever(),
        llms={"hyde": _SpyChain(stop), "chat_fast": chat_fast, "chat": _SpyChain(stop)},
        gateway=_FakeGateway(), fact_engine=None, fact_cache=None, store={})
    svc.answer("帮我算下BMI", user_profile="身高170cm，体重70kg")
    assert "[工具计算结果]" not in chat_fast.last_messages[0]["content"]


def test_cache_key_profile_fingerprint_when_tools_triggered():
    """工具触发时缓存 key 含画像指纹：同问题不同画像不共享缓存（防旧画像答案串用）。"""
    from health_tools import ToolResult
    t = [ToolResult(name="calculate_bmi", title="BMI 计算", content="x")]
    k1 = PipelineService._cache_key_entities(
        "帮我算下BMI", False, t, "身高170cm，体重70kg")
    k2 = PipelineService._cache_key_entities(
        "帮我算下BMI", False, t, "身高180cm，体重80kg")
    assert k1 != k2
    # 无工具触发时画像不参与 key（与旧行为一致，不误伤缓存命中率）
    k0 = PipelineService._cache_key_entities("帮我算下BMI", False, None, "身高180cm，体重80kg")
    assert k0 == PipelineService._cache_key_entities("帮我算下BMI", False, None, None)


def test_rewrite_used_for_retrieval(monkeypatch):
    """多轮指代问句（非禁忌动作）→ 改写后的 retrieval_q 用于检索；原 question 仍进 Prompt。"""
    import pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "REFUSE_ENABLED", False)
    monkeypatch.setattr(pipeline_mod, "REWRITE_ENABLED", True)
    monkeypatch.setattr(pipeline_mod, "CRAG_ENABLED", False)
    monkeypatch.setattr(pipeline_mod, "rewrite_query",
                        lambda q, h, llm: "腰突患者可以做臀桥吗")

    stop = threading.Event()
    retriever = _SearchableRetriever()
    chat_fast = _CaptureChain(stop)
    svc = PipelineService(
        retriever=retriever,
        llms={"hyde": _SpyChain(stop), "chat_fast": chat_fast, "chat": _SpyChain(stop)},
        gateway=_FakeGateway(), fact_engine=None, fact_cache=None, store={})
    svc._append_history("default", "腰突怎么康复", "建议平板支撑等康复动作。")
    svc.answer("那臀桥呢", session_id="default")

    assert retriever.seen_queries and retriever.seen_queries[0] == "腰突患者可以做臀桥吗"
    assert chat_fast.last_messages[-1]["content"] == "那臀桥呢"   # Prompt 用原问题


def test_rewrite_contraindication_intercepted(monkeypatch):
    """多轮改写补充的伤病实体 → 禁忌复查：改写后问题含禁忌动作 → 直接拒绝。

    安全防线断链回归：改写前原问题「那硬拉呢」无伤病实体，Phase A 按「无需禁忌」放行；
    改写后实体合并出「腰突」，必须重跑边界拒绝，否则硬拉（明确禁忌）会被正常推荐。
    """
    import pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "REWRITE_ENABLED", True)
    monkeypatch.setattr(pipeline_mod, "rewrite_query",
                        lambda q, h, llm: "腰突患者可以做硬拉吗")

    stop = threading.Event()
    svc = PipelineService(
        retriever=_DummyRetriever(),   # get_contraindications → {} → 本地禁忌降级数据源
        llms={"hyde": _SpyChain(stop), "chat_fast": _SpyChain(stop),
              "chat": _SpyChain(stop)},
        gateway=_FakeGateway(), fact_engine=None, fact_cache=None, store={})
    svc._append_history("default", "腰突怎么康复", "建议平板支撑等康复动作。")
    result = svc.answer("那硬拉呢", session_id="default")
    assert result.refusal
    assert "硬拉" in result.answer and "禁忌" in result.answer


def test_rewrite_merged_contra_injected_into_prompt(monkeypatch):
    """改写补充伤病但不含禁忌动作 → 不拒绝，但禁忌黑名单注入生成提示（防线仍在）。"""
    import pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "REFUSE_ENABLED", False)
    monkeypatch.setattr(pipeline_mod, "REWRITE_ENABLED", True)
    monkeypatch.setattr(pipeline_mod, "CRAG_ENABLED", False)
    monkeypatch.setattr(pipeline_mod, "rewrite_query",
                        lambda q, h, llm: "腰突患者可以做划船吗")

    stop = threading.Event()
    retriever = _SearchableRetriever()
    chat_fast = _CaptureChain(stop)
    svc = PipelineService(
        retriever=retriever,
        llms={"hyde": _SpyChain(stop), "chat_fast": chat_fast, "chat": _SpyChain(stop)},
        gateway=_FakeGateway(), fact_engine=None, fact_cache=None, store={})
    svc._append_history("default", "腰突怎么康复", "建议平板支撑等康复动作。")
    result = svc.answer("那划船呢", session_id="default")
    assert not result.refusal
    sys_content = chat_fast.last_messages[0]["content"]
    assert "伤病禁忌黑名单" in sys_content and "深蹲" in sys_content   # 合并后的禁忌注入
    assert retriever.seen_queries[0] == "腰突患者可以做划船吗"


def test_rewrite_disabled_uses_original_question(monkeypatch):
    """REWRITE_ENABLED=False → 检索直接用原问题（回归保护）。"""
    import pipeline as pipeline_mod
    monkeypatch.setattr(pipeline_mod, "REFUSE_ENABLED", False)
    monkeypatch.setattr(pipeline_mod, "REWRITE_ENABLED", False)
    monkeypatch.setattr(pipeline_mod, "CRAG_ENABLED", False)

    stop = threading.Event()
    retriever = _SearchableRetriever()
    svc = PipelineService(
        retriever=retriever,
        llms={"hyde": _SpyChain(stop), "chat_fast": _SpyChain(stop),
              "chat": _SpyChain(stop)},
        gateway=_FakeGateway(), fact_engine=None, fact_cache=None, store={})
    svc._append_history("default", "腰突能深蹲吗", "不建议深蹲，会加重腰椎负担。")
    svc.answer("那硬拉呢", session_id="default")
    assert retriever.seen_queries[0] == "那硬拉呢"
