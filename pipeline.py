"""
pipeline.py —— PipelineService：12 步安全流水线（自 app.py 抽取，服务端复用）
=============================================================================
Streamlit 应用重构为 SSE 客户端后，原 app.py 的 Phase 1 流水线迁移至此，
由 FastAPI（api.py lifespan）持有。session_state 全部改为实例变量/参数。

12 步（安全顺序不可乱）：
  1. 实体抽取             FitnessRAGRetriever._extract_entities
  2. 禁忌名单查询         Neo4j（NEO4J_ENABLED=False 时跳过，返回空）
  3. 边界拒绝检查         核心诉求=禁忌动作 → 直接拒绝
  4. HyDE + 三路检索      hyde_retrieve（小模型），一次检索同时出 docs 供引用
  5. 知识盲区 → CRAG      博查联网搜索（触发词单一来源 config）
  6. grounding 拒答       无依据 → 拒答（生成前判定，确定性强）
  7. 分层 Prompt          普通/伤病/计划 + 禁忌黑名单注入
  8. 令牌预算守卫         gateway.guard_token_budget（级联截断历史/上下文）
  9. 降级链生成           FallbackChain（云 → 本地 Ollama）
  10. 硬性禁忌过滤        确定性规则，不依赖 LLM
  11. Fact-Check          四类校验（小模型）+ 缓存 + CRAG 修正
  12. 结构化引用          联网 / 图谱 / 本地KB 分流（与回答正文分离）

线程模型（五阶段 A-F）：
  锁内（A/C/E）：实体/禁忌/边界拒绝、检索（Milvus Lite 非线程安全）、令牌预算/历史/网关状态
  锁外（B/D/F）：HyDE 草稿、LLM 重排、CRAG 联网、grounding 判定、生成/校验/引用——云端无状态并行；
                 FactCache 自带线程锁；会话历史写回时重新取锁（同会话并发按完成顺序追加，网关限流兜底）
"""

from __future__ import annotations

import re
import threading
import uuid
from dataclasses import dataclass, field

from langchain_community.chat_message_histories import ChatMessageHistory
from pymilvus import MilvusClient

from config import *
from crag_search import build_search_query, format_search_results, search_fitness, crag_retrieve
from gateway import Gateway, GatewayConfig
from grounding import assess_grounding, refusal_message
from hyde import generate_hyde_drafts, hyde_retrieve
from llm_adapter import FallbackChain, build_embeddings, build_llm, to_runnable
from retriever import FitnessRAGRetriever, load_bm25_from_pickle


# ============================================================
# 三套分层 Prompt（自 app.py 原样迁移）
# ============================================================
PROMPT_GENERAL = """你是一个专业健身教练AI，会根据用户的身体数据和健身目标，并参考检索到的健身动作知识，给出安全、个性化的训练建议。

用户画像：
{user_profile}

你可以参考以下动作信息：
{context}

如果用户的问题与健身无关，请礼貌地引导回健身话题。不确定的内容请标注"建议进一步咨询专业教练"。
"""

PROMPT_INJURY = """你是一个运动康复顾问AI。用户问题涉及伤病、疼痛或体态异常。你必须严格按照「参考知识」作答。

=== 回答结构（三部分，缺一不可）===
1.【基于知识的建议】— 每条建议必须在「参考知识」中有原文依据。引用具体动作名/禁忌/肌群。
2.【知识库外推断】— 你的合理推断，必须标注「此为一般性推断，因人而异」。
3.【需要就医的情况】— 明确指出哪些情况必须咨询骨科/康复科医生。

=== 核心约束（违反即错误）===
- 第1部分禁止编造「参考知识」中未出现的动作名称、康复周期天数、具体负重公斤数。
- 如果「参考知识」资料不足，回复"现有资料不足以制定完整安全方案，建议咨询专业医师"，禁止拼凑虚假方案。
- 「绝对禁忌」中的动作/器械严禁以任何形式出现（含变式、替代、"轻重量"版本）。

=== 绝对禁忌 ===
{contraindications}

用户画像：
{user_profile}

参考知识：
{context}
"""

PROMPT_PLAN = """你是健身计划制定AI。你必须仅基于「参考知识」制定训练计划。

=== 计划结构（三部分，缺一不可）===
1.【可验证部分】— 每个动作、组数、频率必须在「参考知识」中有原文依据。
2.【推断部分】— 标注「以下为一般性推断，因人而异」。
3.【安全声明】— 列出所有假设和限制条件。

=== 硬性约束 ===
- 禁止编造「参考知识」中不存在的动作名称。
- 用户指定的器械类型必须严格遵守，禁止擅自替换。
- 「绝对禁忌」中的动作严禁以任何形式出现（含变式、替代版本）。
- 无法在排除禁忌后生成完整计划时，回复"现有资料不足，请咨询专业教练"，禁止拼凑方案。

=== 绝对禁忌 ===
{contraindications}

用户画像：
{user_profile}

参考知识：
{context}
"""


# ============================================================
# 结果结构
# ============================================================

@dataclass
class PipelineResult:
    request_id: str
    answer: str = ""
    grounded: bool = True
    refusal: bool = False
    citations: list = field(default_factory=list)   # [Citation dict]
    usage: list = field(default_factory=list)       # [UsageInfo]
    fallback_active: bool = False
    error: str | None = None


# ============================================================
# PipelineService
# ============================================================

class PipelineService:
    """12 步安全流水线。单全局锁串行化（Milvus Lite 非线程安全）。"""

    def __init__(self, retriever: FitnessRAGRetriever, llms: dict[str, FallbackChain],
                 gateway: Gateway, fact_engine=None, fact_cache=None,
                 store: dict | None = None):
        self._retriever = retriever
        self._llms = llms
        self._gateway = gateway
        self._fact_engine = fact_engine
        self._fact_cache = fact_cache
        self._store = store if store is not None else {}
        self._last_access: dict[str, float] = {}   # session_id → 最近访问时间（LRU 驱逐用）
        self._lock = threading.Lock()

    # ----------------------------------------------------------------
    # 对外接口
    # ----------------------------------------------------------------

    def answer(self, question: str, session_id: str = "default",
               user_profile: str | None = None,
               request_id: str | None = None,
               deep_thinking: bool = False,
               on_stage=None) -> PipelineResult:
        """完整流水线（非流式）。SSE 缓冲模式与 /v1/chat 共用此入口。

        五阶段线程模型（锁内只做共享状态/Milvus 操作，LLM/网络全部锁外并行）：
          A（锁内） 实体 → 禁忌 → 边界拒绝 → 分层 —— 纯本地字符串，无 LLM/网络
          B（锁外） HyDE 草稿生成 —— 云端 LLM（复合伤病三路并行）
          C（锁内） 检索 —— Milvus Lite 非线程安全，必须串行
          D（锁外） LLM 重排 + CRAG 联网 + grounding 判定 —— 云端/网络无共享状态
          E（锁内） 令牌预算（裁剪 store）+ 组装 messages + 缓存快速路径
          F（锁外） 生成 → 硬过滤 → Fact-Check → 引用 →（回锁）写历史

        on_stage: 阶段进度回调（SSE status 帧数据源）；None = 不播报
        """
        rid = request_id or uuid.uuid4().hex[:12]
        result = PipelineResult(request_id=rid)

        # ===== 阶段 A（锁内）：实体 → 禁忌 → 边界拒绝 → 分层（纯本地，无 LLM/网络）=====
        with self._lock:
            self._stage(on_stage, "正在分析问题与禁忌约束…")
            entities = FitnessRAGRetriever._extract_entities(question)
            injury_names = entities.get("injury", [])
            forbidden: list[str] = []
            contra_text = "（当前查询无伤病，无需禁忌约束）"
            contra_hit = False
            if injury_names:
                contra_map = self._retriever.get_contraindications(injury_names)
                if not contra_map and not NEO4J_ENABLED:
                    # 本地降级：图谱停机时用内置禁忌数据副本（核心安全数据不依赖单一外部服务）
                    contra_map = self._local_contraindications(injury_names)
                if contra_map:
                    contra_hit = True
                    forbidden = list({a for acts in contra_map.values() for a in acts})
                    lines = ["\n\n【伤病禁忌黑名单 — 绝对禁止出现在回答中】"]
                    for inj, actions in contra_map.items():
                        lines.append(f"- {inj}禁忌: {', '.join(actions)}")
                    contra_text = "\n".join(lines)
                else:
                    contra_text = "（该伤病在知识库中暂无禁忌记录）"

            # 边界拒绝（核心诉求=禁忌动作）
            hit_action = self._contraindicated_request(question, forbidden)
            if hit_action:
                reason = self._contraindication_reason(injury_names, hit_action)
                result.answer = self._reject_message(hit_action, injury_names, reason)
                result.refusal = True
                return result

            # 分层生成策略：按查询复杂度分配模型与预算
            tier = self._select_tier(question)
            tier_cfg = dict(LLM_TIERS.get(tier, LLM_TIERS["simple"]))
            # 深度思考开关（用户自选）：复杂层默认快速（plus 关思考），开启后切 chat（思考开）+ 长预算
            if deep_thinking and tier in ("injury", "plan"):
                tier_cfg["role"] = "chat"
                tier_cfg["max_tokens"] = {"injury": 2048, "plan": 3072}.get(tier, 2048)

        self._stage(on_stage, "已完成安全分析，正在检索知识库…")

        # ===== 阶段 B（锁外）：HyDE 草稿生成（云端 LLM，无共享状态；复合伤病三路并行）=====
        drafts = None
        if HYDE_ENABLED:
            try:
                drafts = generate_hyde_drafts(question, self._llms["hyde"])
            except Exception as e:
                drafts = None
                self._gateway.log_cycle("error", request_id=rid, event_detail="hyde_gen_error",
                                        error=str(e)[:200])

        # ===== 阶段 C（锁内）：检索（Milvus Lite 非线程安全；重排候选池留给锁外）=====
        with self._lock:
            try:
                if drafts:
                    ctx, cite_docs = hyde_retrieve(
                        question, self._llms["hyde"], self._retriever,
                        top_k=tier_cfg.get("retrieve_k"), drafts=drafts)
                else:
                    cite_docs = self._retriever.invoke(question)
                    ctx = self._format_docs(cite_docs)
                # 重排层取候选池（RERANKER_MAX_CANDIDATES）；simple 层直接收窄到 context_docs
                retr_k = RERANKER_MAX_CANDIDATES if tier_cfg["rerank"] \
                    else tier_cfg.get("context_docs", CONTEXT_DOCS_MAX)
                scored_docs = self._retriever.search_with_scores(
                    question, k=retr_k, use_rerank=False)
            except Exception as e:
                ctx, cite_docs, scored_docs = "", [], []
                self._gateway.log_cycle("error", request_id=rid, event_detail="retrieval_error",
                                        error=str(e)[:200])
            self._gateway.log_retrieval(rid, question, self._docs_to_log(scored_docs))

        self._stage(on_stage, "检索完成，正在校验回答依据…")

        # ===== 阶段 D（锁外）：LLM 重排 + CRAG 联网 + grounding 判定（云端/网络无共享状态）=====
        if tier_cfg["rerank"] and scored_docs:
            scored_docs = self._retriever.rerank(
                question, scored_docs,
                top_k=tier_cfg.get("context_docs", CONTEXT_DOCS_MAX))

        # 知识盲区 → CRAG
        crag_raw: list = []
        has_external = False
        knowledge_gap = bool(injury_names and not contra_hit)
        ctx_short = len(ctx) < 200
        needs_external = any(kw in question for kw in CRAG_EXTERNAL_KEYWORDS)
        if knowledge_gap or (injury_names and ctx_short) or (injury_names and needs_external):
            if CRAG_ENABLED:
                try:
                    crag_raw = search_fitness(build_search_query(question, []))
                    crag_ctx = format_search_results(crag_raw) if crag_raw else None
                    if crag_ctx:
                        ctx = f"[联网检索资料]\n{crag_ctx}\n\n[本地知识库]\n{ctx}"
                        has_external = True
                except Exception:
                    pass

        # grounding 拒答（生成前判定）：语义相关性仅对无实体查询生效（见 grounding 注释）
        relevance: float | None = None
        if scored_docs:
            try:
                import numpy as np
                q_emb = np.array(self._retriever._embed(question))
                sims = []
                for d, _ in scored_docs[:3]:
                    d_emb = np.array(self._retriever._embed(d.page_content))
                    sims.append(float(np.dot(q_emb, d_emb) /
                                      (np.linalg.norm(q_emb) * np.linalg.norm(d_emb) + 1e-9)))
                relevance = max(sims)
            except Exception:
                relevance = None
        verdict = assess_grounding(ctx, scored_docs, has_external,
                                   relevance=relevance,
                                   has_entities=bool(entities))
        if REFUSE_ENABLED and not verdict.grounded:
            result.answer = refusal_message(question, verdict)
            result.grounded = False
            result.refusal = True
            self._gateway.log_cycle("refusal", request_id=rid, reason=verdict.reason,
                                    relevance=round(relevance, 3) if relevance else None,
                                    n_docs=verdict.n_docs, ctx_chars=verdict.ctx_chars)
            return result

        # ===== 阶段 E（锁内）：Prompt 组装 + 令牌预算（裁剪 store）+ 缓存快速路径 =====
        with self._lock:
            system_prompt = self._select_prompt(question)
            if contra_text and "暂无" not in contra_text and "无需" not in contra_text:
                ctx = contra_text + "\n" + ctx

            ctx = self._gateway.guard_token_budget(
                system_prompt, self._store, session_id, ctx, question)

            # 缓存快速路径：同问题同模式已校验过的回答直接复用（跳过生成，省 LLM 调用）
            if self._needs_fact_check(question) and self._fact_cache is not None:
                cached = self._fact_cache.get(
                    question, self._cache_key_entities(question, deep_thinking))
                if cached:
                    cached += self._mode_hint(tier, deep_thinking)
                    result.citations = self._build_citations(cite_docs, crag_raw, entities)
                    result.answer = cached
                    self._append_history(session_id, question, cached)
                    self._gateway.log_answer(rid, cached)
                    return result

            # 组装 messages（历史读取在锁内）
            sys_content = system_prompt.format(
                contraindications=contra_text,
                user_profile=user_profile or "暂无身体数据",
                context=ctx,
            )
            tier_hint = tier_cfg.get("hint", "")
            if deep_thinking and tier in ("injury", "plan"):
                # 深度思考专属输出要求：与快速模式形成可感知差异。
                # 实测教训（gateway.log 49s 请求）：提示「篇幅可以更长」会诱导思考 token
                # 耗尽 thinking_budget(2048) → MaaS 硬停生成 → 答案截断在句中间。
                # 故收紧：结论先行 + 全文上限 + 避免过度展开，思考消耗留在预算安全区内。
                tier_hint = ("【深度思考模式】请先给出核心结论，再分点说明依据"
                             "（伤病机制、推荐/禁忌动作的医学依据、风险-收益权衡），"
                             "全文控制在1200字以内，避免过度展开。")
            if tier_hint:
                sys_content += "\n\n输出要求：" + tier_hint
            messages = (
                [{"role": "system", "content": sys_content}]
                + self._history_messages(session_id)
                + [{"role": "user", "content": question}]
            )
            self._gateway.log_prompt(rid, "chat", self._messages_to_text(messages))
            # 内存守卫仅对本地主链生效：云端推理不占本地内存，无需降级（守卫状态在锁内变更）
            base_llm = self._llms.get(tier_cfg["role"], self._llms["chat"])
            chat_llm = base_llm
            providers = chat_llm.active_providers
            if providers and providers[0] == "ollama":
                chat_llm = self._gateway.get_active_llm(chat_llm)

        # ======== 阶段 F（锁外）：LLM 生成 —— 云端无状态，跨请求可并行 ========
        # 锁外执行依据：云 LLM 调用不碰 Milvus/BM25/会话存储；FactCache 自带线程锁。
        gen = dict(
            question=question, session_id=session_id, ctx=ctx,
            contra_text=contra_text, forbidden=forbidden, tier_cfg=tier_cfg,
            tier=tier, deep_thinking=deep_thinking,
            base_llm=base_llm, chat_llm=chat_llm, messages=messages,
            cite_docs=cite_docs, entities=entities, crag_raw=crag_raw,
            on_stage=on_stage,
        )
        return self._generate(gen, rid, result)

    # ----------------------------------------------------------------
    # 各步骤实现（自 app.py 迁移，session_state → 实例/参数）
    # ----------------------------------------------------------------

    def _generate(self, gen: dict, rid: str, result: PipelineResult) -> PipelineResult:
        """阶段二（锁外）：降级链生成 → 硬过滤 → Fact-Check → 引用 → 写历史。

        同会话并发时历史按完成顺序追加（可接受；网关限流已抑制同会话高频请求）。
        """
        question = gen["question"]
        session_id = gen["session_id"]
        ctx = gen["ctx"]
        contra_text = gen["contra_text"]
        forbidden = gen["forbidden"]
        tier_cfg = gen["tier_cfg"]
        base_llm = gen["base_llm"]
        chat_llm = gen["chat_llm"]
        messages = gen["messages"]

        # ---- Step 9: 降级链生成（云 → 本地）----
        self._stage(gen.get("on_stage"), "正在生成回答…")
        resp = chat_llm.invoke(messages, max_tokens=tier_cfg.get("max_tokens"),
                               request_id=rid)
        answer = resp.content
        # fallback_active 覆盖两类降级：供应商降级链 + 内存守卫切小模型
        result.fallback_active = resp.fallback or (chat_llm is not base_llm)
        if resp.usage:
            result.usage.append(resp.usage)
            self._gateway.log_usage(rid, "chat", resp.usage)
        if resp.error_kind:
            result.error = resp.error_kind

        # 回答长度守卫：快速模式（关思考）偶发退化输出（实测 12 token）→ 重生成一次。
        # 重试用 chat_nothink（思考模式重试实测 37s，感知太慢）；再次退化由 💡 深度思考兜底
        if (not gen.get("deep_thinking") and gen.get("tier") in ("injury", "plan")
                and len(answer.strip()) < 150 and not result.refusal):
            self._gateway.log_cycle("answer_short_upgrade", request_id=rid,
                                    tier=gen["tier"], len=len(answer.strip()))
            retry_resp = self._llms["chat_nothink"].invoke(
                messages, max_tokens=tier_cfg.get("max_tokens"), request_id=rid)
            answer = retry_resp.content
            result.fallback_active = result.fallback_active or retry_resp.fallback
            if retry_resp.usage:
                result.usage.append(retry_resp.usage)
                self._gateway.log_usage(rid, "chat", retry_resp.usage)

        # ---- Step 10: 硬性禁忌过滤 ----
        answer, was_filtered = self._hard_filter(answer, forbidden)
        if was_filtered:
            answer += "\n\n🛡️ [安全拦截：已自动剔除禁忌动作]"

        # ---- Step 11: Fact-Check ----
        self._stage(gen.get("on_stage"), "正在安全校验…")
        if self._needs_fact_check(question):
            answer = self._run_fact_check(question, answer, ctx, contra_text, forbidden,
                                          deep_thinking=gen.get("deep_thinking", False),
                                          request_id=rid)

        # 复杂层快速模式答后提示（用户可开深度思考重问；simple 层不提示）
        answer += self._mode_hint(gen.get("tier", ""), gen.get("deep_thinking", False))

        # ---- Step 12: 结构化引用 ----
        result.citations = self._build_citations(gen["cite_docs"], gen["crag_raw"],
                                                 gen["entities"])
        result.answer = answer

        # 会话历史写回（重新获取锁；注入检测等阻断路径不写）
        with self._lock:
            self._append_history(session_id, question, answer)
            self._gateway.log_answer(rid, answer)
        return result

    @staticmethod
    def _stage(on_stage, name: str) -> None:
        """阶段进度回调（SSE status 帧数据源）；on_stage 为 None 时静默跳过。"""
        if on_stage:
            on_stage(name)

    def _select_prompt(self, query: str) -> str:
        entities = FitnessRAGRetriever._extract_entities(query)
        has_injury = bool(entities.get("injury") or entities.get("body_part"))
        has_plan = any(kw in query for kw in ["计划", "方案", "每周", "安排", "周期", "定制"])
        if has_injury:
            return PROMPT_INJURY
        if has_plan:
            return PROMPT_PLAN
        return PROMPT_GENERAL

    @staticmethod
    def _select_tier(query: str) -> str:
        """分层生成策略：simple（快模型+短预算）| injury（主模型+校验）| plan（主模型+长预算）。"""
        entities = FitnessRAGRetriever._extract_entities(query)
        if entities.get("injury") or entities.get("body_part"):
            return "injury"
        if any(kw in query for kw in ["计划", "方案", "每周", "安排", "周期", "定制"]):
            return "plan"
        return "simple"

    @staticmethod
    def _contraindicated_request(query: str, forbidden: list[str]) -> str | None:
        for action in forbidden:
            if len(action) >= 2 and action in query:
                return action
        return None

    @staticmethod
    def _local_contraindications(injury_names: list[str]) -> dict[str, list[str]]:
        """Neo4j 停机时的本地禁忌降级：按别名/全称匹配内置数据，仅取「禁忌动作」关系。"""
        from contra_data import INJURY_ACTION_MAP, INJURY_ALIASES
        result = {}
        for name in injury_names:
            normalized = INJURY_ALIASES.get(name, name)
            for key, actions in INJURY_ACTION_MAP.items():
                if normalized == key or normalized in key or key in normalized:
                    forbidden = [a for a, rel, _d in actions if "禁忌" in rel]
                    if forbidden:
                        result[key] = forbidden
        return result

    @staticmethod
    def _contraindication_reason(injuries: list[str], action: str) -> str | None:
        """从禁忌数据单一来源查拒绝原因（contra_data 与图谱构建共用一份数据）。"""
        from contra_data import INJURY_ACTION_MAP, INJURY_ALIASES
        for name in injuries:
            canon = INJURY_ALIASES.get(name, name)
            for key, acts in INJURY_ACTION_MAP.items():
                if canon == key or canon in key or key in canon:
                    for a, rel, reason in acts:
                        if a == action and "禁忌" in rel:
                            return reason
        return None

    @staticmethod
    def _reject_message(action: str, injuries: list[str], reason: str | None = None) -> str:
        injury_text = "、".join(injuries) if injuries else "该伤病"
        reason_text = reason or "该动作会给受伤部位带来过高压力，可能加重损伤。"
        return (
            f"您的核心诉求「{action}」属于{injury_text}的明确禁忌动作，无法基于此制定训练方案。\n\n"
            f"禁忌原因：{reason_text}\n\n"
            f"建议咨询骨科/康复科医师，获取适合您当前阶段的安全替代训练方案。"
        )

    # 否定语境词：行内含这些词时按「安全提醒」处理（如「避免深蹲」），保留而非误删
    _NEGATION_WORDS = ("避免", "不要", "禁止", "不建议", "切勿", "严禁", "不宜", "不做")
    # 2 字动作名后接这些器械后缀时不算命中（「划船」≠「划船机」）
    _EQUIP_SUFFIXES = "机凳架垫器绳"

    @classmethod
    def _action_in_line(cls, action: str, line: str) -> bool:
        """动作名是否真正出现在行内（含 2 字动作的器械后缀启发式）。"""
        idx = line.find(action)
        while idx != -1:
            nxt = line[idx + len(action):idx + len(action) + 1]
            if len(action) <= 2 and nxt in cls._EQUIP_SUFFIXES:
                idx = line.find(action, idx + 1)  # 命中器械名（划船机），继续找下一处
                continue
            return True
        return False

    @classmethod
    def _hard_filter(cls, response: str, forbidden: list[str]) -> tuple[str, bool]:
        """硬性过滤：逐行扫描，删掉包含禁忌动作名的行。不依赖 LLM。

        例外一：否定语境行（「避免深蹲」）是安全建议，保留——删除会让用户失去警示；
        例外二：2 字动作名后接器械后缀（「划船机」）不算命中，防子串误伤。
        仅否定行命中时不触发「已剔除」警告（未删除任何内容）。
        """
        if not forbidden:
            return response, False
        removed = []
        kept = []
        for line in response.split("\n"):
            hit_action = None
            for action in forbidden:
                if cls._action_in_line(action, line):
                    hit_action = action
                    break
            if hit_action is None:
                kept.append(line)
            elif any(w in line for w in cls._NEGATION_WORDS):
                kept.append(line)   # 否定语境行保留（深层校验由 Fact-Check 兜底）
            else:
                removed.append(hit_action)
        if removed:
            unique = list(dict.fromkeys(removed))
            warning = (f"\n\n[!] 以下禁忌动作已被自动剔除（与伤病冲突）：{'、'.join(unique)}。"
                       "请咨询康复医师获取安全替代方案。")
            return "\n".join(kept) + warning, True
        return response, False

    @staticmethod
    def _format_docs(docs, max_docs: int = 5, max_chars: int = 2400) -> str:
        """组装检索上下文：截断到 max_docs 条，总字数不超过 max_chars。"""
        csv_docs = [d for d in docs if d.metadata.get("source") == "fitness_data.csv"]
        pdf_docs = [d for d in docs if d.metadata.get("source") != "fitness_data.csv"]
        ordered = (csv_docs + pdf_docs)[:max_docs]

        parts = []
        total = 0
        for doc in ordered:
            text = PipelineService._summarize_chunk(doc)
            if total + len(text) > max_chars:
                remaining = max_chars - total
                if remaining > 50:
                    parts.append(text[:remaining] + "...")
                break
            parts.append(text)
            total += len(text)
        return "\n\n---\n".join(parts)

    @staticmethod
    def _summarize_chunk(doc) -> str:
        meta = doc.metadata
        if meta.get("动作名称"):
            header = f"[{meta['动作名称']}]"
            if meta.get("目标肌群"):
                header += f" 肌群:{meta['目标肌群']}"
            if meta.get("器械"):
                header += f" 器械:{meta['器械']}"
            if meta.get("难度"):
                header += f" 难度:{meta['难度']}"
            return header
        text = doc.page_content.replace("\n", " ")
        return text[:150] + ("..." if len(text) > 150 else "")

    def _needs_fact_check(self, query: str) -> bool:
        if self._fact_engine is None:
            return False
        entities = FitnessRAGRetriever._extract_entities(query)
        return bool(entities.get("injury") or entities.get("body_part"))

    @staticmethod
    def _mode_hint(tier: str, deep_thinking: bool) -> str:
        """复杂层快速模式的答后提示（缓存路径与生成路径共用，保持体验一致）。"""
        if tier in ("injury", "plan") and not deep_thinking:
            return ("\n\n💡 本次使用快速模式回答。该问题涉及伤病/复杂分析，"
                    "可开启「深度思考模式」重新提问，获得更深入的分析。")
        return ""

    @staticmethod
    def _cache_key_entities(question: str, deep_thinking: bool = False) -> list[str]:
        """缓存 key 的实体部分：实体名 + 模式标签 + 提示词版本（同问题不同模式不共享缓存）。"""
        names = [n for names in FitnessRAGRetriever._extract_entities(question).values()
                 for n in names]
        names.append(f"mode:{'deep' if deep_thinking else 'fast'}")
        names.append(f"pv:{FACT_CACHE_VERSION}")   # 改提示词后 bump config.FACT_CACHE_VERSION
        return names

    def _run_fact_check(self, question: str, answer: str, context: str,
                        contraindications: str, forbidden_actions: list[str],
                        deep_thinking: bool = False,
                        request_id: str | None = None) -> str:
        """事实校验：缓存 → 小模型四类校验 → 失败走 CRAG 修正。任何环节异常不阻断。"""
        # 1. 查缓存（key 含模式维度；生成前已有快速路径，此处兜底）
        if self._fact_cache is not None:
            cached = self._fact_cache.get(
                question, self._cache_key_entities(question, deep_thinking))
            if cached:
                return cached

        # 2. 小模型四类校验（含字符串预检硬拦截）
        result = self._fact_engine.check(
            question, answer, context, contraindications, forbidden_actions)

        if result.passed:
            if self._fact_cache is not None:
                self._fact_cache.set(
                    question, answer, self._cache_key_entities(question, deep_thinking))
            return answer

        # 3. 校验失败 → CRAG 联网修正（降级链生成）
        if CRAG_ENABLED:
            try:
                crag_ctx = crag_retrieve(question, result.failed_categories)
                if crag_ctx:
                    messages = [
                        {"role": "system", "content":
                            "你是一个专业健身教练。请根据以下来自权威来源的参考信息，"
                            "给出安全准确的回答。如果信息之间存在冲突，优先采纳更保守（更安全）的建议。\n\n"
                            f"参考信息：\n{crag_ctx}"},
                        {"role": "user", "content": question},
                    ]
                    resp = self._llms["chat"].invoke(messages)
                    corrected = resp.content
                    if resp.usage and request_id:
                        # 修正路径 usage 记日志（实测截断事故中此调用日志缺失，排障时不可见）
                        self._gateway.log_usage(request_id, "chat_crag_fix", resp.usage)
                    if corrected and corrected.strip():
                        # 重生成回答复检：与主链路相同的硬性禁忌过滤（安全防线不因修正路径短路）
                        corrected, _was_filtered = self._hard_filter(corrected, forbidden_actions)
                        answer = corrected
            except Exception:
                pass  # 降级：网络不可用，保留原回答

        # 4. 缓存修正后的回答
        if self._fact_cache is not None:
            self._fact_cache.set(
                question, answer, self._cache_key_entities(question, deep_thinking))
        return answer

    _SENT_END = re.compile(r"[。！？；!?;]")

    @staticmethod
    def _clean_snippet(text: str, max_chars: int = 120) -> str:
        """引用片段清洗：空白归一 → 尾部数字残渣 → 句子边界截断。

        - OCR 行内换行折叠为空格，多空格合并
        - 剔除尾部孤立数字/标点残渣（OCR 常见 "单系整健 2," 类噪声尾巴）
        - 截断取完整句子边界，不再从词中间切断（原 [:80] 硬切的展示问题）
        注：混入正文内部的 OCR 误识别字符（合法汉字）规则无法识别，
        根治靠摄入层质检与替换合规 PDF，此处只做展示层清理。
        """
        if not text:
            return ""
        t = re.sub(r"\s+", " ", text).strip()
        if not t:
            return ""
        t = re.sub(r"[\d，,.;；。:：\-—]+$", "", t).strip()
        if not t:
            return ""
        if len(t) > max_chars:
            cut = t[:max_chars]
            end = 0
            for m in PipelineService._SENT_END.finditer(cut):
                end = m.end()
            # 句子边界太靠前（片段会过短）时退回硬截断 + 省略号
            t = cut[:end] if end >= max_chars * 0.5 else cut + "…"
        return t

    @staticmethod
    def _build_citations(cite_docs: list, crag_raw: list, entities: dict) -> list:
        """结构化引用：联网优先 → 图谱 → 本地KB。与回答正文分离。"""
        citations = []
        for r in crag_raw[:3]:
            citations.append({
                "kind": "web",
                "source": r.get("title", "")[:80],
                "snippet": PipelineService._clean_snippet(r.get("snippet", ""), 120),
                "url": r.get("url", ""),
            })
        has_entities = any(v for v in entities.values())
        if has_entities:
            for doc in cite_docs[:5]:
                meta = doc.metadata
                name = meta.get("动作名称")
                if not name and meta.get("source") == "fitness_data.csv":
                    # CSVLoader 1.x metadata 只有 {source, row}：从 page_content 首行解析动作名
                    m = re.match(r"动作名称[:：]\s*([^\n]+)", doc.page_content)
                    name = m.group(1).strip() if m else None
                if name:
                    # CSV 动作条目：动作名 + 肌群/器械摘要
                    parts = []
                    for key in ("目标肌群", "器械"):
                        m = re.search(rf"{key}[:：]\s*([^\n]+)", doc.page_content)
                        if m:
                            parts.append(m.group(1).strip())
                    snippet = " | ".join(parts) if parts else \
                        PipelineService._clean_snippet(doc.page_content, 120)
                    citations.append({"kind": "kb", "source": name, "snippet": snippet})
                elif meta.get("page"):
                    snippet = PipelineService._clean_snippet(doc.page_content, 120)
                    citations.append({
                        "kind": "kb",
                        "source": meta.get("source", "未知来源"),
                        "page": meta.get("page"),
                        "snippet": snippet or f"（该页 OCR 片段质量较差，详见原文档第 {meta.get('page')} 页）",
                    })
                elif meta.get("source") and meta.get("source") != "neo4j":
                    # 文本直抽源（TEXT_KB_SOURCES）：段落级文档，无页码
                    citations.append({
                        "kind": "kb",
                        "source": meta["source"],
                        "snippet": PipelineService._clean_snippet(doc.page_content, 120),
                    })
                elif meta.get("hops"):
                    citations.append({
                        "kind": "graph",
                        "source": f"图谱·{meta['hops']}跳",
                        "snippet": PipelineService._clean_snippet(doc.page_content, 120),
                    })
                elif meta.get("source") == "neo4j":
                    citations.append({
                        "kind": "graph",
                        "source": "图谱",
                        "snippet": PipelineService._clean_snippet(doc.page_content, 120),
                    })
        return citations

    # ----------------------------------------------------------------
    # 会话历史（替代 RunnableWithMessageHistory）
    # ----------------------------------------------------------------

    def _history_messages(self, session_id: str) -> list:
        session = self._store.get(session_id)
        if session is None:
            return []
        self._touch(session_id)  # 读取也是访问（LRU 语义）
        out = []
        for m in session.messages:
            # 兼容 dict（新版 langchain_community 存储格式）与 BaseMessage 两种形态
            if isinstance(m, dict):
                mtype = m.get("type") or m.get("role") or "user"
                role = {"human": "user", "ai": "assistant", "system": "system"}.get(mtype, "user")
                out.append({"role": role, "content": m.get("content", "")})
            else:
                role = {"human": "user", "ai": "assistant", "system": "system"}.get(m.type, "user")
                out.append({"role": role, "content": m.content})
        return out

    def _append_history(self, session_id: str, question: str, answer: str) -> None:
        if session_id not in self._store:
            self._store[session_id] = ChatMessageHistory()
        session = self._store[session_id]
        session.add_message({"type": "human", "content": question})
        session.add_message({"type": "ai", "content": answer})
        self._touch(session_id)
        self._evict_lru()

    def _touch(self, session_id: str) -> None:
        """记录会话最近访问时间（读取历史也算访问，LRU 语义）。"""
        import time
        self._last_access[session_id] = time.time()

    def _evict_lru(self) -> None:
        """会话数超过 SESSION_MAX_COUNT 时按最久未访问驱逐（防内存只增不减）。"""
        if len(self._store) <= SESSION_MAX_COUNT:
            return
        excess = len(self._store) - SESSION_MAX_COUNT
        oldest = sorted(self._last_access.items(), key=lambda kv: kv[1])[:excess]
        for sid, _ts in oldest:
            self._store.pop(sid, None)
            self._last_access.pop(sid, None)

    # ----------------------------------------------------------------
    # 日志辅助
    # ----------------------------------------------------------------

    @staticmethod
    def _docs_to_log(scored_docs: list) -> list:
        return [
            {"source": d.metadata.get("source", "未知"), "page": d.metadata.get("page"),
             "score": round(s, 4), "snippet": d.page_content.replace("\n", " ")[:120]}
            for d, s in scored_docs
        ]

    @staticmethod
    def _messages_to_text(messages: list) -> str:
        return "\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)


# ============================================================
# 组装（api.py lifespan 调用）
# ============================================================

def build_pipeline() -> PipelineService:
    """组装全部依赖：索引 / 适配器 / 网关 / 校验 / 缓存。"""
    embeddings = build_embeddings()  # cloud（MaaS）或本地 Ollama，按 config.EMBEDDING_PROVIDER
    milvus_client = MilvusClient(uri=MILVUS_URI, grpc_options=MILVUS_GRPC_OPTIONS)
    milvus_client.load_collection(MILVUS_COLLECTION)

    bm25_idx, bm25_docs = load_bm25_from_pickle(BM25_INDEX_PATH)

    if NEO4J_ENABLED:
        from neo4j import GraphDatabase
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    else:
        neo4j_driver = None

    reranker = None
    if RERANKER_ENABLED:
        from reranker import FitnessReranker
        reranker = FitnessReranker(llm=build_llm("rerank"))

    retriever = FitnessRAGRetriever(
        milvus_client=milvus_client,
        bm25_index=(bm25_idx, bm25_docs),
        neo4j_driver=neo4j_driver,
        embedding_fn=embeddings.embed_query,
        w_milvus=FUSION_WEIGHT_MILVUS,
        w_bm25=FUSION_WEIGHT_BM25,
        w_neo4j=FUSION_WEIGHT_NEO4J,
        fusion_threshold=FUSION_THRESHOLD,
        reranker=reranker,
        milvus_factor=RERANK_MILVUS_FACTOR if RERANKER_ENABLED else 3,
        bm25_factor=RERANK_BM25_FACTOR if RERANKER_ENABLED else 3,
        neo4j_factor=RERANK_NEO4J_FACTOR if RERANKER_ENABLED else 2,
        neo4j_depth=NEO4J_DEPTH,
        neo4j_max_depth=NEO4J_MAX_DEPTH,
    )

    gateway = Gateway(GatewayConfig.from_module())

    # 适配器：chat/hyde 主链；fact_check 经 to_runnable 进入 FactCheckEngine 的 LCEL 链
    # 后台角色（hyde/rerank/fact_check）usage 统一走日志（request_id=background）
    def _bg_usage(role, usage):
        gateway.log_usage("background", role, usage)

    llms = {
        "chat": build_llm("chat"),
        "chat_nothink": build_llm("chat_nothink", on_usage=_bg_usage),  # 复杂层默认快速（plus 关思考）
        "chat_fast": build_llm("chat_fast", on_usage=_bg_usage),        # 分层策略：simple 查询快模型
        "hyde": build_llm("hyde", on_usage=_bg_usage),
    }

    fact_engine = None
    fact_cache = None
    if FACT_CHECK_ENABLED:
        try:
            from fact_cache import FactCache
            from fact_checker import FactCheckEngine
            fact_engine = FactCheckEngine(to_runnable(build_llm("fact_check", on_usage=_bg_usage)))
            fact_cache = FactCache(path=FACT_CACHE_PATH, max_entries=FACT_CACHE_MAX)
        except Exception:
            fact_engine = None  # 模型不可用时自动降级

    return PipelineService(
        retriever=retriever,
        llms=llms,
        gateway=gateway,
        fact_engine=fact_engine,
        fact_cache=fact_cache,
    )
