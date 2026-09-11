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

import contextlib
import re
import threading
import time
import uuid
from dataclasses import dataclass, field

from pymilvus import MilvusClient

from config import *
from cost_guard import DEGRADED as COST_DEGRADED
from crag_search import build_search_query, format_search_results, search_fitness, crag_retrieve
from gateway import Gateway, GatewayConfig
from grounding import assess_grounding, refusal_message
from health_tools import resolve_health_tools
# 会话存储已抽到 session_store.py（持久化 + LRU）；别名保留既有引用不变
from session_store import SessionHistory as _SessionHistory, build_session_store
from hyde import generate_hyde_drafts, hyde_retrieve, needs_rewrite, rewrite_query
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
    """12 步安全流水线。

    并发模型：一把全局锁（Milvus Lite 非线程安全）保护四段临界区——
    A_analyze / C_retrieve / E_budget / F_history。**但网络调用尽量排在锁外**：
    query embedding（Phase B）与健康工具决策都在锁外做完再传进去，因为云端
    embedding 单次实测中位 158ms，放锁内等于让所有并发请求排队等这一次 HTTP
    （实测移出后并发 32 吞吐 878 → 1682 tok/s）。
    临界区的**等待**与**持有**分开打点（见 `_phase_lock`），用于回答
    「瓶颈到底是不是这把锁」——只看总延迟分不出这两者。
    """

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
               on_stage=None, on_delta=None,
               stop_event: threading.Event | None = None) -> PipelineResult:
        """完整流水线（非流式）。SSE 缓冲模式与 /v1/chat 共用此入口。

        五阶段线程模型（锁内只做共享状态/Milvus 操作，LLM/网络全部锁外并行）：
          A（锁内） 实体 → 禁忌 → 边界拒绝 → 分层 —— 纯本地字符串，无 LLM/网络
          B（锁外） HyDE 草稿生成 —— 云端 LLM（复合伤病三路并行）
          C（锁内） 检索 —— Milvus Lite 非线程安全，必须串行
          D（锁外） LLM 重排 + CRAG 联网 + grounding 判定 —— 云端/网络无共享状态
          E（锁内） 令牌预算（裁剪 store）+ 组装 messages + 缓存快速路径
          F（锁外） 生成 → 硬过滤 → Fact-Check → 引用 →（回锁）写历史

        on_stage: 阶段进度回调（SSE status 帧数据源）；None = 不播报
        on_delta: token 级增量回调（真流式 SSE 数据源）；None = 非流式生成
        stop_event: 客户端断开取消信号；置位时流水线在阶段边界及时退出（SSE 断开保护）
        """
        rid = request_id or uuid.uuid4().hex[:12]
        result = PipelineResult(request_id=rid)

        # ===== 阶段 A（锁内）：实体 → 禁忌 → 边界拒绝 → 分层（纯本地，无 LLM/网络）=====
        with self._phase_lock(rid, "A_analyze"):
            self._stage(on_stage, "正在分析问题与禁忌约束…")
            entities = FitnessRAGRetriever._extract_entities(question)
            injury_names = entities.get("injury", [])
            contra_map, forbidden, contra_text, contra_hit = \
                self._resolve_contraindications(injury_names)

            # 边界拒绝（核心诉求=禁忌动作）；多轮改写后的复查见 Phase C
            hit_action = self._contraindicated_request(question, forbidden)
            if hit_action:
                reason = self._contraindication_reason(injury_names, hit_action)
                result.answer = self._reject_message(hit_action, injury_names, reason)
                result.refusal = True
                return result

            # 分层生成策略：按查询复杂度分配模型与预算
            tier = self._select_tier(question)
            tier_cfg = dict(LLM_TIERS.get(tier, LLM_TIERS["simple"]))

            # 成本降级（软阈值）：关掉唯一的「可选奢侈品」——深度思考的思考 token
            # 是单请求最大的成本乘数，且完全由用户勾选，关掉不影响任何安全环节。
            #
            # 刻意**不**降级的东西：禁忌判定 / 硬过滤 / 事实核查 / CRAG / 分层检索。
            # 前四项是安全链路；CRAG 与 Fact-Check 只对伤病类触发，为省钱关掉它们
            # 等于在最需要完整的回答上偷工减料——那类流量交给硬阈值统一兜底（直接拒绝），
            # 而不是给一个「更便宜的伤病建议」。
            #
            # 必须在下方 deep_thinking 分支**之前**覆盖：该变量还会进缓存 key，
            # 若在生成时才改，缓存 key 会声明思考态而实际没思考 → 跨模式串答案。
            if deep_thinking and self._gateway.cost_state == COST_DEGRADED:
                deep_thinking = False
                self._gateway.log_cycle("cost_degraded_deep_thinking_off", request_id=rid)

            # 深度思考开关（用户自选）：复杂层默认快速（plus 关思考），开启后切 chat（思考开）+ 长预算
            if deep_thinking and tier in ("injury", "plan"):
                tier_cfg["role"] = "chat"
                tier_cfg["max_tokens"] = {"injury": 2048, "plan": 3072}.get(tier, 2048)

            # 多轮查询改写标记：指代消解后检索（「那硬拉呢」需要上一轮「腰突」上下文）。
            # 历史快照在锁内读取，改写本身放锁外（云端 LLM 调用）
            history_snapshot: list = []
            do_rewrite = False
            phase_a_injuries: list = []   # 改写前的伤病快照（Phase C 复查时比对新增）
            if REWRITE_ENABLED:
                history_snapshot = self._history_messages(session_id)
                do_rewrite = needs_rewrite(question, history_snapshot)
                phase_a_injuries = list(injury_names)

        self._stage(on_stage, "已完成安全分析，正在检索知识库…")
        if self._cancelled(stop_event):
            return result

        # ===== 阶段 B（锁外）：查询改写 + HyDE 草稿生成（云端 LLM，无共享状态）=====
        # 多轮改写先行：改写后的 retrieval_q 仅用于检索（Phase C）；
        # 原 question 仍用于 Prompt / 分层判定 / FactCache key
        retrieval_q = question
        if do_rewrite:
            retrieval_q = rewrite_query(question, history_snapshot, self._llms["hyde"])
            # 改写后的查询可能补充伤病上下文（「那硬拉呢」→「腰突患者可以做硬拉吗」）：
            # 合并实体供 Phase C 补查禁忌 + 重跑边界拒绝——原问题无伤病实体时
            # Phase A 已按「无需禁忌约束」放行，多轮语境必须复查（防安全防线断链）
            _rewritten = FitnessRAGRetriever._extract_entities(retrieval_q)
            for etype in set(entities) | set(_rewritten):
                entities[etype] = list(dict.fromkeys(entities.get(etype, [])
                                                     + _rewritten.get(etype, [])))
            injury_names = entities.get("injury", [])
        # simple 层跳过 HyDE 直查（HYDE_SKIP_SIMPLE）：基础问答直查已满分，省 1 次 LLM 调用
        drafts = None
        if HYDE_ENABLED and not (tier == "simple" and HYDE_SKIP_SIMPLE):
            try:
                drafts = generate_hyde_drafts(retrieval_q, self._llms["hyde"])
            except Exception as e:
                drafts = None
                self._gateway.log_cycle("hyde_gen_error", request_id=rid,
                                        error=str(e)[:200])
        # query 向量在**锁外**预算：云端 embedding 是一次网络往返（实测中位 158ms、
        # 最坏 800ms+）。放在锁内等于让所有并发请求的检索段排队等这一次 HTTP——
        # 实测 simple 层 C_retrieve 持有时长的 ~68% 就是它。
        # 放在这里（与 HyDE 同段，都是网络调用、无共享状态）不改变依赖顺序：
        # 检索要用的是 retrieval_q，此时已定稿。
        precomputed_vec = None
        if not self._cancelled(stop_event):
            try:
                precomputed_vec = self._retriever._embed(retrieval_q)
            except Exception as e:
                # 预算失败不致命：检索侧会自行补算（退回原行为），只是慢一点
                self._gateway.log_cycle("embed_precompute_error", request_id=rid,
                                        error=str(e)[:120])

        if self._cancelled(stop_event):
            return result

        # ===== 阶段 C（锁内）：检索（Milvus Lite 非线程安全；重排候选池留给锁外）=====
        # 检索侧 OOM 防线:可用内存过低 → 降级为 BM25-only(稀疏索引内存占用极小)
        if RETRIEVAL_MEMORY_GUARD_ENABLED:
            try:
                from config import get_available_memory_gb
                avail = get_available_memory_gb()
                low = avail < RETRIEVAL_MEMORY_THRESHOLD_GB
                self._retriever.set_degraded(low)
                if low:
                    self._gateway.log_cycle("retrieval_degraded_sparse_only", request_id=rid,
                                            avail_gb=round(avail, 2))
            except Exception:
                pass   # 内存检测失败不影响主链路
        with self._phase_lock(rid, "C_retrieve"):
            # 多轮改写复查：改写补充了伤病实体 → 补查禁忌 + 重跑边界拒绝
            # （Phase A 按原问题实体放行；此处用改写后合并实体重新上安全闸门）
            if do_rewrite and injury_names != phase_a_injuries:
                contra_map, forbidden, contra_text, contra_hit = \
                    self._resolve_contraindications(injury_names)
                hit_action = self._contraindicated_request(question, forbidden)
                if hit_action:
                    reason = self._contraindication_reason(injury_names, hit_action)
                    result.answer = self._reject_message(hit_action, injury_names, reason)
                    result.refusal = True
                    return result
            q_emb = None
            try:
                # 注：hyde_retrieve 必须先于 search_with_scores——其内部检索会把
                # _last_query_embedding 覆盖为草稿向量，后者的 search 才写回 query 向量
                if drafts:
                    ctx, cite_docs = hyde_retrieve(
                        retrieval_q, self._llms["hyde"], self._retriever,
                        top_k=tier_cfg.get("retrieve_k"), drafts=drafts)
                # 重排层取候选池（RERANKER_MAX_CANDIDATES）；simple 层直接收窄到 context_docs
                retr_k = RERANKER_MAX_CANDIDATES if tier_cfg["rerank"] \
                    else tier_cfg.get("context_docs", CONTEXT_DOCS_MAX)
                scored_docs = self._retriever.search_with_scores(
                    retrieval_q, k=retr_k, use_rerank=False,
                    query_vec=precomputed_vec)   # 锁外算好的向量，锁内直接用
                if not drafts:
                    # 直查路径：search_with_scores 结果即引用来源（simple 层跳过 HyDE 后
                    # 不再单独 invoke 重复检索——原实现同 query 同 k 搜了两次）
                    cite_docs = [d for d, _ in scored_docs]
                    ctx = self._format_docs(cite_docs)
                # 锁内读取 query 向量缓存：grounding 相关性判定复用，免重复 embedding
                # （搜索成功后该属性必为本查询向量；锁外读取会被并发检索覆盖，故在锁内取）
                q_emb = self._retriever._last_query_embedding
            except Exception as e:
                ctx, cite_docs, scored_docs, q_emb = "", [], [], None
                self._gateway.log_cycle("retrieval_error", request_id=rid,
                                        error=str(e)[:200])
            # 记录实际检索用 query（改写场景可审计原始 vs 改写后）
            self._gateway.log_retrieval(rid, retrieval_q, self._docs_to_log(scored_docs))
        if self._cancelled(stop_event):
            return result

        self._stage(on_stage, "检索完成，正在校验回答依据…")

        # ===== 阶段 D（锁外）：LLM 重排 + CRAG 联网 + grounding 判定（云端/网络无共享状态）=====
        # ⚠️ 仅在「无 HyDE 草稿」时重排。有草稿时 ctx 来自 hyde_retrieve，而它内部的
        # similarity_search 已经对每路草稿重排过（use_rerank 默认 True）——此处再排一次，
        # 重排的却是另一组文档（use_rerank=False 的直接检索结果），且该组只用于 grounding，
        # 属于纯浪费（injury/plan 层每请求多一次 LLM 调用）。
        # 无草稿（HyDE 失败降级）时，ctx 直接来自未重排的检索，此处的重排才是唯一一次。
        if tier_cfg["rerank"] and scored_docs and not drafts:
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
                if q_emb is None:
                    q_emb = self._retriever._embed(question)   # 稀疏检索降级路径：向量未算过
                q_vec = np.array(q_emb)
                sims = []
                for d, _ in scored_docs[:3]:
                    d_emb = np.array(self._retriever._embed(d.page_content))
                    sims.append(float(np.dot(q_vec, d_emb) /
                                      (np.linalg.norm(q_vec) * np.linalg.norm(d_emb) + 1e-9)))
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
        # REFUSE_ENABLED=False:不拒答 → 生成"通用知识 + 免责声明"(grounded 标记保留)
        unfounded = not verdict.grounded
        if self._cancelled(stop_event):
            return result

        # ===== 工具决策（锁外）：模型自主调用 tools 协议，关键词兜底 =====
        # 放锁外：模型调用是网络 IO，进锁会阻塞并发请求的检索段（Milvus 串行）。
        # 且仅当能抽到身高/体重/年龄时才发起——参数全无时任何工具都算不出来，
        # 多打一次 LLM 纯属浪费（这也是把模型调用限制在少数请求上的关键）。
        tool_results: list = []
        tool_source = "none"
        if HEALTH_TOOLS_ENABLED:
            try:
                _router = (self._llms.get("tool_router")
                           if HEALTH_TOOLS_MODEL_DECISION else None)
                tool_results, tool_source = resolve_health_tools(
                    question, user_profile, _router)
            except Exception:
                tool_results, tool_source = [], "none"
        if tool_results:
            self._gateway.log_cycle("tool_call", request_id=rid,
                                    source=tool_source,
                                    tools=[t.name for t in tool_results])
        if self._cancelled(stop_event):
            return result

        # ===== 阶段 E（锁内）：Prompt 组装 + 令牌预算（裁剪 store）+ 缓存快速路径 =====
        with self._phase_lock(rid, "E_budget"):
            # 工具计算结果注入上下文顶部
            # （位于令牌预算守卫之前：头部位置不会被尾部截断误伤）
            if tool_results:
                tool_ctx = "\n".join(f"- [{t.title}] {t.content}" for t in tool_results)
                ctx = f"[工具计算结果]\n{tool_ctx}\n\n{ctx}"

            system_prompt = self._select_prompt(question)
            if contra_text and "暂无" not in contra_text and "无需" not in contra_text:
                ctx = contra_text + "\n" + ctx

            # 传入实际激活的供应商：令牌预算按其上下文窗口取值（而非恒按本地 8192）
            ctx = self._gateway.guard_token_budget(
                system_prompt, self._store, session_id, ctx, question,
                provider=self._active_provider(tier_cfg))

            # 缓存快速路径：同问题同模式已校验过的回答直接复用（跳过生成，省 LLM 调用）
            if self._needs_fact_check(question) and self._fact_cache is not None:
                cached = self._fact_cache.get(
                    question, self._cache_key_entities(question, deep_thinking,
                                                       tool_results, user_profile))
                if cached:
                    cached += self._mode_hint(tier, deep_thinking)
                    result.citations = self._build_citations(
                        cite_docs, crag_raw, entities, tool_results)
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
            tool_results=tool_results, user_profile=user_profile,
            on_stage=on_stage, on_delta=on_delta,
            unfounded=unfounded, stop_event=stop_event,
        )
        return self._generate(gen, rid, result)

    # ----------------------------------------------------------------
    # 各步骤实现（自 app.py 迁移，session_state → 实例/参数）
    # ----------------------------------------------------------------

    @contextlib.contextmanager
    def _phase_lock(self, rid: str, phase: str):
        """获取全局锁，并记录**等待**与**持有**时长。

        为什么分别记两个数：等待时长是「串行化程度」的直接证据，持有时长是「临界区
        本身多长」。并发升高时——
          等待↑、持有→  ⇒ 瓶颈就是这把锁（要拆锁/外置）
          两者都→      ⇒ 瓶颈不在这里，分布式锁/Redis 解决不了任何问题
        只看总延迟是分不出这两者的，这正是压测归因要回答的问题。
        """
        t0 = time.perf_counter()
        self._lock.acquire()
        wait_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        try:
            yield
        finally:
            hold_ms = (time.perf_counter() - t1) * 1000.0
            self._lock.release()
            # 纯度量埋点不得成为硬依赖：gateway 为 None 时静默跳过。
            # 否则「没传 gateway 的轻量构造」会因为一行日志而崩，而这条路径原本不碰网关。
            if self._gateway is not None:
                self._gateway.log_cycle("lock_phase", request_id=rid, phase=phase,
                                        wait_ms=round(wait_ms, 1),
                                        hold_ms=round(hold_ms, 1))

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
        user_profile = gen.get("user_profile")

        # ---- Step 9: 降级链生成（云 → 本地）----
        self._stage(gen.get("on_stage"), "正在生成回答…")
        on_delta = gen.get("on_delta")
        # usage 日志里必须写**真实角色**（chat / chat_fast / chat_nothink）：
        # 原实现三处都硬编码 "chat"，日志分不出用量来自哪一层 →
        # 成本归属无从查起（想优化成本先得知道钱花在哪个角色上）。
        gen_role = tier_cfg.get("role", "chat")
        if on_delta is not None:
            # 真流式:stream_events 逐 token 回调 + 末块 usage(流中降级标记随文本透出)
            parts: list[str] = []
            last_usage = None
            fb_switch = False
            for ev in chat_llm.stream_events(messages, max_tokens=tier_cfg.get("max_tokens"),
                                             request_id=rid):
                if self._cancelled(gen.get("stop_event")):
                    break   # 客户端断开：及时退出 LLM 流式循环
                if ev.text:
                    parts.append(ev.text)
                    on_delta(ev.text)
                if ev.usage:
                    last_usage = ev.usage
                if ev.fallback_switch:
                    fb_switch = True
            answer = "".join(parts)
            # fallback_active 覆盖三类降级:供应商降级链 + 内存守卫切小模型 + 流中切换
            result.fallback_active = fb_switch or (chat_llm is not base_llm)
            if last_usage is not None:
                result.usage.append(last_usage)
                self._gateway.log_usage(rid, gen_role, last_usage)
        else:
            resp = chat_llm.invoke(messages, max_tokens=tier_cfg.get("max_tokens"),
                                   request_id=rid)
            answer = resp.content
            # fallback_active 覆盖两类降级：供应商降级链 + 内存守卫切小模型
            result.fallback_active = resp.fallback or (chat_llm is not base_llm)
            if resp.usage:
                result.usage.append(resp.usage)
                self._gateway.log_usage(rid, gen_role, resp.usage)
            if resp.error_kind:
                result.error = resp.error_kind

        # 客户端断开（stop_event 置位）：丢弃部分结果，跳过重试/校验/历史写回
        if self._cancelled(gen.get("stop_event")):
            result.answer = answer   # 保留已生成的部分内容（SSE 侧已断开，仅占位不展示）
            return result

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
                self._gateway.log_usage(rid, "chat_nothink", retry_resp.usage)

        # ---- Step 10: 硬性禁忌过滤 ----
        answer, was_filtered = self._hard_filter(answer, forbidden)
        if was_filtered:
            answer += "\n\n🛡️ [安全拦截：已自动剔除禁忌动作]"

        # ---- Step 11: Fact-Check ----
        self._stage(gen.get("on_stage"), "正在安全校验…")
        if self._needs_fact_check(question):
            answer = self._run_fact_check(question, answer, ctx, contra_text, forbidden,
                                          deep_thinking=gen.get("deep_thinking", False),
                                          tool_results=gen.get("tool_results"),
                                          user_profile=user_profile,
                                          request_id=rid)

        # 复杂层快速模式答后提示（用户可开深度思考重问；simple 层不提示）
        answer += self._mode_hint(gen.get("tier", ""), gen.get("deep_thinking", False))

        # 无依据生成分支（REFUSE_ENABLED=False）:附加免责声明
        if gen.get("unfounded"):
            answer += UNFOUNDED_DISCLAIMER
            result.grounded = False

        # ---- Step 12: 结构化引用 ----
        result.citations = self._build_citations(gen["cite_docs"], gen["crag_raw"],
                                                 gen["entities"],
                                                 gen.get("tool_results"))
        result.answer = answer

        # 会话历史写回（重新获取锁；注入检测等阻断路径不写）
        with self._phase_lock(rid, "F_history"):
            self._append_history(session_id, question, answer)
            self._gateway.log_answer(rid, answer)
        return result

    @staticmethod
    def _stage(on_stage, name: str) -> None:
        """阶段进度回调（SSE status 帧数据源）；on_stage 为 None 时静默跳过。"""
        if on_stage:
            on_stage(name)

    @staticmethod
    def _cancelled(stop_event) -> bool:
        """SSE 断开取消信号：置位时流水线在阶段边界及时退出（见 answer 的 stop_event 参数）。"""
        return stop_event is not None and stop_event.is_set()

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

    def _resolve_contraindications(self, injury_names: list[str]) -> tuple[dict, list[str], str, bool]:
        """伤病名列表 → (contra_map, forbidden, contra_text, contra_hit)。

        图谱优先（NEO4J_ENABLED），停机时本地降级（contra_data 副本，单一数据源）。
        Phase A（原问题实体）与 Phase C（多轮改写合并实体）共用，保证两处口径一致。
        """
        if not injury_names:
            return {}, [], "（当前查询无伤病，无需禁忌约束）", False
        contra_map = self._retriever.get_contraindications(injury_names)
        if not contra_map:
            # 本地降级：图谱未命中/停机时用内置禁忌数据副本
            # （核心安全数据不依赖单一外部服务）
            # ⚠️ 条件只看「图谱是否返回数据」，不看 NEO4J_ENABLED：
            #    前者只覆盖“配置关闭”，图谱“已启用但运行时挂掉”时降级会失效 →
            #    禁忌黑名单为空 → 边界拒绝与硬过滤同时失守（安全网消失）。
            contra_map = self._local_contraindications(injury_names)
        if contra_map:
            forbidden = list({a for acts in contra_map.values() for a in acts})
            lines = ["\n\n【伤病禁忌黑名单 — 绝对禁止出现在回答中】"]
            for inj, actions in contra_map.items():
                lines.append(f"- {inj}禁忌: {', '.join(actions)}")
            return contra_map, forbidden, "\n".join(lines), True
        return {}, [], "（该伤病在知识库中暂无禁忌记录）", False

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
    def _cache_key_entities(question: str, deep_thinking: bool = False,
                            tool_results: list | None = None,
                            user_profile: str | None = None) -> list[str]:
        """缓存 key 的实体部分：实体名 + 模式标签 + 提示词版本（同问题不同模式不共享缓存）。

        tool_results 非空时追加画像指纹：同问题不同画像的工具答案不同
        （「帮我算下BMI」在身高170/体重70 与 180/80 下结果不同），不能复用旧缓存。
        """
        from contra_data import data_fingerprint

        names = [n for names in FitnessRAGRetriever._extract_entities(question).values()
                 for n in names]
        names.append(f"mode:{'deep' if deep_thinking else 'fast'}")
        names.append(f"pv:{FACT_CACHE_VERSION}")   # 改提示词后 bump config.FACT_CACHE_VERSION
        # 禁忌表**内容**指纹：pv 只跟提示词，改 contra_data（增删禁忌）它不动，
        # 缓存会继续吐按旧禁忌表生成的答案 —— 安全数据不能靠「记得手动 bump」。
        # 取内容哈希后改表即自动失效（实测：改一条禁忌 → 指纹变化 → 缓存必然 miss）
        names.append(f"cd:{data_fingerprint()}")
        if tool_results and user_profile:
            import hashlib
            names.append(f"profile:{hashlib.md5(user_profile.encode('utf-8')).hexdigest()[:8]}")
        return names

    def _active_provider(self, tier_cfg: dict) -> str | None:
        """解析当前层实际会使用的供应商（降级链首位）。

        与生成阶段的选型同一口径（`_llms.get(tier_cfg["role"])`），供令牌预算决定
        上下文窗口大小。取不到/无适配器时返回 None → 网关回退本地配置。
        """
        llm = self._llms.get(tier_cfg.get("role", "")) or self._llms.get("chat")
        providers = getattr(llm, "active_providers", None)
        return providers[0] if providers else None

    def _run_fact_check(self, question: str, answer: str, context: str,
                        contraindications: str, forbidden_actions: list[str],
                        deep_thinking: bool = False,
                        tool_results: list | None = None,
                        user_profile: str | None = None,
                        request_id: str | None = None) -> str:
        """事实校验：缓存 → 小模型四类校验 → 失败走 CRAG 修正。任何环节异常不阻断。

        ⚠️ user_profile 必须与生成前快速路径（见 answer() 内缓存快速路径）传同一口径：
        工具触发时缓存 key 会追加画像指纹，漏传会导致「同问题不同画像」串用旧答案
        （原缺陷：此处两处调用均未传 user_profile，而快速路径传了）。
        """
        # 1. 查缓存（key 含模式维度；生成前已有快速路径，此处兜底）
        if self._fact_cache is not None:
            cached = self._fact_cache.get(
                question, self._cache_key_entities(question, deep_thinking,
                                                   tool_results, user_profile))
            if cached:
                return cached

        # 2. 小模型四类校验（含字符串预检硬拦截）
        result = self._fact_engine.check(
            question, answer, context, contraindications, forbidden_actions)

        if result.passed:
            if self._fact_cache is not None:
                self._fact_cache.set(
                    question, answer, self._cache_key_entities(question, deep_thinking,
                                                               tool_results, user_profile))
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
                question, answer, self._cache_key_entities(question, deep_thinking,
                                                           tool_results, user_profile))
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
    def _build_citations(cite_docs: list, crag_raw: list, entities: dict,
                         tool_results: list | None = None) -> list:
        """结构化引用：工具优先 → 联网 → 图谱 → 本地KB。与回答正文分离。"""
        citations = []
        for t in (tool_results or []):
            citations.append({"kind": "tool", "source": t.title, "snippet": t.content})
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
            # 兼容 dict（{type, content} 存储格式）与 BaseMessage 两种形态
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
            self._store[session_id] = _SessionHistory()
        session = self._store[session_id]
        session.add_message({"type": "human", "content": question})
        session.add_message({"type": "ai", "content": answer})
        self._touch(session_id)
        self._evict_lru()
        self._persist_store()

    def _touch(self, session_id: str) -> None:
        """记录会话最近访问时间（读取历史也算访问，LRU 语义）。"""
        touch = getattr(self._store, "touch", None)
        if touch is not None:          # SessionStore：持久化存储自带 LRU
            touch(session_id)
            return
        import time
        self._last_access[session_id] = time.time()

    def _evict_lru(self) -> None:
        """会话数超过上限时按最久未访问驱逐（防内存只增不减）。"""
        evict = getattr(self._store, "evict_lru", None)
        if evict is not None:          # SessionStore：驱逐与持久化同处一地
            evict()
            return
        if len(self._store) <= SESSION_MAX_COUNT:
            return
        excess = len(self._store) - SESSION_MAX_COUNT
        oldest = sorted(self._last_access.items(), key=lambda kv: kv[1])[:excess]
        for sid, _ts in oldest:
            self._store.pop(sid, None)
            self._last_access.pop(sid, None)

    def _persist_store(self) -> None:
        """会话落盘。store 为普通 dict 时静默跳过（测试/轻量用法不受影响）。

        落盘失败不阻断请求：内存态照常工作，退化为原来的非持久行为。
        每请求一次原子写——会话数 ≤ 64、单文件量级 KB，代价可接受；
        若将来会话量级上升，改为「脏标记 + 定时刷盘」即可。
        """
        save = getattr(self._store, "save", None)
        if save is None:
            return
        try:
            save()
        except Exception:
            pass

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
        # 走 graph_client：显式超时 + 熔断。裸 driver 在图谱停机时单次调用要
        # 34.53s（实测，驱动默认重试预算吃满），一次伤病问句含 2~3 次调用
        from graph_client import make_graph_client
        neo4j_driver = make_graph_client(NEO4J_URI, (NEO4J_USER, NEO4J_PASSWORD))
    else:
        neo4j_driver = None

    # 网关先建：下面 reranker 就要用它提供的记账回调（_bg_usage 依赖 gateway）
    gateway = Gateway(GatewayConfig.from_module())

    # 适配器：chat/hyde 主链；fact_check 经 to_runnable 进入 FactCheckEngine 的 LCEL 链
    # 后台角色（hyde/rerank/fact_check）usage 统一走日志（request_id=background）
    def _bg_usage(role, usage):
        gateway.log_usage("background", role, usage)

    reranker = None
    if RERANKER_ENABLED:
        from reranker import FitnessReranker
        # on_usage 必须挂：rerank 是一次真实的 LLM 调用（伤病/计划层每请求一次）。
        # 原实现漏挂 → 它既不进日志、也不进成本账本，成了唯一一条**不记账的调用路径**，
        # 与 cost_guard.py 里「记账必须挂在所有 LLM 调用的公共漏斗上」的前提直接冲突。
        # doc_max_chars 必须显式传：reranker 自己的默认值是 200，而 config 里是 400。
        # 不传 → 线上按 200 跑、eval_testset 按 400 跑，**评测测的不是线上跑的东西**。
        # 统一到 config 的 400（= 已发布的 Hit@k 指标所在的那个口径）。
        reranker = FitnessReranker(llm=build_llm("rerank", on_usage=_bg_usage),
                                   doc_max_chars=RERANKER_DOC_MAX_CHARS)

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

    llms = {
        "chat": build_llm("chat"),
        # 主链角色（chat_nothink/chat_fast）的 usage 由 pipeline 按 request_id 记一次日志；
        # 若挂 on_usage 会在 FallbackChain 内以 request_id=background 再记一条 → 双记。
        # 后台角色（hyde/rerank/fact_check）无请求上下文，统一记 request_id=background。
        "chat_nothink": build_llm("chat_nothink"),                      # 复杂层默认快速（plus 关思考）
        "chat_fast": build_llm("chat_fast"),                            # 分层策略：simple 查询快模型
        "hyde": build_llm("hyde", on_usage=_bg_usage),
        # 工具路由：仅产出 tool_calls（无正文），走小模型短预算
        "tool_router": build_llm("tool_router", on_usage=_bg_usage),
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
        # 会话记忆持久化：进程重启后多轮上下文不丢（见 session_store.py）
        store=build_session_store(),
    )
