import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from pymilvus import MilvusClient
from neo4j import GraphDatabase
from config import *
from hyde import hyde_retrieve
from retriever import FitnessRAGRetriever, load_bm25_from_pickle
from crag_search import build_search_query, format_search_results

st.set_page_config(page_title="AI 健身教练", page_icon="🏋️")
st.title("🏋️ AI 健身教练（Milvus Lite + BM25 + Neo4j AuraDB）")

@st.cache_resource
def load_retriever():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Milvus Lite（嵌入式模式，uri 指向本地文件）
    milvus_client = MilvusClient(uri=MILVUS_URI)
    # 加载 collection 到内存（Milvus Lite 默认 released 状态）
    milvus_client.load_collection(MILVUS_COLLECTION)

    # BM25
    bm25_idx, bm25_docs = load_bm25_from_pickle(BM25_INDEX_PATH)

    # Neo4j AuraDB
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def embed_fn(text: str):
        return embeddings.embed_query(text)

    # Reranker（零内存：复用已有 LLM，temperature=0 确保确定性排序）
    # 用 import config（非 from import *）绕过 Streamlit 模块热加载缓存问题
    import config as _cfg
    reranker = None
    _rr_enabled = getattr(_cfg, 'RERANKER_ENABLED', False)
    if _rr_enabled:
        from reranker import FitnessReranker
        reranker_llm = ChatOllama(model=LLM_MODEL, temperature=0)
        reranker = FitnessReranker(
            llm=reranker_llm,
            max_candidates=getattr(_cfg, 'RERANKER_MAX_CANDIDATES', 15),
            doc_max_chars=getattr(_cfg, 'RERANKER_DOC_MAX_CHARS', 200),
        )

    return FitnessRAGRetriever(
        milvus_client=milvus_client,
        bm25_index=(bm25_idx, bm25_docs),
        neo4j_driver=neo4j_driver,
        embedding_fn=embed_fn,
        w_milvus=FUSION_WEIGHT_MILVUS,
        w_bm25=FUSION_WEIGHT_BM25,
        w_neo4j=FUSION_WEIGHT_NEO4J,
        fusion_threshold=FUSION_THRESHOLD,
        reranker=reranker,
        milvus_factor=getattr(_cfg, 'RERANK_MILVUS_FACTOR', 3) if _rr_enabled else 3,
        bm25_factor=getattr(_cfg, 'RERANK_BM25_FACTOR', 3) if _rr_enabled else 3,
        neo4j_factor=getattr(_cfg, 'RERANK_NEO4J_FACTOR', 2) if _rr_enabled else 2,
        neo4j_depth=getattr(_cfg, 'NEO4J_DEPTH', 1),
        neo4j_max_depth=getattr(_cfg, 'NEO4J_MAX_DEPTH', 3),
    )

retriever = load_retriever()
llm = ChatOllama(model=LLM_MODEL, temperature=0.7)

# ============================================================
# Fact-Check 资源（资源隔离：小模型，不占 7B 算力）
# ============================================================
_fact_check_llm = None
_fact_cache = None
_fact_engine = None

if FACT_CHECK_ENABLED:
    try:
        _fact_check_llm = ChatOllama(model=FACT_CHECK_MODEL, temperature=0, num_predict=128)
        from fact_cache import FactCache
        _fact_cache = FactCache(path=FACT_CACHE_PATH, max_entries=FACT_CACHE_MAX)
        from fact_checker import FactCheckEngine
        _fact_engine = FactCheckEngine(_fact_check_llm)
    except Exception:
        FACT_CHECK_ENABLED = False  # 模型不可用时自动降级


def _needs_fact_check(query: str) -> bool:
    """分流减负：普通健身问答跳过，仅伤病/体态查询开启校验。"""
    if not FACT_CHECK_ENABLED:
        return False
    from retriever import FitnessRAGRetriever
    entities = FitnessRAGRetriever._extract_entities(query)
    return bool(entities.get("injury") or entities.get("body_part"))


def _get_forbidden_actions() -> list[str]:
    """从 session_state 获取禁忌动作扁平列表（由聊天循环预计算）。"""
    return st.session_state.get("_forbidden_actions", [])


def _hard_filter_contraindications(response: str, forbidden: list[str]) -> tuple[str, bool]:
    """硬性过滤：逐行扫描，删掉包含禁忌动作名的行。不依赖 LLM。"""
    if not forbidden:
        return response, False
    violations = []
    lines = response.split("\n")
    kept = []
    for line in lines:
        hit = False
        for action in forbidden:
            if action in line:
                violations.append(action)
                hit = True
                break
        if not hit:
            kept.append(line)
    if violations:
        unique = list(dict.fromkeys(violations))
        warning = f"\n\n[!] 以下禁忌动作已被自动剔除（与伤病冲突）：{'、'.join(unique)}。请咨询康复医师获取安全替代方案。"
        return "\n".join(kept) + warning, True
    return response, False


def _is_contraindicated_request(query: str, forbidden: list[str]) -> str | None:
    """检查用户核心诉求是否为禁忌动作本身。返回命中的动作名。"""
    for action in forbidden:
        if len(action) >= 2 and action in query:
            return action
    return None


def _gap_fallback_note() -> str:
    return (
        "\n\n[!] 本地知识库未收录该伤病资料，联网检索暂不可用。"
        "请基于你的通用医学知识回答，但必须在开头注明"
        "「本地知识库暂无此伤病数据，以下建议基于通用知识，请咨询专业医生核实」。"
        "禁止编造具体康复周期、负重数值或用药建议。"
    )


def _format_citations() -> str:
    """引用来源：联网优先 → 健身实体匹配时展示本地KB → 否则标注通用知识。"""
    docs = st.session_state.get("_last_docs", [])
    crag_results = st.session_state.get("_crag_results", [])
    entities = st.session_state.get("_last_entities", {})
    has_entities = any(v for v in entities.values())

    lines = ["\n---\n**参考来源：**"]
    has_any = False

    if crag_results:
        for r in crag_results[:3]:
            title = r.get("title", "")[:80]
            url = r.get("url", "")
            if url:
                lines.append(f"- \U0001f310 [{title}]({url})")
            else:
                lines.append(f"- \U0001f310 {title}")
        has_any = True

    if has_entities:
        for doc in docs[:5]:
            src = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page", "")
            hops = doc.metadata.get("hops", "")
            if page:
                lines.append(f"- {src} 第{page}页")
                has_any = True
            elif hops:
                snippet = doc.page_content.replace("\n", " ")[:80]
                lines.append(f"- [图谱·{hops}跳] {snippet}...")
                has_any = True
            elif src == "neo4j":
                snippet = doc.page_content.replace("\n", " ")[:80]
                lines.append(f"- [图谱] {snippet}...")
                has_any = True

    if not has_any:
        if has_entities:
            lines.append("- （本地知识库无匹配资料）")
        else:
            lines.append("- （回答基于模型通用知识）")

    return "\n".join(lines)


def _reject_contraindicated_request(action: str, injuries: list[str]) -> str:
    """生成边界拒绝回答。"""
    injury_text = "、".join(injuries) if injuries else "该伤病"
    return (
        f"您的核心诉求「{action}」属于{injury_text}的明确禁忌动作，无法基于此制定训练方案。\n\n"
        f"禁忌原因：{action}会给受伤部位带来过高压力，可能加重损伤。\n\n"
        f"建议咨询骨科/康复科医师，获取适合您当前阶段的安全替代训练方案。"
    )


def _run_fact_check(query: str, response: str) -> str:
    """
    事实校验入口（带缓存 + CRAG + 降级保障）。
    返回：原回答 或 校验修正后的回答。任何环节异常不阻断。
    """
    import config as _cfg
    crg_enabled = getattr(_cfg, 'CRAG_ENABLED', True)

    # 1. 查缓存
    if _fact_cache is not None:
        from retriever import FitnessRAGRetriever
        entities = FitnessRAGRetriever._extract_entities(query)
        entity_names = [n for names in entities.values() for n in names]
        cached = _fact_cache.get(query, entity_names)
        if cached:
            return cached

    # 2. 获取检索上下文 + 禁忌黑名单（文本）+ 禁忌动作列表（代码预检用）
    context = st.session_state.get("_last_context", "")
    contraindications = st.session_state.get("_contraindications", "")
    forbidden_actions = _get_forbidden_actions()

    # 3. 小模型做四类校验（含字符串预检硬拦截）
    result = _fact_engine.check(query, response, context, contraindications, forbidden_actions)

    if result.passed:
        # 校验通过 → 缓存并返回原回答
        if _fact_cache is not None:
            from retriever import FitnessRAGRetriever
            entities = FitnessRAGRetriever._extract_entities(query)
            entity_names = [n for names in entities.values() for n in names]
            _fact_cache.set(query, response, entity_names)
        return response

    # 4. 校验失败 → CRAG 联网修正
    if crg_enabled:
        try:
            from crag_search import crag_retrieve
            crag_ctx = crag_retrieve(query, result.failed_categories)
            if crag_ctx:
                # 用 CRAG 资料 + 7B 模型二次生成
                crg_prompt = ChatPromptTemplate.from_messages([
                    ("system", "你是一个专业健身教练。请根据以下来自权威来源的参考信息，给出安全准确的回答。如果信息之间存在冲突，优先采纳更保守（更安全）的建议。\n\n参考信息：\n{context}"),
                    ("human", "{question}"),
                ])
                crg_chain = crg_prompt | llm | StrOutputParser()
                corrected = crg_chain.invoke({"question": query, "context": crag_ctx})
                if corrected and corrected.strip():
                    response = corrected
        except Exception:
            pass  # 降级：网络不可用，保留原回答

    # 5. 缓存修正后的回答
    if _fact_cache is not None:
        from retriever import FitnessRAGRetriever
        entities = FitnessRAGRetriever._extract_entities(query)
        entity_names = [n for names in entities.values() for n in names]
        _fact_cache.set(query, response, entity_names)

    return response

# ============================================================
# 三套分层 Prompt（根据查询类型动态选择）
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

# 默认 prompt 模板（将在聊天循环中动态替换）
prompt = ChatPromptTemplate.from_messages([
    ("system", PROMPT_GENERAL),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

def format_docs(docs, max_docs: int = 5, max_chars: int = 2400):
    """
    组装检索上下文：截断到 max_docs 条，总字数不超过 max_chars。
    优先保留含实体标签（CSV 动作条目）的文档。
    """
    # 优先 CSV 来源（有结构化元数据）→ PDF 来源
    csv_docs = [d for d in docs if d.metadata.get("source") == "fitness_data.csv"]
    pdf_docs = [d for d in docs if d.metadata.get("source") != "fitness_data.csv"]
    ordered = csv_docs + pdf_docs
    ordered = ordered[:max_docs]

    parts = []
    total = 0
    for doc in ordered:
        text = _summarize_chunk(doc)
        if total + len(text) > max_chars:
            remaining = max_chars - total
            if remaining > 50:
                parts.append(text[:remaining] + "...")
            break
        parts.append(text)
        total += len(text)
    return "\n\n---\n".join(parts)


def _summarize_chunk(doc) -> str:
    """对单个 chunk 做轻量摘要：CSV 来源提取结构化字段，PDF 截断。"""
    meta = doc.metadata
    if meta.get("动作名称"):
        # CSV 动作条目 → 结构化摘要
        header = f"[{meta['动作名称']}]"
        if meta.get("目标肌群"):
            header += f" 肌群:{meta['目标肌群']}"
        if meta.get("器械"):
            header += f" 器械:{meta['器械']}"
        if meta.get("难度"):
            header += f" 难度:{meta['难度']}"
        return header
    else:
        # PDF → 截断到前 150 字
        text = doc.page_content.replace("\n", " ")
        return text[:150] + ("..." if len(text) > 150 else "")

if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

if "user_profiles" not in st.session_state:
    st.session_state.user_profiles = {}

def get_user_profile(session_id):
    return st.session_state.user_profiles.get(session_id, "暂无身体数据")

def _select_prompt(query: str) -> str:
    """根据查询类型选择对应的专属提示词模板。"""
    from retriever import FitnessRAGRetriever
    entities = FitnessRAGRetriever._extract_entities(query)
    has_injury = bool(entities.get("injury") or entities.get("body_part"))
    has_plan = any(kw in query for kw in ["计划", "方案", "每周", "安排", "周期", "定制"])

    if has_injury:
        return PROMPT_INJURY   # 伤病 → 最严格约束
    elif has_plan:
        return PROMPT_PLAN     # 计划 → 中度约束
    else:
        return PROMPT_GENERAL  # 普通 → 宽松约束


def build_chain(prompt_template=None):
    """构建 RAG 链。prompt_template 为 ChatPromptTemplate，默认用通用 prompt。"""
    if prompt_template is None:
        prompt_template = prompt

    # 检索函数（HyDE 增强：假想文档 → 向量检索）
    # 同时将 context 存储到 session_state，供 fact-check 使用
    def retrieve_context(input_dict):
        question = input_dict.get("question", "")
        _s = st.session_state.get("_pipeline_status")

        # Step A: HyDE 假想文档生成（如启用）
        if _s and HYDE_ENABLED:
            _s.update(label="正在生成 HyDE 假想文档...")

        if HYDE_ENABLED:
            ctx = hyde_retrieve(question, llm, retriever)
        else:
            docs = retriever.invoke(question)
            ctx = format_docs(docs)
        st.session_state["_last_context"] = ctx

        # 存储原始 docs 供引用来源展示
        try:
            _cite_docs = retriever.invoke(question)
            st.session_state["_last_docs"] = _cite_docs[:5]
        except Exception:
            st.session_state["_last_docs"] = []

        # 防线一：查询 Neo4j 禁忌动作 → 注入上下文 + 存储原始数据供硬过滤
        from retriever import FitnessRAGRetriever
        entities = FitnessRAGRetriever._extract_entities(question)
        injury_names = entities.get("injury", [])

        if injury_names:
            if _s:
                _s.update(label=f"正在查询图谱禁忌：{'、'.join(injury_names)}...")
            contra_map = retriever.get_contraindications(injury_names)
            if contra_map:
                st.session_state["_contraindications_map"] = contra_map
                all_forbidden = set()
                for actions in contra_map.values():
                    all_forbidden.update(actions)
                if _s:
                    _s.update(label=f"图谱命中 {len(all_forbidden)} 个禁忌动作：{'、'.join(list(all_forbidden)[:8])}")
                lines = ["\n\n【伤病禁忌黑名单 — 绝对禁止出现在回答中】"]
                for inj, actions in contra_map.items():
                    lines.append(f"- {inj}禁忌: {', '.join(actions)}")
                contra_text = "\n".join(lines)
                ctx = contra_text + "\n" + ctx
            else:
                st.session_state["_contraindications_map"] = {}
                if _s:
                    _s.update(label="图谱中无该伤病禁忌记录")

        # 知识盲区：伤病查询但本地知识库无匹配 → 主动 CRAG（不可用时诚实告知）
        st.session_state["_crag_sources"] = ""
        knowledge_gap = bool(injury_names and not st.session_state.get("_contraindications_map"))
        ctx_short = len(ctx) < 200

        if knowledge_gap or (injury_names and ctx_short):
            if _s:
                _s.update(label="知识库数据不足，正在联网搜索...")
            import config as _cfg
            crag_ctx = None
            if getattr(_cfg, 'CRAG_ENABLED', True):
                try:
                    from crag_search import crag_retrieve
                    crag_ctx = crag_retrieve(question, [])
                except Exception:
                    pass

            if crag_ctx:
                if _s:
                    _s.update(label="联网搜索完成，正在整合资料...")
                prefix = (
                    "[重要：本地知识库缺少该伤病的详细资料。"
                    "以下信息来自联网检索，请以此为主要参考，"
                    "但需标注'来自外部搜索，建议核实']"
                )
                ctx = f"{prefix}\n{crag_ctx}\n\n[本地知识库（内容有限）]\n{ctx}"
                st.session_state["_crag_sources"] = crag_ctx
                from langchain_core.documents import Document
                crag_docs = [Document(page_content=crag_ctx[:500], metadata={"source": "crag"})]
                st.session_state["_last_docs"] = crag_docs + st.session_state.get("_last_docs", [])
            else:
                if _s:
                    _s.update(label="联网搜索不可用，使用模型通用知识...")
                recall_prompt = (
                    f"你是一个医学知识助手。请简要列出关于以下伤病的核心信息（症状、禁忌动作、推荐康复训练），"
                    f"每项一句话，不要编造具体数据：{question}"
                )
                try:
                    recall_resp = llm.invoke(recall_prompt)
                    recall_text = recall_resp.content if hasattr(recall_resp, "content") else str(recall_resp)
                    if recall_text.strip():
                        gap_note = (
                            "\n\n[!] 本地知识库未收录该伤病资料，联网检索暂不可用。"
                            "以下背景知识来自模型通用训练数据（非医学权威来源），仅供参考：\n"
                            f"{recall_text[:600]}\n"
                            "[重要约束] 你必须在回答开头注明「本地知识库暂无此伤病数据，以下建议基于通用知识，请咨询专业医生核实」"
                            "禁止编造具体的康复周期、负重数值或用药建议。"
                        )
                        ctx = gap_note + "\n" + ctx
                    else:
                        ctx = _gap_fallback_note() + "\n" + ctx
                except Exception:
                    ctx = _gap_fallback_note() + "\n" + ctx
                st.session_state["_last_docs"] = []

        if _s:
            _s.update(label="检索完成，正在生成回答...")
        return ctx

    # 使用 RunnablePassthrough.assign 自动构建输入字典
    rag_chain = (
        RunnablePassthrough.assign(context=retrieve_context)
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

# 默认链（普通问答）
chain_with_history = build_chain()

# --- 侧边栏 ---
st.sidebar.header("⚙️ 我的身体数据")
with st.sidebar.form("profile_form"):
    height = st.number_input("身高（cm）", min_value=100, max_value=250, value=170)
    weight = st.number_input("体重（kg）", min_value=30, max_value=300, value=70)
    goal = st.selectbox("健身目标", ["增肌", "减脂", "塑形", "力量提升"])
    submitted = st.form_submit_button("保存画像")
    if submitted:
        session_id = "default_user"
        profile_str = f"身高{height}cm，体重{weight}kg，目标：{goal}"
        st.session_state.user_profiles[session_id] = profile_str
        st.sidebar.success("✅ 身体数据已保存")

st.sidebar.markdown("---")
st.sidebar.caption("数据保存后，教练将基于你的画像给出建议。")

# --- 聊天区域（稳定版） ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt_input := st.chat_input("请输入你的健身问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    session_id = "default_user"
    input_data = {
        "question": prompt_input,
        "user_profile": get_user_profile(session_id),
    }

    import config as _cfg
    from retriever import FitnessRAGRetriever

    # 清除上轮 CRAG 缓存
    st.session_state["_crag_results"] = []

    # ================================================================
    # Phase 1: 全部后端处理（不产生任何 DOM 更新，= 0 次 removeChild）
    # ================================================================
    entities = FitnessRAGRetriever._extract_entities(prompt_input)
    st.session_state["_last_entities"] = entities

    selected_system_prompt = _select_prompt(prompt_input)

    # 查询 Neo4j 禁忌名单
    injury_names = entities.get("injury", [])
    forbidden_actions: list[str] = []
    if injury_names:
        contra_map = retriever.get_contraindications(injury_names)
        if contra_map:
            st.session_state["_contraindications_map"] = contra_map
            for actions in contra_map.values():
                forbidden_actions.extend(actions)
            forbidden_actions = list(set(forbidden_actions))
            lines = ["\n\n【伤病禁忌黑名单 — 绝对禁止出现在回答中】"]
            for inj, actions in contra_map.items():
                lines.append(f"- {inj}禁忌: {', '.join(actions)}")
            st.session_state["_contraindications"] = "\n".join(lines)
        else:
            st.session_state["_contraindications_map"] = {}
            st.session_state["_contraindications"] = "（该伤病在知识库中暂无禁忌记录）"
    else:
        st.session_state["_contraindications_map"] = {}
        st.session_state["_contraindications"] = "（当前查询无伤病，无需禁忌约束）"

    # 边界拒绝检查
    contra_action = _is_contraindicated_request(prompt_input, forbidden_actions)
    if contra_action:
        full_response = _reject_contraindicated_request(contra_action, injury_names)
        full_response += _format_citations()
        with st.chat_message("assistant"):
            st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.stop()

    # HyDE + 三路检索
    if HYDE_ENABLED:
        ctx = hyde_retrieve(prompt_input, llm, retriever)
    else:
        docs = retriever.invoke(prompt_input)
        ctx = format_docs(docs)

    # 引用来源
    try:
        _cite_docs = retriever.invoke(prompt_input)
        st.session_state["_last_docs"] = _cite_docs[:5]
    except Exception:
        st.session_state["_last_docs"] = []

    # 禁忌注入上下文
    contra_text = st.session_state.get("_contraindications", "")
    if contra_text and "暂无" not in contra_text and "无需" not in contra_text:
        ctx = contra_text + "\n" + ctx

    # 知识盲区 → CRAG
    _ext_kws = ["争议","研究","最新","指南","学界","临床","文献","证据","进展","共识","综述","急性期","慢性期","恢复期","术后","膨出","脱出","游离","哪个医院","手术"]
    knowledge_gap = bool(injury_names and not st.session_state.get("_contraindications_map"))
    ctx_short = len(ctx) < 200
    needs_external = any(kw in prompt_input for kw in _ext_kws)
    if knowledge_gap or (injury_names and ctx_short) or (injury_names and needs_external):
        crag_ctx = None
        if getattr(_cfg, 'CRAG_ENABLED', True):
            try:
                import requests as _req
                _search_q = build_search_query(prompt_input, [])
                _resp = _req.post("https://api.bochaai.com/v1/web-search",
                    headers={"Content-Type":"application/json","Authorization":"Bearer sk-fd46e722a04749f6b6d758317fa535fc"},
                    json={"query":_search_q,"count":5}, timeout=5)
                _resp.raise_for_status()
                _data = _resp.json()
                _pages = _data.get("data",{}).get("webPages",{}).get("value",[])
                _crag_raw = [{"title":p.get("name",""),"snippet":p.get("snippet",""),"url":p.get("url","")} for p in _pages if p.get("snippet","").strip()]
                st.session_state["_crag_results"] = _crag_raw
                crag_ctx = format_search_results(_crag_raw) if _crag_raw else None
            except Exception:
                pass
        if crag_ctx:
            ctx = f"[联网检索资料]\n{crag_ctx}\n\n[本地知识库]\n{ctx}"
            st.session_state["_crag_sources"] = crag_ctx

    st.session_state["_last_context"] = ctx
    st.session_state["_forbidden_actions"] = forbidden_actions

    # 构建 chain
    contra_text_final = st.session_state.get("_contraindications", "（无）")
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", selected_system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]).partial(contraindications=contra_text_final)

    def _precomputed_context(_d):
        return ctx
    rag_chain = (
        RunnablePassthrough.assign(context=_precomputed_context)
        | chat_prompt
        | llm
        | StrOutputParser()
    )
    chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="question", history_messages_key="history",
    )

    # ================================================================
    # Phase 2: 回答展示（st.spinner + 单次 st.markdown，0 次 removeChild）
    # ================================================================
    with st.chat_message("assistant"):
        with st.spinner("正在生成回答..."):
            # 收集完整响应（不流式写 DOM，避免前端 removeChild 竞态）
            full_response = ""
            for chunk in chain.stream(
                input_data,
                config={"configurable": {"session_id": session_id}}
            ):
                full_response += chunk

            # 硬过滤
            full_response, was_filtered = _hard_filter_contraindications(full_response, forbidden_actions)
            if was_filtered:
                full_response += "\n\n🛡️ [安全拦截：已自动剔除禁忌动作]"

            # Fact-Check
            if _needs_fact_check(prompt_input):
                full_response = _run_fact_check(prompt_input, full_response)

            # 引用来源
            full_response += _format_citations()

        st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})