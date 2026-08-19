"""
HyDE (Hypothetical Document Embeddings) 模块
==============================================
为健身 RAG 管道提供假想文档生成能力：
- 关键词规则分类器 → 选择模板 → LLM 生成假想文档 → 向量检索

两套生成模板：
  - 简单问答（50-100字简短草稿）：基础问答、单伤病问答、计划生成等
  - 复合伤病（200-400字详细分析文档）：多伤病/多体态问题组合查询
"""

from config import (
    HYDE_COMPOUND_MARKERS, HYDE_TOP_K,
    STEP_BACK_ENABLED, DECOMPOSE_ENABLED,
    MULTI_QUERY_TOP_K, MAX_SUB_QUESTIONS, MAX_TOTAL_DOCS,
)

import re

# 中文字符（用于复合标记词的独立出现判定）
_HAN_CHAR = re.compile(r"[一-鿿]")


# ============================================================
# Query Classification
# ============================================================

def classify_query(question: str) -> str:
    """
    基于规则分类，将用户查询分为三类（simple / single_injury / compound_injury）。

    Args:
        question: 用户原始查询字符串

    Returns:
        "simple"         — 无伤病关键词的普通健身问答
        "single_injury"  — 涉及单个伤病/体态问题的查询
        "compound_injury" — 涉及多个伤病或体态问题的复合查询

    实体计数用 retriever 实体词典贪心最长匹配（"肩袖损伤"整体算 1 个实体——
    早期按关键词粗计数会把"肩袖"+"损伤"拆成 2 个，误判复合伤病，检索走 3-hop 浪费）。
    """
    q = question.strip()

    # 1. 实体词典贪心抽取伤病/部位（单字组合词由最长匹配合并，如"半月板损伤"算 1 个）
    from retriever import FitnessRAGRetriever
    entities = FitnessRAGRetriever._extract_entities(q)
    injuries = entities.get("injury", [])

    # 2. 无伤病实体 → 简单问答
    if not injuries:
        return "simple"

    # 3. 部位名是伤病名组成部分时不叠加（"膝关节积液"自带"膝关节"，仍算单伤病）
    extra_parts = [p for p in entities.get("body_part", [])
                   if not any(p in i for i in injuries)]

    # 4. 双伤病 / 伤病+独立部位 / 复合标记词 → 复合伤病
    #    多字标记词要求独立出现（前后非汉字）——"梨状肌综合征"里的"综合"是伤病名
    #    子串，不是连接标记（"综合征" 的意外命中会把单伤病误判为复合）
    def _standalone(marker: str) -> bool:
        idx = q.find(marker)
        if idx == -1:
            return False
        if len(marker) == 1:
            return True   # 单字连接词（加/和/兼/及…）只要出现即参与
        prev_ok = idx == 0 or not _HAN_CHAR.match(q[idx - 1])
        next_ok = idx + len(marker) >= len(q) or not _HAN_CHAR.match(q[idx + len(marker)])
        return prev_ok and next_ok

    has_compound_marker = any(_standalone(m) for m in HYDE_COMPOUND_MARKERS)
    if len(injuries) >= 2 or (injuries and extra_parts) or has_compound_marker:
        return "compound_injury"

    return "single_injury"


# ============================================================
# Multi-turn Query Rewrite — 指代消解（多轮语境下检索前的 query 变换）
# ============================================================

REWRITE_HINT_WORDS = (
    "它", "他", "她", "这个", "那个", "该", "此", "刚才", "上面",
    "以上", "这些", "那些", "呢",
)
REWRITE_HISTORY_MAX_TURNS = 6


def needs_rewrite(question: str, history_msgs: list) -> bool:
    """多轮语境下是否需要改写后再检索（纯规则，~0.01ms）。

    历史非空 且（问题含指代词 OR 无任何实体且非计划类问题）→ True。
    含实体/计划关键词的问题通常自包含（「腰突能深蹲吗」「帮我制定周计划」），无需改写。
    """
    if not history_msgs:
        return False
    if any(w in question for w in REWRITE_HINT_WORDS):
        return True
    from retriever import FitnessRAGRetriever
    entities = FitnessRAGRetriever._extract_entities(question)
    if any(v for v in entities.values()):
        return False
    if any(kw in question for kw in ("计划", "方案", "每周", "安排", "周期", "定制")):
        return False
    return True


def _history_text(history_msgs: list, max_turns: int = REWRITE_HISTORY_MAX_TURNS) -> str:
    """最近几轮历史 → "user: ...\nassistant: ..." 文本（dict 与 BaseMessage 兼容）。"""
    rows = []
    for m in history_msgs[-max_turns:]:
        if isinstance(m, dict):
            role = {"human": "user", "ai": "assistant"}.get(
                m.get("type") or m.get("role"), "user")
            content = m.get("content", "") or ""
        else:
            role = {"human": "user", "ai": "assistant"}.get(getattr(m, "type", ""), "user")
            content = getattr(m, "content", "") or ""
        rows.append(f"{role}: {content}")
    return "\n".join(rows)


REWRITE_PROMPT = (
    "你是对话查询改写助手。根据对话历史，把用户最新问题改写成一个可独立检索的"
    "完整中文问题（补充指代对象与省略内容，如「那硬拉呢」→「腰突患者可以做硬拉吗」）。\n"
    "只输出改写后的问句本身：不要回答原问题，不要加引号，不要任何解释。\n\n"
    "对话历史：\n{history}\n\n最新问题：{question}"
)


def rewrite_query(question: str, history_msgs: list, llm) -> str:
    """LLM 改写查询；异常 / 空输出 / 超长 → 回退原问题（检索质量兜底）。

    复用 "hyde" 角色小模型（qwen2.5:0.5b / qwen3.7-flash，max_tokens=64）。
    模式沿用 generate_hypothetical_doc 的「模板 + invoke + 回退」三件套。
    """
    try:
        prompt_text = REWRITE_PROMPT.format(
            history=_history_text(history_msgs), question=question)
        response = llm.invoke([{"role": "user", "content": prompt_text}], max_tokens=64)
        content = response.content if hasattr(response, "content") else str(response)
        out = (content or "").strip().strip('"').strip("'").strip()
        if not out or len(out) > 200:
            return question
        return out
    except Exception:
        return question


# ============================================================
# HyDE Generation Templates
# ============================================================

HYDE_SIMPLE_TEMPLATE = """你是健身教练。为以下问题写一段检索用的参考草稿(80-150字)，必须包含具体的实体名称（肌群名/伤病名/器械名/动作名），关键词密度越高检索越精准。

问题：{question}

按格式输出——直接列实体后写草稿:
肌群: [具体肌肉名称]
伤病: [伤病名，如无则写"无"]
器械: [具体器械名]
禁忌: [应避免的动作名]
动作: [推荐动作名]

草稿:"""


HYDE_COMPOUND_TEMPLATE = """你是运动康复专家。用户同时存在多种伤病。为以下问题撰写康复分析草稿(200-300字)，逐条列出实体名用于检索。

问题：{question}

按格式输出:
1.伤病: [逐一列出伤病名，用、分隔]
2.肌群: [逐一列出相关肌肉名]
3.禁忌动作: [逐一列出 + 禁忌原因]
4.康复动作: [逐一列出 → 目标肌群 → 器械]

草稿:"""


# ============================================================
# Step Back & Decompose Templates（仅对复合伤病触发）
# ============================================================

STEP_BACK_TEMPLATE = """你是一个健身教练。以下是一个具体的健身/伤病问题。请将它抽象成一个更通用、更原理性的问题，用于检索基础知识。

原始问题：{question}

请生成一个更抽象的问题（例如从"腰突能不能深蹲"抽象为"腰椎伤病患者的运动禁忌和选择原则"，从"半月板损伤加肥胖怎么减脂"抽象为"下肢关节损伤人群的安全减脂方法论"）：

抽象问题："""


DECOMPOSE_TEMPLATE = """你是一个健身教练。以下是一个复合健身/伤病问题，涉及多个条件或目标。请将它拆分成 2-4 个独立的子问题，每个子问题聚焦于一个方面。

复合问题：{question}

请输出子问题列表，每行一个，用换行分隔（不要编号，不要其他前缀）："""


# ============================================================
# HyDE Generation & Retrieval
# ============================================================

def generate_hypothetical_doc(question: str, llm, classification: str) -> str:
    """
    根据分类结果选择模板，调用 LLM 生成假想文档。

    Args:
        question: 用户原始查询
        llm: 统一适配器 FallbackChain 实例
        classification: classify_query 返回的分类标签

    Returns:
        生成的假想文档字符串；异常时回退为原始 query
    """
    if classification == "compound_injury":
        template = HYDE_COMPOUND_TEMPLATE
    else:
        # "simple" 和 "single_injury" 都用简单模板
        template = HYDE_SIMPLE_TEMPLATE

    prompt_text = template.format(question=question)

    try:
        response = llm.invoke(prompt_text)
        content = response.content if hasattr(response, "content") else str(response)
        # 空输出或过短（<20字）回退
        content = content.strip() if isinstance(content, str) else ""
        if not content or len(content) < 20:
            return question
        return content
    except Exception:
        # LLM 调用失败时回退到原始查询
        return question

# ============================================================
# Step Back — 抽象提问
# ============================================================

def generate_step_back(question: str, llm) -> str:
    """
    生成 Step Back 抽象问句：将具体问题抽象为更通用的原理级问题。

    Args:
        question: 用户原始查询
        llm: 统一适配器 FallbackChain 实例

    Returns:
        抽象问句字符串；异常时回退为原始 query
    """
    prompt_text = STEP_BACK_TEMPLATE.format(question=question)

    try:
        response = llm.invoke(prompt_text)
        content = response.content if hasattr(response, "content") else str(response)
        if not content or not content.strip():
            return question
        return content.strip()
    except Exception:
        return question


# ============================================================
# Decompose — 子问题拆分
# ============================================================

def decompose_question(question: str, llm) -> list:
    """
    将复合问题拆解为 2-4 个独立子问题。

    Args:
        question: 用户原始查询
        llm: 统一适配器 FallbackChain 实例

    Returns:
        子问题字符串列表；异常时回退为 [question]
    """
    prompt_text = DECOMPOSE_TEMPLATE.format(question=question)

    try:
        response = llm.invoke(prompt_text)
        content = response.content if hasattr(response, "content") else str(response)
        if not content or not content.strip():
            return [question]

        # 按换行拆分，过滤空行和编号前缀
        lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
        # 去除可能的编号前缀（如 "1. ", "1、", "- " 等）
        import re
        cleaned = []
        for line in lines:
            cleaned_line = re.sub(r'^[\d]+[\.\、\)\s\-]+', '', line).strip()
            if cleaned_line:
                cleaned.append(cleaned_line)

        if not cleaned:
            return [question]

        # 限制子问题数量
        return cleaned[:MAX_SUB_QUESTIONS]
    except Exception:
        return [question]


# ============================================================
# Multi-Query Retrieval — 三路并行召回
# ============================================================

def _retrieve_docs(query_text: str, retriever, k: int) -> list:
    """单路检索：query → 向量检索 → 返回父块（完整上下文）。"""
    return retriever.similarity_search(query_text, k=k)


def _merge_deduplicate(doc_lists: list, max_total: int) -> list:
    """
    合并多路文档列表，按 page_content MD5 去重，保留首次出现的文档。

    Args:
        doc_lists: [[doc, ...], [doc, ...], ...] 各路文档列表
        max_total: 合并后最多保留文档数

    Returns:
        去重后的文档列表
    """
    import hashlib
    seen = set()
    merged = []
    for docs in doc_lists:
        for doc in docs:
            key = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
            if key not in seen:
                seen.add(key)
                merged.append(doc)
    return merged[:max_total]


def generate_hyde_drafts(question: str, llm, label: str | None = None) -> list[str]:
    """
    锁外生成全部检索草稿（云端 LLM，无共享状态）：
      - simple / single_injury: 单条 HyDE 假想文档
      - compound_injury: 三路并行（HyDE + Step Back + 子问题拆分）

    检索仍在调用方锁内执行（Milvus Lite 非线程安全）——本函数从检索管道中拆出，
    供 pipeline 两阶段线程模型使用（草稿生成锁外并行，检索锁内串行）。

    Returns:
        草稿查询列表（生成失败回退为原始 question，列表恒非空）
    """
    label = label or classify_query(question)
    if label != "compound_injury":
        return [generate_hypothetical_doc(question, llm, label)]

    # 三路草稿 LLM 生成并行（云调用无共享状态）
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_hyde = ex.submit(generate_hypothetical_doc, question, llm, "compound_injury")
        f_sb = ex.submit(generate_step_back, question, llm) if STEP_BACK_ENABLED else None
        f_sq = ex.submit(decompose_question, question, llm) if DECOMPOSE_ENABLED else None
        hyde_doc = f_hyde.result()
        step_back_q = f_sb.result() if f_sb else None
        sub_questions = f_sq.result() if f_sq else []

    drafts = [hyde_doc]
    if step_back_q:
        drafts.append(step_back_q)
    drafts.extend(sub_questions)
    return drafts


def multi_query_retrieve(question: str, llm, retriever, drafts: list[str] | None = None) -> list:
    """
    三路并行召回（仅对 compound_injury 触发），返回 Document 对象列表：
      路1: HyDE 假想文档检索
      路2: Step Back 抽象问句检索
      路3: 拆解子问题 × 各自检索

    Args:
        question: 用户原始查询
        llm: 统一适配器 FallbackChain 实例
        retriever: FitnessRAGRetriever 实例
        drafts: 锁外已生成的草稿（pipeline 两阶段用法）；None 时内部生成（旧调用方兼容）

    Returns:
        合并去重后的 LangChain Document 列表
    """
    drafts = drafts if drafts else generate_hyde_drafts(question, llm)

    all_doc_lists = [_retrieve_docs(d, retriever, MULTI_QUERY_TOP_K) for d in drafts]
    return _merge_deduplicate(all_doc_lists, MAX_TOTAL_DOCS)


def hyde_retrieve(question: str, llm, retriever, top_k: int | None = None,
                  drafts: list[str] | None = None) -> tuple[str, list]:
    """
    HyDE 检索管道：分类 → 选择检索策略 → 格式化。

    - compound_injury: 三路并行召回（HyDE + Step Back + 子问题拆分）
    - simple / single_injury: 单路 HyDE 假想文档检索

    Args:
        question: 用户原始查询
        llm: 统一适配器 FallbackChain 实例（建议 HYDE_MODEL 小模型，不占用主生成算力）
        retriever: FitnessRAGRetriever 实例
        top_k: 单路 HyDE 检索 top-k（None=config.HYDE_TOP_K；plan 层传更大值加宽上下文）
        drafts: 锁外已生成的草稿（pipeline 两阶段用法：草稿生成锁外，检索锁内）

    Returns:
        (格式化后的检索上下文字符串, 原始 Document 列表) — docs 供引用来源展示，
        调用方无需为引用再执行一次检索。
    """
    # 1. 分类
    label = classify_query(question)

    # 2. 复合伤病 → 三路召回；其他 → 单路 HyDE
    if label == "compound_injury":
        docs = multi_query_retrieve(question, llm, retriever, drafts=drafts)
        return "\n\n".join(doc.page_content for doc in docs), docs

    # 3. 简单/单伤病：单路 HyDE 检索 → 返回父块
    hyde_doc = drafts[0] if drafts else generate_hypothetical_doc(question, llm, label)
    k = top_k or HYDE_TOP_K
    docs = retriever.similarity_search(hyde_doc, k=k)
    return "\n\n".join(doc.page_content for doc in docs), docs
