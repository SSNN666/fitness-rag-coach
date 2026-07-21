"""
Ollama Server Environment Variables (set BEFORE starting Ollama, then restart):

  Windows GUI (recommended):
    Start Menu → "Edit environment variables for your account" → add User variables:

    OLLAMA_FLASH_ATTENTION=1       # CUDA/GPU Flash Attention; harmless on CPU, future-proof
    OLLAMA_KV_CACHE_TYPE=q8_0      # ★ KV cache f16→q8_0, attention memory bandwidth ~-50%
    OLLAMA_CONTEXT_LENGTH=8192     # Server default context cap (matches project usage)
    OLLAMA_NUM_PARALLEL=1          # Single request at a time (CPU constraint)
    OLLAMA_MAX_LOADED_MODELS=2     # Embedding model + one LLM resident simultaneously

  Or via setx (run once as Administrator):
    setx OLLAMA_FLASH_ATTENTION "1"
    setx OLLAMA_KV_CACHE_TYPE "q8_0"
    setx OLLAMA_CONTEXT_LENGTH "8192"
    setx OLLAMA_NUM_PARALLEL "1"
    setx OLLAMA_MAX_LOADED_MODELS "2"
"""

LLM_MODEL = "qwen2.5:7b"            # 主流式生成 + CRAG 兜底
EMBEDDING_MODEL = "nomic-embed-text" # 向量嵌入（不变）
HYDE_MODEL = "qwen2.5:0.5b"          # HyDE 假想文档生成（0.5B instruct，极速草稿）
RERANK_MODEL = LLM_MODEL              # Reranker 与主生成共用 qwen2.5:7b（串行复用）
CSV_FILE = "fitness_data.csv"
RETRIEVE_TOP_K = 3

# ============================================================
# Milvus Lite 向量数据库（嵌入式模式，无需 Docker）
# ============================================================
MILVUS_URI = "./milvus.db"
MILVUS_COLLECTION = "fitness_rag"
MILVUS_DIM = 768                     # nomic-embed-text 输出维度

# ============================================================
# Neo4j AuraDB 知识图谱（云端免费实例）
# ============================================================
NEO4J_URI = "neo4j+s://04e61057.databases.neo4j.io"
NEO4J_USER = "04e61057"
NEO4J_PASSWORD = "bX_pIf4q6cpTV-JyeuIs2S1R8Y03LCojIunrDYnkS7Y"
NEO4J_DATABASE = "04e61057"

# ============================================================
# BM25 关键词检索（纯 Python rank-bm25）
# ============================================================
BM25_INDEX_PATH = "./bm25_index.pkl"

# ============================================================
# 三路检索融合权重 & 降噪阈值（w_milvus + w_bm25 + w_neo4j = 1.0）
# ============================================================
FUSION_WEIGHT_MILVUS = 0.40          # Milvus 语义向量权重（原 0.50，给图谱让空间）
FUSION_WEIGHT_BM25 = 0.30            # BM25 关键词权重（不变）
FUSION_WEIGHT_NEO4J = 0.30           # Neo4j 图谱关联权重（原 0.20，提升以发挥多跳推理价值）
FUSION_THRESHOLD = 0.15              # 全局最低分数阈值，低于此值丢弃

# ============================================================
# Neo4j 多跳推理配置
# ============================================================
NEO4J_DEPTH = 1                      # 单伤病默认 1-hop
NEO4J_MAX_DEPTH = 3                  # 复合伤病最大 3-hop
NEO4J_PATH_BOOST_CONTRAIND = 1.5     # 经过禁忌关系的路径 boost 倍数

# ============================================================
# 网关路由权重：根据查询类型动态切换 (Milvus, BM25, Neo4j)
# ============================================================
ROUTE_WEIGHTS_DEFAULT = (0.45, 0.35, 0.20)   # 普通问答：向量为主
ROUTE_WEIGHTS_SINGLE  = (0.30, 0.25, 0.45)   # 单伤病：图谱提权
ROUTE_WEIGHTS_COMPOUND = (0.20, 0.25, 0.55)  # 复合伤病：图谱主导

# ============================================================
# Fact-Check 事实校验配置
# ============================================================
FACT_CHECK_ENABLED = True              # 是否启用事实校验
FACT_CHECK_MODEL = "qwen2.5:1.5b"    # 校验用小模型（1.5B instruct，结构化输出纯净无干扰）
FACT_CHECK_NUM_PREDICT = 128          # instruct 模型直接输出，无需 thinking token 预算
FACT_CHECK_TRIGGER_TYPES = ["single_injury", "compound_injury"]  # 触发校验的查询类型

# CRAG 联网检索（校验失败时自动触发）
CRAG_ENABLED = True
CRAG_SEARCH_ENGINE = "bocha"           # 博查 AI 搜索（国内可用，个人免费套餐）
CRAG_MAX_RESULTS = 5                   # 每次搜索返回条数
CRAG_TIMEOUT = 5                       # 联网超时秒数
BOCHA_API_KEY = "sk-fd46e722a04749f6b6d758317fa535fc"

# CRAG 外部知识触发词：含这些关键词的伤病查询自动联网（本地知识库覆盖不足）
CRAG_EXTERNAL_KEYWORDS = [
    "争议", "研究", "最新", "指南", "学界", "临床", "文献",
    "证据", "进展", "共识", "综述", "随机对照",
    "机制", "原理", "为什么", "原因",
    "急性期", "慢性期", "恢复期", "术后",
    "膨出", "脱出", "游离",
    "2024", "2025", "2026",
    "哪个医院", "手术",
]

# 事实缓存（高频伤病问答缓存校验通过的结果）
FACT_CACHE_ENABLED = True
FACT_CACHE_MAX = 200                   # 最多缓存条数
FACT_CACHE_PATH = "fact_cache.json"

# ============================================================
# Ollama Engine Configuration (passed per-request to Ollama API)
# ============================================================
OLLAMA_NUM_CTX = 8192                 # Context window: matches project ~4K chars ≈ 5K tokens
OLLAMA_NUM_THREAD = None              # CPU threads: None = Ollama auto-detects physical cores

# ============================================================
# 上下文裁剪 & 网关截断配置
# ============================================================
CHAT_MAX_HISTORY_TOKENS = 2048         # Chat history max tokens (leaves room for system prompt + context)
CHAT_TOKEN_ESTIMATE_RATIO = 1.8        # Chinese token estimate multiplier (adjust if content gets truncated)
MAX_QUERY_CHARS = 1200                 # User query hard limit (~600-800 tokens for Chinese)
CONTEXT_BUDGET_RATIO = 0.55            # Retrieved context ceiling as fraction of num_ctx
ANSWER_BUDGET_MIN = 500                # Minimum tokens reserved for LLM generation output

# ============================================================
# 切片配置（token 级，基于 tiktoken cl100k_base 编码）
# ============================================================
CHUNK_CHILD_TOKENS = 200
CHUNK_PARENT_TOKENS = 500
CHUNK_OVERLAP_RATIO = 0.15

# ============================================================
# HyDE (Hypothetical Document Embeddings) 配置
# ============================================================
HYDE_ENABLED = True
HYDE_TOP_K = 3

# 伤病关键词（≥2字，避免单字误匹配健身动作名）
HYDE_INJURY_KEYWORDS = [
    "腰间盘", "腰突", "腰肌劳损", "腰痛", "腰疼", "腰椎",
    "颈椎", "半月板", "膝内扣", "膝超伸",
    "肩袖", "肩胛骨", "肩峰", "翼状肩",
    "扁平足", "富贵包", "圆肩", "驼背", "高低肩", "头前伸",
    "骨盆前倾", "骨盆后倾", "脊柱侧弯",
    "弹响", "抽筋", "劳损", "侧弯", "扭伤", "网球肘",
    "椎间盘", "关节炎", "足底筋膜", "跟腱", "髌骨",
    "腕管", "滑囊", "盂唇", "静脉曲张", "椎管狭窄", "坐骨神经",
    "酸痛", "疼痛", "撕裂", "拉伤", "挫伤", "脱位", "错位",
    "麻痹", "麻木", "肿胀", "积液", "粘连", "酸胀",
    "痉挛", "过紧", "无力", "退化", "磨损",
    "骨刺", "骨裂", "骨折",
    "伤病", "损伤", "炎症", "突出", "变直", "僵硬",
    "后倾", "前倾", "不适", "受限", "活动度", "代偿",
    "筋膜", "肌腱", "韧带", "软骨", "关节窝", "关节盂",
]

HYDE_COMPOUND_MARKERS = [
    "加", "和", "兼", "合并", "叠加", "同时", "两", "双", "复合",
    "整套", "综合", "搭配", "及", "与", "伴有", "伴随", "并发",
    "还有", "以及", "另外", "再加上",
]

# ============================================================
# Step Back + 多查询召回配置
# ============================================================
STEP_BACK_ENABLED = True
DECOMPOSE_ENABLED = True
MULTI_QUERY_TOP_K = 3
MAX_SUB_QUESTIONS = 4
MAX_TOTAL_DOCS = 12

# ============================================================
# 重排序 (Reranker) 配置 — 复用现有 Ollama LLM 做 listwise 重排
# ============================================================
RERANKER_ENABLED = True
RERANK_MILVUS_FACTOR = 5       # 原 3，扩大候选池给 reranker 选择
RERANK_BM25_FACTOR = 5         # 原 3
RERANK_NEO4J_FACTOR = 3        # 原 2
RERANKER_MAX_CANDIDATES = 15   # 送入 LLM 排名的最大候选数
RERANKER_DOC_MAX_CHARS = 200   # 每个候选文档截断字符数

# ============================================================
# 内存自适应 keep_alive（Windows GlobalMemoryStatusEx，零外部依赖）
# ============================================================

def get_available_memory_gb() -> float:
    """返回 Windows 可用物理内存 (GB)。非 Windows 或检测失败返回 99（默认高内存模式）。"""
    try:
        import ctypes as _ctypes

        class _MEMORYSTATUSEX(_ctypes.Structure):
            _fields_ = [
                ("dwLength", _ctypes.c_ulong),
                ("dwMemoryLoad", _ctypes.c_ulong),
                ("ullTotalPhys", _ctypes.c_ulonglong),
                ("ullAvailPhys", _ctypes.c_ulonglong),
                ("ullTotalPageFile", _ctypes.c_ulonglong),
                ("ullAvailPageFile", _ctypes.c_ulonglong),
                ("ullTotalVirtual", _ctypes.c_ulonglong),
                ("ullAvailVirtual", _ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", _ctypes.c_ulonglong),
            ]
        _m = _MEMORYSTATUSEX()
        _m.dwLength = _ctypes.sizeof(_MEMORYSTATUSEX)
        _ctypes.windll.kernel32.GlobalMemoryStatusEx(_ctypes.byref(_m))
        return _m.ullAvailPhys / (1024 ** 3)
    except Exception:
        return 99.0


def detect_keep_alive(threshold_gb: float = 10.0,
                       high: str = "10m", low: str = "3m") -> str:
    """根据可用内存返回 keep_alive 时长。
    ≥ threshold_gb → high（流畅模式），否则 → low（低内存模式）。
    7B Q4_K_M ~5GB + embedding ~0.3GB + OS overhead ~3GB → threshold 10GB 留余量。
    """
    return high if get_available_memory_gb() >= threshold_gb else low


# ============================================================
# 网关配置 (Gateway) — 速率限制 / 请求降噪 / 令牌预算 / 内存降级
# 每个组件有独立 _ENABLED 开关，出问题可单独关闭而不影响其他
# ============================================================

# --- 主开关 ---
GATEWAY_ENABLED = True
GATEWAY_LOG_PATH = "gateway.log"          # 结构化日志路径 (JSON Lines)
GATEWAY_LOG_LEVEL = "INFO"                # DEBUG | INFO | WARNING | ERROR

# --- 速率限制 (滑动窗口，防止快速重复请求压垮 Ollama) ---
RATE_LIMIT_ENABLED = True
RATE_LIMIT_MAX = 10                       # 滑动窗口内最大请求数
RATE_LIMIT_WINDOW = 60                    # 窗口秒数
RATE_LIMIT_COOLDOWN = 30                  # 触发限流后的冷却秒数

# --- 请求降噪 (bigram Jaccard 检测近重复查询) ---
NOISE_REDUCTION_ENABLED = True
NOISE_SIMILARITY = 0.85                   # 相似度阈值 (0-1)，越高越严格
NOISE_WINDOW = 5                          # 与最近 N 条查询比较

# --- 令牌预算守卫 Strategy A (复用已有 CONTEXT_BUDGET_RATIO=0.55 / ANSWER_BUDGET_MIN=500) ---
TOKEN_BUDGET_ENABLED = True
TOKEN_BUDGET_TRUNCATE_WARN = 0.30         # 截断超过 30% 时在 UI 展示警告

# --- 内存守卫 Strategy B ---
MEMORY_GUARD_ENABLED = True
MEMORY_GUARD_THRESHOLD = 3.0              # 可用内存低于此值 (GB) 触发降级
MEMORY_GUARD_RECOVERY = 4.0               # 可用内存高于此值 (GB) 恢复 (迟滞防抖)
MEMORY_GUARD_DEGRADED_MODEL = "qwen2.5:1.5b"
