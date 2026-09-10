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

import os

from dotenv import load_dotenv

load_dotenv()  # 从 .env 加载密钥（NEO4J_PASSWORD / BOCHA_API_KEY），密钥不入库

LLM_MODEL = "qwen2.5:7b"            # 主流式生成 + CRAG 兜底
EMBEDDING_MODEL = "nomic-embed-text" # 向量嵌入（本地 Ollama，不变）
# 云端 Embedding（消除并发时的本地 embedding 排队；切换供应商必须重建索引）
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "cloud")   # "cloud" | "ollama"
EMBEDDING_CLOUD_MODEL = "qwen3.7-text-embedding"  # MaaS 实测：原生 1024 维，指定 768 与现有 schema 一致
EMBEDDING_CLOUD_DIM = 768                         # 必须等于 MILVUS_DIM
HYDE_MODEL = "qwen2.5:0.5b"          # HyDE 假想文档生成（0.5B instruct，极速草稿）
RERANK_MODEL = LLM_MODEL              # Reranker 与主生成共用 qwen2.5:7b（串行复用）
CSV_FILE = "fitness_data.csv"
PDF_SOURCE_NAME = os.getenv("PDF_SOURCE_NAME", "康养公开资料.pdf")  # 仅图片 OCR 路径（占位扫描件）的来源显示名
# 知识库文本直抽源（合规公开资料，零 OCR）：(文件, 引用来源显示名)
# 配置后 build_index 优先走文本路径，跳过 pdf_pages/ 图片 OCR（占位版权扫描件路径自动停用）
TEXT_KB_SOURCES = [
    {"file": "kb_18fa.txt",   "name": "科学健身18法（国家体育总局体科所）"},
    {"file": "kb_zhinan.txt", "name": "全民健身指南（国家体育总局）"},
]
RETRIEVE_TOP_K = 5

# ============================================================
# Milvus Lite 向量数据库（嵌入式模式，无需 Docker）
# ============================================================
MILVUS_URI = "./milvus.db"
MILVUS_COLLECTION = "fitness_rag"
MILVUS_DIM = 768                     # nomic-embed-text 输出维度

# gRPC keepalive 覆盖：pymilvus 3.x 默认 grpc.keepalive_time_ms=10000（每 10s ping），
# Milvus Lite 内嵌服务端 ping 限频更严 → 每约 2 分钟一次 GOAWAY "too_many_pings" 断连重连
# （日志 E0814 chttp2_transport 噪音，连接自动恢复）。本地嵌入式连接无需 keepalive 探活，
# INT_MAX 为 gRPC 禁用语义；如换远端 Milvus 服务可按需改回较小值。
MILVUS_GRPC_OPTIONS = {"grpc.keepalive_time_ms": 2147483647}

# ============================================================
# Neo4j AuraDB 知识图谱（云端免费实例；默认停用 NEO4J_ENABLED=False）
# 连接信息全部走 .env —— 历史版本曾硬编码实例 URI（已移除，凭证已轮换）
# ============================================================
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")  # 密钥位于 .env，勿硬编码
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

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
FUSION_THRESHOLD = 0.20              # 融合**加权分**阈值（多路认可的最低加权分）
# 向量路专属的**原始余弦**地板：加权分阈值对最低权重那一路是失效的——
#   有效门槛 = FUSION_THRESHOLD / w_route → 普通 0.50 | 单伤病 0.667 | 复合 1.00
# 复合伤病问句要求余弦 ≥ 1.00，即向量路**永远**被丢弃（实测：Milvus 返回的
# 0.576/0.559/0.527 高相关内容全被丢，融合结果 5/5 只有图谱）。
#
# 取值 0.40 = 沿用 grounding 的实测校准口径（无关类余弦 max≈0.29-0.38，相关类 0.35-0.67）。
# 三个口径各重复测 2~4 轮（单次测量不足以下结论——这一点我在这轮里栽过两次）：
#   ① 改前（只用加权阈值）  Hit@1 0.827 / Hit@3 0.910 / Hit@5 0.953 / MRR 0.871
#   ② 原始分地板 0.20       Hit@1 0.795 / Hit@3 0.903 / Hit@5 0.948 / MRR 0.849
#   ③ 原始分地板 0.40       Hit@1 0.800 / Hit@3 0.905 / Hit@5 0.960 / MRR 0.857
# 0.40 三项优于 0.20，且**定性行为正确**：无关问题（「如何挑选股票」余弦 0.29-0.33）
# 一条都进不来；相关问题的康复内容（余弦 0.49-0.55）正常进入。
# 0.20 太松：无关内容混入后经 _entity_match_boost 提权会挤掉正确内容——
# 实测 demo「腰突患者适合做什么康复训练」的引用一度变成「杠铃深蹲」（禁忌动作）。
VECTOR_RAW_FLOOR = 0.40
CONTEXT_DOCS_MAX = 3                  # 送入 LLM 的最终文档数上限
FUSION_SEMANTIC_DEDUP_ENABLED = False  # 语义去重：余弦相似度 > 阈值时仅保留得分高者
FUSION_SEMANTIC_DEDUP_THRESHOLD = 0.85  # 语义去重余弦相似度阈值
FUSION_MIN_DOCS = 3                  # 语义去重后至少保留的文档数

# ============================================================
# Neo4j 多跳推理配置
# ============================================================
NEO4J_DEPTH = 1                      # 单伤病默认 1-hop
NEO4J_MAX_DEPTH = 3                  # 复合伤病最大 3-hop
NEO4J_PATH_BOOST_CONTRAIND = 1.5     # 经过禁忌关系的路径 boost 倍数

# ============================================================
# 网关路由权重：根据查询类型动态切换 (Milvus, BM25, Neo4j)
# ============================================================
ROUTE_WEIGHTS_DEFAULT = (0.40, 0.35, 0.25)   # 普通问答：图谱微提权
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
BOCHA_API_KEY = os.getenv("BOCHA_API_KEY", "")  # 密钥位于 .env，勿硬编码

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
FACT_CACHE_PATH = os.getenv("FACT_CACHE_PATH", "fact_cache.json")
# 提示词版本（缓存 key 的一部分）：改动分层 Prompt / 层级 hint 后 bump 此值，
# 旧缓存自动失效（实测教训：改 600 字约束后重启，旧长回答仍从缓存吐出，新提示词不生效）
FACT_CACHE_VERSION = 2

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
SESSION_MAX_COUNT = 64                 # 会话 store 上限（超出按最久未访问驱逐，防止内存只增不减）

# 会话记忆持久化：进程内 dict → 落盘（跨重启保留多轮上下文）
# 关掉即退回原「进程内 LRU、重启即失」行为
SESSION_PERSIST_ENABLED = True
SESSION_PERSIST_PATH = os.getenv("SESSION_PERSIST_PATH", "sessions.json")  # 含用户对话内容，已加入 .gitignore

# ============================================================
# 切片配置（token 级，基于 tiktoken cl100k_base 编码）
# ============================================================
CHUNK_CHILD_TOKENS = 300
CHUNK_PARENT_TOKENS = 600
CHUNK_OVERLAP_RATIO = 0.15

# ============================================================
# HyDE (Hypothetical Document Embeddings) 配置
# ============================================================
HYDE_ENABLED = True
HYDE_TOP_K = 3
# simple 层跳过 HyDE 直查：基础问答直查 Hit@3=1.00 已满分，草稿改写增益有限却费时
# （省 1 次 LLM 调用 + 1 次重复检索，simple 层首 token 更快）；置 False 恢复 HyDE 草稿检索
HYDE_SKIP_SIMPLE = True

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
RERANKER_DOC_MAX_CHARS = 400   # 每个候选文档截断字符数

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
GATEWAY_LOG_PATH = os.getenv("GATEWAY_LOG_PATH", "gateway.log")   # 结构化日志 (JSON Lines)
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
NOISE_TTL = 600                           # 降噪状态存活秒数（超时后不再与该访客的旧提问比对）

# --- 成本上限（进程级累计 token 账本，保护被刷时的真金白银）---
# 两级闸门：软阈值 → 强制走 cheapest 层（服务仍可用但变便宜）
#           硬阈值 → 直接拒绝（只在真正失控时触发）
# 阈值按 token 计（供应商真实返回的计量，可核对）。要用金额就自行换算：
#     预算 token = 预算金额 ÷ 单价(元/千token) × 1000
# 默认值按「单请求约 3-6k token」估算，≈1000 次请求触发降级、≈3000 次触发拒绝；
# 上线前请按自己的实际预算改这两个值。
COST_GUARD_ENABLED = True
# 走环境变量：容器部署时改预算不该需要重建镜像（见 Dockerfile 步骤）
COST_SOFT_LIMIT_TOKENS = int(os.getenv("COST_SOFT_LIMIT_TOKENS", "5000000"))    # 累计超过 → 关掉深思考
COST_HARD_LIMIT_TOKENS = int(os.getenv("COST_HARD_LIMIT_TOKENS", "20000000"))   # 累计超过 → 拒绝新请求
COST_WINDOW = 86400                   # 统计窗口秒数（默认 24h，到期自动清零）
COST_GUARD_PATH = os.getenv("COST_GUARD_PATH", "cost_state.json")  # 重启不清零；不持久化就是绕过限额的捷径

# --- 会话级状态回收（限流/降噪按 session_id 建键）---
# 会话隔离修复后 session_id 不再固定，状态键随访客数增长；不回收即无上限增长。
# 原实现 session_id 恒为 "default_user"，这个问题被掩盖。
GW_STATE_MAX_KEYS = 512                   # 状态键硬上限（超过按最久未活动驱逐）
GW_STATE_SWEEP_INTERVAL = 60              # 两次清扫的最小间隔秒数（把 O(n) 摊薄）

# --- 令牌预算守卫 Strategy A (复用已有 CONTEXT_BUDGET_RATIO=0.55 / ANSWER_BUDGET_MIN=500) ---
TOKEN_BUDGET_ENABLED = True
TOKEN_BUDGET_TRUNCATE_WARN = 0.30         # 截断超过 30% 时在 UI 展示警告

# --- 内存守卫 Strategy B ---
MEMORY_GUARD_ENABLED = True
MEMORY_GUARD_THRESHOLD = 3.0              # 可用内存低于此值 (GB) 触发降级
MEMORY_GUARD_RECOVERY = 4.0               # 可用内存高于此值 (GB) 恢复 (迟滞防抖)
MEMORY_GUARD_DEGRADED_MODEL = "qwen2.5:1.5b"

# ============================================================
# ★ 统一大模型适配器（康养 Demo 改造）
#   主链路云端（阿里云百炼 DashScope → DeepSeek → 千帆 ERNIE），Ollama 仅本地调试/兜底
# ============================================================
LLM_PROVIDER_PRIMARY = "dashscope"      # 主链路供应商："dashscope" | "ollama"（全本地调试）
LLM_FALLBACK_CHAIN = ["dashscope", "deepseek", "qianfan", "ollama"]  # 降级链顺序；未配 Key 的供应商自动跳过
LLM_LOCAL_FALLBACK_ENABLED = True       # 云端全部失败时是否落本地 Ollama

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")   # 密钥位于 .env，勿硬编码
DASHSCOPE_BASE_URL = os.getenv(                         # 私有 MaaS 部署用自定义 Host，公共云用 dashscope.aliyuncs.com
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)  # OpenAI 兼容模式
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")     # DeepSeek 开放平台（OpenAI 兼容）
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
QIANFAN_API_KEY = os.getenv("QIANFAN_API_KEY", "")       # 可选第二云供应商（千帆 ERNIE）
QIANFAN_BASE_URL = "https://qianfan.baidubce.com/v2"
OLLAMA_BASE_URL = "http://localhost:11434"

LLM_TIMEOUT = 60.0                      # 云端单次请求超时秒数
LLM_LOCAL_TIMEOUT = 300.0               # 本地 Ollama 超时（CPU 推理慢）
LLM_MAX_RETRIES = 2                     # 单供应商内可重试次数（额度不足/鉴权/上下文超长不盲目重试）
LLM_RETRY_BACKOFF = (1.0, 2.0)          # 重试退避秒数

# 各供应商的上下文窗口（tokens）
# ⚠️ 令牌预算必须按**实际激活的供应商**取值。原实现恒用 OLLAMA_NUM_CTX(=8192)，
#    云端主链（qwen3.7 128k）可用上下文被白扔 16 倍——上下文预算只有 4505 tokens。
#    本地 Ollama 仍按其服务端配置（OLLAMA_CONTEXT_LENGTH）。
LLM_CONTEXT_WINDOWS = {
    "dashscope": 131072,        # qwen3.7-plus / flash（MaaS 私有部署按实际型号调整）
    "deepseek": 65536,          # deepseek-chat
    "qianfan": 131072,          # ernie-4.5-turbo-128k
    "ollama": OLLAMA_NUM_CTX,   # 本地模型：与服务端 OLLAMA_CONTEXT_LENGTH 保持一致
}

# 角色 → 各供应商模型映射 + 生成参数（vision 仅 qwen3-vl-plus：VL 只做图像理解，本地无 VL → 云失败返回明确提示）
# 注：私有 MaaS 部署无 qwen-plus 公共型号，按 models.list() 实际可用名映射
# thinking: 混合思考开关（仅 DashScope 生效）。qwen3.7 默认思考模式极慢（实测 8.4s vs 0.5s），
#           短回答类任务（快速问答/重排/校验/判分）全部关闭；复杂伤病问答保留思考保质量
LLM_ROLES = {
    "chat":         {"dashscope": "qwen3.7-plus", "deepseek": "deepseek-chat", "qianfan": "ernie-4.5-turbo-128k", "ollama": "qwen2.5:7b", "temperature": 0.7, "max_tokens": None, "thinking": True, "thinking_budget": 2048},
    "chat_nothink": {"dashscope": "qwen3.7-plus", "deepseek": "deepseek-chat", "qianfan": "ernie-4.5-turbo-128k", "ollama": "qwen2.5:7b", "temperature": 0.7, "max_tokens": None, "thinking": False},
    "chat_fast":    {"dashscope": "qwen3.7-flash", "deepseek": "deepseek-chat", "qianfan": "ernie-4.5-turbo-32k", "ollama": "qwen2.5:7b", "temperature": 0.7, "max_tokens": 400, "thinking": False},
    "hyde":         {"dashscope": "qwen3.7-flash", "deepseek": "deepseek-chat", "qianfan": "ernie-4.5-turbo-32k", "ollama": "qwen2.5:0.5b", "temperature": 0.7, "max_tokens": 256, "thinking": False},
    "fact_check":   {"dashscope": "qwen3.7-flash", "deepseek": "deepseek-chat", "qianfan": "ernie-4.5-turbo-32k", "ollama": "qwen2.5:1.5b", "temperature": 0.0, "max_tokens": 128, "thinking": False},
    "rerank":       {"dashscope": "qwen3.7-flash", "deepseek": "deepseek-chat", "qianfan": "ernie-4.5-turbo-32k", "ollama": "qwen2.5:7b", "temperature": 0.0, "max_tokens": 32, "thinking": False},
    "judge":        {"dashscope": "qwen3.7-flash", "deepseek": "deepseek-chat", "qianfan": "ernie-4.5-turbo-32k", "ollama": "qwen2.5:7b", "temperature": 0.0, "max_tokens": 8, "thinking": False},
    # 工具路由：只决定「调哪个工具」，不产出正文 → 短预算小模型足够
    "tool_router":  {"dashscope": "qwen3.7-flash", "deepseek": "deepseek-chat", "qianfan": "ernie-4.5-turbo-32k", "ollama": "qwen2.5:1.5b", "temperature": 0.0, "max_tokens": 128, "thinking": False},
    "vision":       {"dashscope": "qwen3-vl-plus-2025-12-19", "ollama": None, "temperature": 0.3, "max_tokens": 1024},
}

# ============================================================
# 分层生成策略：简单问题快速生成，生成时间随复杂度递增
#   默认全部关思考（快）；用户开启「深度思考」后 injury/plan 层切 chat（思考开）
#   simple: 快模型 + 短预算（400 token）+ 跳过 LLM 重排
#   injury: 主模型关思考 + 中预算 + 重排 + Fact-Check（答后提示可开深度思考）
#   plan:   主模型关思考 + 长预算 + 重排（答后提示可开深度思考）
# ============================================================
LLM_TIERS = {
    # retrieve_k: HyDE 检索 top-k；context_docs: 送入 LLM 的文档数（plan 层加宽——计划需要多文档依据）
    "simple": {"role": "chat_fast", "max_tokens": 400,  "rerank": False,
               "retrieve_k": 3, "context_docs": 3,
               "hint": "请简明扼要回答，控制在200字以内，直接给结论，不要展开。"},
    "injury": {"role": "chat_nothink", "max_tokens": 1024, "rerank": True,
               "retrieve_k": 3, "context_docs": 3,
               "hint": "请聚焦用户的具体问题作答，控制在600字以内，条理清晰，直接给建议与依据。"
                      "（深度思考模式下无此限制）"},
    "plan":   {"role": "chat_nothink", "max_tokens": 2048, "rerank": True,
               "retrieve_k": 6, "context_docs": 6, "hint": ""},
}

# ============================================================
# 检索融合模式：weighted（原加权融合）| rrf（Reciprocal Rank Fusion）
# ============================================================
FUSION_MODE = "weighted"                # 默认保留原加权融合，可切 "rrf"（eval 对比后定）
RRF_K = 60                              # RRF 常数 score(d)=Σ 1/(k+rank(d))

# ============================================================
# Neo4j 知识图谱开关（2026-09-10 启用）
# 数据源：本地 Docker 实例（容器名 neo4j-fitness，bolt://localhost:7687）
#   - 原 AuraDB 云实例凭证曾硬编码泄露，已弃用；本地实例无凭证泄露面
#   - 本地实例不会因闲置被暂停/删除（AuraDB Free：3 天暂停 → 30 天删除），
#     面试演示不依赖公网与第三方服务可用性
# 停用即回退双路检索（Milvus + BM25）；禁忌安全数据自动走 contra_data 本地副本
# ============================================================
NEO4J_ENABLED = True

# ============================================================
# FastAPI 服务（康养 Demo 唯一后端；Streamlit 改为 SSE 客户端）
# ============================================================
API_HOST = "127.0.0.1"
API_PORT = 8000
API_KEY_AUTH = os.getenv("API_KEY_AUTH", "")   # 演示级鉴权：非空则要求 X-API-Key 头；空=本地免鉴权
SSE_BUFFER_FIRST = False               # 已改为真流式（token 级 delta 实时透出）；保留开关仅供兼容
SSE_CHUNK_CHARS = 24                   # 真流式下不再使用（仅兼容保留）

# ============================================================
# 防护：Prompt 注入检测 + 百度内容审核
# ============================================================
PROMPT_INJECTION_ENABLED = True
PROMPT_INJECTION_BLOCK_SCORE = 4       # 加权分 ≥ 此值拦截（3 分仅记日志放行）
BAIDU_CENSOR_ENABLED = True
BAIDU_AK = os.getenv("BAIDU_AK", "")   # 内容审核 AK/SK（百度智能云，非千帆 Key）
BAIDU_SK = os.getenv("BAIDU_SK", "")
CENSOR_TIMEOUT = 3.0                   # 审核超时秒数（超时 fail-open 放行 + ERROR 日志）

# ============================================================
# 拒答（grounding）：知识库无依据时拒绝回答（生成前判定）
# ============================================================
REFUSE_ENABLED = True
UNFOUNDED_DISCLAIMER = (
    "\n\n> ⚠️ 提示：知识库未检索到直接相关依据，以上内容基于通用知识生成，"
    "仅供科普参考，不构成专业建议，请咨询专业人士。"
)
# 检索侧 OOM 防线：可用内存低于阈值 → 仅 BM25 稀疏检索（Milvus/图谱跳过）
RETRIEVAL_MEMORY_GUARD_ENABLED = True
RETRIEVAL_MEMORY_THRESHOLD_GB = 1.0
GROUNDING_MIN_CTX_CHARS = 80           # 上下文低于此字数且无外部资料 → 拒答
GROUNDING_MIN_SCORE = None             # weighted 模式最低融合分（None=沿用 FUSION_THRESHOLD）
GROUNDING_MIN_SIM = 0.40               # 无实体查询的语义相关性硬门槛：query 与 top 文档最大余弦低于此值 → 拒答
                                       # qwen3.7-text-embedding 实测校准：无关类 max≈0.29-0.38，无实体健身类 0.35-0.67，
                                       # 阈值取 0.40（宁拒少答，拒答时引导换问法；漏放由 PROMPT_GENERAL 引导兜底）
                                       # 注：切换 Embedding 供应商需重建索引并重新校准此值

# ============================================================
# OCR 乱码质检（摄入层，不进 pdf_ocr 避免污染缓存）
# ============================================================
OCR_QUALITY_ENABLED = True
OCR_QUALITY_MIN_CHARS = 10             # 低于此字数视为空页丢弃
OCR_QUALITY_CHAR_RATIO = 0.6           # 合法字符占比阈值
OCR_QUALITY_DICT_COVERAGE = 0.35       # jieba 词典命中率阈值（康养术语 OOV 多，放宽）
OCR_QUALITY_REPETITION = 0.3           # 最高频单字符占比阈值（>此值判乱码）
OCR_QUALITY_RETRY_SIZE = 2400          # 乱码页提分辨率重试的降采样边长

# ============================================================
# 确定性健康工具层（BMI / 饮水量 / 心率区间——计算注入上下文，
# 不让 LLM 自行算术；注册表结构可演进为 tools 协议）
# ============================================================
HEALTH_TOOLS_ENABLED = True
# 工具调用机制：True = 模型自主决策调哪个工具（tools 协议），关键词触发降级兜底；
# False = 仅关键词触发（旧行为，零额外 LLM 调用）
HEALTH_TOOLS_MODEL_DECISION = True

# ============================================================
# 多轮查询改写（指代消解后检索：会话历史只进 messages 时，
# 第二轮「那硬拉呢」的检索没有上一轮「腰突」上下文）
# ============================================================
REWRITE_ENABLED = True

# ============================================================
# 用户反馈闭环（POST /v1/feedback → feedback.jsonl，
# eval_testset.py --feedback 消费负反馈问题跑质量报告）
# ============================================================
FEEDBACK_PATH = os.getenv("FEEDBACK_PATH", "feedback.jsonl")
FEEDBACK_MAX_RECENT = 200      # 内存保留最近应答数（反馈按 request_id 解析）

# ============================================================
# 评测供应商（默认本地 Ollama 防烧钱；--cloud 切云端）
# ============================================================
EVAL_PROVIDER = "ollama"
