# 🗺️ ROADMAP —— 后续计划与当前状态

> **用途**：记录项目当前状态与后续待办，防止会话/上下文丢失后无从接手。
> **维护约定**：每完成一项就更新状态；新增待办写在对应阶段下。
> **最后更新**：2026-09-10

---

## 📍 当前状态（截至 2026-09-10）

### 已完成

| 批次 | 内容 | 验证 |
|---|---|---|
| Neo4j 启用 | 本地 Docker 实例（`neo4j-fitness`）替代 AuraDB；`NEO4J_ENABLED=True` | 真实实例评测 3/3 PASS |
| 图谱缺陷修复 | 5 个 bug（见下） | 195 节点 / 337 关系，禁忌召回 100% |
| 知识库清理 | `fitness_data.csv` 72 → 62 行（移除 10 行混入的架构笔记） | 索引残留 0 条 |
| 索引重建 | Milvus 271 chunk，`entity_labels` 完整 | 实测抽样确认 |
| 评测落盘 | `eval_results/latest.json`（含配置/git commit/逐样本明细） | 全量 100 条跑通 |
| 全量评测 | Hit@1/3/5 = 0.79 / 0.91 / 0.95，MRR 0.851 | 2026-09-10 |

**已修复的 5 个 bug**（详见 [INTERVIEW_STORIES.md](INTERVIEW_STORIES.md)）：
1. `CSVLoader` 未传 `metadata_columns` → 图谱缺 3 种关系（0 条）+ `entity_labels` 全空
2. 两处 Cypher 未读 `r.relation` 属性 → 关系标签永不匹配，全落默认分 0.5
3. `_neo4j_one_hop` 的 `LIMIT` 无 `ORDER BY` → 打分前任意截断
4. **安全**：`_resolve_contraindications` 降级条件写成 `not NEO4J_ENABLED` → 图谱运行时故障时不降级
5. `graph_view._load_from_neo4j` 缺别名归一 → 重复伤病节点

**另外两个（2026-09-10 补修）**：
6. `_run_fact_check` 未接收 `user_profile` → 缓存 key 与快速路径口径不一致，跨画像串用答案（已加回归测试）
7. 阶段 D 重排在 HyDE 路径下冗余（ctx 已在 `similarity_search` 内重排过）→ 每请求浪费一次 LLM 调用

---

## ✅ 已完成：三件小事（2026-09-10）

- [x] **A1. 提交所有改动**
  `fitness_rag_coach` → 分支 `fix/graph-verification-and-eval`（2 个提交）
  `user_insight_bot` → 分支 `feat/factcheck-benchmark`（1 个提交）
  ⚠️ 提交在**特性分支**上，未推送到远端；合并到 master/main 需手动操作。

- [x] **A2. eval 增量落盘**
  `compute_retrieval_metrics()` 跑完后先落一次盘（`meta.partial=true`），RAGAS 跑完再落完整结果。
  `latest.json` 只指向完整跑，中间结果不覆盖权威指针。

- [x] **A3. Neo4j 接入 start.py**
  `_ensure_neo4j()`：检测容器状态，未运行则 `docker start` 并等 7687 就绪；
  docker 缺失/容器不存在只提示不阻断。两条路径均已实测。

---

## 🎯 待办 B：Agent 三件套（1-2 周）★ 对齐目标岗位

> 背景：目标岗位是 AI 应用 / Agent 开发；当前项目的短板是「Agent 部分薄」——
> 工具是关键词触发而非模型决策、记忆是进程内 LRU 重启即失。

- [x] **B1. 记忆持久化**（2026-09-10 完成）
  新增 `session_store.py`：`SessionHistory` + `SessionStore(dict)`（对 pipeline/网关透明），
  JSON 原子落盘（tmp → `os.replace`）；`config` 加 `SESSION_PERSIST_ENABLED` / `_PATH`。
  坏文件空存储启动并自愈；落盘失败静默降级为内存态；普通 dict 时全部逻辑跳过（测试不受影响）。
  **实测**：进程 1 问答落盘 → 进程 2（模拟重启）恢复 2 条历史 →
  `needs_rewrite('那臀桥呢', 恢复的历史) = True`（对照组无历史 = False），
  **多轮指代消解跨重启存活**。测试 164 项通过（新增 13 项）。
  单测还抓出并修正了本项目一个真 bug：未 touch 的会话不在 `_last_access` 里，
  只对其排序会把**有**时间戳的会话当最旧驱逐，与意图完全相反。

- [x] **B2. 模型决策的工具调用**（2026-09-10 完成）
  `health_tools.py` 新增 `tool_definitions()` / `execute_tool()` / `resolve_health_tools()`；
  `config` 新增 `tool_router` 角色 + `HEALTH_TOOLS_MODEL_DECISION` 开关；
  pipeline 把工具决策**从锁内移到锁外**（模型调用是网络 IO，进锁会阻塞检索段）。
  **关键设计**：参数不由模型提供——身高/体重/年龄走确定性抽取，模型只选工具。
  **成本控制**：仅当能抽到参数时才发起模型调用。
  实测（真实 DashScope）：模型正确调用 BMI/饮水/心率，且对无关问题不调；
  其中「我每天该补充多少液体」**关键词命中不了、模型选对了**。
  测试 151 项通过（新增 7 项）。

- [x] **B3. 上下文管理**（2026-09-10 完成）
  修了 2 个真 bug + 补了设计文档 [CONTEXT_MANAGEMENT.md](CONTEXT_MANAGEMENT.md)：
  1. **预算恒按本地 8192 算** → 云端主链 128k 窗口被白扔 16 倍（预算只有 4505）。
     改为按实际激活供应商取窗口（`LLM_CONTEXT_WINDOWS`，未知安全回退）。
  2. **截断后重组吞掉原标点**：`split` 已吃掉分隔符，再用 `"。".join()` 拼回
     → `！？；` 与**换行**全变成「。」。黑名单按 `\n` 分条，被压成一行跑文。
     改为捕获组 split + 原样拼回。
  文档覆盖：预算取值 / 级联截断顺序 / **头部保留（安全数据不可裁）** /
  分层生成预算 / 可观测性 / 边界限制。测试 172 项通过（新增 8 项）。

---

## ✅ 已完成：C MCP 封装（2026-09-10）

新增 `mcp_server.py`（依赖 `mcp>=2.2.0`），暴露 6 个工具：

- [x] `search_knowledge_base(query, k)` —— 三路检索（懒加载；Milvus 单进程独占见下）
- [x] `check_contraindication(injury, action)` —— 禁忌判定（**独家能力**，别名归一 + 子串匹配）
- [x] `get_injury_graph(injury, depth)` —— 图谱多跳，未启用/连不上时给可读指引
- [x] `calculate_bmi` / `estimate_water_intake` / `heart_rate_zone` —— 由 `TOOL_REGISTRY` 自动生成

**实测**（真实 MCP 协议握手，非 mock）：协议版本 `2025-11-25`，6 个工具全部可调用；
`get_injury_graph('腰突', 2)` 返回 13 条路径，含 `腰突 → 硬拉 → 竖脊肌` 这类间接关联。

⚠️ MCP SDK 2.x 把 `FastMCP` 改名为 `MCPServer`（`from mcp.server.mcpserver import MCPServer`），
`list_tools()` 为异步——照搬旧版 API 会踩坑。

⚠️ Milvus Lite 单进程独占：API 在跑时 MCP 的检索类工具无法打开向量库
（禁忌判定与健康计算不受影响）。要并存需让 MCP 改走 API 的 HTTP 接口。

测试 186 项通过（新增 14 项）。

---

## 📦 待办 D：RAG 工程完备性（按需）

- [x] **D1. 统一文档归一化层**（2026-09-10 完成）
  新增 `doc_loaders.py`：注册表（`.txt/.md/.csv/.pdf/.docx/.xlsx` + 图片 OCR）+ 统一
  `LoadedDoc(markdown, metadata)`。`build_index` 改为走注册表，格式差异不再泄漏进索引构建。
  依赖新增 `python-docx`、`openpyxl`（mcp 已在 C 阶段加入）。
  CSV loader 输出与改造前逐字段一致（62 行、metadata 键相同、正文相同）。

- [x] **D3. 跨页表格处理**（2026-09-10 完成，与 D1 同一批）
  - **误判过滤**：`extract_tables()` 会把竖排文本识别成单列表格。实测某扫描件抽出
    10 个「表格」全是误判——不加校验会把正常正文重排成无意义表格。现加结构校验
    （≥2 行 / ≥2 列 / 非空≥30% / ≥2 行有实质内容），修复后该文件 0 个误判。
  - **跨页合并**：上页末表 + 本页首表 + 列数相同 + 表尾贴页底/表头贴页顶 → 合并并去重表头。
    无坐标信息时保守不合并。单测覆盖 5 个场景。
  - **切块配合**：表格独立成块，不与正文混排（否则会被逐行打散）；超大表也切在行边界。

- [ ] **D2. SHA256 去重 + 增量索引**：上传前算哈希，已存在则跳过/更新，避免全量重建
- [ ] **D4. 公网只读 demo**：预置答案池 + 检索详情可查，部署 HF Spaces / 函数计算
      （绕开 Milvus Lite 单进程与 API Key 暴露）

---

## 🔧 已知遗留问题（低优先级）

| 问题 | 位置 | 说明 |
|---|---|---|
| 图谱不可用时的连接超时 | `graph_view.py` / `retriever.py` | `connection_timeout=8`，无 Neo4j 时测试从 3s → 39s，线上请求同样吃延迟。建议加熔断或缩短超时 |
| `entity_labels` 列建成但不回读 | `retriever.py:704-710` | 现依赖运行时从 `page_content` 现抽（兜底路径），Milvus 列未使用 |
| 孤儿文件 | 仓库根目录 | `eval_results.csv` / `eval_results_threepath.csv` / `cleanup_c.bat` 无任何引用 |
| 死配置 | `config.py` | `RERANKER_DOC_MAX_CHARS`（实际硬编码 200）、`HYDE_INJURY_KEYWORDS`（零引用）、`PROMPT_INJECTION_ENABLED`（零引用）、`SSE_CHUNK_CHARS`（死导入） |
| `contra_data` 重复条目 | `contra_data.py` | 「腰突」是「腰间盘突出」的截断副本，别名归一后永不可达 |
| `fact_cache` 失效不含禁忌表变更 | `fact_cache.py` | 改 `contra_data` 后未 bump `FACT_CACHE_VERSION` 会吐旧答案 |
| 文档陈旧 | `README.md` 等 | 部分 docstring 描述与实现不符（如 `pipeline.py:143` 仍写「单全局锁串行化」） |

---

## 📎 相关文档

| 文档 | 用途 |
|---|---|
| [INTERVIEW_STORIES.md](INTERVIEW_STORIES.md) | 面试故事集（8 个故事 + 讲法 + 追问预案） |
| [DEMO_SCRIPT.md](DEMO_SCRIPT.md) | 6 步演示剧本 |
| [CHANGELOG.md](CHANGELOG.md) | 正式变更记录 |
| `eval_results/latest.json` | 最近一次评测的完整产物（配置 + commit + 逐样本） |
