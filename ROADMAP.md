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

- [ ] **B3. 上下文管理**
  现状：已有令牌预算级联截断（`gateway.guard_token_budget`，头部保留 + 句子边界截断）。
  目标：把这块做深并讲清楚——分层预算、截断策略、超限告警。**这是三个里最有存量优势的。**

---

## 🔌 待办 C：MCP 封装（3-5 天）

把整条能力暴露成 MCP Server，供任意 MCP 客户端（Claude Code 等）调用：

- [ ] `search_knowledge_base(query, k)` —— 三路检索
- [ ] `check_contraindication(injury, action)` —— 禁忌判定（**独家能力**，图谱 + 本地双源）
- [ ] `get_injury_graph(injury, depth)` —— 图谱多跳
- [ ] `calculate_bmi / water_intake / heart_rate_zone` —— 确定性计算（`TOOL_REGISTRY` 直接映射）

> 这一步把「一个问答应用」变成「其他 Agent 可消费的能力」，是 reviewer 点名的方向。

---

## 📦 待办 D：RAG 工程完备性（按需）

- [ ] **D1. 统一文档归一化层**：loader 注册表（pdf / docx / **xlsx** / 图片 → 统一 Markdown 中间表示）
- [ ] **D2. SHA256 去重 + 增量索引**：上传前算哈希，已存在则跳过/更新，避免全量重建
- [ ] **D3. 跨页表格处理**：`pdfplumber.extract_tables()` 版面分析 → 表格作原子块不参与递归切分
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
