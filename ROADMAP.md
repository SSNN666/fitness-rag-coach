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

## ✅ 待办 A：三件小事（约 1 小时）

- [ ] **A1. 提交所有改动**
  `fitness_rag_coach` 与 `user_insight_bot` 均有未提交改动。提交后评测记录里的
  `git_dirty: True` 会变成 `False`，可追溯性才干净。

- [ ] **A2. eval 增量落盘**
  现状：`save_results()` 只在**全部算完**后调用一次。风险：RAGAS 阶段若卡死（实测当天卡过两次），
  前面已跑完的检索指标会**一起丢失**。
  改法：`compute_retrieval_metrics()` 跑完后先落盘一次（可标记 `partial: true`），RAGAS 跑完后再覆盖。

- [ ] **A3. Neo4j 接入 start.py**
  容器 `neo4j-fitness` 创建时**未设 `--restart`**，重启电脑后需手动 `docker start neo4j-fitness`。
  风险：面试演示前忘记启动 → 图谱页为空、三路检索退化成双路。
  改法：`start.py` 启动时检测容器状态，未运行则自动 `docker start`（docker 不可用时跳过并提示）。

---

## 🎯 待办 B：Agent 三件套（1-2 周）★ 对齐目标岗位

> 背景：目标岗位是 AI 应用 / Agent 开发；当前项目的短板是「Agent 部分薄」——
> 工具是关键词触发而非模型决策、记忆是进程内 LRU 重启即失。

- [ ] **B1. 记忆持久化**
  现状：`_SessionHistory` 进程内 LRU（`SESSION_MAX_COUNT=64`），重启即失。
  目标：会话历史落盘（SQLite / JSON），跨重启可恢复；用户画像可跨会话复用。

- [ ] **B2. 模型决策的工具调用**
  现状：`health_tools.py` 的 `TOOL_REGISTRY` 是现成形状，但触发靠**关键词匹配**（`run_health_tools`）。
  目标：把工具定义暴露给模型，由模型自主决定调用哪个（多步）；保留关键词触发作为降级兜底。
  注：`llm_adapter.py` 已实现 tools 协议（`tool_calls` 流式组装），但 pipeline 零使用。

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
