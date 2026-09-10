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

- [x] **D2. SHA256 去重 + 嵌入缓存**（2026-09-10 完成）
  新增 `ingest_cache.py`：
  - `IngestManifest`：源文件 SHA256 指纹清单（**按内容判重**，不看文件名/mtime——
    mtime 会被 checkout/复制改掉，同一内容换名字也是重复）。重建前比对输出
    新增/变更/未变/移除；**全部未变则整轮跳过**。
  - `EmbeddingCache`：以**内容哈希**为键缓存向量。文档改一处，其余块复用向量。
  - `--force` 忽略两者强制全量重建。
  - **实测**（271 chunk）：全部未变 → 整轮跳过；改 1 个文件 → 缓存命中 271/未命中 1
    （99.6%）；还原后重建 → 命中 100%，**零嵌入 API 调用**。
  - 边界：整文件级哈希，不是增量更新；清单/缓存为本地产物（已 gitignore）；
    `embed_cache.json` 约 4.6MB/271 向量，语料变大时需换紧凑格式。
  测试 225 项通过（新增 18 项）。
- [x] **D4. 公网只读 demo**（2026-09-10 完成）
  新增 `build_demo.py` + `demo/`（index.html 自包含、无 CDN 依赖）。
  8 个用例覆盖各链路亮点：简单/伤病/禁忌拦截/工具计算/图谱/问法重述/弱相关/无依据拒答；
  每张卡片可展开**引用来源**与**检索详情（三路融合得分）**——不只给结果，还给中间过程。
  页面顶部明确标注「离线快照，非实时服务」，不误导。
  部署：GitHub Pages / OSS / HF Spaces 任选（`demo/README.md` 有步骤）。

  ⚠️ 过程中又发现一处**文档声明未经实测**：DEMO_SCRIPT 写「量子力学和健身的关系 →
  知识库无依据拒答」，实测该问题相关度 **0.52 > 阈值 0.40**，系统如实作答「无直接关联」。
  **系统行为是对的，错的是那句没测过的文档**——已修正 DEMO_SCRIPT 并把该用例改为
  「弱相关问题」，另补真正会拒答的 `如何挑选股票`（相关度 0.00）。

---

## ✅ 已完成：P0 三个安全/成本缺口（2026-09-10）

> 背景：把服务当「真实在线服务」审视后发现的三个问题——都不是功能缺失，
> 是**上线就会出事**的那种。顺序：先堵洞，再压测（见下）。

- [x] **P0-1. user_profile 未过安全闸门**（`guardrails.py` / `api.py`）
  `user_profile` 与 `question` 一样被拼进三套 Prompt 模板（`{user_profile}` 占位符），
  却只检测了 `question`——把注入载荷放进画像字段即可整段绕过。
  - `detect_injection_multi(fields)`：按**整个请求**跨字段累加权重，命中记录带字段名；
    `detect_injection` 退化为其单字段特例。
  - 跨字段累加而非逐字段判定：防「question 放半句、profile 放半句」的拆分投毒。
  - 画像同时并入内容审核送检（一次调用合并，不增加审核延迟）。
  - 顺手接上了零引用的 `PROMPT_INJECTION_ENABLED` 开关（此前是「写着能关、其实关不掉」）。
  - **实测**：载荷放 profile → 旧路径 score=0 放行，新路径 score=5 拦截
    （`fields=["user_profile"]`）；正常画像 score=0 不误伤。

- [x] **P0-2. 会话隔离**（`app.py` / `gateway.py`）
  UI 硬编码 `SESSION_ID = "default_user"` → 所有访客共用一份会话，后果有三：
  ① 共用对话历史（**A 的多轮上下文会被拼进 B 的 Prompt**，隐私）
  ② 共用限流桶（A 刷满额度 → B 收到 429）
  ③ 共用降噪窗口（B 正常提问可能被判成「A 刚问过的重复问题」而 409）
  改为随 `st.session_state` 生成本次访客的随机 ID（`ui-<16hex>`，无 cookie、无身份信息）。
  - **连带修**：会话隔离后状态键随访客数增长，而 `gw_state` 此前**没有任何回收**
    （只因 session_id 恒为一个值才没暴露）。新增 `Gateway._sweep_state`：
    按限流窗口 / 降噪 TTL 回收过期键，超硬上限则按最久未活动驱逐；
    清扫挂在 `check_rate_limit`（每请求必经）上并按 interval 摊薄。
    只回收自己的前缀，不动调用方放在同一 dict 里的其他键。
  - **实测**：AppTest 两个会话拿到不同 ID 且同一会话内 rerun 稳定；
    A 用尽额度后 B 用不同 ID 问同一问题 → 200（修复前为 409）。

- [x] **P0-3. 成本上限**（新增 `cost_guard.py`）
  此前**只有用量日志、没有任何上限**——公开服务挂着 API Key，被刷就是真金白银的损失。
  - 进程级累计 token 账本 + 两级闸门：**软阈值**关掉深思考（服务仍可用但更便宜），
    **硬阈值**直接拒绝（只在真正失控时触发）。落盘，重启不清零。
  - 记账挂在 `Gateway.log_usage`（所有 LLM 调用的公共漏斗），且**在 `enabled` 判断之外**——
    否则关掉网关日志就出现一条完全不记账的调用路径。
  - **刻意不降级的东西**：禁忌判定 / 硬过滤 / 事实核查 / CRAG / 分层检索。
    前四项是安全链路；CRAG 与 Fact-Check **只对伤病类触发**，为省钱关掉它们
    等于在最需要完整的回答上偷工减料——那类流量交给硬阈值统一兜底（直接拒绝），
    而不是给一个「更便宜的伤病建议」。
  - 顺手修：usage 日志的 `role` 字段三处硬编码 `"chat"` → 写真实层级
    （`chat` / `chat_fast` / `chat_nothink`）。不改的话日志分不出用量来自哪一层，
    **成本归属无从查起**——想优化成本先得知道钱花在哪个角色上。
  - **实测**（真实服务）：硬阈值触发后 `/v1/chat` → 429、SSE → error 帧，
    且被拒的两次请求 **token 增量为 0**（确认拒绝发生在花钱之前）；
    软阈值下同一类问题 `chat`(3756 tokens) → `chat_nothink`(1826 tokens)，
    **降 51%**，而两次的 `fact_check` 都照常执行（安全链路未受影响）。

**测试 225 → 266 项。**

---

## ⏭️ 下一步：压测 → 按判据决定 Redis → Dockerfile

> 预先承诺的判据（不做事后合理化）：压测后
> **瓶颈在会话/限流状态 → 接 Redis**；**在 LLM 并发配额 / Milvus 锁 / 根本没到顶 → 不接**。
> 界于此：**三个 P0 + 压测 + 按判据的 Redis + Dockerfile，到此为止**。
> demo 部署押后（用户决定）。

---

## 🔧 已知遗留问题（低优先级）

| 问题 | 位置 | 说明 |
|---|---|---|
| 图谱不可用时的连接超时 | `graph_view.py` / `retriever.py` | `connection_timeout=8`，无 Neo4j 时测试从 3s → 39s，线上请求同样吃延迟。建议加熔断或缩短超时 |
| `entity_labels` 列建成但不回读 | `retriever.py:704-710` | 现依赖运行时从 `page_content` 现抽（兜底路径），Milvus 列未使用 |
| 孤儿文件 | 仓库根目录 | `eval_results.csv` / `eval_results_threepath.csv` / `cleanup_c.bat` 无任何引用 |
| 死配置 | `config.py` | `RERANKER_DOC_MAX_CHARS`（实际硬编码 200）、`HYDE_INJURY_KEYWORDS`（零引用）、`SSE_CHUNK_CHARS`（死导入）。（`PROMPT_INJECTION_ENABLED` 已于 P0-1 接上，移出本表） |
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
