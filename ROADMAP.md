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
| 全量评测（重跑） | Hit@1/3/5 = **0.83 / 0.91 / 0.95**，MRR **0.872**；RAGAS 忠实度 0.51 / 相关性 0.93 / 上下文精度 0.73 / 召回 0.63 | 2026-09-10 22:16，commit 42d147b |

**已修复的 5 个 bug**（详见 [INTERVIEW_STORIES.md](INTERVIEW_STORIES.md)）：
1. `CSVLoader` 未传 `metadata_columns` → 图谱缺 3 种关系（0 条）+ `entity_labels` 全空
2. 两处 Cypher 未读 `r.relation` 属性 → 关系标签永不匹配，全落默认分 0.5
3. `_neo4j_one_hop` 的 `LIMIT` 无 `ORDER BY` → 打分前任意截断
4. **安全**：`_resolve_contraindications` 降级条件写成 `not NEO4J_ENABLED` → 图谱运行时故障时不降级
5. `graph_view._load_from_neo4j` 缺别名归一 → 重复伤病节点

**另外两个（2026-09-10 补修）**：
6. `_run_fact_check` 未接收 `user_profile` → 缓存 key 与快速路径口径不一致，跨画像串用答案（已加回归测试）
7. 阶段 D 重排在 HyDE 路径下冗余（ctx 已在 `similarity_search` 内重排过）→ 每请求浪费一次 LLM 调用

**评测口径修正（2026-09-10 晚，核对简历数字时发现）**：
- 旧的 `latest.json`（13:04）对应 commit `c5a8ce0`，而 `retriever.py`（图谱评分 / LIMIT 截断）
  与 `build_index.py`（切块）在那之后都改过，**索引重建（14:20）还发生在评测（13:04）之后**
  → 那份数字对应的代码在仓库里找不回来，已重跑（22:16，commit `42d147b`）
- 重跑后 **「单伤病 + 复合伤病两类 Hit@3 均满分」不再成立**：单伤病 0.96（24/25），
  复合伤病仍 1.00。两次独立重跑都是 0.96 → 不是噪声
- 实测**跑动噪声约 ±1 个百分点**（Hit@1 0.82→0.83、Hit@5 0.96→0.95），
  100 条样本里逐样本命中排名变动 2 条。**引用这些数字时不应精确到小数点后两位**
- 修了 `eval_testset.py` 的 `git_dirty` 假阳性（详见 [INTERVIEW_STORIES.md](INTERVIEW_STORIES.md) 故事 25）

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
  新增 `build_demo.py` + `docs/`（index.html 自包含、无 CDN 依赖）。
  目录取名 `docs/` 而非 `demo/`：GitHub Pages 分支部署只支持 `/` 与 `/docs`。
  9 个用例覆盖各链路亮点：简单/伤病/禁忌拦截/工具计算/图谱/问法重述/弱相关/无依据拒答/注入拦截；
  每张卡片可展开**引用来源**与**检索详情（三路融合得分）**——不只给结果，还给中间过程。
  页面顶部明确标注「离线快照，非实时服务」，不误导。
  部署：GitHub Pages / OSS / HF Spaces 任选（`docs/README.md` 有步骤）。

  ⚠️ 过程中又发现一处**文档声明未经实测**：DEMO_SCRIPT 写「量子力学和健身的关系 →
  知识库无依据拒答」，实测该问题相关度 **0.52 > 阈值 0.40**，系统如实作答「无直接关联」。
  **系统行为是对的，错的是那句没测过的文档**——已修正 DEMO_SCRIPT 并把该用例改为
  「弱相关问题」，另补真正会拒答的 `如何挑选股票`。
  （后经实测修正：该用例的 relevance 是 **None**——三路融合后没有任何文档通过阈值，
   不是「相关度 0.00」；两者触发的是不同判据，见故事 23 同类教训）

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

## ✅ 已完成：压测与耗时归因（2026-09-10）

> 目的不是「跑个数字」，是**回答瓶颈在哪一处**——这是要不要引入 Redis 的唯一判据。

### 怎么归因（**未改动任何业务逻辑**）

1. SSE 的 `status` 帧本就标记了阶段边界 → 客户端记到达时刻即得时间线
2. `gateway.log` 每条 `llm_usage` 带 `latency_ms` → 按 `request_id` 求和得 LLM 净耗时
3. 新增 `PipelineService._phase_lock`：把四处 `with self._lock` 换成带埋点的上下文管理器，
   分别记录**等待**（串行化代价）与**持有**（临界区工作量），按 `phase` 打点
   —— 只看总延迟分不出这两者，而这正是本阶段要回答的问题

### 数据（100 条测试集之外的独立压测，simple 层，云端主链）

| 并发 | 墙钟 | 总延迟 p50/p95 | LLM净耗时 | **锁等待** | 吞吐 |
|---|---|---|---|---|---|
| 1 | 4.1s | 4120 / 4120 ms | 1418ms | 0ms | 110 tok/s |
| 2 | 4.3s | 4343 / 4343 ms | 1891ms | 131ms | 192 tok/s |
| 4 | 6.1s | 5639 / 6075 ms | 1442ms | **956ms** | 279 tok/s |
| 8 | 6.6s | 6089 / 6564 ms | 1517ms | **971ms** | 376 tok/s |
| 16 | 9.1s | 8184 / 9084 ms | 1545ms | **2957ms** | 567 tok/s |
| 32 | 11.4s | 10674 / 11293 ms | 1522ms | **5461ms** | 878 tok/s |

### 三条结论

**① 瓶颈不在 LLM。** 净耗时在 1→32 并发全程持平（1418→1522ms）。
供应商侧没有排队、没有触发降级链。**这一条就否掉了「加机器/加配额」方向。**

**② 瓶颈在全局锁，且锁里装的是网络调用。** 按阶段拆 63 次锁获取：

| 阶段 | 等待中位 | 持有中位 | 持有占临界区 |
|---|---|---|---|
| A_analyze | 159.5ms | **0.0ms** | 0.1% |
| **C_retrieve** | 1365.3ms | **185.9ms** | **99%** |
| E_budget（会话/预算状态） | 935.3ms | **0.3ms** | 0.5% |
| F_history（会话写回） | 0.0ms | **1.8ms** | 0.6% |

`A_analyze` 自己只持有 0.0ms 却要等 159ms —— 它排的是 `C_retrieve` 的队。

再往下拆 `C_retrieve` 的 186ms：**云端 query embedding 单次实测中位 158ms**
（`EMBEDDING_PROVIDER=cloud`，qwen3.7-text-embedding）。
**即锁里 ~85% 的时间是一次 HTTP 往返**，不是 Milvus 计算。

**③ 会话/限流状态只占临界区的 1.1%**（163ms / 15042ms）。

### 🎯 Redis 判定：**不接**（按预先承诺的判据，非事后合理化）

判据原文：*瓶颈在会话/限流状态 → 接；在 LLM 配额 / Milvus 锁 / 根本没到顶 → 不接。*

- 会话/限流状态在临界区里占 **1.1%** —— 把它整个搬到 Redis，
  最好的情况也只能改善这 1.1%，而 Redis 自己还要加上网络往返
- 锁等待的真实来源是**云端 embedding 往返**，Redis 对它一点办法都没有
- 吞吐在 32 并发下仍在上升，**没到顶**

### 📌 数据反过来指出的一个便宜优化（未做，待定）

临界区 186ms 里 ~158ms 是 embedding 网络往返。把 query embedding **移到锁外**
（Phase B 与 HyDE 并行算好，向量传进检索）预计能让临界区从 186ms 降到 ~30ms。
- 收益：锁的理论上限从 ≈1/0.239s ≈ **4.2 req/s** 提升到 ≈ 1/0.08s ≈ **12 req/s**
- 代价：小（改动集中在 retriever 的 query 向量传递与 `_last_query_embedding` 的时序约定）
- **注意**：这是「锁外计算、锁内使用」的改动，需要处理那段共享状态注释
  （`retriever.py:99-100` 明确写了「锁外读取方需自行保证时序」）

> 该优化**不在** Q15 划定的范围内（A + Dockerfile），故先记录不执行。

---

## ✅ 已完成：Dockerfile（部署最小集，2026-09-10）

新增 `Dockerfile` + `.dockerignore` + [DEPLOY.md](DEPLOY.md)。**全部实建实跑验证过**，不是写完就交。

| 阶段 | 镜像体积 | 怎么降的 |
|---|---|---|
| 第一版 | 2.97 GB | — |
| 去掉 torch 系 | 1.7 GB | cnocr 移到 pyproject 的 `ocr` extra（声明式，不是列黑名单） |
| 去掉 uv 下载缓存 | 1.38 GB | `uv sync --no-cache`——实测缓存 **919MB** 留在镜像层里 |
| 去掉建索引资料 | 1.38 GB | `pdf_pages/` 125MB + 源 PDF 32MB |

**跑通验证**（真实容器，非 mock）：`/healthz` 5 秒就绪；容器内跑完整流水线问答返回 200 + 3 条引用；
画像注入在容器里同样 403；容器内可达宿主机 Neo4j（195 节点 / 337 关系）；
成本账本**跨容器重建保留**（`docker stop` 后用同一卷重启，329 tokens 没归零）。

**过程中撞出来的三个问题**（都已修，见 [INTERVIEW_STORIES.md](INTERVIEW_STORIES.md) 故事 21）：

1. **密钥被打进镜像**：`.dockerignore` 漏了 `.env`，`COPY . .` 把 1095 字节的
   DASHSCOPE/NEO4J 密钥写进了镜像层，`docker run --rm <img> ls /app/.env` 直接可见。
   → `.env` / `*.pem` / `*.key` 全部排除，配置一律走 `docker run -e`。
2. **排除清单是错的，被冒烟检查抓住**：以为 `pandas` 只有评测用，
   实际 `pymilvus/orm/schema.py` 模块级就 `import pandas`——它是服务链路硬依赖。
   → 加了构建期 `import api` 冒烟检查，这类错误**构建期就失败**。
3. **`--no-install-package` 不排除依赖树**：排了 `cnocr`，`triton`/`wandb`/`ultralytics`
   照样装进来。→ 改用 pyproject 的 `ocr` extra 做声明式排除。

**另外两处配套改动**：
- 5 个运行时状态路径（`COST_GUARD_PATH` / `SESSION_PERSIST_PATH` / `FACT_CACHE_PATH` /
  `GATEWAY_LOG_PATH` / `FEEDBACK_PATH`）支持环境变量覆盖。
  容器里这很关键：**`cost_state.json` 不持久化 = 重启即绕过成本上限**（已验证持久化生效）。
- `pyproject.toml`：`cnocr` 从 `dependencies` 移到 `[project.optional-dependencies] ocr`。
  ⚠️ 本地跑 `uv sync` 会把 cnocr 从 venv 移除，需要 OCR 建索引时用 `uv sync --extra ocr`。

**明确没做**（Q15 划的线）：CI/CD、密钥服务、编排、多副本、灰度回滚。
多副本要注意：会话/限流/成本状态都是进程内的，**多副本会各算各的**（限流与成本上限放大 N 倍）。

---

## 🏁 本轮范围到此结束

> 三个 P0 + 压测 + 按判据的 Redis（结论：不接）+ Dockerfile，**全部完成**。
> demo 部署押后（用户决定）。
> 数据反向指出的一个优化（把 query embedding 移到锁外）**已记录未执行**，见上方压测一节。

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
| **短问法检索质量** | `retriever.py` | **重新生成 demo 时发现**：「腰突怎么康复」这类**短、且不含动作名**的伤病问法，图谱的禁忌关系（腰突→深蹲/硬拉）主导召回，Milvus 又返回深蹲变体 → 模型拿到的「参考知识」全是**禁忌动作** → 按 Prompt 规则保守拒答（77 字），且**引用里列的就是禁忌动作本身**（对用户是误导）。换成「腰突患者适合做什么康复训练」→ 928 字，正常命中臀桥等康复动作。根因未查明（图谱权重过高？康复动作召回不足？），**已从 demo 用例中规避，但缺陷仍在**。 |

---

## 📎 相关文档

| 文档 | 用途 |
|---|---|
| [INTERVIEW_STORIES.md](INTERVIEW_STORIES.md) | 面试故事集（8 个故事 + 讲法 + 追问预案） |
| [DEMO_SCRIPT.md](DEMO_SCRIPT.md) | 6 步演示剧本 |
| [CHANGELOG.md](CHANGELOG.md) | 正式变更记录 |
| `eval_results/latest.json` | 最近一次评测的完整产物（配置 + commit + 逐样本） |
