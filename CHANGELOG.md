# 变更记录

## 🛡️ 2026-09-11 图谱熔断与超时防护

**前置**：ROADMAP「已知遗留问题」记有「图谱不可用时的连接超时」，原文写
`connection_timeout=8`。动手前量基线，发现**比记录严重一个数量级**：

| 层面 | 实测 |
|---|---|
| TCP 连接被拒（OS 地板） | 2.04s |
| `execute_query`（**热路径现状**） | **34.53s** |
| `verify_connectivity` | 4.08s |

34.5s 不是连接超时，是驱动默认 `max_transaction_retry_time=30s` 的重试退避吃满
（1.0 → 2.4 → 3.7 → 7.0s）。**而原文那个 8s 是 `graph_view.py` 的参数——
热路径 `pipeline.py` 建 driver 时没传任何超时参数**。热路径一次伤病问句含
2~3 次图谱调用 → 最坏 ~103s 纯等待，且每个请求都付。

| # | 项 | 文件 | 说明 |
|---|---|---|---|
| 1 | 图谱客户端防护层 | graph_client.py（新）/ config.py | `GraphClient` 与 driver **同签名**（drop-in）：① 显式超时（`connection_timeout=3.0` + `max_transaction_retry_time=0`，不重试因为调用方已有完整本地降级）② 熔断（连续失败 3 次 → 打开 → 冷却 30s → 半开**只放一个探测**）。短路抛 `GraphUnavailable`（普通 `Exception`）→ **调用方的 `except Exception` 降级分支一行未改** |
| 2 | 接线 + 复用 | pipeline.py / mcp_server.py / graph_view.py | ⚠️ mcp_server 与 graph_view **原本每次调用新建 driver**——熔断状态活在客户端里，按调用新建等于每次冷启动、失败累加不到阈值。已改模块级复用。`mcp_server.get_injury_graph` 顺带从 `session + s.run` 统一到 `execute_query`（四处图谱调用口径一致）。建索引/评测保持裸驱动：一次性工具要「响亮地失败」 |
| 3 | 熔断状态可观测 | api.py / retriever.py | `/healthz` 增加 `graph` 字段（state / 连续失败 / 冷却剩余）。熔断是**安静地降级**，不暴露就没人知道它一直开着 |
| 4 | 测试 | tests/test_graph_client.py（新）/ test_api.py / test_retriever 无关 | +13 项：熔断状态机 / 冷却与半开 / **半开单探测（并发下不惊群）** / drop-in 签名保真 / **熔断时安全链路仍产出完整禁忌黑名单**。**293/293 通过**（原 280） |

**验证**（真实 Neo4j 容器，非 mock）：

- **图谱可用时行为不变**（先验这条，它是前提）：同一 Cypher 在裸 driver 与
  GraphClient 上结果**逐字一致**；`graph_view` 图谱模式 vs 本地副本节点集/边集**完全一致**；
  `get_injury_graph` 改写前后同为 13 条路径
- **失败路径**：冷启动 34.53s/次 → 第 1~3 次各 4.08s、之后 **0.0000s**；
  热中断 首次 0.001s → 4.07 / 4.11s → **0.0000s**；
  **端到端一次复合伤病问句的图谱段 103.6s → 第 1 个请求 12.24s、第 2 个起 0.00s**
- **恢复**：真实 30s 冷却 + 真实容器重启 → 半开探测 0.958s 成功 → closed

⚠️ **判据未达标，是判据订错了**：我写「单次失败 ≤ 2.5s」（依据端口层单地址 2.0s），
实测 **4.08s**——驱动对 `localhost` 会试 `::1` 与 `127.0.0.1` **两个地址**，各吃满 2.0s。
订正判据为「≤ 2 × OS 单地址代价（≈4.2s）」后达标。另：`.env` 用
`bolt://127.0.0.1:7687` 可省掉 IPv6 那一跳，实测 **2.055s**（未改用户配置，仅记录）。

---

## 🔒 2026-08-19 安全修复 + 工程收尾批次

**前置**：产品闭环批次自检发现——多轮改写把伤病上下文带进了检索 query，但禁忌检查仍基于原问题实体，
「那硬拉呢」改写后安全防线（禁忌黑名单/边界拒绝/硬过滤）全部失效 → 修复；另收尾弃用依赖与性能弹药。

| # | 项 | 文件 | 说明 |
|---|---|---|---|
| 1 | 多轮改写禁忌复查 | pipeline.py | 改写后（Phase B）合并实体 → Phase C 锁内补查禁忌 + **重跑边界拒绝**：原问题「那硬拉呢」改写为「腰突患者可以做硬拉吗」后，合并出「腰突」→ 硬拉（明确禁忌）直接拒绝；非禁忌追问（「那臀桥呢」）不拒绝但禁忌黑名单注入生成提示。抽出 `_resolve_contraindications` 供 Phase A/C 共用，两处口径一致 |
| 2 | 去弃用依赖 | pipeline.py / gateway.py | langchain_community ChatMessageHistory（官方已停维）→ 自带 `_SessionHistory`（10 行，接口兼容）；测试唯一 DeprecationWarning 消除 |
| 3 | 压测脚本 | bench_stream.py（新）/ README.md | 并发档独立 session（绕过网关降噪）；P50/P95/P99 首 token/总延迟 + token 吞吐；实测：1→4 并发吞吐 189→368 token/s（约 2×，封顶为 MaaS 单 Key 并发额度），首 token 1.5s 固定开销 |
| 4 | 测试 | tests/test_pipeline.py | +2 项（改写禁忌拦截 / 改写合并禁忌注入），改 1 项（原「那硬拉呢」检索用例改为非禁忌「那臀桥呢」——修复后拒绝是正确行为）；**143/143** |

---

## 🎯 2026-08-19 产品闭环批次（工具层 / 多轮改写 / 反馈闭环 / 检索 debugger）

**前置**：与 GitHub 同类项目（rag-health-assistant / medical-rag-assistant 等）对比发现四个真实差距——
① llm_adapter 已实现 tools 协议但 pipeline 零使用；② 会话历史只进 messages，检索无指代消解；
③ 测试集一次性人工构建，无用户反馈回路；④ gateway.log 有全量检索数据但无对外接口。全部补齐。

| # | 项 | 文件 | 说明 |
|---|---|---|---|
| 1 | 确定性健康工具层 | health_tools.py（新）/ pipeline.py / config.py | BMI/饮水量/心率区间三工具：注册表结构（可演进 tools 协议）、关键词触发、参数优先取画像、缺参数不硬算；结果注入上下文顶部（令牌预算截断前，头部不被误伤）+ kind=tool 引用；**缓存 key 联动**：工具触发时追加画像指纹——同问题不同画像不复用旧缓存（原 key 不含 user_profile 的既有隐患） |
| 2 | 多轮查询改写 | hyde.py / pipeline.py | `needs_rewrite`（指代词/无实体判定）+ `rewrite_query`（hyde 小模型 64 token，异常/空/超长回退原问题）；改写后 retrieval_q 仅用于检索（HyDE 草稿与 search 同用），原问题保留在 Prompt/分层/FactCache key；检索日志记录实际 query（可审计） |
| 3 | 反馈闭环 | api.py / app.py / eval_testset.py | `POST /v1/feedback`：recent_answers（200 条内按 request_id 解析问题/回答）→ feedback.jsonl；Streamlit 👍/👎 按钮（历史+当前 run 双渲染，key 防冲突）；`eval_testset.py --feedback`：负反馈问题回流评测（无金标准 → 跳过 Hit@k 口径，跑逐条检索诊断 + faithfulness/relevancy） |
| 4 | 检索 debugger | log_reader.py（新）/ api.py / app.py | `GET /v1/debug/retrieval?request_id=` 与 `/recent?limit=`（require_api_key，limit clamp 1-50）；解析 gateway.log JSON Lines（坏行/轮转容错）；UI「🔍 检索详情」展开器按 request_id 展示每路来源/得分 |
| 5 | 测试 | tests/ | 新增 40 项：工具计算/触发/参数抽取、改写判定与三路回退、工具注入 ctx、改写检索透传（开关回归）、反馈落盘/404/鉴权、debug 端点/clamp —— 延续 mock 打桩风格，零外部服务 |

**验证**：pytest **141/141** 通过（原 101 + 新增 40），零回归；CI（mock 打桩）无需外部服务。

---

## ⚡ 2026-08-19 检索优化批次（simple 直查提速 + 图谱专项评测）

**前置**：上批工程加固后，按 P1 优化方向落地两项——simple 层跳过 HyDE（省 1 次 LLM + 1 次检索）、
图谱 multi-hop 专项评测（mock 确定性评测驱动修复 4 个图谱逻辑 bug）。

| # | 项 | 文件 | 说明 |
|---|---|---|---|
| 1 | simple 层跳过 HyDE | config.py / pipeline.py | `HYDE_SKIP_SIMPLE=True`：基础问答直查 Hit@3=1.00 已满分；实测 20 条对比直查 0.95/1.00/1.00 vs HyDE 1.00/1.00/1.00（仅 1 条首命中降级）；**省 1 次 LLM 调用 + 1 次重复检索**（原 simple 层同 query 同 k 搜两次） |
| 2 | 图谱专项评测 | eval_graph.py | mock 图谱（contra_data 自动构造）确定性评测：单伤病禁忌召回/复合双伤病覆盖/评分校验/矛盾标注；`--neo4j` 切真实实例 |
| 3 | 关系评分双源匹配 | retriever.py | `_score_relation` 曾从 description 判分，但标签在关系类型名里（Neo4j 停用期不可见）→ `_relation_label` 双源提取 |
| 4 | multi-hop 评分公式 | retriever.py | `1/hops` 让 1-hop 康复=1.0 与禁忌无区分 → 标签基础分（禁忌1.0/康复0.9/谨慎0.7）× 深度衰减 + 禁忌 boost |
| 5 | 先排序再截断 | retriever.py | one_hop/multi_hop 曾先 `[:k]` 截断后排序 → 高分禁忌路径被挤出 top-k |
| 6 | 复合判定修复 | retriever.py / hyde.py | ①部位名是伤病名组成部分不算复合（"膝关节积液"）；②实体词典贪心计数（"肩袖损伤"整体 1 个）；③标记词独立判定（"综合征"含"综合"子串不触发） |
| 7 | 分类器回归测试 | tests/test_hyde.py | 单伤病误判复合的 5 类回归用例 + 独立标记判定 |

**评测结果**：单伤病禁忌召回@5 68.4%→**100%**、Top-1 禁忌 100%、复合双伤病覆盖 0%→**100%**、
校验 3/3 PASS；pytest **101/101**。

---

## 🛠️ 2026-08-18 工程加固批次（代码分析驱动的修复 + 优化）

**前置**：全量代码走读发现 3 个中等缺陷（SSE 断连线程泄漏 / enable_thinking 字段泄漏 / 图谱冲突检测死代码）+
 4 个低危项（注释不符 / usage 双记 / banner 死代码 / 冗余 embedding）→ 全部修复并补测试。

| # | 项 | 文件 | 说明 |
|---|---|---|---|
| 1 | enable_thinking 参数隔离 | llm_adapter.py | DashScope 混合思考专属字段不再下发 DeepSeek/千帆（OpenAI 兼容端点可能 400 拒绝 → 链路形同虚设） |
| 2 | 图谱冲突检测修复 | retriever.py | 关系标签写入 doc metadata + 标签归一比较（禁忌/康复/谨慎），「矛盾标注」提示首次可触发 |
| 3 | 语义去重开关接线 | retriever.py | FUSION_SEMANTIC_DEDUP_ENABLED 此前从未被读取 → 融合无条件跑候选两两余弦 + N 次 embedding；默认关 |
| 4 | SSE 断开保护 | api.py / pipeline.py | GeneratorExit → stop_event → 流水线阶段边界/流式循环逐块检查退出，worker 不再空转（最长 300s LLM 调用） |
| 5 | SSE 事件驱动排空 | api.py | 50ms 固定轮询 → `queue.get(timeout)` 阻塞等待，delta 一到即透出、无感知延迟 |
| 6 | query 向量复用 | retriever.py / pipeline.py | `_last_query_embedding` 锁内缓存，grounding 相关性判定复用检索向量，每请求省 1 次 embedding |
| 7 | usage 双记修复 | pipeline.py | chat_nothink/chat_fast 去掉 background on_usage，主链调用只按 request_id 记一次 |
| 8 | 内存降级横幅接线 | api.py / app.py | meta 帧透出 `banner` + UI st.warning 展示（此前 get_degraded_banner 无调用方） |
| 9 | 卫生 | .gitignore | `屏幕截图*.png` 不入库 |
| 10 | 测试 | tests/ | 新增 11 项（参数隔离/冲突检测/去重开关/向量复用/断连取消/stop_event 透传），**92/92 通过** |

**修复过程中额外发现**：① `FUSION_SEMANTIC_DEDUP_ENABLED=False` 配置形同虚设（死开关，真实开销比预想大）；
② pipeline 取消路径返回值需保留部分答案；③ `from config import *` 使测试 monkeypatch 需作用于 pipeline 模块本身。

---

## 🏁 2026-08-14 今日总览（14 项工作，详见下方各节）

**上午 → 深夜全链路**：项目分析 → 图谱可视化 → 知识库换血 → 安全细化 → 一键启动 → 两批工程改进 → 评测体系重建。

| # | 工作 | 核心产出 | 关键数据 |
|---|---|---|---|
| 1 | 项目全量分析 | 改进清单（P0-P3，17 项） | 全部落地 |
| 2 | Neo4j 可用性诊断 | 确认 AuraDB 实例已删除（DNS NXDOMAIN），保留本地降级路线 | 代码路径完好 |
| 3 | 知识图谱可视化页签 | graph_view.py + ECharts 力导向图（离线内置） | 27 伤病+30 动作+64 边，浏览器实测渲染 |
| 4 | 引用片段清洗 | `_clean_snippet`（空白归一/残渣剔除/句子边界截断） | web/kb/图谱三类引用接入 |
| 5 | 知识库换血 | 合规文本直抽路径（TEXT_KB_SOURCES）+ 两份官方资料 | 282 chunks，引用零乱码 |
| 6 | 拒绝链路细化 | 边界拒绝用 contra_data 具体原因 + 两类拒绝 UI 区分 | 「膝盖屈伸负重挤压半月板」 |
| 7 | 一键启动 | start.py + start.bat（双击可用） | 健康检查/端口预检/Ctrl+C 全停 |
| 8 | 线程模型重构 | 五阶段 A-F（锁内只剩 Milvus/共享状态） | **3 并发 19.5s vs 串行 56.9s（2.9×）** |
| 9 | 安全与工程批次 | CRAG 复检/否定句保留/日志脱敏/端点防护对称/会话 LRU/vision MIME/器械后缀 | pytest **77/77** |
| 10 | eval 体系修复 | SIM_THRESHOLD 重校准 0.75→0.60 | 敏感度曲线实测校准 |
| 11 | 测试集重建 + 全量评测 | rebuild_testset.py（可复现）+ 100 条 KB 锚定样本 | **Hit@3 0.62 / MRR 0.566 / relevancy 0.90** |
| 12 | 运行日志治理 + 图谱解释入口 | st.iframe 迁移 / gRPC keepalive 治理 / API 根路由 / 聊天页一键跳图谱聚焦 | pytest 80/80 |
| 13 | 响应速度优化 | SSE 阶段进度帧 / injury 600字约束 / 长度守卫重试提速 | 生成 13s 瓶颈实测定位，pytest 81/81 |
| 14 | Vision 图文问答 UI 入口 | 第三视图：上传报告图片 → /v1/vision 多模态解析 | 多模态能力可演示，503 降级提示 |

**今日新增文件**：graph_view.py、start.py、start.bat、rebuild_testset.py、tests/test_graph_view.py、
kb_18fa.txt、kb_zhinan.txt、static/、.streamlit/config.toml

**运行实测修出的隐蔽 bug（不跑起来发现不了）**：st.components.html 不存在（→ v1.iframe 静态页方案）、
srcdoc iframe 内脚本不加载、静态服务默认关闭、bat GBK 乱码/系统 Python 无依赖/✓ 编码崩溃。

**遗留待办**：① ~~git 提交~~（已完成：`0ebcee1` 后端架构 / `2f8b11a` UI+图谱+KB / `b210f75` 评测+文档，共 45 文件 5900+ 行）
② 轮换 git 历史泄露的博查 Key（外部动作，需去博查控制台；可选 git filter-repo 清历史）
③ eval --rrf 全量对比（可选）④ push 远程（本地分支 feat/kangyang-rag-api 未推送）。

---

## 2026-08-14 · 运行日志治理 + 图谱解释入口（用户反馈驱动）

| 项 | 文件 | 说明 |
|---|---|---|
| st.iframe 迁移 | app.py / graph_view.py | `st.components.v1.iframe` 弃用（2026-06-01 移除）→ 新版 `st.iframe`；新版签名无 `scrolling` 参数（上线实测 TypeError 后移除） |
| gRPC keepalive 治理 | config.py / pipeline.py / build_index.py / eval_testset.py / ingest_pdf.py | pymilvus 3.x 默认 `grpc.keepalive_time_ms=10000`，Milvus Lite 内嵌服务端 ping 限频更严 → 每约 2 分钟 GOAWAY "too_many_pings" 断连重连（E 级日志噪音，连接自愈）。新增 `MILVUS_GRPC_OPTIONS`（INT_MAX = gRPC 禁用 keepalive 语义，本地嵌入式连接无需探活），应用于全部 5 处 MilvusClient 创建点；临时库冒烟测试验证参数透传 |
| API 根路由 | api.py | 浏览器直访 :8000 的 `GET /` 与 `/favicon.ico` 404 噪音 → `/` 302 跳 /docs，favicon 204 |
| 图谱解释入口 | app.py / graph_view.py | 伤病相关回答下方出现「🕸️ 查看伤病关系图谱」按钮（禁忌拒绝/硬过滤时措辞为「禁忌关系」）→ 跨视图跳转 + 自动聚焦涉及伤病（别名归一，腰突→腰间盘突出）。初版仅拒绝/硬过滤触发，用户实测「腰突」咨询型问答无入口 → 扩为全部伤病命中触发。st.tabs 无程序化选中 API → 导航改 segmented_control + session_state 状态驱动；回答先写历史再渲染按钮（防 st.rerun() 中断丢消息）。**用户实测「没跳转」→ AppTest 最小复现定位两处根因并修复**：① 按钮在 st.chat_input 块内，点击触发的 rerun 中块被跳过、按钮不实例化 → 点击事件丢失（按钮迁至历史渲染循环，key 按消息序号唯一，focus 元信息随消息入历史）；② 点击送达后直写 widget key（nav）抛 StreamlitAPIException（widget 实例化后禁写其 session_state）→ 跳转改经 st.query_params 中转（刷新不丢） |
| 单测 | tests/test_graph_view.py | +3 项（match_focus 别名归一/无命中空列表/multiselect 值合法性），pytest **80/80**；Streamlit AppTest 双视图渲染零异常 |

> 日志中 `/.well-known/appspecific/com.chrome.devtools.json` 404 为 Chrome DevTools 自动探测请求，非应用问题。
> keepalive 修复需重启 API 进程生效（运行中进程仍持旧通道配置）。

---

## 2026-08-14 · 响应速度优化（用户反馈驱动）

gateway.log 实测定位：injury 查询生成阶段 13s（qwen3.7-plus 长回答 1100+ 字，MaaS ~55-75 tok/s）占全程 75%+；长度守卫升级路径曾实测 37s（思考模式）。

| 项 | 文件 | 说明 |
|---|---|---|
| SSE 阶段进度帧 | pipeline.py / api.py / app.py | 缓冲模式等待期零反馈是「感觉慢」的主因。流水线新增 `on_stage` 回调（分析→检索→校验依据→生成→安全校验 5 帧），api.py 后台线程执行 + 250ms 轮询播报 `status` 帧，UI 显示 🔄 阶段文案。不缩短总耗时但感知延迟大幅下降；后台异常兜底转 error 帧（不悬挂 SSE） |
| injury 层长度约束 | config.py | hint 加「600字以内」约束（生成 token 是主瓶颈）；深度思考模式无此限制 |
| 长度守卫重试提速 | pipeline.py | 快速模式退化输出（<150字）的升级重生成从思考模式（实测 37s）改 `chat_nothink`（≈13s）；再次退化由 💡 深度思考兜底 |
| 单测 | tests/ | +1 on_stage 播报单测；test_api 事件序更新（meta→status→delta→citations→done）；pytest **81/81** |
| 深度思考截断修复 | pipeline.py | 用户实测「腰突」深度思考：49.2s 且答案截断在「腰」字。gateway.log 定位：completion 2855 = 思考 2048（thinking_budget 耗尽硬停）+ 答案 807 token；原提示词「篇幅可以更长」诱导思考超预算。收紧深度思考提示词（结论先行 + 全文≤1200字 + 避免过度展开），思考消耗留安全余量；thinking_budget 保持 2048 防失控。**事故复盘**：用户随后看到的完整回答并非新提示词生效（API 未重启，日志无新主生成调用），而是截断回答 → Fact-Check FAIL → CRAG 联网修正路径重新生成的完整版本——防御体系分层补位的真实案例；修正路径 usage 补记日志（chat_crag_fix），此前该调用在日志中不可见。**修复后实测（骨盆前倾）**：快速 10.8s（completion 552，旧 13.0s）、深度 38.0s 完整收尾（completion 2189，旧 49.2s 截断）；深度回答以「核心结论」开头，新提示词标记（600字/1200字/核心结论）均已在 prompt 中确认 |
| 缓存版本化 | config.py / pipeline.py | 实测第二次事故：重启后重问「腰突」仍返回旧长回答。gateway.log 定位：4 个请求全部命中 FactCache 快速路径（检索+answer、无 prompt/生成调用）——旧提示词答案已缓存且跨重启持久化，新提示词无机会执行。修复：缓存 key 增加 `pv:{FACT_CACHE_VERSION}` 维度，改提示词后 bump 版本即自动失效（旧条目随 FIFO 自然淘汰）；已删除旧 fact_cache.json |

---

## 2026-08-14 · Vision 图文问答 UI 入口（面试演示闭环）

/v1/vision 接口已有但 UI 无入口——多模态能力面试无法演示。补第三视图：

| 项 | 文件 | 说明 |
|---|---|---|
| 图文问答视图 | app.py | 导航加「🖼️ 图文问答」：上传体检报告图（PNG/JPG ≤8MB 客户端预检）→ 问题输入 → 调 /v1/vision（multipart，120s 超时）→ 展示图片 + 回答 + 模型信息；结果存 session_state 跨 rerun 保留 |
| 降级提示 | app.py | 后端 503（本地无 VL 模型）时展示 API 返回的明确提示；403/连接失败等错误路径分别处理 |
| 文档 | README.md / CHANGELOG.md | 结构说明更新；今日第 14 项 |

> 面试演示动线：上传体检报告图片 → 问「这份报告有哪些异常指标？」→ 10-30 秒出结构化解读；
> 再讲降级：未配 DASHSCOPE_API_KEY → 503 明确提示（本地无 VL 模型的诚实降级）。

---

## 2026-08-14 · 测试集重建 + 全量评测（用户反馈驱动）

| 项 | 文件 | 说明 |
|---|---|---|
| 测试集重建 | rebuild_testset.py（新增）/ test_100_full.csv | 旧测试集含 14 条旧系统元问题（GraphRAG/MCP/蒸馏——描述已不存在的架构）+ 86 条旧扫描书 KB 黄金参考；重建为 100 条锚定当前 KB（CSV 动作库/18法/指南/禁忌数据）的健身域问题，分布：基础30/单伤病25/复合15/计划15/纠错10/定制5；旧版备份 test_100_full_legacy.csv |
| 全量评测 | README.md | `--cloud` 100 条：Hit@3 **0.62** / MRR **0.566** / relevancy 0.90；基础问答与动作纠错 Hit@3=1.00；复合伤病 0.00 为「单文档 Hit 指标 vs 多文档证据场景」的固有口径限制，该题型有效口径为 context_recall（0.46） |

> 生成方式：rebuild_testset.py 用项目自身 flash 模型按 KB 内容锚点逐条生成（参考文本贴近原文可检索命中，
> 黄金答案为完整回答），去重 + 长度校验 + 抽查质检；生成脚本入库，测试集可复现重建。

## 2026-08-14 · 换库后评测重跑 + eval 阈值重校准

| 项 | 文件 | 说明 |
|---|---|---|
| SIM_THRESHOLD 重校准 | eval_testset.py | 0.75（nomic 时代）→ 0.60（qwen 嵌入 doc↔ref 分离度实测）。敏感度曲线：th=0.60 → Hit@3=0.40/MRR=0.352（与旧库持平）；th=0.75 → Hit@3=0.00（全量误判）。切换 Embedding 必须重校准的第二处落实（第一处 GROUNDING_MIN_SIM 0.40） |
| 换库后指标 | README.md | weighted：Hit@3 0.40 / MRR 0.343 / precision 0.45（+200%）/ recall 0.35（+600%）/ faithfulness 0.30；rrf：MRR 0.339 与 weighted 差距收窄——「RRF 稀释排名」旧结论是旧库内容分布产物 |

> 前 20 条样本全为基础问答题型；RAGAS 为 LLM-judge 20 条小样本，波动 ±0.1 属噪声。

## 2026-08-14 · 改进批次二：端点防护对称 + 会话治理 + 打磨（用户反馈驱动）

| 项 | 文件 | 说明 |
|---|---|---|
| /v1/chat 防护对称 | api.py | 非流式端点补齐网关限流（429）+ 降噪（409，模式 tag 与 SSE 一致）——此前只有 SSE 端点有防护 |
| vision MIME 透传 | api.py | PNG 等按真实 content_type 发送（此前一律 image/jpeg）；冗余表达式清理 |
| 会话 LRU 驱逐 | pipeline.py / config.py | 会话数超 SESSION_MAX_COUNT（64）按最久未访问驱逐，防内存只增不减；读取即算访问 |
| 硬过滤器械后缀 | pipeline.py | 2 字动作名后接器械后缀（机/凳/架/垫/器/绳）不算命中——「划船」不再误删「划船机」行 |
| crag 日志规范化 | crag_search.py | 移除绕过 Gateway 封装的 getLogger("gateway") 直写 + 无意义事件名 crag_v4，改用标准模块 logger |
| graph_view 单测 | tests/test_graph_view.py（新增） | 6 项：图数据形态/别名合并/关系色映射/聚焦过滤/HTML 引导 |
| 网关与端点单测 | tests/test_gateway.py / test_api.py | 限流器窗口/冷却/会话隔离 2 项 + /v1/chat 409/429 布线 2 项 + LRU 驱逐 1 项 + 后缀启发式 1 项 |
| eval 时效标注 | README.md | 评估指标标注为旧知识库时期数据，换库后待重跑 |

> 实测：/v1/chat 同问重发 → 409 duplicate；换问法 → 200。限流经单元测试验证
> （实测串行请求每个耗时 ~15s，60s 滑动窗口无法在真实 LLM 延迟下击穿，属预期行为）。
> pytest 77/77。

## 2026-08-14 · 改进批次：线程模型 + 安全闭环 + 卫生（用户反馈驱动）

| 项 | 文件 | 说明 |
|---|---|---|
| **五阶段线程模型** | pipeline.py / hyde.py / retriever.py | 锁内只做共享状态/Milvus 操作（A 实体禁忌 / C 检索 / E 预算组装），LLM 与网络全部锁外并行（B HyDE 草稿 / D 重排+CRAG+grounding / F 生成）。修复原「HyDE 草稿与 LLM 重排锁内串行」的线程模型偏差。**实测 3 并发 19.5s vs 串行 56.9s（2.9×，接近线性）** |
| CRAG 修正复检 | pipeline.py | 校验失败联网重生成的回答重新过硬性禁忌过滤（安全防线不因修正路径短路） |
| 硬过滤否定句保留 | pipeline.py | 「避免深蹲」类安全提醒行不再被误删；未删行时不报「已剔除」警告；+3 单测 |
| 日志脱敏 | gateway.py | log_prompt/log_answer 中用户画像（身高/体重/目标）替换为 [已脱敏]；问题与回答正文保留用于审计 |
| 卫生 | README.md / grounding.py / fact_cache.py | README 测试数 44→62；grounding 默认值死代码 0.70→0.40；fact_cache 注释 LRU→FIFO（实际按插入序淘汰） |
| 单测 | tests/test_pipeline.py | +6 项（硬过滤 3 + 已有 3），pytest 65/65 |

## 2026-08-14 · 一键启动

| 改动 | 文件 | 说明 |
|---|---|---|
| 一键启动器 | start.py（新增）/ start.bat（新增） | `python start.py`（或双击 bat）：API → 健康检查（最长 120s）→ UI → 自动开浏览器；Ctrl+C 全停；端口占用预检（已有健康实例时跳过 API）；`--skip-api` 仅图谱页模式 |
| 运行文档 | README.md | 一键启动置顶，手动分步降级为调试用 |

**双击实测踩坑修复（用户环境验证）**：

| 问题 | 现象 | 修复 |
|---|---|---|
| bat 中文注释 GBK 乱码 | UTF-8 注释被 cmd 按 GBK 读 → 乱码被当命令执行 | start.bat 全 ASCII 化 |
| 系统 Python 无依赖 | 双击用 PATH 里的 Python 3.12（无 fastapi） | bat 优先 `.venv\Scripts\python.exe` |
| GBK 控制台编码崩溃 | `UnicodeEncodeError: 'gbk' codec can't encode '✓'` → start.py 中途退出 | stdout 检测到 GBK 时 `reconfigure(errors="replace")` |
| 双开撞端口 | 已有实例时重复启动报错 | 预检发现健康 API/UI → 跳过并提示「使用现有实例」 |

## 2026-08-14 · 拒绝链路细化（用户反馈驱动）

| 改动 | 文件 | 说明 |
|---|---|---|
| 具体禁忌原因 | pipeline.py | 边界拒绝文案改用 contra_data 的原始原因（「膝盖屈伸负重挤压半月板」），替代通用「过高压力」文案；查不到时回退通用文案 |
| 两类拒绝区分 | app.py | refusal + grounded=True → 「🛡️ 该动作属于伤病禁忌，已自动拒绝」；grounded=False → 「知识库无相关依据」——之前两类拒绝都显示「无依据」误导用户 |
| 单测 | tests/test_pipeline.py | +3 项（原因查找/未知动作/拒绝文案），pytest 62/62 |

## 2026-08-14 · 知识库换血：合规文本直抽（用户反馈驱动）

| 改动 | 文件 | 说明 |
|---|---|---|
| 文本直抽路径 | build_index.py / config.py | 新增 `TEXT_KB_SOURCES`：配置后 build_index 优先文本直抽（零 OCR），跳过 pdf_pages/ 图片 OCR（版权扫描件路径自动停用，保留为兜底） |
| 合规资料入库 | kb_18fa.txt / kb_zhinan.txt（新增） | 《科学健身18法》（国家体育总局体科所技术支持）+《全民健身指南》（国家体育总局 2017 发布）——公开官方科普资料，txt 抽取版入库，二进制 gitignore |
| CSV 引用紧凑化 | pipeline.py | CSVLoader 1.x metadata 只有 {source,row} →「动作名称」分支为死代码；改为从 page_content 解析动作名/肌群/器械，恢复紧凑引用（如「高脚杯深蹲 \| 股四头肌/臀大肌/核心 \| 哑铃」） |
| 无页码文本源引用 | pipeline.py | 文本直抽源（段落级、无页码）补 kb 引用分支 |

> 根治前情：旧知识库为版权书扫描件（《肌肉力量训练彩色图谱》第 88 页等），OCR 版面交错+
> 图形标签乱码（「第飞必聚创储」类）。实测 2400px 重 OCR 修字符错误（房司→肩同、弯古→弯曲）
> 但修不了版面交错 → 结论：换合规文本型资料是唯一根治路径。重建后 282 chunks，
> 引用全部干净；OCR 质检链路（text_quality/提分辨率重试）保留供扫描件场景复用。

## 2026-08-14 · 引用片段清洗（用户反馈驱动）

| 改动 | 文件 | 说明 |
|---|---|---|
| `_clean_snippet` | pipeline.py | 引用片段三层清洗：① OCR 行内换行折叠/多空格合并 ② 尾部孤立数字标点残渣剔除（"单系整健 2," → 去 "2,"）③ 句子边界截断（原 `[:80]` 从词中间硬切的展示问题），窗口无句号时退化为省略号截断 |
| 引用窗口加宽 | pipeline.py | kb/web 片段 80→120 字符，清洗后加宽不增加垃圾可见度 |
| 空片段兜底 | pipeline.py | 清洗后为空的 PDF 片段 → "（该页 OCR 片段质量较差，详见原文档第 N 页）" |
| 单测 | tests/test_pipeline.py | +5 项（残渣/空白/边界/省略号/空值），pytest 59/59 |

> 边界说明：混入正文内部的 OCR 误识别字符（"第飞必聚创储"类合法汉字序列）规则无法识别——
> 实测词典覆盖率（乱码 0.797 vs 正常 0.848）与 OOV 比（0.32 vs 0.28）均无分离度。
> 根治靠替换合规 PDF（占位资料待办）或提分辨率重 OCR，不做伪检测。

## 2026-08-14 · 知识图谱可视化（Streamlit 页签）

| 改动 | 文件 | 说明 |
|---|---|---|
| 图谱可视化页签 | app.py / graph_view.py（新增）/ static/echarts.min.js（新增） | ECharts 力导向图：伤病(红)/动作(蓝)节点，边按关系着色（禁忌红/谨慎黄/康复绿），拖拽缩放、悬停 tooltip、点选高亮邻接、聚焦伤病子图筛选 |
| 数据源双模式 | graph_view.py | 默认读 contra_data.py 本地副本（与禁忌安全流水线共用单一数据源）；NEO4J_ENABLED=True 时直连 Cypher 实时查询，连接失败自动回退本地（fail-open） |
| 别名归一 | graph_view.py | 腰突/腰间盘突出 等别名条目合并为全称节点（28 键 → 27 伤病节点 + 30 动作节点 + 64 关系） |
| 离线可用 | static/echarts.min.js | echarts 5.5.1 本地内置（npmmirror 源），无 CDN 依赖，国内网络无忧 |

**运行实测发现并修复（headless Edge CDP 驱动验证）**：

| 问题 | 修复 |
|---|---|
| 本版本 Streamlit 无 `st.components.html` 别名 | 改用规范 API `st.components.v1.html`（后进一步改为 `components.v1.iframe`） |
| srcdoc iframe 内 `<script>`（内联 1MB / 外链）均不被浏览器加载 | 改静态页方案：每次 rerun 生成 `static/graph.html` → `components.v1.iframe` 加载，标准浏览器行为 |
| Streamlit 静态服务默认关闭（未知路由回退首页） | `.streamlit/config.toml` 显式开启 `enableStaticServing`，默认启动命令即可用 |

> 背景：原 AuraDB 免费实例已被删除（DNS NXDOMAIN，2026-08-14 实测），
> 可视化选择直接基于本地禁忌数据副本，不依赖任何外部图数据库。
> 浏览器实测：27 伤病 + 30 动作 + 64 关系，canvas 正常绘制；`static/graph.html` 运行时生成已入 .gitignore。

## 2026-08-13 · 康养 Demo 改造日（未提交，分支 feat/kangyang-rag-api）

当天从"项目分析"到"云端全链路 + 体验调优"共五个阶段，25 个文件变更，新增模块约 2800 行。pytest 54/54 全绿。

---

### 阶段一：原有代码重构（项目分析发现的问题修复）

| 改动 | 文件 | 说明 |
|---|---|---|
| 密钥迁移 | config.py / crag_search.py / app.py / .env(.example) | Neo4j 密码与博查 Key 从代码迁入 `.env`（git 忽略），删除 3 处硬编码 |
| 网关接线 | app.py | 之前 gateway.py 写好了从未接入——限流/降噪/令牌预算/内存降级全部生效 |
| HyDE 专用小模型 | app.py / eval_testset.py | HyDE 改用 HYDE_MODEL（0.5B），不再占用 7B |
| 管线收敛 | app.py（-131 行） | 删除 build_chain/retrieve_context 死代码；CRAG 统一走 crag_search 模块；检索从两次降为一次（hyde_retrieve 返回 (ctx, docs)） |
| 数据去重 | build_index.py | "坐骨神经痛"重复条目合并 |
| 兼容修复 | pipeline 相关 | langchain_community 新版 ChatMessageHistory 存 dict 的历史兼容 |

### 阶段二：康养 RAG 系统改造（12 步计划全部落地）

**新增模块（8 个）**

| 模块 | 职责 | 关键设计 |
|---|---|---|
| llm_adapter.py | 统一大模型适配器 + 多供应商降级链 | 错误分类表（超时/429/额度/上下文超长/鉴权/5xx/网络）驱动重试与降级；`DashScopeAdapter`/`QianfanAdapter`/`OllamaAdapter` 三实现；`build_llm(role)` 角色工厂 |
| pipeline.py | 12 步安全流水线（自 app.py 抽取） | 两阶段线程模型：检索/状态锁内串行（Milvus Lite 单进程约束），LLM 生成锁外并行 |
| api.py | FastAPI 唯一后端 | `/healthz` `/v1/chat` `/v1/chat/stream`(SSE) `/v1/vision`；X-API-Key 鉴权（compare_digest）；缓冲模式保证审核先于展示 |
| guardrails.py | Prompt 注入检测 | 7 条正则加权，≥4 分拦截、3 分告警放行（确定性、零成本） |
| grounding.py | 无依据拒答（生成前判定） | 实体+语义双信号；qwen 嵌入实测校准阈值 0.40 |
| content_moderation.py | 百度内容审核 | text_censor/v2；QPS 限流退避重试；fail-open |
| text_quality.py | OCR 乱码质检 | 字符/词典/重复率三维判定；提分辨率重试→丢弃统计 |
| contra_data.py | 伤病禁忌数据（单一数据源） | AST 从 build_index 提取 28 类伤病映射；图谱停机时本地降级 |
| tests/（6 文件 54 项） | 全 mock 打桩测试 | 适配器降级链/注入/拒答/乱码/API 集成/网关 tag/本地禁忌降级 |

**改造（10 文件）**：app.py 重写为 SSE 客户端（零索引依赖）；retriever 加 RRF 融合模式 + Neo4j 可插拔（NEO4J_ENABLED=False）；gateway 日志扩展（llm_usage/retrieval/prompt/answer 四类事件）+ 降级模型适配器化；build_index/ingest_pdf 接入 OCR 质检 + 来源去硬编码；eval 统一适配器（--cloud/--rrf 开关）。

### 阶段三：云端全链路接入（用户提供全部 Key）

- **MaaS 私有部署适配**：237 个模型探测 → 无 qwen-plus 公共型号 → 映射 `qwen3.7-plus`（生成）/`qwen3.7-flash`（短任务）/`qwen3-vl-plus-2025-12-19`（视觉）；`DASHSCOPE_BASE_URL` 支持自定义 Host
- **千帆 ERNIE** 第二云供应商就位（降级链：dashscope → qianfan → ollama）
- **百度审核**真实调用验证（免费档 QPS=1，重试吸收）
- **embedding 上云**：qwen3.7-text-embedding（dimensions=768 与 schema 一致）；⚠️ 重建索引（不同模型向量空间不兼容）+ 拒答阈值重新校准（0.70→0.40，qwen 嵌入分离度实测远好于 nomic）

### 阶段四：体验调优（用户反馈驱动）

| 优化 | 效果 |
|---|---|
| 分层生成策略（LLM_TIERS） | simple 层 flash+400 token+跳过重排 **42s→8s**；injury/plan 层按复杂度加预算 |
| 混合思考开关（enable_thinking） | 短任务关思考 **8.4s→0.5s（17×）**；快模型基准实测 flash 优于 deepseek-flash/GLM-fast |
| 并行化 | 复合伤病三路召回草稿并行；管线拆锁云端生成跨请求并行（3 并发 1.3-1.6×） |
| 深度思考用户开关 | 侧边栏/API deep_thinking 字段；默认快速+💡答后引导；深度版专属输出要求（分期机制+依据+权衡） |
| 思考预算上限（thinking_budget=2048） | 修复思考失控（**3 分钟→63s**） |
| 模式感知缓存 | 缓存 key 加模式维度（修复"两次回答一样"）；缓存命中跳过生成 + 模式提示一致 |
| 回答长度守卫 | 快速模式退化输出（实测 12 token）自动升级思考模式重生成 |
| 降噪器模式标签 | 同问题切深度思考重问不再被误拦（💡 引导与防重复的冲突修复） |
| 增肌计划数据补盲 | CSV +1 行（3 天增肌计划示例）+ plan 层检索宽度 6 文档（修复计划类拒答） |

### 当天修复的 Bug（11 个）

1. FallbackChain 重试循环吞掉上下文超长截断重试（max_retries=0 时）
2. langchain 新版历史存 dict 导致 _history_messages/令牌预算崩溃
3. 内存守卫误降级云端主链（云推理不占本地内存）
4. fallback_active 误报（对比对象应为 tier 基础模型而非固定 chat 链）
5. 事实缓存 key 缺模式维度 → 深度答案被快速缓存顶掉
6. 缓存路径漏加 💡 提示
7. 降噪器拦截"开深度思考重问"（💡 引导与防重复冲突）
8. plan 层查询全拒答（KB 缺增肌计划数据 + 检索宽度不足）
9. 伤病问答退化"无法确定"（Neo4j 停用后禁忌数据缺失 → 本地降级数据源）
10. 快速模式关思考退化输出（12 token）→ 长度守卫自动升级
11. 深度思考失控 3 分钟（思考 token 无上限）→ thinking_budget

### 关键实测数据

| 指标 | 值 |
|---|---|
| simple 层端到端 | ~8s（原 42s） |
| injury 层快速/深度 | ~17s / ~63s（预算封顶，原失控 3min+） |
| 快模型基准 | flash 0.3s vs deepseek-flash 0.9s vs GLM-fast 0.9s（校验任务） |
| embedding 云端 | 批量 41 条/s，8 并发 0.26s 无排队 |
| 拒答阈值（qwen 嵌入校准） | 0.40（无关类 max≈0.29-0.38，无实体健身类 0.35+） |
| eval 回归（20 条，Neo4j 停用） | weighted：Hit@3 0.40/MRR 0.328；rrf：recall 0.20 但 MRR 0.183（排名稀释，默认保留 weighted） |
| 测试 | pytest 54/54 |

### 待办（面试前）

- [ ] 换合规公开康养 PDF（`build_index.py --fast --source-dir` + `PDF_SOURCE_NAME`）
- [ ] 轮换泄露过的 Neo4j 密码 / 博查 Key（git 历史中有旧值）
- [ ] 提交：建议拆两个 commit（① 密钥迁移+管线重构 ② 康养 Demo 改造）
