# 🏋️ 康养知识库智能问答 RAG 系统

> ### 🔗 在线演示：<https://ssnn666.github.io/fitness-rag-coach/>
> **不用 clone、不用配环境、不用 API Key，点开就能看到各条链路的真实输出**——
> 每条回答都能展开「引用来源」和「检索详情（三路融合得分）」，不只给结果，也给中间过程。
> 覆盖：分层生成 / 伤病问答 / 禁忌拦截 / 工具计算 / 图谱检索 / 问法重述 / 弱相关 / 无依据拒答 / **Prompt 注入拦截**。
>
> ⚠️ 那是**离线快照**（`docs/` 由 `build_demo.py` 跑完整管线导出），不是实时服务。

基于检索增强生成（RAG）的康养问答 Demo：**FastAPI 唯一后端 + Streamlit SSE 客户端**，Milvus 向量 + BM25 关键词双路检索（Neo4j 图谱可插拔），统一大模型适配器（云端主链路 + 多供应商降级），HyDE 查询改写、拒答判定、Prompt 注入检测、内容审核、完整日志。

## ✨ 核心特性

- **统一大模型适配器**：主链路阿里云百炼 DashScope（MaaS 部署 qwen3.7-plus，`DASHSCOPE_BASE_URL` 可切公共云/私有化），DeepSeek / 千帆 ERNIE 可选，本地 Ollama 兜底；超时/429 限流（按 Retry-After 退避）/额度不足/上下文超长自动分类 → 重试退避 → 降级链
- **分层生成策略**：按查询复杂度分配模型与预算——简单问题快模型 + 短预算 + 跳过重排（**实测中位 3.2s**），伤病/计划问题主模型 + LLM 重排 + Fact-Check（**实测中位 4.8s**）；深度思考为**用户可选**，开启后走思考模式（**实测 64.8s**，耗时约 13 倍，故默认关闭）；混合思考开关按角色配置（短回答类任务关思考提速 17 倍）
- **双路检索 + 图谱可插拔**：Milvus Lite 语义向量 + BM25 关键词，RRF / 加权融合双模式可切换；Neo4j 伤病禁忌图谱默认停用、config 一键切回三路
- **安全流水线（12 步）**：实体抽取 → 禁忌黑名单 → 边界拒绝 → HyDE 检索 → CRAG 补盲 → **拒答判定** → 分层 Prompt → 令牌预算 → 降级链生成 → 硬性过滤 → Fact-Check → 结构化引用
- **禁忌数据双源降级**：Neo4j 图谱停机时自动切 [contra_data.py](contra_data.py) 本地副本（28 类伤病禁忌，与图谱构建共用单一数据源）——核心安全数据不依赖单一外部服务
- **接口层防护**：X-API-Key 鉴权（演示级）、Prompt 注入检测（规则加权）、百度内容审核（输入/输出双向，fail-open）
- **完整日志**：query / 检索片段 / prompt / 模型输出 / 报错 / token 消耗（JSON Lines，轮转）
- **SSE 真流式**：token 级 delta 实时透出（降级链 stream_events，流中降级标记随文显示）+ 权威全文帧（事实核查/审核后覆盖增量区）+ 引用/降级/token 元信息帧
- **边界加固**：OCR 乱码质检（提分辨率重试→丢弃统计）、知识库无依据拒答、多模态 503 明确降级
- **SSE 断开保护**：客户端断开 → 取消信号贯穿流水线（阶段边界 + 流式循环逐块检查），worker 线程不再空转；delta 事件驱动排空（替代 50ms 轮询）
- **供应商参数隔离**：`enable_thinking`（DashScope 混合思考专属）不泄漏到 DeepSeek/千帆；语义去重开关接线（默认关，省每请求 N 次候选 embedding）；query 向量跨检索/grounding 复用
- **确定性健康工具层**：BMI / 每日饮水量 / 心率区间三工具（注册表结构、关键词触发、参数优先取画像）——数值计算确定性注入上下文，不让 LLM 自行算术（可复现、零 token 成本）；注册表可直接演进为 tools 协议
- **多轮查询改写**：指代消解后检索（「那硬拉呢」自动补上上一轮的「腰突」上下文），小模型改写 + 异常回退原问题；改写仅作用于检索，原问题仍用于 Prompt/分层/缓存 key
- **反馈闭环 + 检索可观测**：👍/👎 反馈 → feedback.jsonl → `eval_testset.py --feedback` 回流评测（真实用户不满意的样本驱动迭代）；`/v1/debug/retrieval` 按 request_id 查检索得分——演示时当场展示三路中间过程

## 🛠️ 技术栈

| 层 | 组件 |
|---|---|
| 云端 LLM | DashScope（MaaS 私有部署，OpenAI 兼容模式）: qwen3.7-plus / qwen3.7-flash / qwen3-vl-plus |
| 可选云 LLM | 百度千帆: ernie-4.5-turbo-128k / ernie-speed-128k（OpenAI 兼容 v2） |
| 本地 LLM | Ollama: qwen2.5:7b（兜底）/ 1.5b（校验）/ 0.5b（HyDE） |
| Embedding | qwen3.7-text-embedding（云端，768d 指定维度）+ nomic-embed-text（本地兜底） |
| 向量库 | Milvus Lite（嵌入式，单进程独占） |
| 关键词 | BM25（rank-bm25 + jieba） |
| 知识图谱 | Neo4j AuraDB（伤病→禁忌/康复，默认停用可插拔） |
| API | FastAPI + Pydantic + SSE；Streamlit（纯 API 客户端） |
| 内容审核 | 百度智能云 text_censor/v2 |
| 评估 | 自研 LLM-judge（Hit@k / MRR / faithfulness / relevancy / precision / recall） |

## 🏗️ 架构

```
┌───────────────┐  POST /v1/chat/stream (SSE, X-API-Key)   ┌────────────────────────────────┐
│  app.py       │ ───────────────────────────────────────▶ │  api.py (FastAPI 唯一后端)      │
│  Streamlit    │ ◀─────────────────────────────────────── │  ├─ require_api_key 鉴权         │
│  SSE 客户端    │   meta / status* / delta* / citations / │  ├─ 注入检测 + 百度输入审核       │
│  (零索引/LLM   │   answer / done / error                  │  ├─ 网关: 限流/降噪/预算/内存降级  │
│   依赖)       │                                          │  └─ /healthz /v1/chat /v1/vision │
└───────────────┘                                          └───────────────┬────────────────┘
                                                                          │ 单进程内（Milvus Lite 约束）
                                                      ┌───────────────────▼───────────────────┐
                                                      │ pipeline.py PipelineService（12 步）   │
                                                      │  实体→禁忌(Neo4j可选)→边界拒绝→        │
                                                      │  HyDE双路检索→CRAG→拒答→分层Prompt→    │
                                                      │  预算守卫→生成→硬过滤→FactCheck→引用    │
                                                      └───┬───────────────┬───────────────────┘
                                            ┌─────────────▼──────┐  ┌─────▼──────────────────────┐
                                            │ llm_adapter.py      │  │ retriever (Milvus+BM25      │
                                            │ FallbackChain 降级链 │  │   +[Neo4j] weighted/RRF)    │
                                            │  dashscope→qianfan?  │  └────────────────────────────┘
                                            │  →ollama            │
                                            └─────────────────────┘
```

**完整请求链路**：X-API-Key → Prompt 注入检测 → 百度输入审核（流式前）→ 网关限流/降噪 → 实体抽取 → Neo4j 禁忌名单（停用时安全跳过）→ 边界拒绝检查 → 多轮改写（指代消解，仅检索）→ HyDE + 双路检索（加权/RRF 融合 + 实体 boost + LLM 重排）→ 知识盲区 CRAG 联网 → grounding 拒答判定（生成前）→ 分层 Prompt + 禁忌注入 + 工具计算结果（BMI/饮水量/心率，确定性计算）→ 令牌预算级联截断 → 降级链生成 → 硬性禁忌过滤 → Fact-Check 四类校验（缓存 + CRAG 修正）→ 百度输出审核（展示前）→ 结构化引用 → SSE 分块输出。

## 📦 安装

```bash
uv venv && uv pip install -r requirements.txt
ollama pull qwen2.5:7b qwen2.5:1.5b qwen2.5:0.5b nomic-embed-text
cp .env.example .env   # 填入密钥（.env 不入库）
```

## 🚀 运行

**一键启动（推荐）**：`python start.py`（或双击 `start.bat`）——自动拉起 API + UI、等健康检查就绪、打开浏览器；Ctrl+C 全部停止。

```bash
python start.py                  # 一键启动全部 + 自动开浏览器
python start.py --no-browser     # 不开浏览器
python start.py --skip-api       # 仅 UI（知识图谱页签可用）
```

手动分步启动（调试用）：

```bash
# 1. 先启动 API（独占 Milvus Lite；禁用 --reload，reloader 会 fork 子进程导致文件锁冲突）
uvicorn api:app --host 127.0.0.1 --port 8000

# 2. 再启动 UI（纯 SSE 客户端）
streamlit run app.py

# 索引构建（Milvus Lite 单进程独占：重建索引/跑评测前需停 API 服务）
python build_index.py --fast            # CSV + TEXT_KB_SOURCES 文本直抽 → Milvus + BM25
python build_index.py --fast --source-dir <页面图片目录>   # 回退图片 OCR 路径（扫描件场景）
```

curl 冒烟：

```bash
curl http://127.0.0.1:8000/healthz
curl -N -X POST http://127.0.0.1:8000/v1/chat/stream \
  -H "X-API-Key: <你的API_KEY_AUTH>" -H "Content-Type: application/json" \
  -d '{"question":"深蹲主要锻炼哪些肌群"}'
# 反馈 + 检索 debugger（request_id 取自 meta 帧）
curl -X POST http://127.0.0.1:8000/v1/feedback \
  -H "X-API-Key: <你的API_KEY_AUTH>" -H "Content-Type: application/json" \
  -d '{"request_id":"<request_id>","vote":"down","comment":"太笼统"}'
curl "http://127.0.0.1:8000/v1/debug/retrieval?request_id=<request_id>" -H "X-API-Key: <你的API_KEY_AUTH>"
```

## 🔑 环境变量（.env）

| 变量 | 用途 | 缺省行为 |
|---|---|---|
| DASHSCOPE_API_KEY | 云端主链路（含 /v1/vision） | 跳过 DashScope，直接本地 Ollama |
| QIANFAN_API_KEY | 可选第二云供应商（ERNIE） | 降级链自动跳过 |
| BAIDU_AK / BAIDU_SK | 百度内容审核（text_censor/v2） | NullCensor 直通放行 + 日志标注 |
| API_KEY_AUTH | 接口鉴权 Key（自定字符串） | 空 = 本地免鉴权 |
| NEO4J_PASSWORD | Neo4j AuraDB（NEO4J_ENABLED=True 时） | — |
| BOCHA_API_KEY | CRAG 联网搜索 | 联网补盲降级为无外部资料 |

## ⚡ 分层生成策略（按复杂度分配生成时间）

| 层级 | 判定 | 模型 | 预算 | 重排 | 思考模式 | 实测耗时 |
|---|---|---|---|---|---|---|
| simple | 无伤病/体态/计划关键词 | qwen3.7-flash | 400 token + "200字内"提示 | 跳过 | 关 | **3.2s**（原 42s） |
| injury | 含伤病/体态实体 | qwen3.7-plus | 1024 token | ✓ | **默认关**（答后提示可开深度思考） | **4.8s**（含 Fact-Check） |
| plan | 含计划类关键词 | qwen3.7-plus | 2048 token | ✓ | **默认关**（答后提示可开深度思考） | 按计划长度 |

- **深度思考模式**：侧边栏开关 / API `deep_thinking` 字段——injury/plan 层默认快速（plus 关思考），开启后切思考模式（**实测 4.8s → 64.8s，约 13 倍**，推理更深入）；simple 层不受影响
  > 上表耗时为本机一次实测（云端主链，随网络与模型负载波动）；此外**命中事实缓存时会显著更快**（伤病类同一问题二次提问实测 4.5s）
- 快速模式回答后附加 💡 提示引导开启深度思考（仅 injury/plan 层）

- `LLM_TIERS` 在 config 集中配置；重排/校验角色（rerank/fact_check/judge/hyde）全部关闭思考模式（实测 8.4s→0.5s）
- 分层判定先于检索：simple 层跳过 LLM 重排，一次查询省一次模型调用
- **simple 层跳过 HyDE 直查**（`HYDE_SKIP_SIMPLE`）：基础问答直查 Hit@3=1.00 已满分，实测 20 条对比直查 0.95/1.00/1.00 vs HyDE 1.00/1.00/1.00（仅 1 条首命中降级），省 1 次 LLM 调用 + 1 次重复检索
- **并行化**：复合伤病三路召回草稿（HyDE/Step-Back/子问题）并行生成；管线两阶段——检索/共享状态锁内串行（Milvus Lite 约束），云端 LLM 生成锁外并行（实测 3 并发请求 1.3-1.6× 加速；封顶因素为 MaaS 单 Key 并发额度）
- **Embedding 上云**：qwen3.7-text-embedding（dimensions=768 与 schema 一致，实测批量 41 条/s、8 并发 0.26s 无排队）——检索与本地资源解耦。⚠️ 切换 `EMBEDDING_PROVIDER` 必须重建索引（不同模型向量空间不兼容），并重新校准 `GROUNDING_MIN_SIM`（qwen 嵌入实测校准 0.40）
- 快模型选型实测：qwen3.7-flash 优于 deepseek-v4-flash / glm-5.2-fast-preview（短任务 0.3s vs 0.9s）

## 📊 压测基准（bench_stream.py）

```bash
python bench_stream.py                    # 默认 2/4/8 并发（各档独立 session 绕过网关降噪）
python bench_stream.py --concurrency 1,2,4 --question "腰突怎么康复"   # 指定档位/题型
```

实测（2026-08-19，DashScope 主链 qwen3.7-flash，simple 层快速模式，问题「深蹲主要锻炼哪些肌群」）：

| 并发 | 墙钟 | 首 token P50 | 总延迟 P50 / P99 | 吞吐 |
|---|---|---|---|---|
| 1 | 2.7s | 1.5s | 2.7s / 2.7s | 189 token/s |
| 2 | 5.3s | 2.6s | 5.3s / 5.3s | 189 token/s |
| 4 | 5.3s | 2.5s | 4.9s / 5.3s | 368 token/s |

> 解读：吞吐 1→4 并发约 2×（非 4×）——**封顶因素为 MaaS 单 Key 并发额度**
> （检索锁内串行是 Milvus Lite 约束，云端生成锁外并行不受影响）；首 token 1.5s 为
> 「网关 + 检索 + grounding 判定」的固定开销，分层策略的 simple 快模型已把生成期压到 ~1s。

## 🔻 降级链触发条件

| 错误分类 | 判定 | 处理 |
|---|---|---|
| TIMEOUT / 429 限流 / 5xx / 网络 | 异常类型 + 状态码 + 关键词 | 退避重试（默认 2 次）→ 下一级供应商 |
| CONTEXT_OVERFLOW（上下文超长） | 错误消息关键词 | 截断历史（system+最后 user）重试一次 → 降级 |
| QUOTA（欠费/额度不足） | Arrearage / 403 | **不盲目重试**，直接降级 |
| AUTH（Key 无效） | 401 / InvalidApiKey | 直接降级 + ERROR 日志 |
| 全部供应商失败 | — | 返回「服务暂时不可用」+ error_kind 标注 |

链序：`dashscope → qianfan（配了 Key 才启用）→ ollama`。SSE done 帧 `fallback_active` + UI 横幅提示降级。

## 🧪 测试

```bash
pytest -q tests/                    # 143 项：适配器降级链/注入/拒答/乱码/API 集成/引用清洗/拒绝原因/网关防护/图谱/断连取消/供应商参数隔离/分类器回归/健康工具/多轮改写/禁忌复查/反馈/检索 debugger（mock 打桩）
python -u eval_graph.py             # 图谱检索专项评测（mock 确定性；--neo4j 切真实实例）
python -u eval_testset.py --skip-groups --limit 100   # 检索+生成评测（本地 Ollama）
python -u eval_testset.py --rrf --limit 20            # RRF 融合模式对比
python -u eval_testset.py --cloud --limit 20          # 云端评测（需 DASHSCOPE_API_KEY）
```

## 📊 评估（测试集重建后全量口径）

> 2026-08-14：测试集重建（`rebuild_testset.py`，100 条全部锚定当前知识库内容，废弃旧系统元问题，
> 旧版备份为 test_100_full_legacy.csv）+ `SIM_THRESHOLD` 随 Embedding 切换重校准 0.75→0.60。
>
> **2026-09-10 复测**：修复图谱元数据缺陷（`CSVLoader` 未传 `metadata_columns`）+ 清理知识库
> （移除 10 行混入的架构笔记）+ 启用真实 Neo4j 实例后重跑全量。
> **结果可复现、已落盘**：[eval_results/latest.json](eval_results/latest.json)（含运行配置 / git commit / 逐样本命中明细）。

| 指标 | 08-14 | **09-10** | 备注 |
|---|---|---|---|
| Hit@1 | 0.51 | **0.79** | 测试集锚定 KB 后的检索命中率 |
| Hit@3 | 0.62 | **0.91** | |
| Hit@5 | 0.65 | **0.95** | |
| MRR | 0.566 | **0.851** | |
| faithfulness | 0.39 | **0.46** | LLM-judge 二元判定（严格） |
| answer_relevancy | 0.90 | **0.94** | |
| context_precision | 0.41 | **0.70** | |
| context_recall | 0.46 | **0.62** | |

按题型 Hit@3 / MRR：

| 题型 | n | **Hit@3** | **MRR** | （08-14 Hit@3） |
|---|---|---|---|---|
| 基础问答 | 30 | **1.000** | **1.000** | 1.00 |
| 单伤病问答 | 25 | **1.000** | **1.000** | 0.44 |
| 复合伤病问答 | 15 | **1.000** | 0.889 | **0.00** |
| 动作纠错 | 10 | 0.900 | 0.900 | 1.00 |
| 计划生成 | 15 | 0.667 | 0.494 | 0.60 |
| 定制长计划 | 5 | 0.600 | 0.490 | 0.60 |

> **解读（面试可讲）**：
> ① **复合伤病 0.00 → 1.00 是本轮最大的修正，也是一个教训**。此前文档把 0.00 解释为
> 「Hit@k 是单文档比对、与复合伤病多文档场景不匹配」——**这个解释是错的**。真实原因是图谱建图时
> 读不到 CSV 元数据，`TARGETS_MUSCLE` 等三种关系一条都没建出来（「伤病→动作→肌群→伤病」的
> 间接禁忌链路是断的），且关系标签存在 `r.relation` 属性而查询读的是 `type(r)`，差异化评分全落默认分。
> **一个"听起来专业"的解释会让人停止调查——要区分「工具真的不合适」和「我的东西坏了但我不想查」。**
> ② 单伤病 0.44 → 1.00 同理：图谱路径此前实际未生效。
> ③ 测试集参考文本由 KB 内容锚定生成（贴近原文），检索侧无泄漏（评测输入是 query，参考只定义正确性）；
> ④ faithfulness 0.46：二元判定（任一主张无依据即 0 分）+ eval 简化 answer_chain（top-3 截断上下文），
> 线上管线有完整 Prompt + Fact-Check 兜底，不可直接类比。
> ⑤ 动作纠错 1.00 → 0.90：10 条样本中 1 条首命中位次变化，小幅波动。

> 原有 5 阶段优化流程（三路检索）历史记录见 [eval_results_final.csv](eval_results_final.csv)：
> Hit@3 0.06→0.38（+533%）、MRR 0.05→0.30、context_precision 0.01→0.14、relevancy 0.92。
> 核心 trade-off：放宽候选池→召回↑精度↓ → 实体过滤拉回精度。

### 图谱检索专项评测（eval_graph.py，2026-08-19 新增）

`python -u eval_graph.py`（mock 图谱确定性评测，零网络；`--neo4j` 切真实实例跑同一评测集）。
图谱证据源（contra_data 28 类伤病）与 KB 文档是两套系统，Hit@k 口径不适用——专项口径：

| 指标 | 得分 |
|---|---|
| 单伤病禁忌召回@5 / Top-1 禁忌 | **100% / 100%** |
| 复合伤病双伤病路径覆盖（top-3） | **100%** |
| 校验断言（差异化评分 / 深度衰减 / 禁忌 boost / 矛盾标注） | 3/3 PASS |

> 评测驱动修出 4 个图谱逻辑 bug（Neo4j 停用期不可见）：① `_score_relation` 从描述判分，
> 但关系标签在类型名里 → 双源匹配；② multi-hop 评分 `1/hops` 让 1-hop 康复=1.0 与禁忌无区分
> → 标签基础分×深度衰减；③ 检索结果先截断后排序 → 高分禁忌路径被挤出 top-k；④ 单伤病误判复合
> （"肩袖损伤"拆词计数 / "综合征"含"综合"标记子串 / 部位名是伤病名组成部分）→ 实体词典贪心计数
> + 标记词独立判定。修复后禁忌召回 68.4%→100%、复合覆盖 0%→100%。

> **⚠️ 2026-09-10 补充：mock 通过 ≠ 真实图谱通过。** 上述评测长期只跑 mock 口径。接入真实
> Neo4j 实例后首次跑 `--neo4j` 只有 **1/3 PASS**（Top-1 禁忌 **0%**、多跳路径全空）。
> 原因是 mock 把关系标签放在 `type(rel)` 而真实图谱放在 `r.relation` 属性，且 mock 自带硬编码
> 多跳边——**mock 掩盖了真实图谱的两处缺陷**（建图时 CSV 元数据未读入 + 检索时标签属性未读）。
> 修复后真实实例 **3/3 PASS**、Top-1 禁忌 **0% → 100%**。
> **教训：mock 通过只证明逻辑自洽，不证明数据通路是通的——两者必须定期对齐口径。**

## ⚖️ 数据合规声明（面试话术）

- **知识库来源（已落地）**：`TEXT_KB_SOURCES` 配置两份公开发布的官方健康科普资料，**文本直抽入库（零 OCR）**：
  - 《科学健身18法》——国家体育总局体育科学研究所技术支持（高校官网公开转载件）
  - 《全民健身指南》——国家体育总局 2017 年发布
  - 原始二进制不入库（`.gitignore`），txt 抽取版随仓库可重建；**不使用网络爬虫**抓取网页内容，规避 robots 协议、版权与数据合规风险。
- **版权资料隔离**：`*.pdf` / `*.docx` 与 `pdf_pages/` 已加入 .gitignore，原始版权文件（旧版扫描书占位资料）已移出知识库，永不入库；OCR 图片路径保留为兜底但默认停用。
- **文本直抽 vs OCR 的取舍**（面试可讲）：公开官方资料以文本型 PDF/文档为主，pdfplumber 直抽零字符错误；OCR 仅用于无文本层的扫描件场景，且版面交错/图形标签乱码是扫描件的固有缺陷（实测 2400px 重 OCR 可修字符错误但修不了版面）。
- **为什么不用 VL 做 PDF 文档提取**：多模态模型逐页解析速度慢、成本高、对纯文本精度不如 OCR 专项模型；VL 仅用于体检报告图片问答（/v1/vision）这类真实多模态场景——职责分离、成本可控。
- **爬虫/版权风险认知**（可展开）：数据采集需区分「公开许可 vs 公开可见」；爬取需遵守 robots.txt、频率限制与网站条款；医疗健康内容需注明来源与时效，AI 输出不构成诊疗建议。

## 📄 文档摄入：统一归一化层（doc_loaders.py）

多格式文档 → **统一 Markdown 中间表示**。格式差异不再泄漏到切块/嵌入/建图里
——加一种格式只需 `@register(".xxx")`，不动索引构建代码。

| 扩展名 | 解析 | 说明 |
|---|---|---|
| `.txt` / `.md` | 直读 | 已归一，原样 |
| `.csv` | csv 模块 | 每行一条；**短列**进 metadata（长文本列留在正文，避免撑爆 `metadata_json` 的 1024 上限） |
| `.pdf` | pdfplumber | 文本 + 表格抽取 + **跨页表格合并** |
| `.docx` | python-docx | 段落层级（标题转 `#`）+ 表格 |
| `.xlsx` | openpyxl | 每个 sheet → 一个 Markdown 表格 |
| 图片 | cnocr（可选） | 未安装时**明确报错**，不静默跳过 |

**两个实测踩出来的坑**：

1. **表格误判**：`pdfplumber.extract_tables()` 会把**竖排文本段落**识别成「单列表格」。
   不加校验直接采用，原本正常的正文会被重排成无意义的表格——**反而破坏已有的抽取质量**。
   现加结构校验：≥2 行、≥2 列、非空占比 ≥30%、且 ≥2 行有实质内容。
   （实测：某扫描件 PDF 修复前抽出 10 个「表格」，全是竖排文本误判；修复后 0 个。）
2. **跨页表格**：表格被分页截断时会拆成两张表。现按「上页末表 + 本页首表 + 列数相同
   + 表尾贴近页底/表头贴近页顶」合并，并去掉续表的重复表头。无坐标信息时**保守不合并**
   （宁可拆开也不要拼错表）。

**切块侧的配合**：表格在切块时**独立成块**，不与正文混排——切分器的分隔符含 `\n`，
而 Markdown 表格按行分隔，混排会被逐行打散成碎片。独立成块后常规表格整张落在同一父块；
即便超大表仍需切分，也会切在**行边界**上（行本身完整）。

### 重复摄入与重复嵌入（ingest_cache.py）

**「同一份文件重复上传怎么办？」** —— 按 **SHA256** 判重，而不是按文件名或 mtime
（mtime 会被 checkout / 复制 / 恢复备份改掉，内容却没变；同一内容换个名字也该判为重复）。

- **源文件指纹清单** `ingest_manifest.json`：`路径 → {sha256, size, indexed_at}`。
  重建前比对，输出「新增 / 变更 / 未变 / 移除」。**全部未变 → 整轮跳过**（不必白跑一遍）。
- **嵌入缓存** `embed_cache.json`：以**内容哈希**为键缓存向量。文档改一处，其余块直接复用向量。

实测（271 个 chunk）：

| 场景 | 结果 |
|---|---|
| 源文件全部未变 | **整轮跳过**（「SHA256 一致，跳过重建」） |
| 改 1 个文件后重建 | 检测到 1 个变更 → **嵌入缓存命中 271 / 未命中 1（99.6%）** |
| 还原后重建 | 命中 **100%**，重建**零嵌入 API 调用** |

`--force` 可忽略两个缓存强制全量重建。

## 🔌 MCP Server（把能力交给其他 Agent）

`mcp_server.py` 把项目从「一个问答应用」变成「**其他 Agent 可消费的能力**」：
任意 MCP 客户端（Claude Code / Claude Desktop 等）都能直接调用这里的检索、
禁忌判定、图谱多跳与确定性健康计算。

| 工具 | 依赖 | 说明 |
|---|---|---|
| `search_knowledge_base` | Milvus + BM25 + Neo4j | 三路检索，返回命中片段、来源与得分 |
| `check_contraindication` | contra_data（+Neo4j 可选） | 某动作对某伤病是否禁忌，**含具体原因** |
| `get_injury_graph` | Neo4j | 伤病关联动作的多跳路径 |
| `calculate_bmi` / `estimate_water_intake` / `heart_rate_zone` | 无 | 确定性健康计算 |

```bash
python mcp_server.py --list       # 列出已注册工具
python mcp_server.py --selftest   # 本地跑一遍各工具，验证可用性
python mcp_server.py              # stdio 模式（供 MCP 客户端接入）
python mcp_server.py --transport sse
```

客户端配置：

```json
{
  "mcpServers": {
    "fitness-rag": {
      "command": "<项目路径>/.venv/Scripts/python.exe",
      "args": ["<项目路径>/mcp_server.py"]
    }
  }
}
```

实测（真实 MCP 协议握手，非 mock）：协议版本 `2025-11-25`，暴露 6 个工具；
`check_contraindication('腰突','硬拉')` → 正确返回「禁忌动作」及原因；
`get_injury_graph('腰突', 2)` → 13 条路径，含 `腰突 → 硬拉 → 竖脊肌` 这类
**文本知识库看不出的间接关联**。

> ⚠️ **Milvus Lite 单进程独占**：API 服务（`api.py`）在跑时，MCP 的检索类工具
> 无法打开向量库；禁忌判定与健康计算不受影响（不碰 Milvus）。

## 🗂️ 结构

```
├── CHANGELOG.md           # 变更记录（2026-08-13 康养 Demo 改造日全记录）
├── ROADMAP.md             # 当前状态与后续计划（防上下文丢失）
├── INTERVIEW_STORIES.md   # 面试故事集（11 个故事 + 讲法 + 追问预案）
├── CONTEXT_MANAGEMENT.md  # 上下文管理设计（预算/级联截断/头部保留）
├── mcp_server.py          # MCP Server：把检索/禁忌判定/图谱/计算暴露给其他 Agent
├── session_store.py       # 会话记忆持久化（JSON 原子写，跨重启恢复）
├── doc_loaders.py         # 统一文档归一化层（pdf/docx/xlsx/csv/图片 → Markdown）
├── start.py / start.bat   # 一键启动（API+UI、健康检查、自动开浏览器、Ctrl+C 全停）
├── kb_18fa.txt            # 《科学健身18法》文本直抽（体科所，合规公开）
├── kb_zhinan.txt          # 《全民健身指南》文本直抽（国家体育总局，合规公开）
├── api.py                 # FastAPI 唯一后端（SSE/鉴权/审核/vision）
├── graph_view.py          # 伤病禁忌图谱可视化（ECharts 力导向图，双数据源）
├── static/                # echarts.min.js（本地内置）+ graph.html（运行时生成，gitignore）
├── .streamlit/config.toml # 开启静态服务（图谱页依赖 /app/static/）
├── pipeline.py            # PipelineService 12 步安全流水线
├── llm_adapter.py         # 统一大模型适配器 + 多供应商降级链
├── guardrails.py          # Prompt 注入检测（规则加权）
├── content_moderation.py  # 百度内容审核（fail-open）
├── grounding.py           # 知识库依据判定（无依据拒答）
├── text_quality.py        # OCR 乱码质检（摄入层）
├── retriever.py           # 双路检索 + weighted/RRF 融合 + [Neo4j 可插拔]
├── reranker.py            # LLM listwise 重排序
├── hyde.py                # HyDE + Step-Back + 问题分解 + 多轮改写
├── health_tools.py        # 确定性健康工具层（BMI/饮水量/心率区间，注册表结构）
├── fact_checker.py        # 四类事实校验
├── fact_cache.py          # 校验结果缓存（LRU）
├── gateway.py             # 网关（限流/降噪/预算/内存）+ 结构化日志
├── log_reader.py          # gateway.log 检索事件读取（/v1/debug/retrieval 数据源）
├── crag_search.py         # CRAG 博查联网搜索
├── app.py                 # Streamlit 客户端（零索引依赖；智能问答 + 图谱 + 图文解读三视图）
├── build_index.py         # 索引构建（OCR 质检 + Milvus + BM25 + [Neo4j]）
├── ingest_pdf.py          # 单 PDF 摄入脚本
├── pdf_ocr.py             # cnocr 并行 OCR + 缓存
├── eval_testset.py        # 评测（适配器统一，--cloud/--rrf/--feedback）
├── bench_stream.py        # 并发压测（P50/P95/P99 首 token/总延迟 + token 吞吐）
├── config.py              # 全局配置（密钥走 .env）
├── tests/                 # pytest 77 项
└── eval_results_final.csv # 历史优化记录
```
