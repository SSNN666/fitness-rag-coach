# 🏋️ AI 健身教练 — 三路检索 RAG + 知识图谱 + 本地 LLM

基于检索增强生成（RAG）的智能健身与运动康复顾问。**Milvus 向量检索 + BM25 关键词 + Neo4j 知识图谱** 三路融合，搭配 HyDE 假想文档生成、Fact-Check 事实校验、CRAG 联网兜底。完全本地运行。

## ✨ 核心特性

- **三路融合检索**：Milvus Lite 语义向量 + BM25 关键词 + Neo4j 知识图谱多跳推理，加权融合 + LLM 重排
- **伤病安全防线**：Neo4j 禁忌动作图谱查询 → System Prompt 黑名单注入 → 硬性逐行过滤 → Fact-Check 四类校验
- **HyDE 假想文档**：0.5B 小模型生成假设文档增强召回，复合伤病自动 Step-Back + 问题分解
- **智能路由**：根据查询类型（普通/伤病/计划）动态切换检索权重和 System Prompt
- **CRAG 联网搜索**：本地知识库不足时触发博查搜索，获取外部资料
- **网关守护**：速率限制、请求降噪、令牌预算、内存自适应降级（7B→1.5B）
- **完全离线**：Ollama 本地运行，无隐私泄露

## 🛠️ 技术栈

| 层 | 组件 |
|---|---|
| LLM | Ollama: qwen2.5:7b (主生成) / 1.5b (校验) / 0.5b (HyDE) |
| Embedding | nomic-embed-text (768d) |
| 向量库 | Milvus Lite (嵌入式, 无需 Docker) |
| 关键词 | BM25 (rank-bm25 + jieba) |
| 知识图谱 | Neo4j AuraDB (伤病→禁忌/康复关系) |
| UI | Streamlit |
| 评估 | RAGAS (faithfulness/relevancy/precision/recall) |

## 📦 安装

```bash
uv venv && uv pip install -r requirements.txt
ollama pull qwen2.5:7b qwen2.5:1.5b qwen2.5:0.5b nomic-embed-text
```

## 🚀 运行

```bash
python build_index.py --fast --skip-neo4j   # 构建索引
streamlit run app.py                         # http://localhost:8501
```

## 📊 评估

```bash
python -u eval_testset.py --skip-groups --limit 50
```

| 指标 | FAISS基线 → 最终 | 涨幅 |
|---|---|---|
| Hit@3 | 0.06 → **0.38** | +533% |
| MRR | 0.05 → **0.30** | +468% |
| context_precision | 0.01 → **0.14** | +1300% |
| context_recall | 0.04 → **0.06** | +50% |
| answer_relevancy | 0.91 → **0.92** | 优质稳定 |

> 完整优化流程见 [eval_results_final.csv](eval_results_final.csv)

## 🗂️ 结构

```
├── app.py              # Streamlit 主应用 (12步安全流水线)
├── config.py           # 全局配置 (不入库)
├── retriever.py        # 三路检索 + 加权融合 + 实体过滤
├── reranker.py         # LLM listwise 重排序
├── hyde.py             # HyDE + Step-Back + 问题分解
├── fact_checker.py     # 四类事实校验
├── gateway.py          # 网关 (限流/降噪/预算/内存)
├── crag_search.py      # CRAG 联网搜索
├── build_index.py      # 索引构建 (Milvus+BM25+Neo4j)
├── eval_testset.py     # 百条测试集评估
├── fitness_data.csv    # 59条动作 + 12条知识补盲
├── test_100_full.csv   # 100条标注测试集
└── eval_results_final.csv  # 最终评估 + 优化流程
```
