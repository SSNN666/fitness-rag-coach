# 🐳 部署（Docker）

把 `api.py` 跑成容器。**范围**：只做「让服务能被部署」的最小集——
镜像构建 + 运行参数 + 依赖说明。不含 CI/CD、密钥服务、编排、多副本。

---

## 1. 构建

```bash
docker build -t fitness-api .
```

国内网络直连 Docker Hub / PyPI 常失败，换镜像源即可（不写死在 Dockerfile 里）：

```bash
docker build -t fitness-api \
  --build-arg BASE_IMAGE=docker.m.daocloud.io/library/python:3.13-slim .
```

> 构建过程包含一次 `import api` 冒烟检查：Dockerfile 里排除了建索引/评测才用的
> 重依赖（torch 系），万一排除清单写错，**构建期就会失败**，不会留到运行时。

## 2. 准备索引（首次或换了 embedding 供应商时）

向量索引是**数据**不是代码，没烘进镜像（详见下方「为什么索引要挂载」）。

```bash
# 容器内建（需要能访问 embedding 供应商）
docker run --rm -v "$PWD/data:/app/data" \
  -e DASHSCOPE_API_KEY=sk-xxx \
  fitness-api uv run --no-sync python build_index.py
```

已有本地索引就直接挂载，跳过这步。

## 3. 运行

```bash
docker run -d --name fitness-api -p 8000:8000 \
  -v "$PWD/milvus.db:/app/milvus.db" \
  -v "$PWD/bm25_index.pkl:/app/bm25_index.pkl" \
  -v fitness-state:/data \
  -e DASHSCOPE_API_KEY=sk-xxx \
  -e API_KEY_AUTH=your-strong-key \
  -e NEO4J_URI=bolt://host.docker.internal:7687 \
  -e NEO4J_USER=neo4j -e NEO4J_PASSWORD=xxx \
  -e COST_GUARD_PATH=/data/cost_state.json \
  -e SESSION_PERSIST_PATH=/data/sessions.json \
  -e FACT_CACHE_PATH=/data/fact_cache.json \
  -e GATEWAY_LOG_PATH=/data/gateway.log \
  --add-host=host.docker.internal:host-gateway \
  fitness-api
```

> `milvus.db` 是**目录**不是文件（Milvus Lite 的本地库：`collections/` `databases/` `LOCK`），
> 照文件挂载会得到 `Permission denied`。

验证：

```bash
curl -s localhost:8000/healthz | python -m json.tool
```

`/healthz` 返回 `cost` 账本（用量与档位）与 `neo4j_enabled`，可据此确认外部依赖。

> **为什么要把四个状态路径指到挂载卷**：它们默认写在容器工作目录 `/app`，
> 容器一重建就全没了。其中 `cost_state.json` 尤其关键——
> **它一丢，成本上限就归零，重启即成为绕过限额的捷径**（P0-3 的落盘设计就是为此）。
> 这四个路径都支持环境变量覆盖，就是为容器场景准备的。

---

## 必须知道的四个约束

### ① 单 worker，不能加

Milvus Lite 是**嵌入式**向量库（进程内文件，不是独立服务），**单进程独占**。
`--workers 2` 会让两个进程争抢 `milvus.db`。

要真正水平扩展，得把 Milvus 换成 standalone server——但**当前没有数据支持做这件事**：
压测显示全局锁的临界区耗时里 ~85% 是**云端 embedding 的网络往返**，
Milvus 本身只占很小一部分（见 [ROADMAP.md](ROADMAP.md) 压测一节）。

### ② Neo4j 是外部依赖，连不上会降级

容器里的 `localhost` 不是宿主机。用 `host.docker.internal`（Linux 需加
`--add-host=host.docker.internal:host-gateway`），或把两个容器放进同一 network。

连不上时服务**不会崩**：禁忌数据自动回退到 `contra_data.py` 本地副本
（安全防线不断链）。但已知遗留问题：连接超时是 8 秒，无 Neo4j 时每个请求都会吃这个延迟。

### ③ 成本上限要按实际预算设

镜像里带的是占位默认值（软 5M / 硬 20M token）。部署前按自己的预算改：

```bash
-e COST_SOFT_LIMIT_TOKENS=... -e COST_HARD_LIMIT_TOKENS=...
```

换算：`预算 token = 预算金额 ÷ 单价(元/千token) × 1000`。
硬阈值触发后服务会拒绝新请求（返回 429 / SSE error 帧），**拒绝发生在花钱之前**。

### ④ 运行时状态是「进程内 + 落盘」，不是共享的

会话记忆、限流窗口、降噪窗口、成本账本都在进程内（部分落盘到文件）。
**单副本没问题；多副本会各算各的**——限流和成本上限都会被放大成 N 倍。

> 这正是当初考虑引入 Redis 的场景。**压测后按预先写定的判据决定不引入**：
> 会话/限流状态只占全局锁临界区的 1.1%，瓶颈不在这里。
> 详细数据与判据见 [ROADMAP.md](ROADMAP.md)。

---

## 为什么索引要挂载而不是烘进镜像

| | 烘进镜像 | 挂载（当前做法） |
|---|---|---|
| 镜像可移植性 | 绑死在建索引时的 embedding 供应商上 | 同一镜像可指向不同索引 |
| 换供应商 | 必须重新 build 镜像 | 换挂载文件即可 |
| 镜像体积 | +索引体积 | 不变 |

`milvus.db` / `bm25_index.pkl` 在 `.gitignore` 里，本来就不入库——它们是产物。

---

## 镜像里装了什么、没装什么

`pyproject.toml` 的依赖同时服务三种角色，只有第一种是**在线服务**需要的：

| 依赖 | 用途 | 镜像里 | 怎么排除的 |
|---|---|---|---|
| langchain / pymilvus / neo4j / fastapi / … | 在线服务 | ✅ | — |
| **cnocr → torch / torchvision / pytorch-lightning / ultralytics / wandb** | **建索引**时的扫描件 OCR | ❌ **数 GB** | 声明式：pyproject 的 `ocr` extra |
| ragas | 离线评测 | ❌ | `uv sync --no-install-package` |
| streamlit | 前端（独立进程，见 `app.py`） | ❌ | 同上 |
| **pandas** | — | ✅ **必须装** | 见下 |

**pandas 不能排**：它看起来只被 `eval_testset.py` 用，但
`pymilvus/orm/schema.py` 在模块级就 `import pandas as pd`——是服务链路的硬依赖。
（这个错误是被 Dockerfile 里的 `import api` 冒烟检查抓出来的，第一版构建直接失败。）

**OCR 那坨为什么不写在 Dockerfile 里**：它原本在 `dependencies` 里。用
`--no-install-package cnocr` 只能排除 cnocr 自己，**它的依赖树照样装**
（实测 `triton`/`wandb`/`ultralytics` 全都进来了）。所以改成声明式的
`[project.optional-dependencies] ocr`：

```bash
uv sync              # 只跑服务 / 跑测试（不含 OCR）
uv sync --extra ocr  # 要建索引（扫描件路径）
```

⚠️ 这个改动意味着：**本地跑 `uv sync` 会把 cnocr 从你的 venv 里移除**。
需要 OCR 建索引时加 `--extra ocr`。当前 `TEXT_KB_SOURCES` 走文本路径，
默认建索引流程不碰 OCR。

## 镜像体积（实测，每一步都是量出来的）

| 阶段 | 体积 |
|---|---|
| 第一版（cnocr 在基础依赖里 + uv 缓存留在层里） | 2.97 GB |
| 去掉 torch 系（移到 `ocr` extra） | 1.7 GB |
| 再去掉 uv 下载缓存（`uv sync --no-cache`，实测缓存 **919MB**） | 1.38 GB |
| 再去掉建索引资料与缓存（`pdf_pages/` 125MB + PDF 32MB） | **1.38 GB** |

## ⚠️ 密钥绝不能进镜像

`.dockerignore` 里的 `.env` 那一条是**实测撞出来的**：第一版漏了它，
`COPY . .` 把 `.env`（1095 字节，含 DASHSCOPE/NEO4J 密钥）打进了镜像层，
`docker run --rm <img> ls /app/.env` 直接可见——**镜像给谁，密钥就给谁**。
现在 `.env` / `*.pem` / `*.key` 全部排除，配置一律走 `docker run -e`。
