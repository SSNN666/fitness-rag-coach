# 注：这里**刻意不写** `# syntax=docker/dockerfile:1`。
# 该指令会让构建先去 Docker Hub 拉 BuildKit 前端镜像——国内直连必然失败，
# 而本文件没用到任何 BuildKit 专属语法（heredoc / --mount），加了只有坏处。
#
# 康养知识库问答 API —— 部署最小集
# =================================
# 目标：让 `api.py` 这个服务能被部署，且镜像里**不装服务跑不到的依赖**。
#
# 关于「最小」：pyproject 的依赖同时服务三种角色——在线服务、离线建索引、离线评测。
# 最大的那块（扫描件 OCR）已经按声明式的做法移到了 pyproject 的 `ocr` extra
# （cnocr → torch/torchvision/pytorch-lightning/ultralytics/wandb，数 GB），
# 所以这里 `uv sync` **不带 --extra ocr** 就自然不含它们。
#
# 剩下两个纯属「非服务角色」，且不在 api 的 import 链上：
#   ragas     → 仅 eval_testset.py（离线评测）
#   streamlit → 仅 app.py（前端，独立进程）
# 用 --no-install-package 在安装期排除（uv.lock 保持原样，本地 uv sync 与 CI 不受影响）。
#
# ⚠️ pandas **不能排**：它看起来只被 eval_testset.py 用，但 pymilvus/orm/schema.py
#    在模块级就 `import pandas as pd` —— 是服务链路的硬依赖。
#    （这个错误正是被下面的冒烟检查抓出来的，第一版构建直接失败。）
#
# ⚠️ 这类「按当前 import 结构推导」的排除会随上游改动失效，所以下一层带了一次
#    `import api` 冒烟检查：真漏了什么，**构建期就失败**，而不是等收到请求才炸。

ARG BASE_IMAGE=python:3.13-slim
FROM ${BASE_IMAGE}

# 国内网络：Docker Hub / PyPI 官方源常不可直连，用 --build-arg 覆盖，不写死在镜像里
#   docker build --build-arg BASE_IMAGE=docker.m.daocloud.io/library/python:3.13-slim .
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN pip install --no-cache-dir -i ${PIP_INDEX} uv

# ---- 依赖层：只要 pyproject/uv.lock 不变就命中缓存 ----
COPY pyproject.toml uv.lock ./
# --no-cache：uv 默认把下载的 wheel 留在 /root/.cache/uv，实测**919MB**，
# 运行时一次都用不到，却整个留在镜像层里。加这一个 flag 直接省掉近 1GB。
RUN uv sync --frozen --no-dev --no-cache \
        --no-install-package ragas \
        --no-install-package streamlit

# ---- 应用代码 ----
COPY . .

# 冒烟检查必须放在 COPY 之后（此前只拷了依赖清单，还没有 api.py 可导入）。
# 它验证的正是上面那串排除是否成立：真漏了依赖，**构建期就失败**。
RUN uv run --no-sync python -c "import api, pipeline, retriever, gateway, cost_guard, guardrails, doc_loaders, crag_search, health_tools, session_store; print('import smoke ok')"

EXPOSE 8000

# Milvus Lite 是嵌入式向量库（进程内文件，不是独立服务），**单进程独占**：
# 多 worker 会争抢 milvus.db。要水平扩展得先换成独立 Milvus——
# 但当前压测数据不支持这么做（锁里 ~85% 是云端 embedding 往返，Milvus 只占小头，
# 见 ROADMAP.md 压测一节）。所以这里固定 --workers 1。
CMD ["uv", "run", "--no-sync", "uvicorn", "api:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
