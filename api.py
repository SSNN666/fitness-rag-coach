"""
api.py —— FastAPI 唯一后端（康养 Demo）
========================================
路由：
  GET  /healthz             健康检查（免鉴权）
  POST /v1/chat             非流式问答
  POST /v1/chat/stream      SSE 流式（默认缓冲模式：完整生成 → 输出审核 → 分块吐出）
  POST /v1/vision           图文问答（P2：体检报告图片 → qwen3-vl-plus）

请求链路（SSE）：X-API-Key 鉴权 → 注入检测 → 百度输入审核（流式前）
             → 网关限流/降噪 → 12 步流水线 → 百度输出审核（展示前）
             → 分块 delta → citations → done

启动（Milvus Lite 单进程约束，禁用 --reload）：
  uvicorn api:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from config import (
    API_KEY_AUTH, LLM_PROVIDER_PRIMARY, MAX_QUERY_CHARS, NEO4J_ENABLED, SSE_CHUNK_CHARS,
)
from content_moderation import build_censor
from guardrails import detect_injection
from llm_adapter import build_llm
from pipeline import build_pipeline, PipelineService

_log = logging.getLogger("api")


# ============================================================
# Pydantic 模型
# ============================================================

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUERY_CHARS,
                          description="用户问题（1-1200 字符）")
    session_id: str = Field(default="default", min_length=1, max_length=64)
    user_profile: str | None = Field(default=None, max_length=500)
    deep_thinking: bool = Field(default=False,
                                description="深度思考模式（伤病/计划层用 plus+思考，更深入但更慢）")


class CitationOut(BaseModel):
    kind: str = "kb"          # "kb" | "graph" | "web"
    source: str = ""
    page: int | None = None
    snippet: str = ""
    url: str | None = None


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    grounded: bool = True
    refusal: bool = False
    citations: list[CitationOut] = []
    fallback_active: bool = False
    error: str | None = None


class VisionResponse(BaseModel):
    request_id: str
    answer: str
    provider: str | None = None
    available: bool = True


# ============================================================
# 鉴权（演示级：X-API-Key + 常数时间比较；API_KEY_AUTH 为空 = 本地免鉴权）
# ============================================================

def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if API_KEY_AUTH and not secrets.compare_digest(x_api_key or "", API_KEY_AUTH):
        raise HTTPException(status_code=401, detail="invalid api key")


# ============================================================
# 生命周期：单进程加载全部索引/模型依赖
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    _log.info("lifespan: building pipeline (Milvus + BM25 + adapters)...")
    app.state.pipeline = build_pipeline()
    app.state.censor = build_censor()
    app.state.gw_state = {}   # 网关限流/降噪的会话状态（替代 st.session_state）
    app.state.vision_llm = build_llm("vision")  # P2：本地无 VL → active_providers 为空
    _log.info("lifespan: ready. chat chain=%s vision chain=%s neo4j=%s",
              app.state.pipeline._llms["chat"].active_providers,
              app.state.vision_llm.active_providers, NEO4J_ENABLED)
    yield
    try:
        app.state.pipeline._retriever._milvus.close()
    except Exception:
        pass


app = FastAPI(title="康养知识库智能问答 API", version="1.0.0", lifespan=lifespan)


# ============================================================
# 防护辅助
# ============================================================

def _guard_input(question: str, censor) -> tuple[bool, str, dict]:
    """注入检测 + 百度输入审核（同步、流式前）。返回 (blocked, message, detail)。"""
    verdict = detect_injection(question)
    if verdict.blocked:
        _log.warning("injection_blocked score=%d hits=%s", verdict.score, verdict.hits)
        return True, "检测到疑似 Prompt 注入，请求已拦截。", {"kind": "injection", "score": verdict.score}

    if censor is not None:
        cr = censor.check_text(question, task="RAG_QA_INPUT")
        if not cr.passed:
            return True, "输入内容未通过安全审核，请修改后重试。", \
                {"kind": "censor", "conclusion": cr.conclusion, "types": cr.blocked_types}
    return False, "", {}


def _usage_to_dict(u) -> dict:
    return {
        "provider": u.provider, "model": u.model,
        "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
        "total_tokens": u.total_tokens, "latency_ms": u.latency_ms,
    }


# ============================================================
# 路由
# ============================================================

@app.get("/", include_in_schema=False)
def root():
    """浏览器直访 API 端口时跳转交互文档（消除 GET / 404 噪音）。"""
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """浏览器自动请求 favicon 时返回 204（消除日志 404）。"""
    return Response(status_code=204)


@app.get("/healthz")
def healthz():
    pipeline: PipelineService = app.state.pipeline
    return {
        "status": "ok",
        "provider": LLM_PROVIDER_PRIMARY,
        "chat_chain": pipeline._llms["chat"].active_providers,
        "memory_degraded": pipeline._gateway.is_degraded,
        "neo4j_enabled": NEO4J_ENABLED,
        "fusion_mode": pipeline._retriever._fusion_mode,
    }


@app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
def chat(req: ChatRequest):
    """非流式问答。防护链路与 SSE 端点对齐：注入检测 → 审核 → 网关限流/降噪。"""
    request_id = uuid.uuid4().hex[:12]
    blocked, msg, detail = _guard_input(req.question, app.state.censor)
    if blocked:
        raise HTTPException(status_code=403, detail={**detail, "message": msg})

    # 网关限流 / 降噪（与 SSE 端点对称；模式 tag 与降噪器一致）
    gateway = app.state.pipeline._gateway
    allowed, reason = gateway.check_rate_limit(req.session_id, app.state.gw_state)
    if not allowed:
        raise HTTPException(status_code=429, detail={"kind": "rate_limit", "message": reason})
    if gateway.is_duplicate(req.question, req.session_id, app.state.gw_state,
                            tag="deep" if req.deep_thinking else "fast"):
        raise HTTPException(status_code=409, detail={
            "kind": "duplicate",
            "message": "检测到与最近请求高度相似的问题，为避免重复调用模型已跳过。"
                       "如需新回答，请换一种问法或补充更多细节。"})

    result = app.state.pipeline.answer(
        req.question, req.session_id, req.user_profile, request_id,
        deep_thinking=req.deep_thinking)

    censor = app.state.censor
    answer = result.answer
    if censor is not None and not result.refusal:
        cr = censor.check_text(answer, task="RAG_QA_OUTPUT")
        if not cr.passed:
            answer = "该回答未通过内容审核，已屏蔽。请换一种问法。"

    return ChatResponse(
        request_id=result.request_id, answer=answer,
        grounded=result.grounded, refusal=result.refusal,
        citations=[CitationOut(**c) for c in result.citations],
        fallback_active=result.fallback_active, error=result.error,
    )


@app.post("/v1/chat/stream", dependencies=[Depends(require_api_key)])
def chat_stream(req: ChatRequest):
    """SSE 流式（缓冲模式：审核先于展示）。事件帧：data: {"event": ..., ...}"""
    request_id = uuid.uuid4().hex[:12]
    return StreamingResponse(_sse_gen(req, request_id), media_type="text/event-stream")


@app.post("/v1/vision", response_model=VisionResponse, dependencies=[Depends(require_api_key)])
def vision(file: UploadFile = File(...),
           question: str = Form(..., min_length=1, max_length=MAX_QUERY_CHARS),
           session_id: str = Form(default="default")):
    """P2 图文问答：体检/健康报告图片 → 云端多模态模型解析 → 问答。

    本地无 VL 模型 → 503 明确提示；图片内容走输出审核后再返回。
    """
    request_id = uuid.uuid4().hex[:12]

    if not app.state.vision_llm.active_providers:
        raise HTTPException(
            status_code=503,
            detail="未配置多模态模型（需 DASHSCOPE_API_KEY 以启用 qwen3-vl-plus）。",
        )

    blocked, msg, detail = _guard_input(question, app.state.censor)
    if blocked:
        raise HTTPException(status_code=403, detail={**detail, "message": msg})

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="仅支持图片文件")
    data = file.file.read()
    if len(data) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="图片过大（>8MB）")

    prompt = f"这是用户上传的健康检查/体检报告图片。请根据图片内容回答问题：{question}"
    # MIME 透传（PNG/JPEG 按真实类型发送；未声明时兜底 image/jpeg）
    resp = app.state.vision_llm.invoke_vision(data, prompt, mime=file.content_type or "image/jpeg")

    answer = resp.content
    censor = app.state.censor
    if censor is not None:
        cr = censor.check_text(answer, task="RAG_QA_OUTPUT")
        if not cr.passed:
            answer = "该回答未通过内容审核，已屏蔽。"

    return VisionResponse(
        request_id=request_id,
        answer=answer,
        provider=resp.usage.provider if resp.usage else None,
        available=bool(app.state.vision_llm.active_providers),
    )


def _sse_gen(req: ChatRequest, request_id: str):
    pipeline: PipelineService = app.state.pipeline
    gateway = pipeline._gateway
    censor = app.state.censor

    def evt(event: str, data: dict) -> str:
        payload = {"event": event, **data}
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    # meta 先行：前端立即有反馈
    yield evt("meta", {"request_id": request_id, "provider": LLM_PROVIDER_PRIMARY})

    # 1. 注入检测 + 输入审核（同步、流式前）
    blocked, msg, detail = _guard_input(req.question, censor)
    if blocked:
        yield evt("error", {"message": msg, **detail})
        return

    # 2. 网关限流 / 降噪
    allowed, reason = gateway.check_rate_limit(req.session_id, app.state.gw_state)
    if not allowed:
        yield evt("error", {"message": reason, "kind": "rate_limit"})
        return
    if gateway.is_duplicate(req.question, req.session_id, app.state.gw_state,
                            tag="deep" if req.deep_thinking else "fast"):
        yield evt("error", {"message": "检测到与最近请求高度相似的问题，为避免重复调用模型已跳过。"
                                       "如需新回答，请换一种问法或补充更多细节。",
                            "kind": "duplicate"})
        return

    # 3. 12 步流水线（后台线程执行；status 帧实时播报阶段进度）
    #    缓冲模式下等待期零反馈是「感觉慢」的主因——进度帧不缩短耗时但大幅改善感知
    progress: list[str] = []
    progress_lock = threading.Lock()

    def _on_stage(name: str) -> None:
        with progress_lock:
            progress.append(name)

    result_box: dict = {}

    def _run_pipeline() -> None:
        try:
            result_box["result"] = pipeline.answer(
                req.question, req.session_id, req.user_profile, request_id,
                deep_thinking=req.deep_thinking, on_stage=_on_stage)
        except Exception as e:  # 兜底：后台异常不悬挂 SSE，转 error 帧
            _log.error("pipeline_worker_error rid=%s err=%s", request_id, str(e)[:200])
            result_box["error"] = str(e)[:200]

    worker = threading.Thread(target=_run_pipeline, daemon=True)
    worker.start()
    sent = 0
    while worker.is_alive():
        with progress_lock:
            pending, sent = progress[sent:], len(progress)
        for name in pending:
            yield evt("status", {"stage": name})
        time.sleep(0.25)
    worker.join()
    with progress_lock:
        pending, sent = progress[sent:], len(progress)
    for name in pending:
        yield evt("status", {"stage": name})

    if result_box.get("error") or result_box.get("result") is None:
        yield evt("error", {"message": "服务处理失败，请重试。"})
        return
    result = result_box["result"]

    # 4. 输出审核（展示前——缓冲模式的核心：审核完成后才发 delta）
    censor_note = None
    if censor is not None and not result.refusal:
        cr = censor.check_text(result.answer, task="RAG_QA_OUTPUT")
        if not cr.passed:
            result.answer = "该回答未通过内容审核，已屏蔽。请换一种问法。"
            censor_note = cr.conclusion

    # 5. 分块吐出（视觉流式）
    text = result.answer
    for i in range(0, len(text), SSE_CHUNK_CHARS):
        yield evt("delta", {"text": text[i:i + SSE_CHUNK_CHARS]})

    yield evt("citations", {
        "docs": result.citations, "grounded": result.grounded, "refusal": result.refusal,
    })
    yield evt("done", {
        "usage": [_usage_to_dict(u) for u in result.usage],
        "fallback_active": result.fallback_active,
        "answer_len": len(text),
        "censor": censor_note,
    })
