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
import queue
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

import log_reader
from config import (
    API_KEY_AUTH, FEEDBACK_MAX_RECENT, FEEDBACK_PATH, LLM_PROVIDER_PRIMARY,
    MAX_QUERY_CHARS, NEO4J_ENABLED, SSE_CHUNK_CHARS,
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


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=64)
    vote: Literal["up", "down"]
    comment: str = Field(default="", max_length=500)


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
    app.state.recent_answers = {}   # 最近应答（request_id → 内容，反馈解析用）
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


def _remember_request(store: dict, request_id: str, question: str, answer: str,
                      session_id: str) -> None:
    """记录最近应答供反馈解析（上限 FEEDBACK_MAX_RECENT，超限驱逐最旧）。

    线程安全说明：dict 单键写 + GIL 原子，demo 规模足够（与网关限流状态同级别）。
    """
    store[request_id] = {
        "request_id": request_id, "question": question, "answer": answer,
        "session_id": session_id, "ts": time.time(),
    }
    if len(store) > FEEDBACK_MAX_RECENT:
        oldest = min(store, key=lambda k: store[k]["ts"])
        store.pop(oldest, None)


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
    _remember_request(app.state.recent_answers, request_id,
                      req.question, result.answer, req.session_id)

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


@app.post("/v1/feedback", dependencies=[Depends(require_api_key)])
def feedback(req: FeedbackRequest):
    """用户反馈（👍/👎）：按 request_id 解析出问题与回答，追加写 feedback.jsonl。

    评测闭环：eval_testset.py --feedback 读取负反馈问题跑质量报告——
    测试集不再是一次性人工构建，真实用户不满意的样本回流到评测。
    """
    entry = app.state.recent_answers.get(req.request_id)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"该 request_id 不存在或已过期（仅保留最近 {FEEDBACK_MAX_RECENT} 条请求）")
    record = {**entry, "vote": req.vote, "comment": req.comment,
              "feedback_ts": datetime.now(timezone.utc).isoformat()}
    try:
        with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"反馈写入失败: {e}")
    return {"ok": True, "request_id": req.request_id}


@app.get("/v1/debug/retrieval", dependencies=[Depends(require_api_key)])
def debug_retrieval(request_id: str):
    """检索详情：按 request_id 返回该请求的检索文档与得分（调试/演示用）。

    演示时当场展示「这个问题三路检索各给了多少分」的中间过程；
    数据来自 gateway.log 结构化日志（pipeline 每请求写一次）。
    """
    ev = log_reader.find_retrieval_event(request_id)
    if ev is None:
        raise HTTPException(status_code=404, detail="未找到该请求的检索记录（可能未启用日志或已轮转）")
    return {"query": ev.get("query", ""), "docs": ev.get("docs", [])}


@app.get("/v1/debug/retrieval/recent", dependencies=[Depends(require_api_key)])
def debug_retrieval_recent(limit: int = 10):
    """最近 N 条检索请求摘要（调试/演示用）。"""
    limit = max(1, min(limit, 50))
    events = log_reader.recent_retrieval_events(limit)
    return {"events": [
        {"request_id": e.get("request_id", ""), "query": e.get("query", ""),
         "n_docs": len(e.get("docs", []))} for e in events]}


def _sse_gen(req: ChatRequest, request_id: str):
    """SSE 真流式:token 级 delta 实时透出(降级链 stream_events),生成后补权威全文。

    事件序:meta → status* → delta*(token 级) → citations → answer → done | error
    - answer 帧为事实核查/硬过滤后的权威全文,前端以此覆盖增量区
    - 输出审核在生成后执行:不通过 → answer 帧替换为屏蔽提示(真流式下的固有取舍,
      增量区已展示的内容由前端按 answer 帧覆盖)
    - 客户端断开(GeneratorExit) → stop_event 通知 worker 线程及时退出,
      避免 LLM 调用继续空转（pipeline 流式循环逐块检查）
    """
    pipeline: PipelineService = app.state.pipeline
    gateway = pipeline._gateway
    censor = app.state.censor

    def evt(event: str, data: dict) -> str:
        payload = {"event": event, **data}
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    # meta 先行：前端立即有反馈（内存降级横幅仅首次降级时非空）
    yield evt("meta", {"request_id": request_id, "provider": LLM_PROVIDER_PRIMARY,
                       "banner": gateway.get_degraded_banner()})

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

    # 3. 流水线（后台线程）:阶段进度 + token 级 delta 经线程安全队列透出
    progress: list[str] = []
    progress_lock = threading.Lock()
    delta_q: queue.Queue = queue.Queue()

    def _on_stage(name: str) -> None:
        with progress_lock:
            progress.append(name)

    def _on_delta(text: str) -> None:
        delta_q.put(text)

    result_box: dict = {}
    stop_event = threading.Event()

    def _run_pipeline() -> None:
        try:
            result_box["result"] = pipeline.answer(
                req.question, req.session_id, req.user_profile, request_id,
                deep_thinking=req.deep_thinking,
                on_stage=_on_stage, on_delta=_on_delta, stop_event=stop_event)
        except Exception as e:  # 兜底：后台异常不悬挂 SSE，转 error 帧
            _log.error("pipeline_worker_error rid=%s err=%s", request_id, str(e)[:200])
            result_box["error"] = str(e)[:200]

    worker = threading.Thread(target=_run_pipeline, daemon=True)
    worker.start()

    sent = 0
    try:
        while worker.is_alive() or not delta_q.empty():
            # 优先排空 token 增量(减小流式延迟),再播报阶段进度
            drained = False
            while True:
                try:
                    text = delta_q.get_nowait()
                except queue.Empty:
                    break
                drained = True
                yield evt("delta", {"text": text})
            with progress_lock:
                pending, sent = progress[sent:], len(progress)
            for name in pending:
                yield evt("status", {"stage": name})
            if not drained:
                # 无增量时阻塞等待（替代固定 50ms 轮询）：delta 一到即透出，无感知延迟
                try:
                    text = delta_q.get(timeout=0.2)
                    yield evt("delta", {"text": text})
                except queue.Empty:
                    pass
        worker.join()
    except GeneratorExit:
        # 客户端断开：通知 worker 及时从 LLM 流式循环退出（daemon 线程不悬挂）
        stop_event.set()
        worker.join(timeout=5.0)
        raise

    if result_box.get("error") or result_box.get("result") is None:
        yield evt("error", {"message": "服务处理失败，请重试。"})
        return
    result = result_box["result"]
    _remember_request(app.state.recent_answers, request_id,
                      req.question, result.answer, req.session_id)

    # 4. 输出审核（生成后;增量区已展示的内容由 answer 帧兜底覆盖）
    censor_note = None
    if censor is not None and not result.refusal:
        cr = censor.check_text(result.answer, task="RAG_QA_OUTPUT")
        if not cr.passed:
            result.answer = "该回答未通过内容审核，已屏蔽。请换一种问法。"
            censor_note = cr.conclusion

    # 5. 权威全文 + 引用 + 完成
    text = result.answer
    yield evt("citations", {
        "docs": result.citations, "grounded": result.grounded, "refusal": result.refusal,
    })
    yield evt("answer", {"text": text})   # 事实核查/硬过滤后的最终全文
    yield evt("done", {
        "usage": [_usage_to_dict(u) for u in result.usage],
        "fallback_active": result.fallback_active,
        "answer_len": len(text),
        "censor": censor_note,
    })
