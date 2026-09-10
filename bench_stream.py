"""bench_stream.py —— /v1/chat/stream 并发压测（P99 延迟 / token 吞吐 / **耗时归因**）
======================================================================
对 SSE 流式端点做并发请求，测每档并发下的：
  - 首 token 延迟（流式体验关键指标：从发起到第一个 delta）
  - 总延迟（到 done 帧）
  - token 吞吐（done 帧 usage 累计 / 墙钟时间）
  - **耗时归因**：LLM 网络耗时 vs 其余（锁等待/检索/审核/排队）

为什么必须归因（而不是只看 P99）：
  单看「并发 8 时变慢 3 倍」无法判断该优化什么——
  可能是 LLM 配额打满，可能是全局锁串行化检索，也可能压根没到顶只是抖动。
  归因后才谈得上「按判据决定要不要引入 Redis」这类结论。

归因怎么做（**无需改动服务端**）：
  1. SSE 的 status 帧本身就标记了阶段边界 → 客户端记到达时刻即得阶段时间线
  2. 服务端 gateway.log 每条 llm_usage 带 latency_ms → 按 request_id 求和得 LLM 净耗时
  3. 总耗时 − LLM 净耗时 = 其余部分（锁等待 / 检索 / 审核 / 排队）
  若并发升高时 LLM 净耗时基本持平、而「其余」线性增长 → 瓶颈在串行段（锁）。
  若 LLM 净耗时本身就随并发暴涨 → 瓶颈在供应商侧（配额/排队），加 Redis 无用。

各并发档用独立 session_id（绕过网关限流/降噪干扰）。

用法:
  python bench_stream.py                                   # 默认 2/4/8 并发
  python bench_stream.py --concurrency 1,2,4,8,16
  python bench_stream.py --question "腰突怎么康复"          # injury 层（含 Fact-Check，更慢）
  python bench_stream.py --concurrency 1,2,4,8 --no-attribute   # 跳过归因（不看日志）

前置：API 已启动（python start.py --no-browser 或 uvicorn api:app）
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time

import httpx

import config as _cfg

API_BASE = f"http://{_cfg.API_HOST}:{_cfg.API_PORT}"
API_KEY = _cfg.API_KEY_AUTH
LOG_PATH = _cfg.GATEWAY_LOG_PATH


def _run_one(idx: int, question: str, barrier, result: dict, level: int) -> None:
    """单请求：SSE 流式读到 done 帧，记录首 token / 总耗时 / token 数 / 阶段时刻。"""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    # session 带档位前缀：各并发档独立会话，避免复用上一档 session 触发网关降噪
    payload = {"question": question, "session_id": f"bench_{level}_{idx}",
               "user_profile": None, "deep_thinking": False}
    t0 = time.time()
    first_delta = None
    total_tokens = 0
    request_id = ""
    first_status = None
    gen_start = None      # 「正在生成回答…」= 检索段结束、生成段开始
    stages: list = []
    try:
        with httpx.stream("POST", f"{API_BASE}/v1/chat/stream",
                          json=payload, headers=headers, timeout=300.0) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                ev = data.get("event")
                if ev == "meta":
                    request_id = data.get("request_id", "")
                elif ev == "status":
                    now = time.time() - t0
                    stages.append((data.get("stage", ""), now))
                    if first_status is None:
                        first_status = now
                    if "生成回答" in (data.get("stage") or ""):
                        gen_start = now
                elif ev in ("delta", "answer") and first_delta is None:
                    first_delta = time.time() - t0
                elif ev == "done":
                    for u in data.get("usage") or []:
                        total_tokens += u.get("total_tokens", 0)
                elif ev == "error":
                    result[idx] = {"error": data.get("message", "error")[:80],
                                   "t_total": time.time() - t0}
                    return
        result[idx] = {"t_total": time.time() - t0, "t_first": first_delta,
                       "tokens": total_tokens, "request_id": request_id,
                       "t_status": first_status, "t_gen": gen_start,
                       "pre_llm": gen_start if gen_start is not None else first_delta}
    except httpx.HTTPError as e:
        result[idx] = {"error": str(e)[:80], "t_total": time.time() - t0}


# ----------------------------------------------------------------
# 归因：从 gateway.log 取每个 request_id 的 LLM 净耗时
# ----------------------------------------------------------------

def _log_size() -> int:
    try:
        return os.path.getsize(LOG_PATH)
    except OSError:
        return 0


def _llm_ms_by_request(since_offset: int) -> dict[str, int]:
    """读 [since_offset, EOF) 的 llm_usage，按 request_id 汇总 latency_ms。

    request_id 形如 "background" 的是 HyDE/校验等后台调用（不进 SSE 的 request_id），
    单独归入 "<background>" 桶，不摊到具体请求上。
    """
    out: dict[str, int] = {}
    try:
        with open(LOG_PATH, encoding="utf-8", errors="replace") as f:
            f.seek(since_offset)
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("event") != "llm_usage":
                    continue
                rid = e.get("request_id") or "<unknown>"
                out[rid] = out.get(rid, 0) + int(e.get("latency_ms") or 0)
    except OSError:
        pass
    return out


def _lock_wait_by_request(since_offset: int) -> dict[str, int]:
    """读 [since_offset, EOF) 的 lock_phase，按 request_id 汇总**等待**时长。

    只累加 wait_ms 不累加 hold_ms：等待才是串行化的代价，持有是临界区本身的工作量。
    """
    out: dict[str, int] = {}
    try:
        with open(LOG_PATH, encoding="utf-8", errors="replace") as f:
            f.seek(since_offset)
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("event") != "lock_phase":
                    continue
                rid = e.get("request_id") or "<unknown>"
                out[rid] = out.get(rid, 0) + int(round(e.get("wait_ms") or 0))
    except OSError:
        pass
    return out


def _stat(vals, key=None):
    vals = [v for v in (vals if key is None else (x.get(key) for x in vals))
            if v is not None]
    if not vals:
        return None
    vals.sort()

    def p(pct):
        return vals[min(len(vals) - 1, int(len(vals) * pct))]

    return {"mean": statistics.mean(vals), "p50": p(0.50),
            "p95": p(0.95), "p99": p(0.99), "n": len(vals)}


def _fmt_ms(v) -> str:
    """秒 → "123ms"。"""
    return f"{v * 1000:.0f}ms" if v is not None else "-"


def _fmt_raw_ms(v) -> str:
    """毫秒 → "123ms"（已经是毫秒的值，不要再乘 1000）。"""
    return f"{v:.0f}ms" if v is not None else "-"


def bench(concurrency: int, question: str, attribute: bool = True) -> dict:
    """一批并发请求；返回聚合统计（含耗时归因）。"""
    result: dict = {}
    barrier = threading.Barrier(concurrency)   # 同时起跑（消除启动偏差）
    log_offset = _log_size() if attribute else 0
    threads = []
    t0 = time.time()
    for i in range(concurrency):
        t = threading.Thread(target=_run_one,
                             args=(i, question, barrier, result, concurrency))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0

    ok = [r for r in result.values() if "error" not in r]
    errs = [r for r in result.values() if "error" in r]

    llm = _llm_ms_by_request(log_offset) if attribute else {}
    locks = _lock_wait_by_request(log_offset) if attribute else {}
    for r in ok:
        r["llm_ms"] = llm.get(r.get("request_id") or "", 0)
        r["lock_wait_ms"] = locks.get(r.get("request_id") or "", 0)
        r["other_ms"] = max(0.0, (r["t_total"] * 1000) - r["llm_ms"])
    bg_ms = sum(v for k, v in llm.items() if k in ("background", "<unknown>"))

    return {
        "concurrency": concurrency, "wall": wall,
        "t_first": _stat(ok, "t_first"), "t_total": _stat(ok, "t_total"),
        "llm_ms": _stat(ok, "llm_ms"), "other_ms": _stat(ok, "other_ms"),
        "lock_wait_ms": _stat(ok, "lock_wait_ms"),
        "bg_llm_ms_total": bg_ms,
        "tokens": sum(r.get("tokens", 0) for r in ok),
        "errors": errs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="SSE 流式端点并发压测")
    ap.add_argument("--concurrency", default="2,4,8",
                    help="并发档位（逗号分隔），每档一批")
    ap.add_argument("--question", default="深蹲主要锻炼哪些肌群",
                    help="压测问题（simple 层快；injury/plan 层慢但覆盖校验路径）")
    ap.add_argument("--no-attribute", action="store_true",
                    help="跳过耗时归因（不读 gateway.log）")
    ap.add_argument("--repeat", type=int, default=1,
                    help="每档重复批数（取中位数抗抖动；默认 1）")
    args = ap.parse_args()

    attribute = not args.no_attribute
    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    print(f"[INFO] 压测目标: {API_BASE}/v1/chat/stream")
    print(f"[INFO] 问题: {args.question}")
    print(f"[INFO] 并发档位: {levels}（每档 {args.repeat} 批，独立 session）")
    print(f"[INFO] 耗时归因: {'开（读 ' + LOG_PATH + '）' if attribute else '关'}\n")

    print(f"{'并发':<5}{'墙钟':>8}{'首token p50':>13}{'总延迟 p50/p95':>22}"
          f"{'LLM净耗时':>11}{'锁等待':>9}{'其余':>9}{'token/s':>9}")
    print("-" * 88)
    for c in levels:
        runs = [bench(c, args.question, attribute) for _ in range(args.repeat)]
        errs = [e for r in runs for e in r["errors"]]
        if errs:
            print(f"{c:<5}{runs[0]['wall']:>7.1f}s  {len(errs)} 个请求出错：")
            for e in errs[:3]:
                print(f"      - {e.get('error')}")
            continue

        def med(key, sub="p50"):
            vals = [r[key][sub] for r in runs if r.get(key)]
            return statistics.median(vals) if vals else None

        wall = statistics.median(r["wall"] for r in runs)
        tps = statistics.median(r["tokens"] / r["wall"] for r in runs)
        print(f"{c:<5}{wall:>7.1f}s  "
              f"{_fmt_ms(med('t_first')):>12}  "
              f"{_fmt_ms(med('t_total')):>10}/{_fmt_ms(med('t_total', 'p95')):<10}"
              f"{_fmt_raw_ms(med('llm_ms')):>10} "
              f"{_fmt_raw_ms(med('lock_wait_ms')):>8} "
              f"{_fmt_raw_ms(med('other_ms')):>8}"
              f"{tps:>9.0f}")
    print()
    print("注：首token = 发起到第一个 delta；总延迟 = 到 done 帧（含 Fact-Check/审核）")
    print("    LLM净耗时 = 该 request_id 的 llm_usage.latency_ms 之和")
    print("    锁等待    = 该 request_id 的 lock_phase.wait_ms 之和（A/C/E/F 四个临界区）")
    print("    其余      = 总延迟 − LLM净耗时（含锁等待、检索、审核、排队、后台调用）")
    print("    ⚠️ 后台调用（HyDE/校验，request_id=background）不摊到具体请求，")
    print("       故「其余」含这部分 LLM 时间；判断串行化看的是「锁等待」那一列。")


if __name__ == "__main__":
    main()
