"""bench_stream.py —— /v1/chat/stream 并发压测（P99 延迟 / token 吞吐）
======================================================================
对 SSE 流式端点做并发请求，测每档并发下的：
  - 首 token 延迟（流式体验关键指标：从发起到第一个 delta）
  - 总延迟（到 done 帧）
  - token 吞吐（done 帧 usage 累计 / 墙钟时间）

各并发档用独立 session_id（绕过网关限流/降噪干扰），默认 2/4/8 并发 × 各一批。

用法:
  python bench_stream.py                                   # 默认 2/4/8 并发
  python bench_stream.py --concurrency 1,2,4,8 --question "深蹲主要锻炼哪些肌群"
  python bench_stream.py --question "腰突怎么康复"          # injury 层（含 Fact-Check，更慢）

前置：API 已启动（python start.py --no-browser 或 uvicorn api:app）
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time

import httpx

import config as _cfg

API_BASE = f"http://{_cfg.API_HOST}:{_cfg.API_PORT}"
API_KEY = _cfg.API_KEY_AUTH


def _run_one(idx: int, question: str, barrier, result: dict, level: int) -> None:
    """单请求：SSE 流式读到 done 帧，记录首 token / 总耗时 / token 数。"""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    # session 带档位前缀：各并发档独立会话，避免复用上一档 session 触发网关降噪
    payload = {"question": question, "session_id": f"bench_{level}_{idx}",
               "user_profile": None, "deep_thinking": False}
    t0 = time.time()
    first_delta = None
    total_tokens = 0
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
                if first_delta is None and data.get("event") in ("delta", "answer"):
                    first_delta = time.time() - t0
                if data.get("event") == "done":
                    for u in data.get("usage") or []:
                        total_tokens += u.get("total_tokens", 0)
                if data.get("event") == "error":
                    result[idx] = {"error": data.get("message", "error")[:80],
                                   "t_total": time.time() - t0}
                    return
        result[idx] = {"t_total": time.time() - t0, "t_first": first_delta,
                       "tokens": total_tokens}
    except httpx.HTTPError as e:
        result[idx] = {"error": str(e)[:80], "t_total": time.time() - t0}


def _fmt_ms(v) -> str:
    return f"{v * 1000:.0f}ms" if v is not None else "-"


def bench(concurrency: int, question: str) -> dict:
    """一批并发请求；返回聚合统计。"""
    result: dict = {}
    barrier = threading.Barrier(concurrency)   # 同时起跑（消除启动偏差）
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

    def _stat(key):
        vals = [r[key] for r in ok if r.get(key) is not None]
        if not vals:
            return None
        vals.sort()
        def p(pct):
            return vals[min(len(vals) - 1, int(len(vals) * pct))]
        return {"mean": statistics.mean(vals), "p50": p(0.50),
                "p95": p(0.95), "p99": p(0.99), "n": len(vals)}

    return {
        "concurrency": concurrency, "wall": wall,
        "t_first": _stat("t_first"), "t_total": _stat("t_total"),
        "tokens": sum(r.get("tokens", 0) for r in ok),
        "errors": errs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="SSE 流式端点并发压测")
    ap.add_argument("--concurrency", default="2,4,8",
                    help="并发档位（逗号分隔），每档一批")
    ap.add_argument("--question", default="深蹲主要锻炼哪些肌群",
                    help="压测问题（simple 层快；injury/plan 层慢但覆盖校验路径）")
    args = ap.parse_args()

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    print(f"[INFO] 压测目标: {API_BASE}/v1/chat/stream")
    print(f"[INFO] 问题: {args.question}")
    print(f"[INFO] 并发档位: {levels}（每档一批，独立 session）\n")

    print(f"{'并发':<6}{'墙钟':>10}{'首token P50/P95/P99':>26}{'总延迟 P50/P95/P99':>26}{'token/s':>10}")
    print("-" * 80)
    for c in levels:
        r = bench(c, args.question)
        if r["errors"]:
            print(f"{c:<6}{r['wall']:.1f}s  {len(r['errors'])} 个请求出错（详见下方）")
            for e in r["errors"][:3]:
                print(f"    - {e.get('error')}")
            continue
        f, t = r["t_first"], r["t_total"]
        tps = r["tokens"] / r["wall"]
        print(f"{c:<6}{r['wall']:>8.1f}s  "
              f"{_fmt_ms(f['p50'])}/{_fmt_ms(f['p95'])}/{_fmt_ms(f['p99']):<8}"
              f"{_fmt_ms(t['p50'])}/{_fmt_ms(t['p95'])}/{_fmt_ms(t['p99']):<8}"
              f"{tps:>8.0f}")
    print()
    print("注：首 token 延迟 = 发起到第一个 delta；总延迟 = 到 done 帧（含 Fact-Check/审核）")


if __name__ == "__main__":
    main()
