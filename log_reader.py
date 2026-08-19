"""
log_reader.py —— gateway.log 检索事件读取器（调试/演示用）
============================================================
从 JSON Lines 结构化日志（gateway.log，RotatingFileHandler 5MB×3）读取
event=="retrieval" 事件，供 /v1/debug/retrieval 端点与 UI 展示——演示时
当场查看「这个问题三路检索各给了多少分」的中间过程。

轮转说明：仅解析当前 gateway.log 文件；gateway.log.1/.2/.3 为轮转备份，
调试场景不追溯（最新请求总是写在当前文件，且 5MB 内可容纳大量请求）。

用法:
    from log_reader import find_retrieval_event, recent_retrieval_events
    ev = find_retrieval_event("abc123")       # → {query, docs:[{source,score,...}]} | None
    recent = recent_retrieval_events(5)       # → 最近 5 条（新在前）
"""

import json

from config import GATEWAY_LOG_PATH


def _iter_events(path: str = GATEWAY_LOG_PATH):
    """逐行解析 JSON Lines；坏行静默跳过（日志轮转瞬间可能截断半行）。"""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return


def find_retrieval_event(request_id: str, path: str = GATEWAY_LOG_PATH) -> dict | None:
    """按 request_id 查找检索事件（含 query 与每路文档得分/来源/摘要）。"""
    found = None
    for ev in _iter_events(path):
        if ev.get("event") == "retrieval" and ev.get("request_id") == request_id:
            found = ev   # 同 request_id 理论上唯一；保留最后一次匹配
    return found


def recent_retrieval_events(limit: int = 10, path: str = GATEWAY_LOG_PATH) -> list[dict]:
    """返回最近 limit 条检索事件（文件按时间顺序写入 → 取末尾 N 条再反转，新在前）。"""
    out: list[dict] = []
    for ev in _iter_events(path):
        if ev.get("event") == "retrieval":
            out.append(ev)
            if len(out) > limit:
                out.pop(0)
    return list(reversed(out))
