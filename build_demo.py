"""
build_demo.py —— 构建只读演示数据
==================================
把若干**代表性问题**离线跑一遍完整管线，产出静态数据（`demo/demo_data.json`），
配合 `demo/index.html` 就是一个**无需后端、无需 API Key、无需 Milvus** 的公网 demo。

为什么用「预置问答」而不是真的在线跑
------------------------------------
在线 demo 需要：云端 API Key（不能公开）、Milvus Lite（单进程独占）、常驻服务（成本）。
而演示的目的是**让面试官点开就能看到系统做了什么**——预置数据同样能展示：
分层策略、引用来源、检索得分、禁忌拦截、拒答判定、工具计算。

诚实的边界：这是**离线快照**，不是实时服务。页面会明确标注这一点。

用法
----
    python build_demo.py                 # 跑完整管线生成数据（需要 API Key / 索引就绪）
    python build_demo.py --list          # 只看会跑哪些问题
"""

from __future__ import annotations

import json
import os
import sys

# 演示用例：覆盖各条链路的亮点，顺序即页面展示顺序
DEMO_CASES: list[dict] = [
    {
        "question": "深蹲主要锻炼哪些肌群",
        "tag": "简单问答",
        "highlight": "分层策略：快模型 + 短预算 + 跳过重排，端到端约 8 秒",
    },
    {
        "question": "腰突怎么康复",
        "tag": "伤病问答",
        "highlight": "伤病层：主模型 + LLM 重排 + Fact-Check 四类校验",
    },
    {
        "question": "腰突能做硬拉吗",
        "tag": "禁忌拦截（安全）",
        "highlight": "生成前边界拒绝——不给模型编造的机会",
    },
    {
        "question": "帮我算下BMI",
        "tag": "工具计算",
        "profile": "身高175cm，体重75kg，目标：减脂",
        "highlight": "确定性计算注入上下文，不让 LLM 自行算术",
    },
    {
        "question": "肩袖损伤能练卧推吗",
        "tag": "图谱检索",
        "highlight": "三路召回中图谱给出『伤病→动作』的禁忌关系",
    },
    {
        "question": "深蹲主要练哪些肌肉",
        "tag": "问法重述",
        "highlight": "与第 1 问同义但措辞不同——检索稳定命中同一批文档",
    },
    {
        "question": "量子力学和健身的关系",
        "tag": "弱相关问题",
        "highlight": "含『健身』故语义相关度 0.52（高于拒答阈值 0.40）→ 如实说明无直接关联，"
                     "而不是硬编或过度拒答",
    },
    {
        "question": "如何挑选股票",
        "tag": "无依据拒答（安全）",
        "highlight": "语义相关度 0.00，知识库完全无依据 → 生成前拒答，不消耗一次生成",
    },
]

OUT_DIR = "demo"
OUT_FILE = os.path.join(OUT_DIR, "demo_data.json")


def _usage_to_dict(usage: list) -> list[dict]:
    """UsageInfo → dict（容错：字段缺失不报错）。"""
    out = []
    for u in usage or []:
        if isinstance(u, dict):
            out.append(u)
            continue
        d = {k: v for k, v in vars(u).items() if not k.startswith("_")} \
            if hasattr(u, "__dict__") else {}
        out.append(d)
    return out


def _citations_to_dict(citations: list) -> list[dict]:
    return [dict(c) if isinstance(c, dict) else str(c) for c in citations or []]


def build() -> None:
    from log_reader import find_retrieval_event
    from pipeline import build_pipeline

    print("正在初始化管线（Milvus + BM25 + 适配器）…")
    svc = build_pipeline()
    print(f"就绪。开始跑 {len(DEMO_CASES)} 个演示用例：\n")

    cases = []
    for i, c in enumerate(DEMO_CASES, 1):
        # 每个用例独立 session，避免多轮上下文互相影响
        sid = f"demo_{i}"
        result = svc.answer(
            c["question"],
            session_id=sid,
            user_profile=c.get("profile"),
        )
        retrieval = find_retrieval_event(result.request_id) or {}

        answer = result.answer or ""
        if result.refusal and not answer:
            answer = "（已拒答）"
        cases.append({
            "question": c["question"],
            "tag": c["tag"],
            "highlight": c["highlight"],
            "profile": c.get("profile"),
            "answer": answer,
            "grounded": bool(result.grounded),
            "refusal": bool(result.refusal),
            "fallback_active": bool(result.fallback_active),
            "citations": _citations_to_dict(result.citations),
            "usage": _usage_to_dict(result.usage),
            "retrieval": {
                "query": retrieval.get("query", c["question"]),
                "docs": (retrieval.get("docs") or [])[:5],
            },
            "answer_len": len(answer),
        })
        flag = "拒答" if result.refusal else "作答"
        print(f"  [{i}/{len(DEMO_CASES)}] {c['question'][:22]:24s} → {flag} "
              f"({len(answer)} 字, 引用 {len(result.citations)} 条)")

    os.makedirs(OUT_DIR, exist_ok=True)
    payload = {
        "generated_by": "build_demo.py",
        "note": "离线快照：由完整管线跑出的真实结果，非实时服务。",
        "n_cases": len(cases),
        "cases": cases,
    }
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n已写出 {OUT_FILE}（{os.path.getsize(OUT_FILE)} 字节）")
    print(f"打开 {OUT_DIR}/index.html 即可查看；部署该目录到任意静态托管即可公网访问。")


def main() -> None:
    if "--list" in sys.argv:
        for i, c in enumerate(DEMO_CASES, 1):
            print(f"{i}. [{c['tag']}] {c['question']}")
            print(f"   {c['highlight']}")
        return
    build()


if __name__ == "__main__":
    main()
