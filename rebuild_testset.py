"""
rebuild_testset.py —— 测试集重建（锚定当前知识库内容）
========================================================
背景：旧 test_100_full.csv 为 GraphRAG/旧扫描书 KB 时代产物——
  14 条架构/评测/总结类元问题描述的是已不存在的旧系统，
  其余 86 条的黄金参考按旧版权书内容撰写，与换库后的知识库（CSV 动作库
  + 科学健身18法 + 全民健身指南）不同步，检索 Hit 指标失真。

重建原则：
  - 全部 100 条均为健身域问题，锚定当前 KB 真实内容（每条 seed 来自 KB）
  - 参考文本贴近 KB 原文表述（可被向量检索命中），黄金答案为完整回答
  - 题型分布：基础问答 30 / 单伤病 25 / 复合伤病 15 / 计划生成 15 /
    动作纠错 10 / 定制长计划 5（元问题类型废弃）
  - 旧文件备份为 test_100_full_legacy.csv

用法: python rebuild_testset.py [--limit 10]   # 生成并覆盖 test_100_full.csv
"""

from __future__ import annotations

import csv
import random
import re
import sys
import time

from llm_adapter import build_llm

OUT = "test_100_full.csv"
LEGACY = "test_100_full_legacy.csv"

# ============================================================
# KB 锚点数据
# ============================================================

def _load_csv_actions() -> list[dict]:
    with open("fitness_data.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_zhinan_paras() -> list[str]:
    """指南正文段落（剔除新闻导语部分，优先方案/原则/强度相关内容）。"""
    text = open("kb_zhinan.txt", encoding="utf-8").read()
    paras = [p for p in text.split("\n\n") if len(p) > 40]
    # 新闻导语特征段（"近日，""参与该指南研制的机构"等）不是指南正文
    paras = [p for p in paras
             if not any(k in p for k in ("近日，", "参与该指南研制的机构", "该项目研究组组长"))]
    # 计划类 seed 优先匹配方案/原则/强度/频率等关键词
    preferred = [p for p in paras
                 if any(k in p for k in ("方案", "原则", "强度", "频率", "次数", "每周", "运动方式"))]
    return preferred or paras


def _load_18fa_blocks() -> list[str]:
    text = open("kb_18fa.txt", encoding="utf-8").read()
    return [b for b in text.split("\n\n") if len(b) > 20]


def _load_injuries() -> dict:
    """伤病 → {禁忌: [(动作, 原因)], 康复: [(动作, 原因)]}（只保留动作在 CSV 中有对应条目者，
    保证参考文本可被检索命中；CSV 动作为全称，如「杠铃深蹲」与伤病动作「深蹲」做包含匹配）。"""
    from contra_data import INJURY_ACTION_MAP, INJURY_ALIASES
    names = {r["动作名称"] for r in _load_csv_actions()}
    out: dict[str, dict] = {}
    for key, acts in INJURY_ACTION_MAP.items():
        canon = INJURY_ALIASES.get(key, key)
        entry = out.setdefault(canon, {"禁忌": [], "康复": []})
        for action, rel, reason in acts:
            if rel == "禁忌动作":
                entry["禁忌"].append((action, reason))
            elif rel == "康复动作":
                entry["康复"].append((action, reason))
    # 过滤：至少 1 条禁忌且其动作名能在 CSV 中找到（含匹配）
    eligible = {}
    for inj, entry in out.items():
        taboos = [t for t in entry["禁忌"] if any(t[0] in n for n in names)]
        rehabs = [t for t in entry["康复"] if any(t[0] in n for n in names)]
        if taboos and rehabs:
            eligible[inj] = {"禁忌": taboos, "康复": rehabs}
    return eligible


# ============================================================
# 生成器
# ============================================================

def _gen(llm, prompt: str) -> dict | None:
    """调用模型生成一行测试样本，解析「问题:/黄金答案:/参考文本:」三行格式。"""
    resp = llm.invoke(prompt, max_tokens=500, temperature=0.8)
    text = resp.content or ""
    q = re.search(r"问题[:：]\s*(.+)", text)
    gt = re.search(r"黄金答案[:：]\s*(.+)", text)
    ref = re.search(r"参考文本[:：]\s*(.+)", text)
    if not (q and gt and ref):
        return None
    return {
        "query": q.group(1).strip(),
        "ground_truth": gt.group(1).strip(),
        "reference_texts": ref.group(1).strip(),
    }


def build(llm, limit: int | None = None) -> list[dict]:
    actions = _load_csv_actions()
    zhinan = _load_zhinan_paras()
    fa18 = _load_18fa_blocks()
    injuries = _load_injuries()
    inj_names = sorted(injuries)
    random.seed(42)

    rows: list[dict] = []
    # 每类型 (数量, 生成函数)
    specs = [
        ("基础问答", 30, lambda i: _gen_basic(llm, actions, i)),
        ("动作纠错", 10, lambda i: _gen_correction(llm, actions, i)),
        ("单伤病问答", 25, lambda i: _gen_single_injury(llm, injuries, inj_names, i)),
        ("复合伤病问答", 15, lambda i: _gen_compound_injury(llm, injuries, inj_names, i)),
        ("计划生成", 15, lambda i: _gen_plan(llm, zhinan, actions, i)),
        ("定制长计划", 5, lambda i: _gen_long_plan(llm, zhinan, injuries, inj_names, actions, i)),
    ]
    for qtype, count, fn in specs:
        done = 0
        attempt = 0
        while done < count and attempt < count * 3:
            attempt += 1
            row = fn(done)
            if row and 5 <= len(row["query"]) <= 45 and 30 <= len(row["reference_texts"]) <= 160:
                row["question_type"] = qtype
                rows.append(row)
                done += 1
                print(f"  [{qtype}] {done}/{count} {row['query']}")
                time.sleep(0.1)
        if limit and len(rows) >= limit:
            break
    return rows


def _gen_basic(llm, actions, i):
    a = actions[i % len(actions)]
    prompt = f"""你是评测集撰写专家。基于以下知识库动作条目写一条基础问答测试样本：
动作条目：{a['动作名称']} | 肌群:{a['目标肌群']} | 器械:{a['器械']} | 步骤:{a['步骤']} | 注意:{a['注意事项']}
问题询问该动作锻炼什么肌群或怎么做才标准。参考文本必须贴近条目原文表述（便于向量检索命中），黄金答案为完整专业回答（30-80字）。
只输出三行：问题: / 黄金答案: / 参考文本:"""
    return _gen(llm, prompt)


def _gen_correction(llm, actions, i):
    a = actions[(i * 7 + 3) % len(actions)]
    prompt = f"""你是评测集撰写专家。基于以下知识库动作条目写一条动作纠错测试样本：
动作条目：{a['动作名称']} | 肌群:{a['目标肌群']} | 步骤:{a['步骤']} | 注意事项:{a['注意事项']}
问题询问该动作的常见错误或注意事项。参考文本必须贴近条目原文表述，黄金答案为完整专业回答（30-80字）。
只输出三行：问题: / 黄金答案: / 参考文本:"""
    return _gen(llm, prompt)


def _gen_single_injury(llm, injuries, inj_names, i):
    """单伤病：伤病×禁忌动作×问法三因子轮换，保证去重后仍有足够多样性。"""
    inj = inj_names[i % len(inj_names)]
    d = injuries[inj]
    taboo, t_reason = d["禁忌"][(i * 3) % len(d["禁忌"])]
    rehab, r_reason = d["康复"][(i * 3 + 1) % len(d["康复"])]
    forms = [
        f"{inj}能做{taboo}吗",
        f"{inj}练{taboo}有什么风险",
        f"{inj}期间{taboo}需要避免吗",
    ]
    prompt = f"""你是评测集撰写专家。基于以下伤病知识写一条单伤病问答测试样本：
伤病：{inj}；禁忌动作：{taboo}（原因：{t_reason}）；康复动作：{rehab}（原因：{r_reason}）。
问题参考形式：{forms[i % len(forms)]}（可变换措辞，但必须聚焦该伤病与该动作）。
参考文本需涵盖禁忌与康复两方面表述（贴近知识库动作条目措辞），黄金答案为完整安全建议（50-100字，含禁忌原因与康复方向）。
只输出三行：问题: / 黄金答案: / 参考文本:"""
    return _gen(llm, prompt)


def _gen_compound_injury(llm, injuries, inj_names, i):
    """复合伤病：组合按 i 系统性错开，避免重复组合。"""
    n = len(inj_names)
    inj1 = inj_names[i % n]
    inj2 = inj_names[(i + 1 + (i // n)) % n]
    d1, d2 = injuries[inj1], injuries[inj2]
    t1 = d1["禁忌"][i % len(d1["禁忌"])][0]
    r1 = d1["康复"][i % len(d1["康复"])][0]
    t2 = d2["禁忌"][(i + 1) % len(d2["禁忌"])][0]
    r2 = d2["康复"][(i + 1) % len(d2["康复"])][0]
    prompt = f"""你是评测集撰写专家。基于以下复合伤病知识写一条复合伤病问答测试样本：
伤病A：{inj1}（禁忌 {t1}，康复 {r1}）；伤病B：{inj2}（禁忌 {t2}，康复 {r2}）。
问题形式如「{inj1}加{inj2}该怎么练」或「同时有{inj1}和{inj2}能做{t1}吗」。
参考文本涵盖两种伤病的关键表述，黄金答案为综合安全建议（60-120字）。
只输出三行：问题: / 黄金答案: / 参考文本:"""
    return _gen(llm, prompt)


def _gen_plan(llm, zhinan, actions, i):
    para = zhinan[(i * 5 + 2) % len(zhinan)]
    a1 = actions[(i * 3) % len(actions)]
    a2 = actions[(i * 3 + 11) % len(actions)]
    prompt = f"""你是评测集撰写专家。基于以下知识库内容写一条计划生成测试样本：
指南片段：{para[:200]}
相关动作：{a1['动作名称']}（{a1['目标肌群']}）、{a2['动作名称']}（{a2['目标肌群']}）
问题询问训练计划（增肌/减脂/新手计划等）。参考文本需贴近指南与动作条目表述，黄金答案为结构化计划要点（50-120字）。
只输出三行：问题: / 黄金答案: / 参考文本:"""
    return _gen(llm, prompt)


def _gen_long_plan(llm, zhinan, injuries, inj_names, actions, i):
    inj = inj_names[i % len(inj_names)]
    d = injuries[inj]
    taboo = d["禁忌"][0][0]; rehab = d["康复"][0][0]
    para = zhinan[(i * 9 + 5) % len(zhinan)]
    a = actions[(i * 5 + 1) % len(actions)]
    prompt = f"""你是评测集撰写专家。基于以下知识库内容写一条定制长计划测试样本（多约束）：
伤病：{inj}（禁忌 {taboo}，康复 {rehab}）；指南片段：{para[:180]}；参考动作：{a['动作名称']}（{a['目标肌群']}）。
问题为含伤病+目标+器械/时间等多项约束的计划请求。参考文本涵盖约束相关的关键表述，黄金答案为多约束计划要点（60-130字）。
只输出三行：问题: / 黄金答案: / 参考文本:"""
    return _gen(llm, prompt)


# ============================================================
# 主流程
# ============================================================

def main():
    limit = None
    for i, arg in enumerate(sys.argv):
        if arg == "--limit" and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1])

    print("[KB] 加载锚点数据 ...")
    llm = build_llm("chat_fast", provider="dashscope")  # flash 快速生成
    rows = build(llm, limit=limit)

    # 去重与校验
    seen, final_rows = set(), []
    for r in rows:
        if r["query"] in seen:
            continue
        seen.add(r["query"])
        final_rows.append(r)

    print(f"\n生成 {len(final_rows)} 条（去重后）")
    # 备份旧文件（legacy 已存在时不覆盖——保留最初的旧版备份）
    import os
    if os.path.exists(OUT) and not os.path.exists(LEGACY):
        os.replace(OUT, LEGACY)
        print(f"旧测试集已备份为 {LEGACY}")

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query", "ground_truth", "reference_texts", "question_type"])
        w.writeheader()
        w.writerows(final_rows)
    print(f"已写入 {OUT}")


if __name__ == "__main__":
    main()
