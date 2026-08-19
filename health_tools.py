"""
health_tools.py —— 确定性健康计算工具层
========================================
为 RAG 流水线提供确定性健康计算（BMI / 每日饮水量 / 运动心率区间），
结果注入检索上下文供生成模型引用——健康数据的数值计算必须确定：
可复现、可审计、零 token 成本，不让 LLM 自行推算（算术类任务 LLM 不可靠）。

设计：
  - TOOL_REGISTRY 注册表：name / title / keywords（触发词）/ compute 函数
  - 参数抽取（身高/体重/年龄）优先取 user_profile（侧边栏画像 "身高170cm，体重70kg"），
    其次从问题文本正则提取；参数缺失不硬算（返回 None，该工具跳过）
  - run_health_tools 只返回「关键词命中且参数齐全」的计算结果

演进路径：llm_adapter 已支持 OpenAI tools 协议，本注册表结构可直接转换为
tools 定义接入 LLM 自主工具调用——计算逻辑与触发机制解耦，两条路都留好。

用法:
    from health_tools import run_health_tools
    results = run_health_tools("帮我算下BMI", "身高170cm，体重70kg")
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ToolResult:
    """单个工具的计算结果（title + 已格式化 content，可直接注入上下文）。"""
    name: str
    title: str
    content: str


# ============================================================
# 参数抽取（身高 cm / 体重 kg / 年龄 岁）
# ============================================================

_RE_HEIGHT = re.compile(r"身高\s*(\d{2,3})\s*(?:cm|厘米|公分)?", re.IGNORECASE)
_RE_WEIGHT = re.compile(r"体重\s*(\d{1,3}(?:\.\d)?)\s*(?:kg|公斤)?", re.IGNORECASE)
_RE_AGE = re.compile(r"(\d{1,3})\s*岁")


def _extract_params(question: str, user_profile: str | None) -> dict:
    """抽取身高/体重/年龄参数。user_profile 优先，question 兜底（画像可能过期）。

    返回 {} 表示一个参数都没有（直接跳过全部工具，省关键词扫描）。
    """
    params: dict = {}
    for text in (user_profile or "", question or ""):
        if not text:
            continue
        if "height" not in params:
            m = _RE_HEIGHT.search(text)
            if m:
                params["height"] = float(m.group(1))
        if "weight" not in params:
            m = _RE_WEIGHT.search(text)
            if m:
                params["weight"] = float(m.group(1))
        if "age" not in params:
            m = _RE_AGE.search(text)
            if m:
                params["age"] = float(m.group(1))
    return params


# ============================================================
# 工具实现（参数不全 → 返回 None，跳过）
# ============================================================

def _calc_bmi(params: dict) -> ToolResult | None:
    """BMI = 体重/身高²；中国成人分档 18.5 / 24 / 28。"""
    h, w = params.get("height"), params.get("weight")
    if not h or not w:
        return None
    bmi = w / ((h / 100) ** 2)
    if bmi < 18.5:
        level = "偏瘦（<18.5）"
    elif bmi < 24:
        level = "正常（18.5-23.9）"
    elif bmi < 28:
        level = "超重（24-27.9）"
    else:
        level = "肥胖（≥28）"
    return ToolResult(
        name="calculate_bmi",
        title="BMI 计算",
        content=f"身高{h:.0f}cm、体重{w:.0f}kg → BMI={bmi:.1f}，属{level}。"
                "BMI 仅作参考，肌肉量大者可能被高估，需结合体脂率判断。",
    )


def _calc_water(params: dict) -> ToolResult | None:
    """每日饮水量 ≈ 体重 × 30ml（成人参考量，中国居民膳食指南口径）。"""
    w = params.get("weight")
    if not w:
        return None
    base = w * 30
    return ToolResult(
        name="estimate_water_intake",
        title="每日饮水量估算",
        content=f"体重{w:.0f}kg → 基础饮水量约 {base:.0f}ml"
                f"（约 {base / 500:.1f} 瓶 500ml），运动日建议额外补充 500-1000ml。",
    )


def _calc_hr(params: dict) -> ToolResult | None:
    """最大心率 ≈ 220-年龄；燃脂 60-70% / 有氧耐力 70-80% / 无氧 80-90%。"""
    age = params.get("age")
    if not age:
        return None
    max_hr = 220 - age
    return ToolResult(
        name="heart_rate_zone",
        title="运动心率区间（最大心率法）",
        content=f"年龄{age:.0f}岁 → 最大心率≈{max_hr:.0f} 次/分；"
                f"燃脂区（60-70%）={max_hr * 0.6:.0f}-{max_hr * 0.7:.0f}，"
                f"有氧耐力区（70-80%）={max_hr * 0.7:.0f}-{max_hr * 0.8:.0f}。",
    )


# ============================================================
# 注册表与入口
# ============================================================

TOOL_REGISTRY = [
    {"name": "calculate_bmi", "title": "BMI 计算", "keywords": ("bmi", "体重指数", "身高", "体重"), "compute": _calc_bmi},
    {"name": "estimate_water_intake", "title": "每日饮水量估算", "keywords": ("喝水", "饮水", "喝多少", "多少水", "多喝水", "补水", "水量", "水分", "水摄入"), "compute": _calc_water},
    {"name": "heart_rate_zone", "title": "运动心率区间", "keywords": ("心率", "燃脂", "最大心率", "靶心率"), "compute": _calc_hr},
]


def run_health_tools(question: str, user_profile: str | None = None) -> list[ToolResult]:
    """关键词命中且参数齐全 → 返回计算结果列表；无命中/缺参数 → []。

    纯本地字符串处理（~0.1ms），pipeline 锁内调用安全。
    """
    params = _extract_params(question, user_profile)
    if not params:
        return []
    results: list[ToolResult] = []
    q_lower = (question or "").lower()
    for tool in TOOL_REGISTRY:
        if any(kw in q_lower for kw in tool["keywords"]):
            r = tool["compute"](params)
            if r is not None:
                results.append(r)
    return results
