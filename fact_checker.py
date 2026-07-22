"""
FactCheckEngine —— 四类事实校验引擎
====================================
用小模型（deepseek-r1:1.5b）逐项检查 AI 回答的：
  1. 动作建议  2. 康复周期  3. 负重限制  4. 伤病禁忌
任一 FAIL → 触发 CRAG 联网修正；全部 PASS → 缓存回答。

用法:
    from fact_checker import FactCheckEngine, FactCheckResult
    engine = FactCheckEngine(check_llm)
    result = engine.check(question, answer, context)
    if result.overall != "PASS":
        print(f"失败类别: {result.failed_categories}")
"""

import re
from dataclasses import dataclass, field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


@dataclass
class FactCheckResult:
    overall: str = "PASS"           # "PASS" | "FAIL"
    details: dict = field(default_factory=dict)  # {"动作": ("PASS",""), ...}
    failed_categories: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.overall == "PASS"


# 四类校验 Prompt（用一小段中文指令 + 结构化输出）
_FACT_CHECK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是健身/骨科康复专家审核员。请审核以下AI回答中的事实准确性。

参考知识（来自知识库）：
{context}

伤病禁忌黑名单（绝对禁止出现在AI回答中的动作/器械）：
{contraindications}

用户问题：{question}
AI回答：{answer}

请逐项检查以下四类事实，每项输出 PASS 或 FAIL（附简要理由）：

1. 动作建议：建议的训练动作是否与用户伤病/体态匹配？是否有禁忌动作被推荐？注意：只要出现了禁忌黑名单中的任何动作（包含变式），直接判定 FAIL。
2. 康复周期：给出的恢复时间/训练频率是否符合骨科康复常规？
3. 负重限制：建议的重量/强度是否适合该伤病阶段？是否有过量风险？
4. 伤病禁忌：是否遗漏了该伤病的已知禁忌注意事项？

严格按此格式输出（每行一项，不要编号）：
动作:PASS:理由（若PASS可简短）
康复:PASS:理由
负重:PASS:理由
禁忌:PASS:理由
整体:PASS（全部PASS）或整体:FAIL（任一FAIL）"""),
])


class FactCheckEngine:
    """用小模型做四类事实校验。资源隔离：不占用主 LLM 算力。"""

    def __init__(self, llm):
        """
        Args:
            llm: ChatOllama 实例（小模型，如 deepseek-r1:1.5b, temperature=0）
        """
        self._llm = llm
        self._chain = _FACT_CHECK_PROMPT | llm | StrOutputParser()

    def check(self, question: str, answer: str, context: str,
              contraindications: str = "",
              forbidden_actions: list[str] | None = None) -> FactCheckResult:
        """调用小模型 → 解析结构化输出 → 返回 FactCheckResult。

        Args:
            question: 用户原始问题
            answer: LLM 生成的回答
            context: 检索到的参考知识
            contraindications: 绝对禁忌黑名单文本
            forbidden_actions: 禁忌动作名的扁平列表（用于字符串预检）

        Returns:
            FactCheckResult — 异常时返回 PASS（不阻断回答流程）
        """
        # 拦截点 3：字符串预检 — 不依赖 LLM，确定性匹配
        if forbidden_actions:
            hits = [a for a in forbidden_actions if a in answer]
            if hits:
                return FactCheckResult(
                    overall="FAIL",
                    details={"动作": ("FAIL", f"回答中包含禁忌动作: {', '.join(hits)}")},
                    failed_categories=["动作"],
                )

        try:
            raw = self._chain.invoke({
                "question": question,
                "answer": answer,
                "context": context[:4000],  # 截断，小模型上下文有限
                "contraindications": contraindications or "（无）",
            })
            return self._parse(raw)
        except Exception:
            # 任何异常都 fallthrough：校验不阻断回答
            return FactCheckResult(overall="PASS")

    @staticmethod
    def _parse(raw: str) -> FactCheckResult:
        """解析 '动作:PASS:理由' 格式的输出。"""
        details = {}
        failures = []

        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue

            # 解析 "类别:判定:理由"
            parts = line.split(":", 2)
            if len(parts) < 2:
                continue

            cat = parts[0].strip()
            verdict = parts[1].strip().upper()
            reason = parts[2].strip() if len(parts) > 2 else ""

            # 标准化类别名
            cat_normalized = _normalize_category(cat)

            if cat_normalized and cat_normalized != "整体":
                details[cat_normalized] = (verdict, reason)
                if "FAIL" in verdict:
                    failures.append(cat_normalized)

            # 整体判定
            if cat_normalized == "整体" or "整体" in cat:
                details["整体"] = (verdict, reason)

        # 推导 overall：四类全部 PASS 且没有 FAIL
        if failures:
            overall = "FAIL"
        else:
            overall = details.get("整体", ("PASS", ""))[0]

        return FactCheckResult(
            overall=overall,
            details=details,
            failed_categories=failures,
        )


def _normalize_category(raw: str) -> str | None:
    """模糊匹配类别名到标准四类 + 整体。"""
    mapping = {
        "动作": "动作", "动作建议": "动作",
        "康复": "康复", "康复周期": "康复", "恢复": "康复",
        "负重": "负重", "负重限制": "负重", "重量": "负重", "强度": "负重",
        "禁忌": "禁忌", "伤病禁忌": "禁忌", "注意事项": "禁忌",
        "整体": "整体", "整体判定": "整体", "总体": "整体",
    }
    for key, val in mapping.items():
        if key in raw:
            return val
    return None
