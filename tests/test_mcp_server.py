"""mcp_server 单测：工具注册 / 禁忌判定 / 确定性计算（不启服务、不依赖 Milvus）。"""
import asyncio

import mcp_server as m


# ================================================================
# 工具注册
# ================================================================

class TestToolRegistration:
    def _tools(self):
        return asyncio.run(m.server.list_tools())

    def test_all_expected_tools_exposed(self):
        names = {t.name for t in self._tools()}
        assert names == {
            "search_knowledge_base",
            "check_contraindication",
            "get_injury_graph",
            "calculate_bmi",
            "estimate_water_intake",
            "heart_rate_zone",
        }

    def test_every_tool_has_description(self):
        """MCP 客户端靠 description 决定何时调用——缺了工具就形同不存在。"""
        for t in self._tools():
            assert t.description and t.description.strip(), t.name


# ================================================================
# 禁忌判定（项目独家能力，零外部依赖）
# ================================================================

class TestCheckContraindication:
    def test_contraindicated_action(self):
        out = m.check_contraindication("腰突", "硬拉")
        assert "禁忌动作" in out
        assert "腰间盘突出" in out          # 别名为全称
        assert "压迫腰椎" in out            # 带具体原因

    def test_alias_resolution(self):
        """别名与全称应得到同一结果。"""
        a = m.check_contraindication("腰突", "硬拉")
        b = m.check_contraindication("腰间盘突出", "硬拉")
        assert "禁忌动作" in a and "禁忌动作" in b

    def test_substring_action_match(self):
        """动作名互为子串时也应命中（「深蹲」vs「负重深蹲」）。"""
        out = m.check_contraindication("半月板损伤", "深蹲")
        assert "禁忌动作" in out

    def test_unknown_injury_is_honest(self):
        """未收录的伤病必须明说「暂无记录」，并提示不代表安全。"""
        out = m.check_contraindication("不存在的伤病", "深蹲")
        assert "暂无" in out
        assert "不代表" in out and "安全" in out

    def test_known_injury_unknown_action_lists_alternatives(self):
        out = m.check_contraindication("腰突", "完全不相干的动作")
        assert "没有与" in out
        assert "不等于安全" in out


# ================================================================
# 确定性健康计算
# ================================================================

class TestHealthCalculators:
    def test_bmi(self):
        out = m.calculate_bmi(height_cm=175, weight_kg=75)
        assert "24.5" in out and "超重" in out      # 75/(1.75²)=24.49

    def test_bmi_missing_params_is_explicit(self):
        out = m.calculate_bmi(weight_kg=75)
        assert "参数不足" in out

    def test_water_intake(self):
        out = m.estimate_water_intake(weight_kg=75)
        assert "2250" in out                        # 75×30

    def test_heart_rate_zone(self):
        out = m.heart_rate_zone(age=30)
        assert "190" in out                         # 220-30
        assert "燃脂" in out

    def test_calculators_are_deterministic(self):
        """同样输入必须同样输出（项目对健康数值的核心要求）。"""
        assert m.calculate_bmi(height_cm=175, weight_kg=75) == \
               m.calculate_bmi(height_cm=175, weight_kg=75)


# ================================================================
# 图谱工具在未启用时的降级
# ================================================================

class TestGraphTool:
    def test_disabled_neo4j_returns_guidance(self, monkeypatch):
        """图谱未启用时应给出明确指引，而不是抛异常。"""
        monkeypatch.setattr(m, "NEO4J_ENABLED", False)
        out = m.get_injury_graph("腰突")
        assert "未启用" in out
        assert "check_contraindication" in out      # 指明替代方案

    def test_unreachable_graph_returns_readable_error(self, monkeypatch):
        """图谱连不上时不抛异常，返回可读错误 + 替代路径。"""
        monkeypatch.setattr(m, "NEO4J_ENABLED", True)
        monkeypatch.setattr(m, "NEO4J_URI", "bolt://127.0.0.1:1")   # 必然连不上
        out = m.get_injury_graph("腰突")
        assert "图谱查询失败" in out
        assert "check_contraindication" in out
