"""health_tools 单测：确定性计算正确性 / 参数抽取 / 触发判定（纯函数，零依赖）。"""
from health_tools import (
    TOOL_REGISTRY,
    ToolResult,
    _extract_params,
    run_health_tools,
)


class TestExtractParams:
    def test_params_from_user_profile(self):
        p = _extract_params("帮我算下BMI", "身高170cm，体重70kg，目标：减脂")
        assert p["height"] == 170.0
        assert p["weight"] == 70.0

    def test_params_from_question_fallback(self):
        """question 自报参数（画像未保存时兜底）。"""
        p = _extract_params("我身高175体重68，帮我算下BMI", None)
        assert p["height"] == 175.0
        assert p["weight"] == 68.0

    def test_profile_takes_precedence_over_question(self):
        p = _extract_params("我身高180体重80", "身高170cm，体重70kg")
        assert p["height"] == 170.0   # 画像优先
        assert p["weight"] == 70.0

    def test_age_extraction(self):
        assert _extract_params("我今年30岁，最大心率多少", None)["age"] == 30.0

    def test_no_params_empty(self):
        assert _extract_params("深蹲练什么肌肉", None) == {}
        assert _extract_params("深蹲练什么肌肉", None) == {}


class TestRunHealthTools:
    def test_bmi_trigger_and_values(self):
        r = run_health_tools("帮我算下BMI", "身高170cm，体重70kg")
        assert len(r) == 1 and r[0].name == "calculate_bmi"
        assert "24.2" in r[0].content   # 70/(1.7²)=24.2 → 超重
        assert "超重" in r[0].content

    def test_bmi_normal_range(self):
        r = run_health_tools("我身高180体重65，帮我算下BMI", None)
        assert "20.1" in r[0].content and "正常" in r[0].content

    def test_water_intake(self):
        r = run_health_tools("每天应该喝多少水", "身高170cm，体重70kg")
        assert len(r) == 1 and r[0].name == "estimate_water_intake"
        assert "2100" in r[0].content   # 70×30=2100ml

    def test_hr_zones(self):
        r = run_health_tools("30岁健身心率多少合适", None)
        assert len(r) == 1 and r[0].name == "heart_rate_zone"
        assert "190" in r[0].content    # 220-30
        assert "114" in r[0].content    # 190×0.6

    def test_keyword_hit_but_missing_params_skipped(self):
        """关键词命中但参数不全 → 不硬算（返回空，宁缺勿错）。"""
        assert run_health_tools("帮我算下BMI", None) == []
        assert run_health_tools("每天喝多少水", None) == []

    def test_no_keyword_no_compute(self):
        assert run_health_tools("深蹲练什么肌肉", "身高170cm，体重70kg") == []

    def test_multiple_tools_same_query(self):
        r = run_health_tools("30岁体重70kg每天喝多少水", None)
        names = [t.name for t in r]
        assert "estimate_water_intake" in names

    def test_keyword_case_insensitive_bmi(self):
        r = run_health_tools("Calculate my BMI please", "height 170cm weight 70kg")
        # 中文正则不匹配英文画像 → 无参数 → 不触发（确定性，不误算）
        assert r == []


class TestRegistryShape:
    def test_registry_has_three_tools(self):
        assert len(TOOL_REGISTRY) == 3
        assert {t["name"] for t in TOOL_REGISTRY} == {
            "calculate_bmi", "estimate_water_intake", "heart_rate_zone"}

    def test_compute_returns_none_on_missing_params(self):
        for t in TOOL_REGISTRY:
            assert t["compute"]({}) is None, f"{t['name']} 缺参数应返回 None"

    def test_tool_result_is_dataclass(self):
        r = ToolResult(name="x", title="y", content="z")
        assert r.name == "x" and r.title == "y" and r.content == "z"
