"""
tests/agent/test_planner.py
v3.1 Planning 模块测试
覆盖: get_planning_context / generate_plan / is_complex_task /
     _extract_platform_hint / _compute_confidence
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.planner import (
    is_complex_task,
    _extract_platform_hint,
    _compute_confidence,
    get_planning_context,
    generate_plan,
)


# ------------------------------------------------------------------
# is_complex_task tests
# ------------------------------------------------------------------

class TestIsComplexTask:
    def test_empty_returns_false(self):
        assert is_complex_task("") is False
        assert is_complex_task("   ") is False

    def test_short_single_keyword_returns_true(self):
        # 采集 is in step_keywords
        assert is_complex_task("采集数据") is True

    def test_long_instruction_always_complex(self):
        # > 40 chars
        assert is_complex_task("请帮我分析一下今天采集到的所有热销商品数据并整理成表格") is True

    def test_multi_step_keywords(self):
        assert is_complex_task("先采集再分析最后导出") is True

    def test_simple_short_returns_false(self):
        # short + no keywords
        assert is_complex_task("ok") is False
        assert is_complex_task("好的") is False
        assert is_complex_task("继续") is False


# ------------------------------------------------------------------
# _extract_platform_hint tests
# ------------------------------------------------------------------

class TestExtractPlatformHint:
    def test_miaoshou(self):
        assert _extract_platform_hint("采集妙手ERP热销榜") == "miaoshou"
        assert _extract_platform_hint("用erp采集") == "miaoshou"

    def test_tiktok(self):
        assert _extract_platform_hint("TikTok马来西亚热销") == "tiktok"
        assert _extract_platform_hint("抖音选品") == "tiktok"

    def test_1688(self):
        assert _extract_platform_hint("1688货源采集") == "1688"
        assert _extract_platform_hint("alibaba找货") == "1688"

    def test_xiaohongshu(self):
        assert _extract_platform_hint("小红书数据") == "xiaohongshu"
        assert _extract_platform_hint("xhs选品") == "xiaohongshu"

    def test_shopee(self):
        assert _extract_platform_hint("Shopee热销") == "shopee"
        assert _extract_platform_hint("虾皮数据采集") == "shopee"

    def test_no_match(self):
        assert _extract_platform_hint("随便采集一些数据") is None


# ------------------------------------------------------------------
# _compute_confidence tests
# ------------------------------------------------------------------

class TestComputeConfidence:
    def test_empty_returns_medium(self):
        # Empty trajectories + empty errors + empty insights = score 0 → else → "medium"
        result = _compute_confidence([], [], {})
        assert result == "medium"

    def test_many_trajectories_high_confidence(self):
        # 4 trajectories: +1 (any) +1 (>=3) = 2, empty insights (+1) = 3 → high
        trajectories = [{}, {}, {}, {}]
        result = _compute_confidence([], trajectories, {})  # {} = empty = not empty after get("empty")
        assert result == "high"

    def test_suppress_errors_reduce_confidence(self):
        # 1 trajectory → +1, suppress errors → -1, empty insights → score = 0 → medium
        trajectories = [{}]
        errors = [{"level": "suppress"}]
        result = _compute_confidence(errors, trajectories, {})
        # score = 0 → medium
        assert result == "medium"

    def test_high_success_rate_boosts(self):
        trajectories = [{}, {}]
        insights = {"empty": False}
        learning = {"success_rate": 0.85}
        # 2 trajs → +1, >=3 → +1 = 2, insights +1 = 3, high SR +1 = 4 → high
        result = _compute_confidence([], trajectories, insights, learning)
        assert result == "high"

    def test_low_success_rate_penalizes(self):
        trajectories = [{}]
        insights = {"empty": False}
        learning = {"success_rate": 0.2}
        # 1 traj → +1, insights +1, low SR -1 = 1 → medium
        result = _compute_confidence([], trajectories, insights, learning)
        assert result == "medium"

    def test_many_errors_and_trajectories(self):
        # 2 trajs → score +1, 3 errors (suppress>=1) → score -1, >=3 errors → -1, insights → +1
        # Score = 0 → low
        trajectories = [{}, {}]
        errors = [{"level": "warn"}, {"level": "warn"}, {"level": "suppress"}]
        insights = {"empty": False}
        result = _compute_confidence(errors, trajectories, insights)
        assert result == "low"

    def test_threshold_high(self):
        # 3 trajs (+1,+1) + insights (+1) + high SR (+1) = 4 → high
        trajectories = [{}, {}, {}]
        insights = {"empty": False}
        learning = {"success_rate": 0.9}
        result = _compute_confidence([], trajectories, insights, learning)
        assert result == "high"


# ------------------------------------------------------------------
# get_planning_context tests
# ------------------------------------------------------------------

class TestGetPlanningContext:
    def test_returns_required_keys(self):
        result = get_planning_context(
            task="采集TikTok热销",
            session_id="test-session",
        )
        assert "task" in result
        assert "errors" in result
        assert "trajectories" in result
        assert "learning_insights" in result
        assert "insights" in result
        assert "generated_at" in result
        assert result["task"] == "采集TikTok热销"

    def test_platform_hint_extracted(self):
        # No exception when platform hint is extracted
        result = get_planning_context(
            task="妙手ERP采集tiktok马来西亚热销",
            session_id="test",
        )
        assert isinstance(result["errors"], list)

    def test_empty_trajectories_graceful(self):
        result = get_planning_context(
            task="简单任务",
            session_id="test",
            max_trajectories=0,
        )
        assert isinstance(result, dict)
        assert result["trajectories"] == []


# ------------------------------------------------------------------
# generate_plan tests
# ------------------------------------------------------------------

class TestGeneratePlan:
    def test_empty_context_returns_empty_string(self):
        result = generate_plan({})
        assert result == ""

    def test_basic_plan_structure(self):
        context = {
            "task": "采集TikTok热销榜",
            "errors": [],
            "trajectories": [],
            "learning_insights": {"success_rate": 0.0},
            "insights": {"empty": True},
        }
        result = generate_plan(context)
        assert "采集TikTok热销榜" in result
        assert "置信度" in result
        assert "执行建议" in result

    def test_plan_with_errors(self):
        context = {
            "task": "采集",
            "errors": [
                {
                    "platform": "miaoshou",
                    "task_type": "tiktok_hot",
                    "error_type": "timeout",
                    "count": 3,
                    "level": "warn",
                    "suggestion": "增加超时参数",
                }
            ],
            "trajectories": [],
            "learning_insights": {},
            "insights": {"empty": True},
        }
        result = generate_plan(context)
        assert "历史风险警告" in result
        assert "timeout" in result
        assert "增加超时参数" in result

    def test_plan_with_trajectories(self):
        context = {
            "task": "采集任务",
            "errors": [],
            "trajectories": [
                {"timestamp": "2026-05-21T10:00:00", "completed": True, "message_count": 10},
                {"timestamp": "2026-05-21T11:00:00", "completed": False, "message_count": 25},
            ],
            "learning_insights": {"success_rate": 0.5},
            "insights": {"empty": True},
        }
        result = generate_plan(context)
        assert "最近执行轨迹" in result
        assert "✅" in result or "❌" in result

    def test_plan_with_learning_insights(self):
        context = {
            "task": "采集",
            "errors": [],
            "trajectories": [],
            "trajectory_analysis": {"success_count": 2, "failure_count": 1},
            "learning_insights": {
                "success_rate": 0.667,
                "model_behavior": "模型倾向于过度思考",
                "recommended_fixes": ["简化任务描述"],
                "top_failure_types": [{"type": "timeout", "count": 3}],
            },
            "insights": {"empty": True},
        }
        result = generate_plan(context)
        assert "学习洞察" in result
        assert "67%" in result
        assert "模型状态" in result

    def test_plan_confidence_icon(self):
        # empty → low → 🔴
        context = {
            "task": "采集",
            "errors": [],
            "trajectories": [],
            "learning_insights": {},
            "insights": {"empty": True},
        }
        result = generate_plan(context)
        assert "🔴" in result
        assert "low" in result

    def test_plan_uses_insights_when_available(self):
        context = {
            "task": "采集",
            "errors": [],
            "trajectories": [],
            "learning_insights": {},
            "insights": {
                "empty": False,
                "overview": {"total_sessions": 20, "total_cost_usd": 1.5},
                "platforms": [{"source": "feishu", "session_count": 15}],
            },
        }
        result = generate_plan(context)
        assert "使用统计" in result or "近7天" in result or "20" in result


# ------------------------------------------------------------------
# Integration: full planning flow
# ------------------------------------------------------------------

class TestPlanningFlowIntegration:
    def test_context_to_plan_full_flow(self):
        """get_planning_context → generate_plan 串联"""
        task = "采集TikTok马来西亚热销榜3C数码"
        context = get_planning_context(task=task, session_id="integration-test")
        plan = generate_plan(context)

        assert plan != ""
        assert task in plan
        # Should have at least: header + risk section + exec suggestion + confidence
        assert plan.count("##") >= 1
