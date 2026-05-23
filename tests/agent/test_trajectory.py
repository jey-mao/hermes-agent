"""
tests/agent/test_trajectory.py
v3.1 Trajectory 模块测试（Learning 进化层）
覆盖: analyze_trajectories / get_learning_insights / extract_task_intent /
     convert_scratchpad_to_think / has_incomplete_scratchpad / save_trajectory
"""
import json
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.trajectory import (
    analyze_trajectories,
    convert_scratchpad_to_think,
    extract_task_intent,
    get_learning_insights,
    has_incomplete_scratchpad,
    save_trajectory,
)


# ------------------------------------------------------------------
# convert_scratchpad_to_think tests
# ------------------------------------------------------------------

class TestConvertScratchpadToThink:
    def test_no_opens_returns_unchanged(self):
        content = "Hello world"
        assert convert_scratchpad_to_think(content) == content

    def test_empty_returns_empty(self):
        assert convert_scratchpad_to_think("") == ""
        assert convert_scratchpad_to_think(None) is None

    def test_converts_both_tags(self):
        content = "<REASONING_SCRATCHPAD>thinking</REASONING_SCRATCHPAD>"
        result = convert_scratchpad_to_think(content)
        assert "<think>" in result
        assert "</REASONING_SCRATCHPAD>" not in result
        assert "</think>" in result


# ------------------------------------------------------------------
# has_incomplete_scratchpad tests
# ------------------------------------------------------------------

class TestHasIncompleteScratchpad:
    def test_complete_returns_false(self):
        content = "<REASONING_SCRATCHPAD>thinking</REASONING_SCRATCHPAD>"
        assert has_incomplete_scratchpad(content) is False

    def test_open_only_returns_true(self):
        assert has_incomplete_scratchpad("<REASONING_SCRATCHPAD>thinking") is True

    def test_empty_returns_false(self):
        assert has_incomplete_scratchpad("") is False
        assert has_incomplete_scratchpad(None) is False

    def test_close_only_returns_false(self):
        # Only close tag, no open → not incomplete (open is what matters)
        assert has_incomplete_scratchpad("</REASONING_SCRATCHPAD>") is False


# ------------------------------------------------------------------
# save_trajectory tests
# ------------------------------------------------------------------

class TestSaveTrajectory:
    def test_saves_completed_to_default_file(self, tmp_path):
        traj = [{"role": "user", "content": "hello"}]
        out_file = tmp_path / "trajectory_samples.jsonl"
        save_trajectory(traj, "test-model", completed=True, filename=str(out_file))
        assert out_file.exists()
        with open(out_file) as f:
            entry = json.loads(f.readline())
        assert entry["model"] == "test-model"
        assert entry["completed"] is True
        assert entry["conversations"] == traj

    def test_saves_failed_to_default_file(self, tmp_path):
        traj = [{"role": "user", "content": "hello"}]
        out_file = tmp_path / "failed_trajectories.jsonl"
        save_trajectory(traj, "test-model", completed=False, filename=str(out_file))
        assert out_file.exists()
        with open(out_file) as f:
            entry = json.loads(f.readline())
        assert entry["completed"] is False

    def test_appends_not_overwrites(self, tmp_path):
        out_file = tmp_path / "trajectory.jsonl"
        save_trajectory([{"role": "user", "content": "first"}], "m1", completed=True, filename=str(out_file))
        save_trajectory([{"role": "user", "content": "second"}], "m2", completed=False, filename=str(out_file))
        with open(out_file) as f:
            lines = f.readlines()
        assert len(lines) == 2


# ------------------------------------------------------------------
# analyze_trajectories tests
# ------------------------------------------------------------------

class TestAnalyzeTrajectories:
    def test_empty_returns_zeros(self):
        result = analyze_trajectories([])
        assert result["total"] == 0
        assert result["success_count"] == 0
        assert result["failure_count"] == 0
        assert result["success_rate"] == 0.0

    def test_all_success(self):
        trajs = [
            {"completed": True, "message_count": 5},
            {"completed": True, "message_count": 10},
        ]
        result = analyze_trajectories(trajs)
        assert result["success_count"] == 2
        assert result["failure_count"] == 0
        assert result["success_rate"] == 1.0
        assert result["avg_success_message_count"] == 7.5

    def test_mixed_results(self):
        trajs = [
            {"completed": True, "message_count": 10},
            {"completed": False, "message_count": 20},
        ]
        result = analyze_trajectories(trajs)
        assert result["success_count"] == 1
        assert result["failure_count"] == 1
        assert result["success_rate"] == 0.5
        assert result["avg_success_message_count"] == 10.0
        assert result["avg_failure_message_count"] == 20.0
        assert result["avg_message_count"] == 15.0

    def test_missing_message_count_graceful(self):
        trajs = [
            {"completed": True},
            {"completed": True, "message_count": None},
        ]
        result = analyze_trajectories(trajs)
        assert result["avg_message_count"] == 0.0
        assert result["total"] == 2


# ------------------------------------------------------------------
# extract_task_intent tests
# ------------------------------------------------------------------

class TestExtractTaskIntent:
    def test_extracts_collection_action(self):
        action, platform = extract_task_intent("采集TikTok马来西亚热销榜")
        assert action == "采集"
        assert platform == "tiktok"

    def test_extracts_miaoshou_platform(self):
        action, platform = extract_task_intent("用妙手ERP采集数据")
        assert platform == "miaoshou"

    def test_extracts_1688(self):
        action, platform = extract_task_intent("1688找货源")
        assert platform == "1688"

    def test_default_platform_is_web(self):
        _, platform = extract_task_intent("随便浏览一下")
        assert platform == "web"

    def test_default_action_is_general(self):
        action, _ = extract_task_intent("ok好的")
        assert action == "通用"

    def test_search_action(self):
        action, _ = extract_task_intent("帮我搜索一下tiktok的数据")
        assert action == "搜索"

    def test_doc_action(self):
        action, _ = extract_task_intent("写一个选品报告")
        assert action == "文档"

    def test_command_action(self):
        action, _ = extract_task_intent("执行一个shell命令")
        assert action == "命令"


# ------------------------------------------------------------------
# get_learning_insights tests
# ------------------------------------------------------------------

class TestGetLearningInsights:
    def test_empty_inputs_returns_defaults(self):
        result = get_learning_insights([], [])
        assert "success_rate" in result
        assert "error_insights" in result
        assert "model_behavior" in result
        assert "top_failure_types" in result
        assert result["success_rate"] == 0.0
        # empty trajectories → "无执行数据"
        assert result["model_behavior"] == "无执行数据"

    def test_high_success_rate(self):
        trajs = [{"completed": True, "message_count": 10}]
        result = get_learning_insights([], trajs)
        assert result["success_rate"] == 1.0
        assert "高成功率" in result["model_behavior"]

    def test_low_success_rate(self):
        trajs = [
            {"completed": False, "message_count": 10},
            {"completed": False, "message_count": 10},
        ]
        result = get_learning_insights([], trajs)
        assert result["success_rate"] == 0.0
        assert result["model_behavior"] == "无成功案例，建议重新评估任务策略"

    def test_error_insights_enriches_patterns(self):
        errors = [
            {"error_type": "timeout", "count": 3, "level": "warn",
             "platform": "miaoshou", "task_type": "tiktok_hot", "suggestion": "增加超时"},
        ]
        result = get_learning_insights(errors, [])
        assert len(result["error_insights"]) == 1
        insight = result["error_insights"][0]
        assert insight["recommended_fix"] == "增加超时参数或检查网络连接"
        assert insight["pattern"] == "miaoshou::tiktok_hot::timeout"
        assert insight["count"] == 3

    def test_top_failure_types(self):
        errors = [
            {"error_type": "timeout", "count": 5, "level": "warn"},
            {"error_type": "no_data", "count": 3, "level": "warn"},
            {"error_type": "generic_err", "count": 1, "level": "warn"},
        ]
        result = get_learning_insights(errors, [])
        assert len(result["top_failure_types"]) == 3
        assert result["top_failure_types"][0]["type"] == "timeout"
        assert result["top_failure_types"][0]["count"] == 5

    def test_recommended_fixes_from_errors(self):
        errors = [
            {"error_type": "timeout", "count": 2, "level": "warn"},
            {"error_type": "no_data", "count": 2, "level": "warn"},
        ]
        result = get_learning_insights(errors, [])
        assert "增加超时参数或检查网络连接" in result["recommended_fixes"]
        assert "验证API参数或更换数据源" in result["recommended_fixes"]

    def test_no_errors_gives_default_fixes(self):
        trajs = [{"completed": True}]
        result = get_learning_insights([], trajs)
        assert "继续收集执行数据" in result["recommended_fixes"]

    def test_trajectory_analysis_included(self):
        trajs = [
            {"completed": True, "message_count": 10},
            {"completed": False, "message_count": 20},
        ]
        result = get_learning_insights([], trajs)
        ta = result["trajectory_analysis"]
        assert ta["success_count"] == 1
        assert ta["failure_count"] == 1
        assert ta["success_rate"] == 0.5

    def test_generated_at_timestamp(self):
        result = get_learning_insights([], [])
        assert "generated_at" in result
        assert isinstance(result["generated_at"], float)


# ------------------------------------------------------------------
# Integration: Learning full flow
# ------------------------------------------------------------------

class TestLearningFlowIntegration:
    def test_analyze_then_learn_full_flow(self):
        """analyze_trajectories → get_learning_insights 串联"""
        trajs = [
            {"completed": True, "message_count": 8},
            {"completed": True, "message_count": 12},
            {"completed": False, "message_count": 30},
        ]
        errors = [
            {"error_type": "timeout", "count": 4, "level": "suppress",
             "platform": "miaoshou", "task_type": "tiktok_hot", "suggestion": "增加超时"},
        ]
        insights = get_learning_insights(errors, trajs)
        assert insights["success_rate"] == pytest.approx(2 / 3)
        assert insights["trajectory_analysis"]["total"] == 3
        assert len(insights["top_failure_types"]) >= 1
        assert "timeout" in [t["type"] for t in insights["top_failure_types"]]
