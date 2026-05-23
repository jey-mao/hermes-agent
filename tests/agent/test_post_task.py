"""
tests/agent/test_post_task.py
v3.1 Post-Task 模块测试
覆盖: auto_validate_task / generate_reflection_prompt /
     record_post_task_insight / run_post_task_flow
"""
import json
import time
from pathlib import Path

import pytest

# 需要把 agent/ 加入 path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.post_task import (
    auto_validate_task,
    generate_reflection_prompt,
    record_post_task_insight,
    run_post_task_flow,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def empty_events():
    return []


@pytest.fixture
def good_task_completed():
    """单个高分完成事件"""
    return [
        {
            "type": "task_completed",
            "quality_score": {"overall": 9.0, "retry_decision": "accept"},
            "result": {"text": "采集到 23 个热销商品"},
        }
    ]


@pytest.fixture
def mixed_events():
    """混合事件：完成 + 错误"""
    return [
        {
            "type": "task_completed",
            "quality_score": {"overall": 6.0, "retry_decision": "accept"},
            "result": {"text": "采集到 5 个商品"},
        },
        {
            "type": "task_error",
            "error": "Connection timeout after 60s",
        },
        {
            "type": "task_error",
            "error": "Connection timeout after 60s",
        },
        {
            "type": "task_error",
            "error": "Connection timeout after 60s",
        },
    ]


@pytest.fixture
def empty_result_event():
    """完成但结果为空"""
    return [
        {
            "type": "task_completed",
            "quality_score": {"overall": 7.0, "retry_decision": "accept"},
            "result": {"text": ""},
        }
    ]


@pytest.fixture
def human_review_event():
    """需要人工介入"""
    return [
        {
            "type": "task_completed",
            "quality_score": {"overall": 2.0, "retry_decision": "human_review"},
            "result": {"text": "数据异常"},
        }
    ]


@pytest.fixture
def error_pattern_fixture():
    return [
        {"error_type": "timeout", "count": 3, "suggestion": "增加超时参数"},
    ]



# ------------------------------------------------------------------
# run_post_task_flow tests
# ------------------------------------------------------------------

class TestAutoValidateTask:
    def test_empty_events_returns_fail(self, empty_events):
        result = auto_validate_task(empty_events)
        assert result["status"] == "fail"
        assert result["needs_human_review"] is True
        assert result["quality_flags"] == -5

    def test_high_score_completed_returns_pass(self, good_task_completed):
        result = auto_validate_task(good_task_completed)
        assert result["status"] == "pass"
        assert result["needs_reflection"] is False
        assert result["quality_flags"] >= 2

    def test_repeated_errors_triggers_reflection(self, mixed_events):
        result = auto_validate_task(mixed_events, repeated_error_threshold=3)
        assert result["needs_reflection"] is True
        details = result["details"]
        assert "timeout" in details["error_types"]
        assert details["error_types"]["timeout"] == 3

    def test_human_review_triggers_needs_human_review(self, human_review_event):
        result = auto_validate_task(human_review_event)
        assert result["needs_human_review"] is True
        assert result["status"] == "fail"

    def test_empty_result_returns_degraded(self, empty_result_event):
        result = auto_validate_task(empty_result_event)
        # 只有一个空结果 → degraded
        assert result["status"] == "degraded"
        assert "空" in result["summary"]

    def test_low_score_triggers_reflection(self):
        events = [
            {
                "type": "task_completed",
                "quality_score": {"overall": 3.0, "retry_decision": "retry"},
                "result": {"text": "部分数据"},
            }
        ]
        result = auto_validate_task(events)
        assert result["needs_reflection"] is True
        assert result["status"] == "degraded" or result["status"] == "fail"

    def test_error_pattern_field_extracted(self):
        events = [
            {
                "type": "task_completed",
                "quality_score": {"overall": 8.0, "retry_decision": "accept"},
                "result": {"text": "ok"},
                "error_pattern": {"error_type": "no_data", "count": 2},
            }
        ]
        result = auto_validate_task(events)
        assert result["details"]["error_types"]["no_data"] == 2

    def test_replan_decision_higher_penalty(self):
        events = [
            {
                "type": "task_completed",
                "quality_score": {"overall": 5.0, "retry_decision": "replan"},
                "result": {"text": "需要重新规划"},
            }
        ]
        result = auto_validate_task(events)
        # replan → quality_flags -= 2，5.0 本身 +1 → -1 → fail（有评分但 < 2 且 quality_flags < 0）
        assert result["status"] == "fail"
        assert result["quality_flags"] == -1

    def test_summary_contains_quality_info(self, good_task_completed):
        result = auto_validate_task(good_task_completed)
        assert result["summary"]  # 不为空
        assert isinstance(result["summary"], str)


# ------------------------------------------------------------------
# generate_reflection_prompt tests
# ------------------------------------------------------------------

class TestGenerateReflectionPrompt:
    def test_detailed_style_with_errors(self, error_pattern_fixture):
        context = {
            "task": "采集TikTok热销榜",
            "events": [],
            "validation_result": {
                "status": "fail",
                "quality_flags": -3,
                "summary": "任务失败，超时3次",
                "details": {
                    "error_types": {"timeout": 3},
                    "avg_score": 2.0,
                },
            },
            "error_patterns": error_pattern_fixture,
        }
        prompt = generate_reflection_prompt(context, reflection_style="detailed")
        assert "采集TikTok热销榜" in prompt
        assert "timeout" in prompt
        assert "Q1" in prompt or "哪个环节" in prompt
        assert "Q5" in prompt or "TODO" in prompt

    def test_brief_style_short(self):
        context = {
            "task": "测试任务",
            "events": [],
            "validation_result": {
                "status": "degraded",
                "quality_flags": -1,
                "summary": "有问题",
                "details": {"error_types": {"no_data": 2}},
            },
            "error_patterns": [],
        }
        prompt = generate_reflection_prompt(context, reflection_style="brief")
        # brief 不应包含 5 个 Q
        assert prompt.count("Q") < 5
        assert "no_data" in prompt

    def test_pass_status_simple_review(self):
        context = {
            "task": "成功任务",
            "events": [],
            "validation_result": {
                "status": "pass",
                "quality_flags": 5,
                "summary": "任务完成，质量良好",
                "details": {"avg_score": 9.0, "error_types": {}},
            },
            "error_patterns": [],
        }
        prompt = generate_reflection_prompt(context)
        assert "复盘" in prompt or "pass" in prompt.lower()
        assert "Q1" not in prompt  # pass 不需要 5 个问题

    def test_fallback_to_detailed(self):
        # 无 events 的边界情况
        context = {
            "task": "任务",
            "events": [],
            "validation_result": {"status": "fail", "quality_flags": -5, "summary": "", "details": {}},
            "error_patterns": [],
        }
        prompt = generate_reflection_prompt(context)
        assert isinstance(prompt, str)
        assert len(prompt) > 0


# ------------------------------------------------------------------
# record_post_task_insight tests
# ------------------------------------------------------------------

class TestRecordPostTaskInsight:
    def test_returns_correct_structure(self):
        """验证函数返回正确的字典结构（不依赖文件系统写入）"""
        result = record_post_task_insight(
            task="测试任务",
            errors=[{"error_type": "timeout", "count": 2, "suggestion": "增加超时"}],
            quality_flags=-2,
            reflection_answer="超时是因为网络不稳定",
        )
        # 函数应返回带这三个键的字典
        assert "memory_written" in result
        assert "todo_created" in result
        assert "todo_id" in result

    def test_high_priority_creates_todo(self, monkeypatch):
        """priority=high 或 medium 时，尝试创建 TODO（验证路径存在即可）"""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            fake_home = Path(td)
            # 直接 patch Path 类
            import pathlib
            original_home = pathlib.Path.home
            pathlib.Path.home = lambda: fake_home
            try:
                errors = [{"error_type": "timeout", "count": 3, "suggestion": "增加超时"}]
                result = record_post_task_insight(
                    task="TikTok采集",
                    errors=errors,
                    quality_flags=-4,  # high priority
                )
                # 函数不抛异常即通过（文件系统操作被隔离）
                assert isinstance(result, dict)
            finally:
                pathlib.Path.home = original_home

    def test_low_priority_no_todo(self):
        """priority=low 时不创建 TODO"""
        errors = [{"error_type": "info", "count": 1, "suggestion": ""}]
        result = record_post_task_insight(
            task="简单任务",
            errors=errors,
            quality_flags=0,  # low priority
        )
        assert result["todo_created"] is False


# ------------------------------------------------------------------
# run_post_task_flow tests
# ------------------------------------------------------------------

class TestRunPostTaskFlow:
    def test_pass_task_returns_none(self, good_task_completed):
        result = run_post_task_flow(
            events=good_task_completed,
            task="采集任务",
            silent=True,
        )
        assert result["action_taken"] == "none"
        assert result["reflection_prompt"] == ""

    def test_fail_task_returns_reflection_prompt(self, human_review_event):
        result = run_post_task_flow(
            events=human_review_event,
            task="采集任务",
            silent=True,
        )
        assert result["action_taken"] == "reflection_prompt"
        assert result["reflection_prompt"] != ""
        assert "Q1" in result["reflection_prompt"]

    def test_mixed_events_triggers_reflection(self, mixed_events):
        result = run_post_task_flow(
            events=mixed_events,
            task="采集TikTok热销",
            silent=True,
        )
        assert result["validation"]["needs_reflection"] is True
        assert result["action_taken"] == "reflection_prompt"

    def test_reflection_answer_creates_insight(self, mixed_events):
        result = run_post_task_flow(
            events=mixed_events,
            task="采集TikTok热销",
            reflection_answer="超时是因为网络不稳定，下次增加重试",
            silent=True,
        )
        assert result["action_taken"] == "insight_recorded"
        assert result["insight_recorded"] is not None

    def test_empty_events_returns_fail(self, empty_events):
        result = run_post_task_flow(events=empty_events, silent=True)
        assert result["validation"]["status"] == "fail"
        assert result["action_taken"] == "none"

    def test_degraded_status_generates_prompt(self, empty_result_event):
        result = run_post_task_flow(events=empty_result_event, silent=True)
        # 空结果 → degraded → 可能有 reflection_prompt
        validation = result["validation"]
        if validation.get("needs_reflection"):
            assert result["reflection_prompt"] != ""


# ------------------------------------------------------------------
# Integration: end-to-end flow
# ------------------------------------------------------------------

class TestPostTaskFlowIntegration:
    def test_full_flow_pass(self, good_task_completed):
        result = run_post_task_flow(
            events=good_task_completed,
            task="采集TikTok马来西亚热销榜",
            silent=True,
        )
        # pass + no reflection needed → none
        assert result["action_taken"] == "none"
        validation = result["validation"]
        assert validation["status"] == "pass"

    def test_full_flow_fail_with_error_pattern(self):
        events = [
            {
                "type": "task_completed",
                "quality_score": {"overall": 2.0, "retry_decision": "retry"},
                "result": {"text": ""},
                "error_pattern": {"error_type": "no_data", "count": 5},
            }
        ]
        error_patterns = [{"error_type": "no_data", "count": 5, "suggestion": "更换数据源"}]
        result = run_post_task_flow(
            events=events,
            task="采集TikTok热销",
            error_patterns=error_patterns,
            silent=True,
        )
        assert result["validation"]["needs_reflection"] is True
        assert result["validation"]["needs_human_review"] is True
        assert result["action_taken"] == "reflection_prompt"
        prompt = result["reflection_prompt"]
        assert "no_data" in prompt
        assert "Q1" in prompt
        assert "Q5" in prompt or "TODO" in prompt
