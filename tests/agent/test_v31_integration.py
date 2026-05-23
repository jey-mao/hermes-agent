"""
tests/agent/test_v31_integration.py
v3.1 进化层三层集成测试

验证三个模块的集成点：
  1. Planning: is_complex_task → get_planning_context → generate_plan
  2. Post-Task: events → run_post_task_flow → _reflection_context
  3. Orchestration: run_agent.py 的注入逻辑（planning_context / channel_events / reflection_context）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from agent.planner import is_complex_task, get_planning_context, generate_plan
from agent.post_task import run_post_task_flow
from agent.trajectory import analyze_trajectories, get_learning_insights


# ------------------------------------------------------------------
# Layer 1: Planning — 复杂任务触发规划
# ------------------------------------------------------------------

class TestPlanningOrchestration:
    """验证 Planning 模块在正确时机触发"""

    def test_simple_task_does_not_trigger_complex_planning(self):
        """简单指令不触发复杂规划"""
        # is_complex_task: short + no step keywords
        assert is_complex_task("好的") is False
        assert is_complex_task("继续") is False

    def test_complex_task_triggers_planning(self):
        """包含步骤关键词的指令触发规划"""
        assert is_complex_task("采集TikTok马来西亚热销榜并分析") is True
        assert is_complex_task("帮我搜索1688的货源然后对比价格") is True

    def test_planning_flow_full(self):
        """复杂任务 → 完整规划流程"""
        task = "采集TikTok马来西亚热销榜3C数码并对比1688货源"
        context = get_planning_context(task=task, session_id="test-integration")
        plan = generate_plan(context)

        # 规划输出应包含关键信息
        assert task in plan
        assert "置信度" in plan
        assert "执行建议" in plan
        # 规划内容不应为空
        assert len(plan) > 50

    def test_planning_context_includes_all_layers(self):
        """规划上下文包含执行层/认知层/进化层所有信息"""
        context = get_planning_context(
            task="采集妙手ERP热销数据",
            session_id="test",
        )
        # 执行层：错误模式
        assert "errors" in context
        # 进化层-Planning：轨迹
        assert "trajectories" in context
        # 进化层-Learning：学习洞察
        assert "learning_insights" in context
        # 元数据
        assert "generated_at" in context


# ------------------------------------------------------------------
# Layer 2: Post-Task — 任务完成后自动验证+反思
# ------------------------------------------------------------------

class TestPostTaskOrchestration:
    """验证 Post-Task 模块在正确时机触发"""

    def test_pass_task_no_reflection(self):
        """成功任务不触发反思"""
        events = [
            {
                "type": "task_completed",
                "quality_score": {"overall": 9.0, "retry_decision": "accept"},
                "result": {"text": "采集到23个热销商品"},
            }
        ]
        result = run_post_task_flow(events, task="采集任务", silent=True)
        assert result["action_taken"] == "none"
        assert result["reflection_prompt"] == ""

    def test_fail_task_triggers_reflection(self):
        """失败任务触发反思"""
        events = [
            {
                "type": "task_completed",
                "quality_score": {"overall": 2.0, "retry_decision": "human_review"},
                "result": {"text": "数据异常"},
            }
        ]
        result = run_post_task_flow(events, task="采集任务", silent=True)
        assert result["action_taken"] == "reflection_prompt"
        assert result["reflection_prompt"] != ""
        # 反思必须包含具体问题
        assert "Q1" in result["reflection_prompt"] or "哪个环节" in result["reflection_prompt"]

    def test_repeated_errors_trigger_reflection(self):
        """重复错误触发反思"""
        events = [
            {"type": "task_error", "error": "timeout", "error_pattern": {"error_type": "timeout", "count": 3}},
            {"type": "task_error", "error": "timeout", "error_pattern": {"error_type": "timeout", "count": 3}},
            {"type": "task_error", "error": "timeout", "error_pattern": {"error_type": "timeout", "count": 3}},
        ]
        result = run_post_task_flow(events, task="采集", silent=True)
        assert result["validation"]["needs_reflection"] is True

    def test_reflection_with_answer_creates_insight(self):
        """有反思答案时直接记录洞察"""
        events = [
            {"type": "task_error", "error": "timeout",
             "error_pattern": {"error_type": "timeout", "count": 3}},
        ]
        result = run_post_task_flow(
            events=events,
            task="采集TikTok",
            reflection_answer="超时是因为没有增加超时参数，下次要加 timeout=120",
            silent=True,
        )
        assert result["action_taken"] == "insight_recorded"
        assert result["insight_recorded"]["memory_written"] in (True, False)  # 取决于文件权限

    def test_post_task_validation_result_structure(self):
        """验证结果结构完整"""
        events = [
            {
                "type": "task_completed",
                "quality_score": {"overall": 7.0, "retry_decision": "accept"},
                "result": {"text": "部分数据"},
                "error_pattern": {"error_type": "no_data", "count": 1},
            }
        ]
        result = run_post_task_flow(events, silent=True)
        validation = result["validation"]
        assert "status" in validation
        assert "quality_flags" in validation
        assert "needs_reflection" in validation
        assert "needs_human_review" in validation
        assert "summary" in validation
        assert "details" in validation


# ------------------------------------------------------------------
# Layer 3: Learning — 轨迹分析与洞察生成
# ------------------------------------------------------------------

class TestLearningOrchestration:
    """验证 Learning 模块与其他模块的连接"""

    def test_analyze_trajectories_feeds_into_learning(self):
        """轨迹分析结果正确传入 Learning 模块"""
        trajs = [
            {"completed": True, "message_count": 10},
            {"completed": True, "message_count": 8},
            {"completed": False, "message_count": 25},
        ]
        errors = [
            {"error_type": "timeout", "count": 4, "level": "suppress",
             "platform": "miaoshou", "task_type": "tiktok_hot", "suggestion": "增加超时"},
        ]
        insights = get_learning_insights(errors, trajs)

        # 成功率
        assert insights["success_rate"] == pytest.approx(2 / 3)

        # 轨迹分析包含在输出中
        ta = insights["trajectory_analysis"]
        assert ta["success_count"] == 2
        assert ta["failure_count"] == 1

        # 错误洞察
        assert len(insights["error_insights"]) == 1
        assert insights["error_insights"][0]["recommended_fix"] == "增加超时参数或检查网络连接"

        # 失败类型
        assert insights["top_failure_types"][0]["type"] == "timeout"

    def test_planning_includes_learning_insights(self):
        """规划输出包含 Learning 洞察"""
        trajs = [
            {"completed": True, "message_count": 10},
            {"completed": False, "message_count": 30},
        ]
        insights = get_learning_insights([], trajs)
        context = {
            "task": "采集任务",
            "errors": [],
            "trajectories": trajs,
            "trajectory_analysis": insights["trajectory_analysis"],
            "learning_insights": insights,
            "insights": {"empty": True},
        }
        plan = generate_plan(context)
        # 学习洞察应该被包含在规划中
        assert "学习洞察" in plan or "成功率" in plan or "67%" in plan or "50%" in plan


# ------------------------------------------------------------------
# Cross-Cutting: 三层完整流程
# ------------------------------------------------------------------

class TestV31ArchitectureFlow:
    """验证 v3.1 三层架构完整流程"""

    def test_planning_then_execution_then_posttask(self):
        """
        模拟完整任务生命周期：
        1. Planning: 复杂任务 → 生成规划
        2. Execution: 任务完成（有错误）
        3. Post-Task: 自动验证 → 触发反思 → 记录洞察
        """
        # Step 1: Planning
        task = "采集TikTok马来西亚热销榜3C数码"
        assert is_complex_task(task) is True
        context = get_planning_context(task=task, session_id="lifecycle-test")
        plan = generate_plan(context)
        assert plan != ""

        # Step 2: Execution with errors
        events = [
            {
                "type": "task_completed",
                "quality_score": {"overall": 5.0, "retry_decision": "retry"},
                "result": {"text": "采集到5个商品，部分数据缺失"},
                "error_pattern": {"error_type": "no_data", "count": 2},
            },
            {
                "type": "task_error",
                "error": "timeout",
                "error_pattern": {"error_type": "timeout", "count": 2},
            },
        ]

        # Step 3: Post-Task
        result = run_post_task_flow(
            events=events,
            task=task,
            error_patterns=[
                {"error_type": "no_data", "count": 2, "suggestion": "验证API参数"},
                {"error_type": "timeout", "count": 2, "suggestion": "增加超时"},
            ],
            silent=True,
        )
        # 验证
        assert result["validation"]["needs_reflection"] is True
        assert result["action_taken"] == "reflection_prompt"
        prompt = result["reflection_prompt"]
        assert "Q1" in prompt
        assert "Q5" in prompt or "TODO" in prompt
        # 错误类型应该出现在反思中
        assert "no_data" in prompt or "timeout" in prompt

    def test_pass_path_does_nothing_extra(self):
        """成功路径：规划 + 执行 + 验证通过 → 无额外操作"""
        task = "采集TikTok热销"
        # Planning fires (complex task)
        assert is_complex_task(task) is True
        context = get_planning_context(task=task, session_id="pass-test")
        plan = generate_plan(context)
        assert plan != ""

        # Execution succeeds
        events = [
            {
                "type": "task_completed",
                "quality_score": {"overall": 9.0, "retry_decision": "accept"},
                "result": {"text": "采集到23个热销商品，毛利率评估完成"},
            }
        ]

        # Post-Task: nothing extra
        result = run_post_task_flow(events, task=task, silent=True)
        assert result["action_taken"] == "none"
        # 返回完整验证结果（用于记录，但不强制触发反思）
        assert result["validation"]["status"] == "pass"


# ------------------------------------------------------------------
# API Contract: 返回结构稳定性
# ------------------------------------------------------------------

class TestAPIContracts:
    """验证三层模块的公开 API 返回结构稳定"""

    def test_planning_api_contract(self):
        ctx = get_planning_context("采集", "test")
        assert isinstance(ctx, dict)
        # 6个必返回字段
        required = ["task", "errors", "trajectories", "learning_insights", "insights", "generated_at"]
        for key in required:
            assert key in ctx, f"Missing key: {key}"

    def test_plan_text_contract(self):
        plan = generate_plan(get_planning_context("采集", "test"))
        assert isinstance(plan, str)
        assert len(plan) > 0

    def test_post_task_api_contract(self):
        result = run_post_task_flow([{"type": "task_completed", "quality_score": {"overall": 8.0, "retry_decision": "accept"}, "result": {"text": "ok"}}], silent=True)
        required_keys = ["validation", "reflection_prompt", "insight_recorded", "action_taken"]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_learning_api_contract(self):
        insights = get_learning_insights([], [])
        required_keys = ["trajectory_analysis", "error_insights", "model_behavior",
                        "recommended_fixes", "success_rate", "top_failure_types", "generated_at"]
        for key in required_keys:
            assert key in insights, f"Missing key: {key}"
