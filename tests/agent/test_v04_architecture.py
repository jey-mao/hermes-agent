"""
Hermes v0.4 架构测试
测试三层：规划层、认知层、进化层
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from agent.task_planner import TaskPlanner, TaskSignature, Step, ExecutionPlan
from agent.self_critic import SelfCritic, StepResult
from agent.evolution_engine import EvolutionEngine, TRAJ_DIR, TRAJ_FILE
import json


class FakeLLM:
    def generate(self, prompt, schema=None):
        if "历史最佳" in prompt:
            return json.dumps([{"step_id":"s1","description":"复用历史方案第1步","tool":"browser_navigate","action":"搜索","expected":"搜索结果"}])
        return json.dumps([
            {"step_id":"s1","description":"搜索TikTok马来西亚热销耳机","tool":"browser_navigate","action":"搜索","expected":"热销商品列表"},
            {"step_id":"s2","description":"分析竞争情况","tool":"terminal","action":"分析","expected":"竞争分析结果"},
            {"step_id":"s3","description":"输出最终选品建议","tool":"terminal","action":"输出","expected":"3个具体选品"},
        ])


# ─── 测试1：TaskSignature 提取 ──────────────────────────────────────────────
def test_task_signature_extraction():
    planner = TaskPlanner()

    sig1 = planner._extract_signature("帮我搜索1688蓝牙耳机货源，马来西亚市场")
    assert sig1.domain == "1688_货源采集"
    assert sig1.action == "搜索"
    assert "蓝牙耳机" in sig1.constraints
    assert "马来西亚" in sig1.constraints
    assert len(sig1.hash) == 8

    sig2 = planner._extract_signature("TikTok东南亚电商选品分析")
    assert sig2.domain == "TikTok_电商选品"
    assert sig2.action == "分析"

    sig3 = planner._extract_signature("帮我看看GitHub上的某个repo")
    assert sig3.domain == "GitHub_开发"
    print("✅ TaskSignature 提取正确")


# ─── 测试2：规划层 — 生成步骤列表 ──────────────────────────────────────────
def test_planner_generates_steps():
    planner = TaskPlanner(llm_client=FakeLLM())

    plan = planner.plan("帮我搜索TikTok马来西亚热销耳机")
    assert isinstance(plan, ExecutionPlan)
    assert len(plan.steps) >= 3
    assert plan.status == "planning"
    assert plan.task_sig is not None
    assert plan.task_sig.domain == "TikTok_电商选品"

    # 每个 step 有完整字段
    for step in plan.steps:
        assert step.step_id
        assert step.description
        assert step.tool in ("browser_navigate","terminal","lingguang_execute","search_files")
        assert step.action
        assert step.expected
    print(f"✅ 规划层生成 {len(plan.steps)} 个步骤")


# ─── 测试3：规划层 — 历史复用（评分≥7时）─────────────────────────────────
def test_planner_reuses_best_when_score_high():
    ev = EvolutionEngine()
    # 写入一条高分记录（评分8.0）
    ev.record_simple(
        task_sig_hash="a3f2b1c0", task_sig_domain="TikTok_电商选品",
        task_sig_action="分析", task_sig_constraints="马来西亚耳机",
        prompt="TikTok马来西亚热销耳机分析", score=8.0,
        evaluation={"scores":{}}, steps=[{"id":"s1","score":8.0}],
        duration_ms=5000, result_summary="成功")

    sig = TaskSignature(domain="TikTok_电商选品", action="分析",
                        constraints="马来西亚耳机", hash="a3f2b1c0")
    best = ev.find_similar(sig)
    assert best is not None
    assert best["score"] == 8.0
    print("✅ 进化层检索到历史 best (score=8.0)")


# ─── 测试4：认知层 — 评分计算 ───────────────────────────────────────────────
def test_critic_scoring():
    critic = SelfCritic()

    # 正常成功结果
    r_ok = StepResult(
        step_id="s1", description="搜索1688", tool="browser_navigate",
        action="搜索", raw_output="找到50个商品，URL已获取", success=True, duration_ms=5000)
    ev = critic.evaluate(r_ok, expected="找到50个商品")
    assert ev.score >= 7.0, f"总分{ev.score}低于7.0（预期关键词'找到50个商品'已匹配）"
    assert ev.retry_decision in ("accept","retry"), f"评分{ev.score}应通过或重试，实际{ev.retry_decision}"
    print(f"✅ 正常结果评分={ev.score}, 决策={ev.retry_decision}")

    # 网络超时
    r_timeout = StepResult(
        step_id="s2", description="访问网站", tool="browser_navigate",
        action="访问", raw_output="", success=False, error_type="timeout", duration_ms=30000)
    ev_t = critic.evaluate(r_timeout)
    assert ev_t.score < 5.0
    assert ev_t.retry_decision in ("retry","replan")
    assert ev_t.classification == "timeout"
    print(f"✅ 超时评分={ev_t.score}, 分类={ev_t.classification}, 决策={ev_t.retry_decision}")

    # 空结果
    r_empty = StepResult(
        step_id="s3", description="搜索货源", tool="terminal",
        action="搜索", raw_output="没有找到任何结果", success=True, duration_ms=2000)
    ev_e = critic.evaluate(r_empty)
    assert ev_e.retry_decision in ("retry","replan","human_review")
    assert "没有" in " ".join(ev_e.weaknesses) or len(ev_e.weaknesses) > 0
    print(f"✅ 空结果评分={ev_e.score}, 弱点={ev_e.weaknesses}")


# ─── 测试5：认知层 — 失败分类 ────────────────────────────────────────────────
def test_critic_classification():
    critic = SelfCritic()
    cases = [
        ("timeout", "timeout"),
        ("connection refused", "network"),
        ("tool not found", "tool"),
        ("permission denied", "auth"),
        ("no results found", "quality"),
    ]
    for err, expected in cases:
        r = StepResult(step_id="test", description="test", tool="terminal",
                       action="test", raw_output="", success=False, error_type=err)
        ev = critic.evaluate(r)
        assert ev.classification == expected, f"{err} -> {ev.classification} != {expected}"
    print("✅ 失败分类全部正确（timeout/network/tool/auth/quality）")


# ─── 测试6：进化层 — 写入 + 检索 ───────────────────────────────────────────
def test_evolution_engine_write_and_retrieve():
    ev = EvolutionEngine()

    ev.record_simple(
        task_sig_hash="b4c5d6e0", task_sig_domain="1688_货源采集",
        task_sig_action="采集", task_sig_constraints="蓝牙耳机",
        prompt="采集1688蓝牙耳机货源",
        score=7.5, evaluation={"scores":{"s1":7.5}},
        steps=[{"id":"s1","desc":"采集","score":7.5}],
        duration_ms=8000, result_summary="采集到50条货源")

    stats = ev.stats()
    assert stats["total"] >= 1
    assert stats["best_score"] >= 7.0
    assert "1688_货源采集" in stats["domains"]
    print(f"✅ 进化层统计: {stats}")

    # 检索
    sig = TaskSignature(domain="1688_货源采集", action="采集",
                         constraints="蓝牙耳机", hash="b4c5d6e0")
    best = ev.find_similar(sig)
    assert best is not None
    assert best["score"] == 7.5
    print("✅ 进化层检索到 1688_货源采集 的记录")


# ─── 测试7：重规划 — 单步失败只重规划该步 ──────────────────────────────────
def test_replan_single_step():
    planner = TaskPlanner()
    sig = TaskSignature(domain="browser_信息搜索", action="搜索",
                        constraints="无特定约束")
    plan = ExecutionPlan(
        plan_id="test-replan",
        task_sig=sig,
        steps=[
            Step(step_id="s1", description="搜索网页", tool="browser_navigate",
                 action="搜索", expected="结果"),
            Step(step_id="s2", description="分析结果", tool="terminal",
                 action="分析", expected="分析报告"),
        ],
        current_step=0,
    )

    failed = plan.steps[0]
    failed.status = "failed"
    plan = planner.replan(plan, failed, reason="网络超时")
    assert plan.steps[0].status == "pending"
    assert plan.current_step == 0
    print("✅ 单步失败：只重规划该步，保留其他步")


# ─── 测试8：v0.4 端到端闭环 ────────────────────────────────────────────────
def test_v04_full_loop():
    from agent.hermes_v04 import HermesV04
    import json

    h = HermesV04(llm_client=FakeLLM())
    result = h.run("帮我搜索TikTok马来西亚热销耳机")

    assert result["status"] in ("complete","replanning")
    assert len(result["steps"]) >= 3
    assert result["task_sig"]["domain"] == "TikTok_电商选品"
    assert result["ev_stats"]["total"] >= 1
    print(f"✅ v0.4 完整闭环：status={result['status']}, steps={len(result['steps'])}, "
          f"replan={result['replan_count']}, ev_total={result['ev_stats']['total']}")


# ─── 测试9：ExecutionPlan 状态转换 ────────────────────────────────────────
def test_execution_plan_state():
    planner = TaskPlanner(llm_client=FakeLLM())
    plan = planner.plan("测试任务")
    assert plan.status == "planning"

    step = plan.next_step()
    assert step.status == "running"
    assert plan.status == "planning"

    plan.advance()
    assert plan.steps[0].status == "done"
    assert plan.current_step == 1
    assert plan.status == "planning"  # 还没完成

    # 全部完成
    while plan.current_step < len(plan.steps):
        plan.advance()
    assert plan.status == "complete"
    print("✅ ExecutionPlan 状态转换正确")


# ─── 运行所有测试 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        test_task_signature_extraction,
        test_planner_generates_steps,
        test_planner_reuses_best_when_score_high,
        test_critic_scoring,
        test_critic_classification,
        test_evolution_engine_write_and_retrieve,
        test_replan_single_step,
        test_v04_full_loop,
        test_execution_plan_state,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"❌ {t.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"结果: {passed}/{passed+failed} 通过")
    if failed == 0:
        print("🎉 全部通过！")
