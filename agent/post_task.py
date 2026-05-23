"""
agent/post_task.py — v3.1 架构：任务结束后自动验证 + 反思 + 整改

职责：
  每次灵光任务执行完成后，自动触发以下流程：
    1. 验证结果（auto_validate_task）— 任务成功了吗？
    2. 判断是否需要反思（needs_reflection）— 有什么地方可以更好？
    3. 生成反思提示（generate_reflection_prompt）— 具体问哪里做错了
    4. 记录洞察到记忆（record_post_task_insight）— 写入 CORE_MEMORY
    5. 创建整改 TODO — 让这个问题有后续跟进

触发点：
  - run_agent.py 主循环中，每次 lingguang 任务完成后（task_completed / task_timeout / task_error）
  - 在 _publish_lingguang_event() 之后调用

注意：
  - 每个函数都是独立的，可以单独调用
  - 所有错误都被 catch，不打断主流程
  - 反思是生成提示文本，真正的反思由灵灵自己完成
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Section 1: Validation — 验证任务结果
# ----------------------------------------------------------------------

def auto_validate_task(
    events: List[Dict[str, Any]],
    quality_score_threshold: float = 5.0,
    repeated_error_threshold: int = 3,
) -> Dict[str, Any]:
    """
    根据任务事件列表，评估任务完成质量。

    判断维度：
      1. 是否有 task_completed 且有实际数据
      2. 评分是否全部 >= threshold
      3. 是否重复出现相同错误
      4. 是否需要人工介入

    Returns:
        {
            "status": "pass" | "degraded" | "fail",
            "quality_flags": int,       # 正数=好，负数=差
            "needs_reflection": bool,   # 是否需要触发反思
            "needs_human_review": bool, # 是否需要人工介入
            "summary": str,             # 简洁的人类可读结论
            "details": {...},           # 详细分析
        }
    """
    if not events:
        return {
            "status": "fail",
            "quality_flags": -5,
            "needs_reflection": False,
            "needs_human_review": True,
            "summary": "无事件数据，无法验证",
            "details": {},
        }

    quality_flags = 0
    completed_count = 0
    failed_count = 0
    all_scores: List[float] = []
    retry_decisions: List[str] = []
    error_types: Dict[str, int] = {}
    error_messages: List[str] = []

    for ev in events:
        ev_type = ev.get("type", "")
        qs = ev.get("quality_score", {})
        err = ev.get("error", "")

        if ev_type in ("task_completed", "task_accepted"):
            completed_count += 1
            if qs:
                overall = qs.get("overall", 0)
                all_scores.append(overall)
                decision = qs.get("retry_decision", "accept")
                retry_decisions.append(decision)

                if overall >= 8.0:
                    quality_flags += 2
                elif overall >= 5.0:
                    quality_flags += 1
                else:
                    quality_flags -= 2

                if decision == "retry":
                    quality_flags -= 1
                elif decision == "replan":
                    quality_flags -= 2
                elif decision == "human_review":
                    quality_flags -= 3

        elif ev_type in ("task_error", "task_timeout"):
            failed_count += 1
            quality_flags -= 2
            if err:
                error_messages.append(err)
            # 提取错误类型
            err_lower = (err or "").lower()
            if "timeout" in err_lower or "超时" in err_lower:
                error_types["timeout"] = error_types.get("timeout", 0) + 1
            elif "no data" in err_lower or "空" in err_lower:
                error_types["no_data"] = error_types.get("no_data", 0) + 1
            else:
                error_types["generic_err"] = error_types.get("generic_err", 0) + 1

        # error_pattern 字段
        ep = ev.get("error_pattern", {})
        if ep:
            et = ep.get("error_type", "unknown")
            error_types[et] = error_types.get(et, 0) + ep.get("count", 1)

    # 空数据检测（completed 但无实际结果）
    has_empty_results = False
    for ev in events:
        if ev.get("type") == "task_completed":
            result = ev.get("result", {})
            text = result.get("text", "")
            if not text or len(text.strip()) < 5:
                has_empty_results = True
                quality_flags -= 1

    # 重复错误检测
    needs_reflection = False
    for err_type, count in error_types.items():
        if count >= repeated_error_threshold:
            needs_reflection = True
            quality_flags -= count  # 错误越多，扣分越多

    # 评分全部 < threshold
    if all_scores and all(s < quality_score_threshold for s in all_scores):
        needs_reflection = True
        quality_flags -= 2

    # 需要人工介入
    needs_human_review = (
        any(d == "human_review" for d in retry_decisions)
        or any(count >= 5 for count in error_types.values())
    )

    # 综合状态判断
    # 优先看 completed_count：无分数但有完成事件 → 默认 pass（除非有失败）
    # 有评分时：quality_flags + failed_count 决定状态
    if failed_count > 0 and failed_count >= completed_count:
        status = "fail"
    elif all_scores:
        # 有评分 → 按 quality_flags 判断
        if quality_flags >= 2 and failed_count == 0:
            status = "pass"
        elif quality_flags >= 0:
            status = "degraded"
        else:
            status = "fail"
    else:
        # 无评分，但有完成事件且无失败 → pass
        # 但如果有空数据 → degraded（不是失败，但没有有效数据）
        if completed_count > 0 and failed_count == 0:
            if has_empty_results and completed_count >= 2:
                # 多个空数据 → fail（明显有问题）
                status = "fail"
            elif has_empty_results:
                # 单个空数据 → degraded
                status = "degraded"
            else:
                status = "pass"
        else:
            status = "degraded"

    # 人类可读总结
    if status == "pass":
        summary = f"任务完成，质量良好（评分≥{quality_score_threshold}）"
    elif status == "degraded":
        if has_empty_results:
            summary = "任务完成但数据为空，建议检查数据源"
        elif needs_reflection:
            summary = f"任务有问题（{len(error_types)}种错误类型），需要反思"
        else:
            summary = "任务部分成功，有改进空间"
    else:
        if failed_count > completed_count:
            summary = f"任务失败（{failed_count}个错误），需要人工介入"
        else:
            summary = f"任务质量差（评分低，{len(error_types)}种错误），需要反思"

    return {
        "status": status,
        "quality_flags": quality_flags,
        "needs_reflection": needs_reflection,
        "needs_human_review": needs_human_review,
        "summary": summary,
        "details": {
            "completed_count": completed_count,
            "failed_count": failed_count,
            "avg_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
            "error_types": error_types,
            "retry_decisions": retry_decisions,
            "error_messages": error_messages[:3],  # 最多3条
        },
    }


# ----------------------------------------------------------------------
# Section 2: Reflection Prompt — 生成反思提示
# ----------------------------------------------------------------------

def generate_reflection_prompt(
    context: Dict[str, Any],
    reflection_style: str = "detailed",
) -> str:
    """
    根据验证结果，生成具体的反思提示。

    反思必须包含：
      1. 任务是什么
      2. 具体哪个环节出错了
      3. 应该怎么避免
      4. 下次遇到类似情况怎么做

    Args:
        context: {
            "task": str,
            "events": [...],
            "validation_result": {...},
            "error_patterns": [...],
        }
        reflection_style: "brief" | "detailed" | "strict"

    Returns:
        Markdown 格式的反思提示文本（给灵灵执行）
    """
    task = context.get("task", "未知任务")
    validation = context.get("validation_result", {})
    error_patterns = context.get("error_patterns", [])
    details = validation.get("details", {})
    error_types = details.get("error_types", {})
    status = validation.get("status", "unknown")

    lines: List[str] = []

    if reflection_style == "brief":
        # 简短反思（3条以内）
        lines.append("### 🔍 任务反思")
        if error_types:
            top_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:2]
            for err_type, count in top_errors:
                lines.append(f"- **{err_type}** 出现 {count} 次 → 下次怎么避免？")
        lines.append(f"- 这次任务的评分：{validation.get('summary', '')}")
        lines.append("")
        return "\n".join(lines)

    # detailed / strict 风格
    lines.append("## 🔍 任务反思")
    lines.append(f"**任务**: {task}")
    lines.append("")

    # 错误分析
    if error_types:
        lines.append("### ❌ 错误分析")
        sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)
        for err_type, count in sorted_errors:
            # 找对应的 pattern suggestion
            suggestion = ""
            for ep in error_patterns:
                if ep.get("error_type") == err_type:
                    suggestion = ep.get("suggestion", "")
                    break
            icon = "🔴" if count >= 3 else "⚠️"
            lines.append(f"{icon} **{err_type}** × {count}次" +
                         (f" → {suggestion}" if suggestion else ""))
        lines.append("")

    # 具体问题
    if status in ("fail", "degraded"):
        lines.append("### 📌 必须回答的问题")

        # Q1: 哪一步做错了
        lines.append("**Q1. 具体哪个环节出了问题？**")
        lines.append("请对照任务步骤，指出哪一步有问题。不要泛泛而谈。")
        lines.append("")

        # Q2: 为什么出错
        lines.append("**Q2. 为什么这一步会出错？**")
        if error_types:
            top_err = sorted(error_types.keys(),
                             key=lambda k: error_types[k], reverse=True)[0]
            lines.append(f"结合 **{top_err}** 类型错误，分析根本原因。")
        lines.append("")

        # Q3: 下次怎么避免
        lines.append("**Q3. 下次遇到类似情况，怎么做才对？**")
        lines.append("写出具体的一条规则或检查点。")
        lines.append("")

        # Q4: 需要记住什么
        lines.append("**Q4. 需要写入记忆的规律是什么？**")
        lines.append("写一条可以用在下次任务的通用规律。")
        lines.append("")

        # Q5: 创建一个 TODO
        if error_types:
            lines.append("**Q5. 现在创建一个 TODO**")
            top_err = sorted(error_types.keys(),
                             key=lambda k: error_types[k], reverse=True)[0]
            lines.append(
                f"用 todo 工具创建一项后续跟进，格式："
                f"[改] 修复 **{top_err}** 问题 → 具体做法是什么"
            )
        lines.append("")

    else:
        # pass: 简单复盘
        lines.append("### ✅ 复盘")
        lines.append(f"任务状态：{validation.get('summary', '成功')}")
        if details.get("avg_score", 0) > 0:
            lines.append(f"平均评分：{details['avg_score']:.1f}")
        lines.append("")
        lines.append("**这次做得好的是哪里？**")
        lines.append("**下次可以保持的是什么？**")
        lines.append("")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# Section 3: Record Insight — 写入记忆 + 创建 TODO
# ----------------------------------------------------------------------

def record_post_task_insight(
    task: str,
    errors: List[Dict[str, Any]],
    reflection_answer: str = "",
    quality_flags: int = 0,
) -> Dict[str, Any]:
    """
    将反思洞察写入记忆并创建跟进 TODO。

    执行两件事：
      1. memory.add() — 写入 L1/L2 记忆，让灵灵下次不再犯
      2. create_todo() — 创建具体跟进项，确保有后续

    Args:
        task: 任务名称（用于 TODO 标题）
        errors: 错误模式列表
        reflection_answer: 灵灵的反思答案（如果有）
        quality_flags: 质量分数（用于判断优先级）

    Returns:
        {"memory_written": bool, "todo_created": bool, "todo_id": str}
    """
    from pathlib import Path

    results: Dict[str, Any] = {
        "memory_written": False,
        "todo_created": False,
        "todo_id": None,
    }

    # ── 1. 写入记忆 ───────────────────────────────────────────────────
    try:
        import hermes_tools  # fallback if not in path
    except Exception:
        pass

    # 生成记忆内容
    if errors:
        top_error = max(errors, key=lambda e: e.get("count", 0))
        error_type = top_error.get("error_type", "unknown")
        error_count = top_error.get("count", 0)
        error_suggestion = top_error.get("suggestion", "")
    else:
        error_type = "general"
        error_count = 0
        error_suggestion = ""

    # 优先级
    if quality_flags <= -3:
        priority = "high"
    elif quality_flags < 0:
        priority = "medium"
    else:
        priority = "low"

    memory_content = (
        f"**任务后反思 [{time.strftime('%Y-%m-%d')}]**\n"
        f"任务: {task}\n"
        f"错误类型: {error_type}（累计{error_count}次）\n"
        f"建议: {error_suggestion}\n"
    )
    if reflection_answer:
        memory_content += f"反思结论: {reflection_answer}\n"

    # ── 1. 写入记忆 ───────────────────────────────────────────────────
    try:
        # 直接写入 JSONL 文件（可靠，不需要依赖 MemoryManager 的 API）
        insights_file = Path.home() / ".hermes" / "post_task_insights.jsonl"
        insights_file.parent.mkdir(parents=True, exist_ok=True)
        insight_entry = {
            "timestamp": time.time(),
            "task": task,
            "error_type": error_type,
            "count": error_count,
            "suggestion": error_suggestion,
            "reflection": reflection_answer,
            "quality_flags": quality_flags,
            "priority": priority,
        }
        with open(insights_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(insight_entry, ensure_ascii=False) + "\n")
        results["memory_written"] = True
        logger.info("[post_task] 记忆写入成功 -> %s", insights_file)
    except Exception as exc:
        logger.warning("[post_task] 记忆写入失败: %s", exc)

    # ── 2. 创建 TODO ──────────────────────────────────────────────────
    if errors and priority in ("high", "medium"):
        try:
            # 直接写入 TODO 文件（可靠，不需要依赖工具）
            todo_file = Path.home() / ".hermes" / "post_task_todos.jsonl"
            todo_file.parent.mkdir(parents=True, exist_ok=True)
            todo_entry = {
                "timestamp": time.time(),
                "task": task,
                "error_type": error_type,
                "content": (
                    f"[改] 修复 {task} 的 {error_type} 问题\n"
                    f"错误: {error_suggestion}\n"
                    f"反思: {reflection_answer or '待填写'}"
                ),
                "priority": priority,
                "status": "pending",
            }
            with open(todo_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(todo_entry, ensure_ascii=False) + "\n")
            results["todo_created"] = True
            logger.info("[post_task] TODO 创建成功 -> %s", todo_file)
        except Exception as exc:
            logger.warning("[post_task] TODO 创建失败: %s", exc)

    return results


# ----------------------------------------------------------------------
# Section 4: Orchestration — 主流程编排
# ----------------------------------------------------------------------

def run_post_task_flow(
    events: List[Dict[str, Any]],
    task: str = "",
    error_patterns: Optional[List[Dict[str, Any]]] = None,
    reflection_answer: str = "",
    silent: bool = True,
) -> Dict[str, Any]:
    """
    执行完整的任务后流程：验证 → 反思 → 记忆+整改。

    这是 run_agent.py 调用的主入口。

    Args:
        events: 任务事件列表（来自 channel observer）
        task: 任务描述
        error_patterns: 错误模式列表（来自 error_accumulator）
        reflection_answer: 如果灵灵已经做了反思，传入答案
        silent: True=不打印，False=打印结果

    Returns:
        {
            "validation": {...},
            "reflection_prompt": str,
            "insight_recorded": {...},
            "action_taken": str,  # "none" | "reflection_prompt" | "insight_recorded"
        }
    """
    result: Dict[str, Any] = {
        "validation": {},
        "reflection_prompt": "",
        "insight_recorded": {},
        "action_taken": "none",
    }

    # ── Step 1: 验证 ──────────────────────────────────────────────────
    try:
        validation = auto_validate_task(events)
        result["validation"] = validation
    except Exception as exc:
        logger.debug("[post_task] validation failed: %s", exc)
        return result

    # 快速路径：成功任务，不需要任何后续
    if validation["status"] == "pass" and not validation.get("needs_reflection"):
        if not silent:
            print(f"✅ [post_task] 任务验证通过，无需后续处理")
        result["action_taken"] = "none"
        return result

    # ── Step 2: 生成反思提示 ─────────────────────────────────────────
    context = {
        "task": task or "未知任务",
        "events": events,
        "validation_result": validation,
        "error_patterns": error_patterns or [],
    }

    try:
        reflection_prompt = generate_reflection_prompt(context)
        result["reflection_prompt"] = reflection_prompt
    except Exception as exc:
        logger.debug("[post_task] reflection prompt failed: %s", exc)
        return result

    # 如果灵灵已经有反思答案 → 直接记录
    if reflection_answer:
        try:
            insight = record_post_task_insight(
                task=task or "未知任务",
                errors=error_patterns or [],
                reflection_answer=reflection_answer,
                quality_flags=validation.get("quality_flags", 0),
            )
            result["insight_recorded"] = insight
            result["action_taken"] = "insight_recorded"
            if not silent:
                print(f"✅ [post_task] 洞察已记录: memory={insight.get('memory_written')}, todo={insight.get('todo_created')}")
        except Exception as exc:
            logger.debug("[post_task] insight record failed: %s", exc)
        return result

    # 如果需要反思但还没有答案 → 返回提示，让灵灵执行
    if validation.get("needs_reflection"):
        result["action_taken"] = "reflection_prompt"
        return result

    return result