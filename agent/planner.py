"""
agent/planner.py — Evolution Layer: Planning Module

职责：
  整合三个数据源（error_accumulator + trajectory + insights），
  生成任务规划上下文，注入 LLM，让模型在行动前先想清楚怎么做。

数据来源：
  - error_accumulator    : 历史错误模式（3次警告/5次压制）
  - trajectory.jsonl    : 最近任务轨迹（成功/失败）
  - insights (SQLite)   : 使用量趋势（平台/工具/会话统计）

触发时机：
  - is_complex_task() 返回 True 时（任务复杂度 > 阈值）
  - 每次新任务开始（run_agent.py 调用 get_planning_context）
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Complexity threshold: tasks with ≥ this many tool-call hints → complex
# ----------------------------------------------------------------------
_COMPLEXITY_THRESHOLD = 3  # 3+ keywords or 3+ steps → trigger planning


# ----------------------------------------------------------------------
# Section 1: Error Patterns (from error_accumulator)
# ----------------------------------------------------------------------

def _get_error_patterns(
    platform: Optional[str] = None,
    task_type: Optional[str] = None,
    max_warn: int = 3,
) -> List[Dict[str, Any]]:
    """Load error patterns from error_accumulator.

    Returns the same format as get_error_context() but with no
    max_warn cap on the list — we want all warn+ suppress patterns
    for planning purposes.
    """
    try:
        from agent.error_accumulator import get_error_context

        patterns = get_error_context(
            platform=platform,
            task_type=task_type,
            max_warn=max_warn,
        )
        return patterns
    except Exception as exc:
        logger.debug("[planner] error_accumulator unavailable: %s", exc)
        return []


# ----------------------------------------------------------------------
# Section 2: Recent Trajectories (from trajectory.jsonl files)
# ----------------------------------------------------------------------

def get_recent_trajectories(
    max_entries: int = 5,
    max_lines: int = 20,
    completed_only: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """
    Read the last N trajectory entries from trajectory JSONL files.

    We check the two standard locations:
      1. ~/.hermes/trajectory_samples.jsonl   (success)
      2. ~/.hermes/failed_trajectories.jsonl  (failure)

    Args:
        max_entries:  Maximum number of trajectory entries to return
        max_lines:    Maximum lines to scan per file (performance guard)
        completed_only: None = all; True = only success; False = only failed

    Returns:
        List of trajectory dicts (keys: conversations, timestamp, model, completed)
    """
    import os

    hermes_dir = Path.home() / ".hermes"
    files = [
        hermes_dir / "trajectory_samples.jsonl",
        hermes_dir / "failed_trajectories.jsonl",
    ]

    results: List[Dict[str, Any]] = []
    seen: set[str] = set()  # dedupe by timestamp+model

    for fpath in files:
        if not fpath.exists():
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # Read in reverse to get most recent first
                for line in reversed(lines[-max_lines:]):
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Filter by completed status
                    if completed_only is not None:
                        if entry.get("completed") != completed_only:
                            continue

                    # Dedupe
                    key = f"{entry.get('timestamp', '')}-{entry.get('model', '')}"
                    if key in seen:
                        continue
                    seen.add(key)

                    # Keep only essential fields for planning
                    short_entry = {
                        "timestamp": entry.get("timestamp"),
                        "model": entry.get("model"),
                        "completed": entry.get("completed", False),
                        "message_count": (
                            len(entry.get("conversations", []))
                            if isinstance(entry.get("conversations"), list)
                            else 0
                        ),
                    }
                    results.append(short_entry)

                    if len(results) >= max_entries:
                        return results

        except Exception as exc:
            logger.debug("[planner] failed reading %s: %s", fpath, exc)

    return results


# ----------------------------------------------------------------------
# Section 3: Usage Insights (from InsightsEngine)
# ----------------------------------------------------------------------

def get_planning_insights(
    days: int = 7,
    platform_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get a lightweight insights snapshot for planning.

    Returns a stripped-down version of InsightsEngine.generate()
    focused on what's useful for task planning:
      - Session count / cost
      - Platform breakdown (which platform used most)
      - Tool success patterns

    Returns empty dict if SQLite not available or no data.
    """
    try:
        from agent.insights import InsightsEngine
        from hermes_state import SessionDB

        db = SessionDB()
        engine = InsightsEngine(db)
        report = engine.generate(days=days, source=platform_filter)

        # Strip large blobs, keep planning-relevant fields
        if report.get("empty"):
            return report

        return {
            "days": report.get("days"),
            "total_sessions": (
                report.get("overview", {}).get("total_sessions", 0)
                if "overview" in report else 0
            ),
            "total_cost_usd": (
                report.get("overview", {}).get("total_cost_usd", 0.0)
                if "overview" in report else 0.0
            ),
            "platforms": report.get("platforms", [])[:5],  # top 5
            "tools": report.get("tools", [])[:5],           # top 5
            "top_sessions": [
                {"session_id": s.get("session_id"), "source": s.get("source")}
                for s in report.get("top_sessions", [])[:3]
            ],
            "empty": False,
        }
    except Exception as exc:
        logger.debug("[planner] insights unavailable: %s", exc)
        return {"empty": True}


# ----------------------------------------------------------------------
# Core: Assemble Planning Context
# ----------------------------------------------------------------------

def get_planning_context(
    task: str,
    session_id: str,
    max_trajectories: int = 5,
) -> Dict[str, Any]:
    """
    Assemble a complete planning context from all three data sources.

    This is the main entry point called by run_agent.py before each
    complex task.

    Args:
        task:        Current user instruction (e.g. "采集TikTok马来西亚热销榜")
        session_id:  Current session ID (for filtering)
        max_trajectories: How many recent trajectories to include

    Returns:
        {
            "task": str,
            "errors": [...],           # error pattern list
            "trajectories": [...],     # raw recent trajectories (backward compat)
            "trajectory_analysis": {...}, # Learning: analyzed stats
            "learning_insights": {...},  # Learning: error + trajectory combined insights
            "insights": {...},         # usage stats from InsightsEngine
            "generated_at": float      # timestamp for cache invalidation
        }
    """
    # Extract platform hint from task (best-effort heuristic)
    platform_hint = _extract_platform_hint(task)

    # Load raw trajectories
    raw_trajectories = get_recent_trajectories(max_entries=max_trajectories)

    # ── Learning module: enrich trajectories ───────────────────────────
    try:
        from agent.trajectory import analyze_trajectories, get_learning_insights
        traj_analysis = analyze_trajectories(raw_trajectories)
        # Get error patterns (reuse _get_error_patterns, needs platform hint)
        error_patterns = _get_error_patterns(platform=platform_hint)
        learning_insights = get_learning_insights(error_patterns, raw_trajectories)
    except Exception:
        traj_analysis = {"total": 0, "success_rate": 0.0}
        learning_insights = {"success_rate": 0.0, "recommended_fixes": ["继续收集数据"]}

    return {
        "task": task,
        "errors": _get_error_patterns(platform=platform_hint),
        "trajectories": raw_trajectories,
        "trajectory_analysis": traj_analysis,
        "learning_insights": learning_insights,
        "insights": get_planning_insights(platform_filter=platform_hint),
        "generated_at": time.time(),
    }


# ----------------------------------------------------------------------
# Core: Generate Plan Text
# ----------------------------------------------------------------------

def generate_plan(context: Dict[str, Any]) -> str:
    """
    Convert a planning context into a human-readable plan text segment
    that will be injected into the LLM's context.

    The plan includes:
      - Task summary
      - Known risks (from error patterns)
      - Historical context (from trajectories)
      - Recommended steps
      - Backup options

    Args:
        context: Result from get_planning_context()

    Returns:
        Markdown-formatted plan text to append to LLM prompt.
    """
    if not context:
        return ""

    task = context.get("task", "")
    errors = context.get("errors", [])
    trajectories = context.get("trajectories", [])
    insights = context.get("insights", {})

    lines: List[str] = []

    # ── Header ──────────────────────────────────────────────────────────
    lines.append("## 🎯 任务规划")
    lines.append(f"**当前任务**: {task}")
    lines.append("")

    # ── Section 1: Error Warnings ──────────────────────────────────────
    if errors:
        lines.append("### ⚠️ 历史风险警告")
        for err in errors[:3]:  # top 3
            icon = "🚨" if err.get("level") == "suppress" else "⚠️"
            lines.append(
                f"{icon} [{err['platform']}::{err['task_type']}::{err['error_type']}] "
                f"累计 {err['count']} 次 → {err.get('suggestion', '建议规避')}"
            )
        lines.append("")

    # ── Section 2: Historical Context ──────────────────────────────────
    if trajectories:
        lines.append("### 📋 最近执行轨迹")
        for t in trajectories[:3]:  # last 3
            status = "✅" if t.get("completed") else "❌"
            ts = t.get("timestamp", "?")[:16]
            mc = t.get("message_count", "?")
            lines.append(
                f"{status} {ts} · {t.get('model','?')} · {mc}条消息"
            )
        lines.append("")

    # ── Section 2.5: Learning Insights (Evolution Layer — Learning) ──
    learning = context.get("learning_insights", {})
    if learning:
        sr = learning.get("success_rate", 0.0)
        traj_analysis = context.get("trajectory_analysis", {})
        success_count = traj_analysis.get("success_count", 0)
        failure_count = traj_analysis.get("failure_count", 0)
        avg_success_mc = traj_analysis.get("avg_success_message_count", 0.0)
        avg_failure_mc = traj_analysis.get("avg_failure_message_count", 0.0)

        lines.append("### 🧠 学习洞察")
        lines.append(f"- 成功率: {success_count}/{success_count+failure_count} ({sr:.0%})")
        if avg_failure_mc > avg_success_mc > 0:
            lines.append(f"- 失败会话平均 {avg_failure_mc:.0f} 条消息 vs 成功 {avg_success_mc:.0f} 条 → 失败更冗长")
        if learning.get("model_behavior"):
            lines.append(f"- 模型状态: {learning['model_behavior']}")
        if learning.get("recommended_fixes"):
            lines.append("- 建议修复:")
            for fix in learning["recommended_fixes"][:2]:
                lines.append(f"  • {fix}")
        if learning.get("top_failure_types"):
            top_types = ", ".join(f"{f['type']}({f['count']}次)" for f in learning["top_failure_types"])
            lines.append(f"- 高频失败: {top_types}")
        lines.append("")

    # ── Section 3: Usage Overview ──────────────────────────────────────
    if not insights.get("empty"):
        overview = insights.get("overview", {})
        if overview:
            total_sessions = overview.get("total_sessions", 0)
            total_cost = overview.get("total_cost_usd", 0.0)
            lines.append("### 📊 使用统计（近7天）")
            lines.append(f"- 总会话数: {total_sessions}，总成本: ~${total_cost:.2f}")

        # Platform breakdown
        if insights.get("platforms"):
            top_platforms = insights["platforms"][:3]
            platform_str = ", ".join(
                f"{p['source']}({p['session_count']})" for p in top_platforms
            )
            lines.append(f"- 活跃平台: {platform_str}")
            lines.append("")

    # ── Section 4: Recommended Approach ──────────────────────────────
    lines.append("### 📌 执行建议")
    if errors:
        # Derive recommendations from error patterns
        for err in errors[:2]:
            lines.append(f"- 规避 [{err['error_type']}] 风险：{err.get('suggestion', '')}")
    else:
        lines.append("- 无已知风险，正常执行")
    lines.append("")

    # ── Section 5: Confidence ─────────────────────────────────────────
    confidence = _compute_confidence(errors, trajectories, insights, learning)
    conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
    lines.append(f"{conf_icon} **置信度**: {confidence}")
    lines.append("")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def is_complex_task(instruction: str) -> bool:
    """
    Decide whether a task is complex enough to warrant planning.

    Trigger planning when:
      - Instruction contains multiple steps (≥3 "、" separated clauses)
      - Or contains any planning-related keywords
      - Or is longer than 40 chars (complex intent)
    """
    if not instruction:
        return False

    # Long instruction → complex
    if len(instruction) > 40:
        return True

    # Step indicators
    step_keywords = [
        "采集", "分析", "选品", "定价", "上架", "对比",
        "搜索", "抓取", "审核", "整理", "导出",
        "批量", "多", "并", "且",
    ]
    matches = sum(1 for kw in step_keywords if kw in instruction)

    # Explicit multi-step
    if "然后" in instruction or "再" in instruction:
        matches += 2
    if "和" in instruction and len(instruction) > 20:
        matches += 1

    return matches >= 1  # any single step keyword → complex


def _extract_platform_hint(task: str) -> Optional[str]:
    """Best-effort extraction of platform hint from task text."""
    platform_keywords = {
        "miaoshou": ["妙手", "erp"],
        "tiktok": ["tiktok", "抖音"],
        "1688": ["1688", "alibaba"],
        "xiaohongshu": ["小红书", "red", "xhs"],
        "feishu": ["飞书", "lark"],
        "shopee": ["shopee", "虾皮"],
    }
    task_lower = task.lower()
    for platform, keywords in platform_keywords.items():
        if any(kw in task_lower for kw in keywords):
            return platform
    return None


def _compute_confidence(
    errors: List[Dict],
    trajectories: List[Dict],
    insights: Dict,
    learning_insights: Optional[Dict] = None,
) -> str:
    """Compute plan confidence based on available context."""
    score = 0

    # More historical data → higher confidence
    if trajectories:
        score += 1
    if len(trajectories) >= 3:
        score += 1

    # Error patterns reduce confidence
    suppress_count = sum(1 for e in errors if e.get("level") == "suppress")
    if suppress_count > 0:
        score -= 1
    if len(errors) >= 3:
        score -= 1

    # Insights data helps
    if not insights.get("empty"):
        score += 1

    # Learning: high success rate boosts confidence, no data reduces it
    if learning_insights:
        sr = learning_insights.get("success_rate", 0.0)
        if sr >= 0.8:
            score += 1
        elif sr > 0 and sr < 0.4:
            score -= 1

    if score >= 3:
        return "high"
    elif score >= 1:
        return "medium"
    else:
        return "low"