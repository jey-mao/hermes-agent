"""Trajectory saving utilities and static helpers.

_convert_to_trajectory_format stays as an AIAgent method (batch_runner.py
calls agent._convert_to_trajectory_format). Only the static helpers and
the file-write logic live here.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def convert_scratchpad_to_think(content: str) -> str:
    """Convert <REASONING_SCRATCHPAD> tags to <think> tags."""
    if not content or "<REASONING_SCRATCHPAD>" not in content:
        return content
    return content.replace("<REASONING_SCRATCHPAD>", "<think>").replace("</REASONING_SCRATCHPAD>", "</think>")


def has_incomplete_scratchpad(content: str) -> bool:
    """Check if content has an opening <REASONING_SCRATCHPAD> without a closing tag."""
    if not content:
        return False
    return "<REASONING_SCRATCHPAD>" in content and "</REASONING_SCRATCHPAD>" not in content


def save_trajectory(trajectory: List[Dict[str, Any]], model: str,
                    completed: bool, filename: str = None):
    """Append a trajectory entry to a JSONL file in ~/.hermes/trajectories/.

    Args:
        trajectory: The ShareGPT-format conversation list.
        model: Model name for metadata.
        completed: Whether the conversation completed successfully.
        filename: Override output filename. Defaults to trajectory_samples.jsonl
                  or failed_trajectories.jsonl based on ``completed``.
    """
    import os as _os
    traj_dir = _os.path.join(_os.path.expanduser("~"), ".hermes", "trajectories")
    _os.makedirs(traj_dir, exist_ok=True)

    if filename is None:
        filename = "trajectory_samples.jsonl" if completed else "failed_trajectories.jsonl"

    filepath = _os.path.join(traj_dir, filename)

    entry = {
        "conversations": trajectory,
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "completed": completed,
    }

    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("Trajectory saved to %s", filepath)
    except Exception as e:
        logger.warning("Failed to save trajectory: %s", e)


# =========================================================================
# Learning Module (Evolution Layer — Learning)
# =========================================================================

def analyze_trajectories(trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze a list of trajectory summaries to extract behavioral patterns.

    Key insights:
      - Success vs failure rate
      - Avg message count (success vs all)
      - Most common failure models
      - Session length patterns

    Args:
        trajectories: List of trajectory dicts from get_recent_trajectories()
                      (keys: timestamp, model, completed, message_count)

    Returns:
        {
            "total": int,
            "success_count": int,
            "failure_count": int,
            "success_rate": float,
            "avg_message_count": float,
            "avg_success_message_count": float,
            "avg_failure_message_count": float,
        }
    """
    if not trajectories:
        return {
            "total": 0,
            "success_count": 0,
            "failure_count": 0,
            "success_rate": 0.0,
            "avg_message_count": 0.0,
            "avg_success_message_count": 0.0,
            "avg_failure_message_count": 0.0,
        }

    success = [t for t in trajectories if t.get("completed", False)]
    failure = [t for t in trajectories if not t.get("completed", False)]

    def avg(lst: List[Dict], key: str) -> float:
        if not lst:
            return 0.0
        vals = [t.get(key, 0) for t in lst]
        return sum(vals) / len(vals) if vals else 0.0

    all_mc = [t.get("message_count", 0) for t in trajectories if t.get("message_count")]
    success_mc = [t.get("message_count", 0) for t in success if t.get("message_count")]
    failure_mc = [t.get("message_count", 0) for t in failure if t.get("message_count")]

    return {
        "total": len(trajectories),
        "success_count": len(success),
        "failure_count": len(failure),
        "success_rate": len(success) / len(trajectories) if trajectories else 0.0,
        "avg_message_count": sum(all_mc) / len(all_mc) if all_mc else 0.0,
        "avg_success_message_count": sum(success_mc) / len(success_mc) if success_mc else 0.0,
        "avg_failure_message_count": sum(failure_mc) / len(failure_mc) if failure_mc else 0.0,
    }


def extract_task_intent(task_text: str) -> tuple[str, str]:
    """
    Extract the core action and platform from a task description.

    Returns:
        (action, platform) — e.g. ("采集", "tiktok"), ("搜索", "1688")

    Action categories:
        采集/搜索/浏览/文档/命令/配置/分析/导出

    Platform hints:
        miaoshou / tiktok / 1688 / xiaohongshu / feishu / shopee / web
    """
    text = task_text.lower()

    # Action extraction
    action_map = [
        ("采集", ["采集", "抓取", "爬", "hot", "热销"]),
        ("搜索", ["搜索", "找", "查找", "查询"]),
        ("浏览", ["浏览", "访问", "打开", "进入", "visit"]),
        ("文档", ["写", "编辑", "创建", "文档", "周报", "报告"]),
        ("命令", ["执行", "运行", "命令", "cmd", "shell"]),
        ("配置", ["配置", "设置", "setup", "install", "安装"]),
        ("分析", ["分析", "审核", "评估", "review"]),
        ("导出", ["导出", "下载", "save"]),
    ]
    action = "通用"
    for label, keywords in action_map:
        if any(kw in text for kw in keywords):
            action = label
            break

    # Platform extraction
    platform_map = [
        ("miaoshou", ["妙手", "erp", "miaoshou"]),
        ("tiktok", ["tiktok", "抖音", "tiktok_hot"]),
        ("1688", ["1688", "alibaba"]),
        ("xiaohongshu", ["小红书", "red", "xhs"]),
        ("feishu", ["飞书", "lark"]),
        ("shopee", ["shopee", "虾皮"]),
        ("web", ["浏览器", "chrome", "http", "www", "访问"]),
    ]
    platform = "web"
    for label, keywords in platform_map:
        if any(kw in text for kw in keywords):
            platform = label
            break

    return action, platform


def get_learning_insights(
    error_patterns: List[Dict[str, Any]],
    trajectories: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Combine error patterns and trajectory analysis into a unified learning
    insight structure that can be injected into the LLM or stored as skill.

    This is the main entry point for the Learning module.

    Returns:
        {
            "trajectory_analysis": {...},   # from analyze_trajectories()
            "error_insights": [...],       # enriched error patterns
            "model_behavior": str,          # free-text summary of model behavior
            "recommended_fixes": [...],     # concrete action items
            "success_rate": float,          # overall success rate
            "top_failure_types": [...],     # most common error types
            "generated_at": float,
        }
    """
    from datetime import datetime
    import time

    traj_analysis = analyze_trajectories(trajectories)

    # Error insights enrichment
    error_insights = []
    for err in error_patterns:
        level = err.get("level", "warn")
        count = err.get("count", 0)
        error_type = err.get("error_type", "unknown")

        # Generate concrete fix recommendation
        fix_map = {
            "timeout": "增加超时参数或检查网络连接",
            "no_data": "验证API参数或更换数据源",
            "low_quality": "人工介入审核数据质量",
            "platform_err": "检查平台登录状态和权限",
            "generic_err": "记录完整错误日志并重试",
        }
        fix = fix_map.get(error_type, "检查错误日志后重试")

        error_insights.append({
            "pattern": f"{err.get('platform', '')}::{err.get('task_type', '')}::{error_type}",
            "count": count,
            "level": level,
            "suggestion": err.get("suggestion", ""),
            "recommended_fix": fix,
        })

    # Top failure types
    type_count: Dict[str, int] = {}
    for err in error_patterns:
        et = err.get("error_type", "unknown")
        type_count[et] = type_count.get(et, 0) + err.get("count", 0)
    top_failure_types = sorted(type_count.items(), key=lambda x: x[1], reverse=True)[:3]

    # Generate model behavior summary
    sr = traj_analysis["success_rate"]
    if sr >= 0.8:
        model_behavior = "高成功率，表现稳定"
    elif sr >= 0.5:
        model_behavior = "中等成功率，存在偶发失败"
    elif sr > 0:
        model_behavior = "低成功率，需要关注错误模式"
    else:
        model_behavior = "无成功案例，建议重新评估任务策略" if trajectories else "无执行数据"

    # Concrete recommended fixes
    recommended_fixes = [ei["recommended_fix"] for ei in error_insights[:3]]
    if not recommended_fixes:
        recommended_fixes = ["继续收集执行数据"]

    return {
        "trajectory_analysis": traj_analysis,
        "error_insights": error_insights,
        "model_behavior": model_behavior,
        "recommended_fixes": recommended_fixes,
        "success_rate": sr,
        "top_failure_types": [{"type": t, "count": c} for t, c in top_failure_types],
        "generated_at": time.time(),
    }
