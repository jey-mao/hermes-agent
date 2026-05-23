"""
error_accumulator — 灵光任务错误自动积累器

职责：
  1. 当灵光任务失败/超时/需人工介入时，提取错误特征（platform + task_type + error_type）
  2. 累计到 ~/.hermes/lingguang_error_patterns.json
  3. 当同一组合（platform × task_type × error_type）失败 ≥ 3 次 → 生成策略建议
  4. 当同一组合失败 ≥ 5 次 → 触发压制（建议跳过或换平台）
  5. 提供 get_error_context() 供 channel_observer 注入 LLM 上下文

错误类型分类：
  - timeout       : 执行超时
  - no_data       : 采集结果为空
  - low_quality   : 评分 < 5，需人工介入
  - platform_err  : 平台级错误（如登录失效、IP被封）
  - generic_err   : 其他错误

用法：
    from agent.error_accumulator import record_error, get_error_context
    # 任务失败后调用
    record_error(platform="miaoshou", task_type="tiktok_hot", error_type="timeout")
    # LLM 上下文注入
    ctx = get_error_context()
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

ERROR_PATTERNS_FILE = Path.home() / ".hermes" / "lingguang_error_patterns.json"

# 压制阈值：同一错误模式失败 ≥ 5 次 → 建议跳过
SUPPRESS_THRESHOLD = 5
# 警告阈值：同一错误模式失败 ≥ 3 次 → 建议换策略
WARN_THRESHOLD = 3


# -----------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------

def _default_structure() -> Dict[str, Any]:
    return {
        "version": 1,
        "last_updated": 0.0,
        "patterns": {},      # {pattern_key: {count, last_seen, examples, first_seen}}
        "suppressed": {},    # {pattern_key: until_timestamp}  until_timestamp=0 表示永久
    }


# -----------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------

def _load() -> Dict[str, Any]:
    try:
        if ERROR_PATTERNS_FILE.exists():
            with open(ERROR_PATTERNS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 确保结构完整
                if "patterns" not in data:
                    data["patterns"] = {}
                if "suppressed" not in data:
                    data["suppressed"] = {}
                return data
    except Exception as exc:
        logger.debug("[error-accumulator] load failed: %s", exc)
    return _default_structure()


def _save(data: Dict[str, Any]) -> None:
    try:
        ERROR_PATTERNS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ERROR_PATTERNS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.debug("[error-accumulator] save failed: %s", exc)


# -----------------------------------------------------------------------
# Pattern Key
# -----------------------------------------------------------------------

def _make_key(platform: str, task_type: str, error_type: str) -> str:
    """生成错误模式唯一标识"""
    return f"{platform}::{task_type}::{error_type}"


# -----------------------------------------------------------------------
# Core API
# -----------------------------------------------------------------------

def record_error(
    platform: str,
    task_type: str,
    error_type: str,
    error_message: str = "",
    retry_decision: str = "",
    quality_score: float = 0.0,
    sample_task_id: str = "",
) -> Dict[str, Any]:
    """
    记录一次错误。

    Returns:
        dict with keys: count, level ("warn"|"suppress"|None), suggestion
    """
    if not platform or not error_type:
        return {"count": 0, "level": None, "suggestion": ""}

    key = _make_key(platform, task_type, error_type)
    now = time.time()
    data = _load()

    # 初始化 pattern entry
    if key not in data["patterns"]:
        data["patterns"][key] = {
            "count": 0,
            "first_seen": now,
            "last_seen": now,
            "examples": [],          # 最多保留 5 条样本
            "platform": platform,
            "task_type": task_type,
            "error_type": error_type,
        }

    entry = data["patterns"][key]
    entry["count"] += 1
    entry["last_seen"] = now

    # 保留样本（去重，最多 5 条）
    if error_message and error_message not in [e["msg"] for e in entry["examples"]]:
        entry["examples"].append({
            "msg": error_message[:300],
            "ts": now,
            "task_id": sample_task_id,
        })
        entry["examples"] = entry["examples"][-5:]

    data["last_updated"] = now
    _save(data)

    # 计算级别
    count = entry["count"]
    if count >= SUPPRESS_THRESHOLD:
        level = "suppress"
        suggestion = _build_suggestion(platform, task_type, error_type, count, level)
        logger.warning(
            "[error-accumulator] pattern %s suppressed (count=%d)", key, count
        )
    elif count >= WARN_THRESHOLD:
        level = "warn"
        suggestion = _build_suggestion(platform, task_type, error_type, count, level)
        logger.info(
            "[error-accumulator] pattern %s warned (count=%d)", key, count
        )
    else:
        level = None
        suggestion = ""

    return {"count": count, "level": level, "suggestion": suggestion}


def get_error_context(
    platform: Optional[str] = None,
    task_type: Optional[str] = None,
    max_warn: int = 3,
) -> List[Dict[str, Any]]:
    """
    获取需要注入 LLM 上下文的错误模式列表。

    Args:
        platform: 可选，只看某平台
        task_type: 可选，只看某任务类型
        max_warn: 最多返回多少条警告（默认 3）

    Returns:
        [{"key": "...", "platform": "...", "task_type": "...", "error_type": "...",
          "count": N, "level": "warn"|"suppress", "suggestion": "..."}, ...]
        按 count 降序排列
    """
    data = _load()
    now = time.time()
    results: List[Dict[str, Any]] = []

    for key, entry in data["patterns"].items():
        # 检查是否被压制且压制期未过
        if key in data["suppressed"]:
            until = data["suppressed"][key]
            if until == 0 or until > now:
                continue
            # 压制期已过，清理
            del data["suppressed"][key]

        # 过滤平台/任务类型
        if platform and entry.get("platform") != platform:
            continue
        if task_type and entry.get("task_type") != task_type:
            continue

        count = entry["count"]
        if count >= SUPPRESS_THRESHOLD:
            level = "suppress"
        elif count >= WARN_THRESHOLD:
            level = "warn"
        else:
            continue

        suggestion = _build_suggestion(
            entry["platform"], entry["task_type"], entry["error_type"], count, level
        )
        results.append({
            "key": key,
            "platform": entry.get("platform", ""),
            "task_type": entry.get("task_type", ""),
            "error_type": entry.get("error_type", ""),
            "count": count,
            "level": level,
            "suggestion": suggestion,
            "last_seen": entry.get("last_seen", 0),
            "examples": entry.get("examples", []),
        })

    results.sort(key=lambda x: x["count"], reverse=True)
    return results[:max_warn]


def clear_pattern(key: Optional[str] = None) -> None:
    """清除指定错误模式（或全部）"""
    data = _load()
    if key:
        data["patterns"].pop(key, None)
        data["suppressed"].pop(key, None)
    else:
        data["patterns"] = {}
        data["suppressed"] = {}
    data["last_updated"] = time.time()
    _save(data)


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------

_ERROR_TYPE_LABELS = {
    "timeout": "执行超时",
    "no_data": "结果为空",
    "low_quality": "质量过低",
    "platform_err": "平台错误",
    "generic_err": "其他错误",
}

_PLATFORM_LABELS = {
    "miaoshou": "妙手ERP",
    "tiktok_sea": "TikTok东南亚",
    "xiaohongshu": "小红书",
    "1688": "1688货源",
    "feishu": "飞书",
    "video": "视频",
}


def _build_suggestion(
    platform: str, task_type: str, error_type: str, count: int, level: str
) -> str:
    plat = _PLATFORM_LABELS.get(platform, platform)
    etype = _ERROR_TYPE_LABELS.get(error_type, error_type)
    task = task_type or "(通用)"

    if level == "suppress":
        return (
            f"⚠️ 【{plat}::{task}】同一错误（{etype}）已连续失败 {count} 次，"
            f"建议暂时跳过或换平台，当前执行风险高。"
        )
    else:
        return (
            f"⚠️ 【{plat}::{task}】同一错误（{etype}）已出现 {count} 次，"
            f"建议检查参数设置或换策略。"
        )


# -----------------------------------------------------------------------
# Convenience: 从 task result / event 自动判断 error_type
# -----------------------------------------------------------------------

def infer_error_type(
    error_message: str = "",
    retry_decision: str = "",
    quality_score: float = 0.0,
    result_text: str = "",
    empty_data: bool = False,
) -> str:
    """
    从原始信息推断错误类型。

    优先级：
      1. timeout keyword in error_message → timeout
      2. empty_data=True → no_data
      3. quality_score < 5 → low_quality
      4. platform/登录/封禁 keyword → platform_err
      5. otherwise → generic_err
    """
    msg_lower = error_message.lower() + " " + result_text.lower()

    if any(kw in msg_lower for kw in ("timeout", "timed out", "超时")):
        return "timeout"
    if empty_data:
        return "no_data"
    if quality_score > 0 and quality_score < 5:
        return "low_quality"
    if any(kw in msg_lower for kw in ("login", "登录", "blocked", "封禁", "banned", "ip")):
        return "platform_err"
    if error_message:
        return "generic_err"
    return "generic_err"
