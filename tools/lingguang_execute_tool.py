#!/usr/bin/env python3
"""
lingguang_execute — Hermes → OpenClaw 任务下发通道

Hermes 收到需要灵光执行的指令后，调用此工具下发任务到 lingguang_realtime_server.py
（通过 Unix Domain Socket），等待结果后返回。

这是 Phase 2 的核心落地：Hermes 作为核心大脑，OpenClaw/灵光 仅作为 specialized 工具执行器。

Socket 协议：
  1. 写入任务文件  → ~/.hermes/lingguang_tasks/queue/{task_id}.json
  2. 发送 wake 指令 → {"action": "wake", "task_id": "...", "task_type": "execute"}
  3. 读取响应        → {status, result, data_quality, ...}
  4. 响应超时：30 秒

支持平台：miaoshou, tiktok_sea, xiaohongshu, feishu, 1688, video
"""

import json
import logging
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOCKET_PATH = Path.home() / ".hermes" / "lingguang_tasks" / "pipe" / "lingguang_command.sock"
QUEUE_DIR = Path.home() / ".hermes" / "lingguang_tasks" / "queue"
RESULTS_DIR = Path.home() / ".hermes" / "lingguang_tasks" / "results"

DEFAULT_TIMEOUT = 60       # seconds to wait for result
SOCKET_TIMEOUT = 30        # seconds for socket operations


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = {
    "name": "lingguang_execute",
    "description": """下发任务给灵光（OpenClaw）执行。

这是 Hermes → OpenClaw 的任务下发通道。适合需要 specialized 执行的场景：
- 妙手ERP采集（Tiktok热销商品、采集箱编辑）
- 1688数据采集
- 小红书/小红书数据
- 飞书消息发送

**注意**：灵光是 specialized 执行器，不是通用大脑。
指令要具体（platform + action + 目标），不要让灵光自己判断做什么。

platform 选项：
  - miaoshou   : 妙手ERP（采集、编辑采集箱）
  - tiktok_sea : TikTok东南亚数据
  - 1688       : 1688货源
  - xiaohongshu: 小红书
  - feishu     : 飞书
  - video      : 视频

task_type 选项：
  - execute : 同步执行（等待结果，最长60秒）
  - background : 异步执行（立即返回 task_id，稍后查结果）
  - query : 查询任务状态（需提供 task_id）
""",
    "parameters": {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": "任务描述（如：'采集TikTok马来西亚热销榜Top10耳机'）",
            },
            "platform": {
                "type": "string",
                "enum": ["miaoshou", "tiktok_sea", "1688", "xiaohongshu", "feishu", "video"],
                "description": "执行平台",
            },
            "task_type": {
                "type": "string",
                "enum": ["execute", "background", "query"],
                "description": "execute=同步等待, background=异步立即返回, query=查状态",
                "default": "execute",
            },
            "task_id": {
                "type": "string",
                "description": "query 模式需要提供 task_id（不提供则自动生成）",
            },
            "action": {
                "type": "string",
                "description": "平台特定动作（如 miaoshou: 'tiktok_hot_full', 'collect_box'）",
            },
            "target": {
                "type": "object",
                "description": "平台特定参数（如 {category: '手机与数码', country: '马来西亚', top_n: 10}）",
                "additionalProperties": True,
            },
            "timeout": {
                "type": "number",
                "description": f"超时秒数（默认 {DEFAULT_TIMEOUT}，最大 300）",
                "minimum": 1,
                "maximum": 300,
            },
        },
        "required": ["instruction", "platform"],
    },
}


def _score_result(
    result_data: Dict[str, Any],
    platform: str,
    instruction: str,
) -> Dict[str, Any]:
    """
    Compute a structured quality score for a task result.

    Returns a dict with:
      - overall: float 0-10
      - accuracy: float 0-10
      - completeness: float 0-10
      - credibility: float 0-10
      - actionability: float 0-10
      - classification: str  (network / tool / quality / timeout / unknown / ok)
      - weaknesses: list[str]
      - retry_decision: str  (accept / retry / replan / human_review)
    """
    classification = "ok"
    weaknesses: List[str] = []
    retry_decision = "accept"
    overall = 8.0  # default optimistic

    accuracy = 8.0
    completeness = 7.0
    credibility = 7.0
    actionability = 7.0

    # ── Miaoshou platform: compute score from item-level data_quality ──────
    # miaoshou_engine stores data_quality INSIDE each item dict.
    # Structure: result['data'][n]['data_quality'] = {source, reliability, ...}
    if platform == "miaoshou":
        # Support multiple field names for miaoshou items:
        # - result['data']            : direct
        # - result['products']         : alias
        # - result['items']            : alias
        # - result['raw_data']['data']: OpenClaw CLI path (raw_data wraps the full miaoshou_engine output)
        if isinstance(result_data, dict):
            items = result_data.get("data") or result_data.get("products") or result_data.get("items")
            if not items:
                # OpenClaw CLI path: miaoshou_engine output wrapped in raw_data
                raw_data = result_data.get("raw_data", {})
                items = raw_data.get("data") or raw_data.get("products") or raw_data.get("items")
            if items and not isinstance(items, list):
                items = []  # guard against unexpected type
        else:
            items = []

        if items:
            total = len(items)

            # OpenClaw CLI path: reliability info lives in raw_data.data_quality (top-level)
            raw_data = result_data.get("raw_data", {})
            top_quality = raw_data.get("data_quality", {})
            top_rel = top_quality.get("reliability", "")

            # credibility: derive from top-level reliability (source of truth for OpenClaw CLI path)
            if top_rel == "high":
                credibility = 10.0
            elif top_rel == "medium":
                credibility = 6.0
            elif top_rel == "low":
                credibility = 2.0
            else:
                credibility = 5.0  # unknown → conservative medium

            # accuracy: fraction of items that pass 40% margin threshold
            if total > 0:
                margin_pass = sum(
                    1 for it in items
                    if it.get("passes_40pct") is True or it.get("margin_pct", 0) >= 40
                )
                accuracy = round(margin_pass / total * 10, 1)
            else:
                accuracy = 0.0

            # completeness: items with valid name field (handle 'name' not 'title')
            good = [
                it for it in items
                if it.get("name") and it.get("name", "").strip() not in ("", ":")
            ]
            completeness = round(len(good) / max(total, 1) * 10, 1)
            actionability = 10.0 if good else 0.0

            overall = round((completeness + credibility + accuracy) / 3, 1)

            # Weakness detection
            if accuracy < 5:
                weaknesses.append(f" Margin pass rate < 50% ({margin_pass}/{total})")
            if not good:
                weaknesses.append(" No valid items with name field")
            if top_rel == "low":
                weaknesses.append(f" Data source reliability=low ({top_quality.get('source', '')})")

            # Decision: use margin pass rate + credibility + completeness
            if completeness >= 8 and credibility >= 8 and accuracy >= 8:
                retry_decision = "accept"
            elif completeness >= 5 or accuracy >= 5:
                retry_decision = "retry"
            else:
                retry_decision = "human_review"
                classification = "quality"
        else:
            classification = "quality"
            weaknesses.append(" Empty result (no items collected)")
            overall = 0.0
            accuracy = completeness = credibility = actionability = 0.0
            retry_decision = "retry"

        return {
            "overall": round(overall, 1),
            "accuracy": round(accuracy, 1),
            "completeness": round(completeness, 1),
            "credibility": round(credibility, 1),
            "actionability": round(actionability, 1),
            "classification": classification,
            "weaknesses": weaknesses,
            "retry_decision": retry_decision,
        }

    # ── Socket result file path: collect_all/summary.json ─────────────────
    raw = result_data.get("raw_data")
    if isinstance(raw, dict):
        products = raw.get("products", [])
        errors = raw.get("errors", [])
        total = len(products)
        # Completeness: products that have non-empty title and price
        good = [p for p in products if p.get("info", {}).get("title", ":") not in ("", ":")]
        completeness = round(len(good) / max(total, 1) * 10, 1)
        # Credibility: ratio of products without errors vs total
        credibility = round(max(total - len(errors), 0) / max(total, 1) * 10, 1)
        # Accuracy: how many fields are actually filled
        filled = sum(
            1 for p in products
            if p.get("info", {}).get("attributes_filled", 0) > 0
        )
        accuracy = round(filled / max(total, 1) * 10, 1)
        # Actionability: at least some valid products to act on
        actionability = 10.0 if good else 0.0

        if completeness < 5:
            classification = "quality"
            weaknesses.append(f" Completeness={completeness}/10")
        if credibility < 5:
            classification = "quality"
            weaknesses.append(f" Credibility={credibility}/10 (errors={len(errors)})")
        if not good:
            classification = "quality"
            weaknesses.append(" No valid products — actionability=0")

        # Decision
        if completeness >= 8 and credibility >= 8:
            retry_decision = "accept"
            overall = min(completeness, credibility, 10.0)
        elif completeness >= 5 or credibility >= 5:
            retry_decision = "retry"
            overall = round((completeness + credibility) / 2, 1)
        else:
            retry_decision = "human_review"
            overall = round((completeness + credibility) / 4, 1)

    # ── Generic result text: basic heuristic ─────────────────────────────
    else:
        result_text = str(result_data.get("result", ""))
        if len(result_text) < 20:
            classification = "quality"
            weaknesses.append(" Result text too short or empty")
            overall = 3.0
            retry_decision = "retry"
        elif "error" in result_text.lower() or "failed" in result_text.lower():
            classification = "unknown"
            weaknesses.append(" Result contains error keywords")
            overall = 4.0
            retry_decision = "human_review"

    return {
        "overall": round(overall, 1),
        "accuracy": round(accuracy, 1),
        "completeness": round(completeness, 1),
        "credibility": round(credibility, 1),
        "actionability": round(actionability, 1),
        "classification": classification,
        "weaknesses": weaknesses,
        "retry_decision": retry_decision,
    }


def _publish_lingguang_event(
    event_type: str,
    tid: str,
    platform: str,
    instruction: str,
    session_id: str,
    task_type: str = "",
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    duration_ms: Optional[int] = None,
) -> None:
    """发布灵光任务事件到 Channel（可观测性 instrumentation）。

    Args:
        session_id: 必须由调用方显式传入（不通过 thread-local），
            因为 dispatch() 在 ThreadPoolExecutor 里执行 handler，
            threading.local() 和 contextvars 都不能跨线程传递。
        task_type: 任务类型（如 miaoshou_tiktok_hot），用于错误积累和可读标签。
    """
    try:
        # Best-effort instrumentation — channel failures never break task execution.
        from agent.channels import publish_to_agent
        from agent.error_accumulator import (
            record_error,
            infer_error_type,
        )
        import time as _time

        event = {
            "type": event_type,
            "task_id": tid,
            "platform": platform,
            "instruction": instruction[:200] if instruction else None,
            "timestamp": _time.time(),
        }
        if result is not None:
            event["result"] = result
            # Compute quality score for task_completed events
            if event_type == "task_completed":
                score = _score_result(result, platform, instruction)
                event["quality_score"] = score
                # P1: auto error accumulation for non-accept decisions
                if score.get("retry_decision") != "accept":
                    err_type = infer_error_type(
                        error_message=error or "",
                        retry_decision=score.get("retry_decision", ""),
                        quality_score=score.get("overall", 0.0),
                        result_text=result.get("text", ""),
                        empty_data=(score.get("classification") == "quality"),
                    )
                    record_result = record_error(
                        platform=platform,
                        task_type=task_type,
                        error_type=err_type,
                        error_message=error or "",
                        retry_decision=score.get("retry_decision", ""),
                        quality_score=score.get("overall", 0.0),
                        sample_task_id=tid,
                    )
                    event["error_pattern"] = {
                        "count": record_result["count"],
                        "level": record_result["level"],
                        "suggestion": record_result["suggestion"],
                    }
        if error is not None:
            event["error"] = error
            # P1: record error for task_error / task_timeout
            if event_type in ("task_error", "task_timeout"):
                err_type = infer_error_type(
                    error_message=error,
                    quality_score=0.0,
                )
                record_result = record_error(
                    platform=platform,
                    task_type=task_type,
                    error_type=err_type,
                    error_message=error,
                    sample_task_id=tid,
                )
                event["error_pattern"] = {
                    "count": record_result["count"],
                    "level": record_result["level"],
                    "suggestion": record_result["suggestion"],
                }
        if duration_ms is not None:
            event["duration_ms"] = duration_ms

        # Publish to TopicChannel (event stream — all subscribers notified)
        publish_to_agent(
            agent_name="supervisor",
            channel_name="events",
            event=event,
            session_id=session_id,
            channel_type="topic",
        )

        # Also write to LastValueChannel (latest result — quick access for next agent)
        from agent.channels import get_channel
        ch = get_channel(
            f"lingguang.task.{tid}",
            channel_type="lastvalue",
            session_id=session_id,
        )
        ch.set(event)

        logger.debug("[lingguang] published channel event: %s %s", event_type, tid)
    except Exception:
        # Channel 失败不影响任务执行 — best effort instrumentation
        logger.debug("[lingguang] channel publish failed (non-fatal): %s")


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handle(
    instruction: str,
    platform: str,
    task_type: str = "execute",
    task_id: Optional[str] = None,
    action: Optional[str] = None,
    target: Optional[Dict[str, Any]] = None,
    timeout: Optional[int] = None,
    **kwargs,
) -> str:
    """
    下发任务给灵光执行器。

    流程：
    1. 生成 task_id（query 模式复用提供的）
    2. 写任务文件到 queue/
    3. Unix Socket 发 wake 指令
    4. 等待结果文件（同步模式）
    5. 读取结果，返回格式化文本
    """
    # dispatch() puts session_id in kwargs — extract it for Channel publishing.
    # This is the ONLY reliable way to get the caller's session_id:
    # threading.local and contextvars both fail across ThreadPoolExecutor threads.
    session_id = kwargs.get("session_id") or "hermes-main"

    # Validate platform
    valid_platforms = {"miaoshou", "tiktok_sea", "1688", "xiaohongshu", "feishu", "video"}
    if platform not in valid_platforms:
        return json.dumps({
            "error": f"Invalid platform: {platform}. Valid: {sorted(valid_platforms)}"
        }, ensure_ascii=False)

    # Validate task_type
    valid_types = {"execute", "background", "query"}
    if task_type not in valid_types:
        return json.dumps({
            "error": f"Invalid task_type: {task_type}. Valid: {sorted(valid_types)}"
        }, ensure_ascii=False)

    timeout_seconds = min(timeout or DEFAULT_TIMEOUT, 300)

    # Generate or use provided task_id
    tid = task_id or f"lingguang_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    # Check lingguang available (OpenClaw CLI or Unix Socket)
    if not _check_lingguang_available():
        _publish_lingguang_event(
            "task_error", tid, platform, instruction, session_id,
            task_type=task_type,
            error="灵光服务未运行",
        )
        return json.dumps({
            "error": "灵光服务未运行。请确保 openclaw CLI 可用（推荐）或 lingguang_realtime_server.py 正在运行。",
            "task_id": tid,
            "status": "server_offline",
        }, ensure_ascii=False)

    # Execute based on task_type
    if task_type == "query":
        return _query_task(tid)
    else:
        return _dispatch_and_wait(tid, instruction, platform, task_type, action, target, timeout_seconds, session_id)


def _check_server_health() -> bool:
    """检查 lingguang server 是否存活（发 status 指令）。"""
    if not SOCKET_PATH.exists():
        return False

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(SOCKET_TIMEOUT)
        sock.connect(str(SOCKET_PATH))
        msg = json.dumps({"action": "status", "task_id": "health_check"}).encode()
        sock.sendall(msg)
        data = sock.recv(8192)
        sock.close()
        resp = json.loads(data.decode("utf-8"))
        return resp.get("status") == "healthy"
    except Exception as e:
        logger.debug("Lingguang health check failed: %s", e)
        return False


def _dispatch_and_wait(
    tid: str,
    instruction: str,
    platform: str,
    task_type: str,
    action: Optional[str],
    target: Optional[Dict[str, Any]],
    timeout_seconds: int,
    session_id: str,
) -> str:
    """下发任务给灵光执行。

    优先方案（推荐）：OpenClaw CLI — 直接调 supervisor agent，不依赖 server 进程。
    兜底方案（旧）：Unix Socket — 写任务文件 + 发 wake + 等结果文件。
    """
    # Ensure directories exist (for socket fallback)
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 方案1: OpenClaw CLI（推荐）
    if shutil.which("openclaw"):
        return _dispatch_via_openclaw_cli(tid, instruction, platform, task_type, action, target, timeout_seconds, session_id)

    # 方案2: Unix Socket（旧）— 不推荐，仅作兜底
    return _dispatch_via_socket(tid, instruction, platform, task_type, action, target, timeout_seconds, session_id)


def _dispatch_via_openclaw_cli(
    tid: str,
    instruction: str,
    platform: str,
    task_type: str,
    action: Optional[str],
    target: Optional[Dict[str, Any]],
    timeout_seconds: int,
    session_id: str,
) -> str:
    """通过 OpenClaw CLI 下发任务给 supervisor agent。"""
    # Build natural-language message from structured params
    parts = []
    if platform:
        platform_zh = {
            "miaoshou": "妙手ERP", "tiktok_sea": "TikTok东南亚",
            "1688": "1688货源", "xiaohongshu": "小红书",
            "feishu": "飞书", "video": "视频",
        }.get(platform, platform)
        parts.append(f"【执行平台】{platform_zh}")

    if action:
        parts.append(f"【动作】{action}")

    if target:
        parts.append(f"【参数】{json.dumps(target, ensure_ascii=False)}")

    parts.append(f"【任务】{instruction}")

    message = "\n".join(parts)

    # Build CLI command
    cmd = ["openclaw", "agent", "--agent", "supervisor", "--message", message, "--json"]
    if timeout_seconds:
        cmd.extend(["--timeout", str(timeout_seconds)])

    logger.info("[lingguang] dispatching via openclaw: %s", cmd)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=min(timeout_seconds, 300),
            text=True,
        )
    except subprocess.TimeoutExpired:
        err = f"灵光执行超时（>{timeout_seconds}s）"
        _publish_lingguang_event("task_timeout", tid, platform, instruction, session_id, task_type=task_type, error=err)
        return json.dumps({
            "task_id": tid,
            "status": "timeout",
            "error": err,
            "executor": "openclaw_cli",
        }, ensure_ascii=False)
    except FileNotFoundError:
        err = "openclaw CLI 不在 PATH 中"
        _publish_lingguang_event("task_error", tid, platform, instruction, session_id, task_type=task_type, error=err)
        return json.dumps({
            "task_id": tid,
            "status": "error",
            "error": err,
            "executor": "openclaw_cli",
        }, ensure_ascii=False)
    except Exception as e:
        err = str(e)
        _publish_lingguang_event("task_error", tid, platform, instruction, session_id, task_type=task_type, error=err)
        return json.dumps({
            "task_id": tid,
            "status": "error",
            "error": err,
            "executor": "openclaw_cli",
        }, ensure_ascii=False)

    # Parse result
    if result.returncode != 0:
        err = result.stderr.strip() or f"openclaw exited with code {result.returncode}"
        _publish_lingguang_event("task_error", tid, platform, instruction, session_id, task_type=task_type, error=err)
        return json.dumps({
            "task_id": tid,
            "status": "error",
            "error": err,
            "executor": "openclaw_cli",
        }, ensure_ascii=False)

    try:
        resp = json.loads(result.stdout)
    except json.JSONDecodeError:
        err = f"openclaw 输出解析失败: {result.stdout[:200]}"
        _publish_lingguang_event("task_error", tid, platform, instruction, session_id, task_type=task_type, error=err)
        return json.dumps({
            "task_id": tid,
            "status": "error",
            "error": err,
            "executor": "openclaw_cli",
        }, ensure_ascii=False)

    if resp.get("status") != "ok":
        err = resp.get("errorMessage") or resp.get("summary") or "unknown error"
        _publish_lingguang_event("task_error", tid, platform, instruction, session_id, task_type=task_type, error=err)
        return json.dumps({
            "task_id": tid,
            "status": "error",
            "error": err,
            "executor": "openclaw_cli",
        }, ensure_ascii=False)

    # Extract text from payloads
    payloads = resp.get("result", {}).get("payloads", [])
    text_parts = [p.get("text", "") for p in payloads if isinstance(p, dict) and p.get("text")]
    text = " ".join(p.strip() for p in text_parts if p.strip())

    meta = resp.get("result", {}).get("meta", {})
    agent_meta = meta.get("agentMeta", {})

    # Extract raw_data for quality scoring (Miaoshou: miaoshou_engine returns structured data here)
    resp_result = resp.get("result", {})
    raw_data = resp_result.get("raw_data", {})

    # Build result dict
    result_dict = {
        "task_id": tid,
        "status": "completed",
        "result": text,
        "platform": platform,
        "executor": "openclaw_cli",
        "run_id": resp.get("runId"),
        "session_id": agent_meta.get("sessionId"),
        "duration_ms": meta.get("durationMs"),
        "usage": agent_meta.get("usage", {}),
        "model": agent_meta.get("model"),
        "provider": agent_meta.get("provider"),
    }

    # Publish to Channel (best effort — failure doesn't block the result)
    _publish_lingguang_event(
        event_type="task_completed",
        tid=tid,
        platform=platform,
        instruction=instruction,
        session_id=session_id,
        task_type=task_type,
        result={"text": text, "run_id": resp.get("runId"), "raw_data": raw_data},
        duration_ms=meta.get("durationMs"),
    )

    return json.dumps(result_dict, ensure_ascii=False, indent=2)


def _dispatch_via_socket(
    tid: str,
    instruction: str,
    platform: str,
    task_type: str,
    action: Optional[str],
    target: Optional[Dict[str, Any]],
    timeout_seconds: int,
    session_id: str,
) -> str:
    """通过 Unix Socket 下发任务给 lingguang_realtime_server（旧方案，兜底用）。"""

    # Build task file
    task = {
        "task_id": tid,
        "instruction": instruction,
        "platform": platform,
        "task_type": task_type,
        "action": action or "",
        "target": target or {},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "creator": "hermes",
    }

    # Write task file
    task_file = QUEUE_DIR / f"{tid}.json"
    try:
        with open(task_file, "w", encoding="utf-8") as f:
            json.dump(task, f, ensure_ascii=False, indent=2)
    except Exception as e:
        _publish_lingguang_event(
            "task_error", tid, platform, instruction, session_id,
            task_type=task_type,
            error=f"Failed to write task file: {e}",
        )
        return json.dumps({"error": f"Failed to write task file: {e}", "task_id": tid}, ensure_ascii=False)

    # Send wake signal via socket
    if not _send_wake(tid, task_type):
        _publish_lingguang_event(
            "task_error", tid, platform, instruction, session_id,
            task_type=task_type,
            error="灵光服务响应失败（Socket 发送 wake 信号失败）",
        )
        return json.dumps({
            "error": "灵光服务响应失败（Socket 发送 wake 信号失败）",
            "task_id": tid,
            "task_file": str(task_file),
        }, ensure_ascii=False)

    # Wait for result (同步模式)
    if task_type == "background":
        _publish_lingguang_event(
            "task_accepted", tid, platform, instruction, session_id,
            task_type=task_type,
            result={"result_file": str(RESULTS_DIR / f"{tid}_result.json")},
        )
        return json.dumps({
            "task_id": tid,
            "status": "accepted",
            "message": f"任务已下发灵光执行（{task_type} 模式）。task_id={tid}",
            "result_file": str(RESULTS_DIR / f"{tid}_result.json"),
            "note": "使用 task_type=query 查询结果",
        }, ensure_ascii=False)

    # 同步等待：轮询结果文件
    result_file = RESULTS_DIR / f"{tid}_result.json"
    deadline = time.time() + timeout_seconds
    poll_interval = 0.5

    while time.time() < deadline:
        if result_file.exists():
            try:
                with open(result_file, encoding="utf-8") as f:
                    result = json.load(f)
                _publish_lingguang_event(
                    "task_completed", tid, platform, instruction, session_id,
                    task_type=task_type,
                    result={"text": result.get("result", ""), "status": result.get("status")},
                )
                return json.dumps({
                    "task_id": tid,
                    "status": result.get("status", "completed"),
                    "result": result.get("result", ""),
                    "data_quality": result.get("data_quality"),
                    "platform": result.get("platform"),
                    "executor": "lingguang_socket",
                    "timestamp": result.get("timestamp"),
                    "validation": result.get("validation"),
                    "raw_data": result.get("raw_data"),
                }, ensure_ascii=False, indent=2)
            except Exception as e:
                _publish_lingguang_event(
                    "task_error", tid, platform, instruction, session_id,
                    task_type=task_type,
                    error=f"Failed to read result: {e}",
                )
                return json.dumps({"error": f"Failed to read result: {e}", "task_id": tid}, ensure_ascii=False)

        time.sleep(poll_interval)
        poll_interval = min(poll_interval * 1.5, 5.0)  # back off

    # Timeout
    _publish_lingguang_event(
        "task_timeout", tid, platform, instruction, session_id,
        task_type=task_type,
        error=f"灵光执行超时（>{timeout_seconds}s）",
    )
    return json.dumps({
        "task_id": tid,
        "status": "timeout",
        "error": f"灵光执行超时（>{timeout_seconds}s）",
        "task_file": str(task_file),
        "result_file": str(result_file),
        "note": "任务已入队列，可稍后用 task_type=query 查询",
    }, ensure_ascii=False)


def _send_wake(tid: str, task_type: str) -> bool:
    """发送 wake 指令到 lingguang server。"""
    if not SOCKET_PATH.exists():
        logger.error("Lingguang socket not found: %s", SOCKET_PATH)
        return False

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(SOCKET_TIMEOUT)
        sock.connect(str(SOCKET_PATH))
        msg = json.dumps({
            "action": "wake",
            "task_id": tid,
            "task_type": task_type,
        }).encode()
        sock.sendall(msg)
        # Read response
        data = sock.recv(4096)
        sock.close()
        resp = json.loads(data.decode("utf-8"))
        return resp.get("status") in ("accepted", "completed", "healthy")
    except Exception as e:
        logger.error("Failed to send wake to lingguang: %s", e)
        return False


def _query_task(tid: str) -> str:
    """查询已有任务的结果。"""
    result_file = RESULTS_DIR / f"{tid}_result.json"
    if result_file.exists():
        try:
            with open(result_file, encoding="utf-8") as f:
                result = json.load(f)
            return json.dumps({
                "task_id": tid,
                "status": result.get("status", "unknown"),
                "result": result.get("result", ""),
                "data_quality": result.get("data_quality"),
                "platform": result.get("platform"),
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Failed to read result: {e}", "task_id": tid}, ensure_ascii=False)
    else:
        return json.dumps({
            "task_id": tid,
            "status": "not_found",
            "message": "任务结果不存在，可能还在执行中或 task_id 错误",
        }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _check_lingguang_available() -> bool:
    """检查灵光服务是否可用。

    优先方案（推荐）：OpenClaw CLI 可执行即认为可用。
    兜底方案（旧）：Unix Socket 存在 + status 响应正常。

    只要任一方案可用，灵光工具就对 LLM 可见。
    """
    # 方案1: OpenClaw CLI（推荐方案 — 不依赖 server 进程）
    import shutil as _shutil
    if _shutil.which("openclaw"):
        return True

    # 方案2: Unix Socket（旧方案 — 依赖 lingguang_realtime_server）
    if not SOCKET_PATH.exists():
        return False
    try:
        import socket as _socket
        sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(str(SOCKET_PATH))
        msg = json.dumps({"action": "status", "task_id": "health_check"}).encode()
        sock.sendall(msg)
        data = sock.recv(4096)
        sock.close()
        resp = json.loads(data.decode("utf-8"))
        return resp.get("status") == "healthy"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------
from tools.registry import registry

registry.register(
    name="lingguang_execute",
    toolset="lingguang",
    schema=SCHEMA,

    handler=lambda args, **kwargs: handle(**args, **kwargs),
    check_fn=_check_lingguang_available,
    emoji="🎯",
)
