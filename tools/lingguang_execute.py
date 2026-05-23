#!/usr/bin/env python3
"""
⚠️ DEPRECATED — 此文件已废弃，请使用 lingguang_execute_tool.py

lingguang_execute — Send tasks to Supervisor agent via OpenClaw CLI.

DEPRECATED: Hermes 使用 tools/lingguang_execute_tool.py 中的 lingguang_execute。
            本文件是探索阶段的产物，不会被 Hermes 注册，不要直接 import。
  browser_navigate → http://127.0.0.1:18789/__openclaw__/c/chat?session=agent%3Asupervisor%3Amain
  → type → Enter → read response

With a simple function call:
  result = lingguang_execute("采集1688蓝牙耳机商品")

Verification (2026-06-20):
  openclaw agent --agent supervisor --message "ping" --json → {"status": "ok", "payloads": [{"text": "收到！"}]}
  Supervisor 记得灵灵（哥哥），当前无待完成任务 ✅
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Verify openclaw CLI is available at module load time
_OPENCLAW_PATH: Optional[str] = shutil.which("openclaw")
if not _OPENCLAW_PATH:
    logger.warning(
        "[lingguang_execute] openclaw CLI not found in PATH. "
        "Tool will return error on every call until openclaw is installed."
    )


# --------------------------------------------------------------------------_
# Schema (module-level so register() can run at import time)
# ---------------------------------------------------------------------------

_HANDLER = _lingguang_execute
_SCHEMA = {
    "description": (
        "Send a task to the Supervisor (灵光) sub-agent via OpenClaw CLI. "
        "Use this instead of manual browser UI when you need to delegate work "
        "to the Supervisor agent (马来西亚产品审核专家). "
        "Supervisor runs in OpenClaw and communicates via its own workspace files. "
        "Returns structured JSON with the supervisor's reply."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": (
                    "Task description in Chinese. Be specific: what to do + "
                    "why + expected output format. "
                    "Example: '请审核以下1688产品：蓝牙耳机 CNY 35 50g，"
                    "返回PASS/FAIL及原因，毛利≥40%为通过标准'"
                ),
            },
            "session_id": {
                "type": "string",
                "description": (
                    "Optional. Use a specific session UUID to resume a prior "
                    "conversation. If omitted, uses supervisor's main session "
                    "(agent:supervisor:main). "
                    "Get session_id from a previous call's result.meta.sessionId."
                ),
            },
            "timeout": {
                "type": "number",
                "description": (
                    "Timeout in seconds. Default: 120. "
                    "60s for simple queries, 120s for browser tasks, "
                    "300s for complex multi-step tasks."
                ),
            },
        },
        "required": ["task"],
    },
}


# ---------------------------------------------------------------------------
# Tool registration (module-level — must be at top level for registry discovery)
# ---------------------------------------------------------------------------

def register(reg):
    """Called by tools/registry.py at discovery time."""
    reg.register(name="lingguang_execute", handler=_HANDLER, schema=_SCHEMA)


try:
    import tools.registry as _reg
except Exception:
    _reg = None

if _reg is not None:
    _reg.register(name="lingguang_execute", handler=_lingguang_execute, schema={
        "description": (
            "Send a task to the Supervisor (灵光) sub-agent via OpenClaw CLI. "
            "Use this instead of manual browser UI when you need to delegate work "
            "to the Supervisor agent (马来西亚产品审核专家). "
            "Supervisor runs in OpenClaw and communicates via its own workspace files. "
            "Returns structured JSON with the supervisor's reply."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "Task description in Chinese. Be specific: what to do + "
                        "why + expected output format. Example: '请审核以下1688产品："
                        "蓝牙耳机 CNY 35，返回PASS/FAIL及原因，毛利≥40%为通过标准'"
                    ),
                },
                "session_id": {
                    "type": "string",
                    "description": (
                        "Optional. Use a specific session UUID to resume a prior "
                        "conversation. If omitted, uses supervisor's main session "
                        "(agent:supervisor:main). "
                        "Get session_id from a previous call's result.meta.sessionId."
                    ),
                },
                "timeout": {
                    "type": "number",
                    "description": (
                        "Timeout in seconds. Default: 120. "
                        "60s for simple queries, 120s for browser tasks, "
                        "300s for complex multi-step tasks."
                    ),
                },
            },
            "required": ["task"],
        },
    })


# ---------------------------------------------------------------------------
# Async implementation (used by model_tools._run_async)
# ---------------------------------------------------------------------------

async def _lingguang_execute(
    task: str,
    session_id: Optional[str] = None,
    timeout: float = 120.0,
    **kwargs,
) -> dict:
    """
    Execute a task by spawning `openclaw agent --agent supervisor --message ... --json`.

    Returns
    -------
    {
        "success": bool,
        "result": {
            "text": str,          # supervisor's reply text
            "session_id": str,   # session used
            "run_id": str,       # OpenClaw run ID
            "duration_ms": int,
            "usage": dict,       # token usage
            "model": str,
            "provider": str,
        } | None,
        "error": str | None,
        "duration_ms": int,
    }
    """
    start_ms = int(time.time() * 1000)

    # Build command
    cmd = [
        "openclaw", "agent",
        "--agent", "supervisor",
        "--message", task,
        "--json",
    ]
    if session_id:
        cmd.extend(["--session-id", session_id])

    # Execute
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={"PATH": subprocess.os.environ.get("PATH", "")},
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return _make_result(
            success=False,
            error=f"TIMEOUT after {timeout}s — supervisor did not respond",
            start_ms=start_ms,
        )
    except FileNotFoundError:
        return _make_result(
            success=False,
            error=(
                "openclaw CLI not found in PATH. "
                "Install OpenClaw or add it to PATH to use lingguang_execute."
            ),
            start_ms=start_ms,
        )
    except Exception as e:
        return _make_result(
            success=False,
            error=f"EXECUTION_ERROR: {e}",
            start_ms=start_ms,
        )

    # Check exit code
    if proc.returncode != 0:
        stderr_text = stderr_bytes.decode(errors="replace").strip()
        return _make_result(
            success=False,
            error=stderr_text or f"openclaw exited with code {proc.returncode}",
            start_ms=start_ms,
        )

    # Parse JSON output
    try:
        result = json.loads(stdout_bytes.decode())
    except json.JSONDecodeError as e:
        return _make_result(
            success=False,
            error=f"JSON_PARSE_ERROR: {e}\nRaw output: {stdout_bytes.decode(errors='replace')[:500]}",
            start_ms=start_ms,
        )

    # Check OpenClaw status
    if result.get("status") != "ok":
        return _make_result(
            success=False,
            error=result.get("errorMessage") or result.get("summary") or "unknown error",
            start_ms=start_ms,
        )

    # Extract reply text from payloads
    payloads: List[dict] = result.get("result", {}).get("payloads", [])
    text_parts = [p.get("text", "") for p in payloads if isinstance(p, dict) and p.get("text")]
    text = " ".join(part.strip() for part in text_parts if part.strip())

    meta: dict = result.get("result", {}).get("meta", {})
    agent_meta: dict = meta.get("agentMeta", {})

    return {
        "success": True,
        "result": {
            "text": text,
            "session_id": agent_meta.get("sessionId"),  # nested inside agentMeta
            "run_id": result.get("runId"),
            "duration_ms": meta.get("durationMs", 0),
            "usage": agent_meta.get("usage", {}),
            "model": agent_meta.get("model"),
            "provider": agent_meta.get("provider"),
        },
        "duration_ms": int(time.time() * 1000) - start_ms,
    }


# ---------------------------------------------------------------------------
# Sync wrapper (for direct tool invocation outside async context)
# ---------------------------------------------------------------------------

def lingguang_execute(task: str, session_id: Optional[str] = None, timeout: float = 120.0) -> dict:
    """
    Synchronous wrapper for use in non-async contexts (e.g. direct import).
    Wraps _lingguang_execute in a fresh event loop.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_lingguang_execute(task, session_id, timeout))
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    success: bool,
    error: Optional[str] = None,
    start_ms: int = 0,
) -> dict:
    """Build a consistent result dict."""
    elapsed = int(time.time() * 1000) - start_ms
    return {
        "success": success,
        "result": None if not success else {},
        "error": error,
        "duration_ms": elapsed,
    }


# ---------------------------------------------------------------------------
# Self-test (run with: python lingguang_execute.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("lingguang_execute self-test")
    print("=" * 60)

    if not _OPENCLAW_PATH:
        print("❌ openclaw not found in PATH — aborting test")
        sys.exit(1)

    print(f"✓ openclaw found at: {_OPENCLAW_PATH}")

    # Test 1: ping
    print("\n[Test 1] ping...")
    result = lingguang_execute("ping", timeout=30)
    print(f"  success: {result['success']}")
    print(f"  text: {result.get('result', {}).get('text', 'N/A')}")
    print(f"  duration_ms: {result.get('duration_ms', 'N/A')}")

    # Test 2: supervisor identity check
    print("\n[Test 2] identity check...")
    result = lingguang_execute("你还记得我是谁吗？简单回复即可", timeout=30)
    print(f"  success: {result['success']}")
    print(f"  text: {result.get('result', {}).get('text', 'N/A')}")
    print(f"  session_id: {result.get('result', {}).get('session_id', 'N/A')}")

    # Test 3: error handling — timeout
    print("\n[Test 3] timeout handling...")
    result = lingguang_execute("x" * 10000, timeout=2)
    print(f"  success: {result['success']} (expected False)")
    print(f"  error snippet: {str(result.get('error', ''))[:100]}")

    print("\n" + "=" * 60)
    print("self-test complete")