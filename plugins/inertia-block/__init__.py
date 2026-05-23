"""inertia-block plugin — code-level hard block on repeated tool patterns.

Unlike prompt-based reminders (which the model can read but ignore), this plugin
intercepts tool calls BEFORE they execute via the pre_tool_call hook.

How it works:
1. Track tool name + argument signature in a ring buffer
2. On each pre_tool_call, check if this is the N-th identical/similar call
3. If threshold exceeded → return {"action": "block", "message": "..."}
4. Tool never executes, model receives error result as if tool failed

This is a true HARD BLOCK — model cannot bypass, cannot ignore, cannot retry
the same call and expect different behavior.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import deque
from typing import Any, Dict, Optional

# ─── Per-instance state (keyed by task_id) ───────────────────────────────────

_lock = threading.Lock()
_buffers: Dict[str, Dict[str, Any]] = {}


def _get_buffer(task_id: str) -> Dict[str, Any]:
    """Get/create the ring buffer for a task."""
    with _lock:
        if task_id not in _buffers:
            _buffers[task_id] = {
                "history": deque(maxlen=30),      # (tool_name, arg_sig, timestamp)
                "block_counts": {},               # tool_name → consecutive block count
                "last_blocked_turn": {},          # tool_name → turn count when last blocked
            }
        return _buffers[task_id]


def _arg_signature(tool_name: str, args: Dict[str, Any]) -> str:
    """Create a hash signature from tool name + args for similarity detection.
    
    For execute_code: hash the code content
    For browser tools: hash the URL pattern (ignore query params)
    For others: hash the full args JSON
    """
    try:
        if tool_name == "execute_code":
            code = args.get("code", "")
            # Extract structural prefix by stripping trailing numeric variations.
            # "print(hello)\n# 0" and "print(hello)\n# 1" → structural "print(hello)\n#"
            _lines = code[:200].split('\n')
            _stripped = '\n'.join([re.sub(r'[0-9]+$', '', line) for line in _lines])
            return hashlib.sha256(_stripped.encode()).hexdigest()[:16]
        
        elif tool_name == "browser_navigate":
            url = args.get("url", "")
            # Strip query params for URL signature
            base = url.split("?")[0].split("#")[0]
            return hashlib.sha256(base.encode()).hexdigest()[:16]
        
        elif tool_name == "browser_snapshot":
            # Snapshot is typically "full" or empty — use the expression if any
            expr = args.get("expression", "")[:100]
            return hashlib.sha256(expr.encode()).hexdigest()[:16]
        
        else:
            # For other tools, hash the canonical JSON of args
            canonical = json.dumps(args, sort_keys=True, default=str)
            return hashlib.sha256(canonical.encode()).hexdigest()[:16]
    except Exception:
        return hashlib.sha256(json.dumps(args, default=str).encode()).hexdigest()[:16]


def _similar_args(sig1: str, sig2: str) -> bool:
    """Check if two argument signatures are similar (for pattern detection)."""
    # Exact match is always similar
    if sig1 == sig2:
        return True
    # For execute_code: check first 20 chars of hash (code structure similarity)
    if sig1[:16] == sig2[:16]:
        return True
    return False


# ─── Block thresholds ────────────────────────────────────────────────────────

# After this many IDENTICAL consecutive calls → hard block
IDENTICAL_THRESHOLD = 3

# After this many SIMILAR consecutive calls → hard block
SIMILAR_THRESHOLD = 5

# Per tool name (tool_name → how many times blocked this session)
MAX_BLOCKS_PER_TOOL = 5


def _on_pre_tool_call(
    tool_name: str,
    args: dict,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
) -> Optional[Dict[str, Any]]:
    """pre_tool_call hook — intercept and block repeated tool patterns."""
    if not task_id:
        return None  # Can't track without task context
    
    buf = _get_buffer(task_id)
    sig = _arg_signature(tool_name, args)
    history = buf["history"]
    
    # ── Step 1: Check identical consecutive calls ────────────────────────────
    identical_count = 0
    similar_count = 0
    recent_same_tool = [h for h in history if h[0] == tool_name]
    
    for i, (name, prev_sig, _) in enumerate(recent_same_tool):
        if name != tool_name:
            continue
        if sig == prev_sig:
            identical_count += 1
            similar_count += 1
        elif _similar_args(sig, prev_sig):
            similar_count += 1
    
    # ── Step 2: Determine if this call should be blocked ───────────────────
    
    # Block condition A: Same tool + identical args N times in a row
    if identical_count >= IDENTICAL_THRESHOLD - 1:
        block_count = buf["block_counts"].get(tool_name, 0)
        if block_count < MAX_BLOCKS_PER_TOOL:
            buf["block_counts"][tool_name] = block_count + 1
            buf["last_blocked_turn"][tool_name] = len(history)
            return _make_block_response(
                tool_name, identical_count, "identical",
                f"You've called {tool_name} with identical arguments {identical_count} times in a row. "
                f"This indicates a loop pattern. Stop repeating the same call and analyze the actual "
                f"results from previous attempts. Find the root cause of why it failed, then try a "
                f"DIFFERENT approach — not the same call again."
            )
    
    # Block condition B: Same tool + similar args many times
    if similar_count >= SIMILAR_THRESHOLD - 1:
        block_count = buf["block_counts"].get(tool_name, 0)
        if block_count < MAX_BLOCKS_PER_TOOL:
            buf["block_counts"][tool_name] = block_count + 1
            buf["last_blocked_turn"][tool_name] = len(history)
            return _make_block_response(
                tool_name, similar_count, "similar",
                f"You've called {tool_name} {similar_count} times with similar patterns. "
                f"Repeated attempts with minor variations are still a loop. "
                f"STOP calling {tool_name} now. Instead: (1) Summarize what you've learned from "
                f"the {similar_count} previous attempts. (2) Identify the root cause. "
                f"(3) Describe the CORRECT approach before calling any tool."
            )
    
    # Block condition C: Same tool called more than 8 times total in history
    # (even with different args — likely wrong strategy)
    total_calls = sum(1 for h in history if h[0] == tool_name)
    if total_calls >= 8 and tool_name in ["execute_code", "browser_navigate"]:
        block_count = buf["block_counts"].get(tool_name, 0)
        if block_count < MAX_BLOCKS_PER_TOOL:
            buf["block_counts"][tool_name] = block_count + 1
            return _make_block_response(
                tool_name, total_calls, "overused",
                f"You've called {tool_name} {total_calls} times total. "
                f"This exceeds the safety threshold — something is fundamentally wrong with your approach. "
                f"STOP. Report to the user: what have you learned? What's the actual problem? "
                f"What will you do differently? Do NOT call any more tools until you can explain this."
            )
    
    # ── Step 3: Record this call in history ─────────────────────────────────
    with _lock:
        history.append((tool_name, sig, len(history)))
    
    # Not blocked
    return None


def _make_block_response(
    tool_name: str,
    count: int,
    block_type: str,
    message: str,
) -> Dict[str, Any]:
    """Create the block response for the plugin hook."""
    header = f"[HARD BLOCK - INERTIA DETECTED ({block_type} x{count})]"
    body = "\n".join([
        f"Tool: {tool_name}",
        f"Reason: {message}",
        f"Next step: Describe the current state and your analysis, then await user response.",
        f"Do NOT retry this tool call or any tool with the same pattern.",
    ])
    return {
        "action": "block",
        "message": header + "\n" + body,
    }

def register(ctx) -> None:
    """Register the pre_tool_call hook."""
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
