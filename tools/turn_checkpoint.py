"""
TurnCheckpoint — Message-level state snapshot for Hermes agent turns.

Snapshots the conversation state at the START of each agent iteration
(API call boundary), before any tool execution.  This is complementary
to CheckpointManager (which snapshots file system state before mutations).

Unlike CheckpointManager (git shadow repos), TurnCheckpoint stores
structured JSON snapshots so they can be:
  - Listed, inspected, and restored programmatically
  - Used for turn-level undo (not just file-level rollback)
  - Queried by session_id or turn_id

Storage layout:
    ~/.hermes/turn_checkpoints/{session_id}/{turn_id:04d}.json

Snapshot format:
    {
        "turn_id": 5,
        "session_id": "abc123",
        "timestamp": "2026-05-20T12:00:00.000Z",
        "api_call_count": 5,
        "reason": "pre_api_call_5",
        "messages": [...],           # copy of messages list at turn start
        "iteration_budget": {
            "used": 5,
            "remaining": 85,
            "max_total": 90
        },
        "tool_results_this_turn": []  # filled in by _snapshot_after_tools
    }

Recovery:
    To resume from turn N, load turn_checkpoints/{session_id}/{N:04d}.json
    and restore messages[:] from it before calling run_conversation() again.
"""

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TURN_CHECKPOINT_BASE = get_hermes_home() / "turn_checkpoints"
TURN_CHECKPOINT_BASE = Path(TURN_CHECKPOINT_BASE)
MAX_TURNS_PER_SESSION = 200   # prune oldest beyond this


# ---------------------------------------------------------------------------
# TurnCheckpoint
# ---------------------------------------------------------------------------

class TurnCheckpoint:
    """
    Manages per-turn message snapshots for an AIAgent session.

    Designed to be owned by AIAgent.  Call ``snapshot()`` at the START of
    each agent iteration (before API call).  Call ``snapshot_after_tools()``
    at the END of each iteration (after all tool results are collected) to
    record what tools ran.

    Parameters
    ----------
    enabled : bool
        Master switch.
    max_turns : int
        Maximum snapshots to keep per session (oldest pruned beyond this).
    """

    def __init__(self, enabled: bool = True, max_turns: int = MAX_TURNS_PER_SESSION):
        self.enabled = enabled
        self.max_turns = max_turns
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def snapshot(
        self,
        turn_id: int,
        session_id: str,
        messages: List[Dict],
        api_call_count: int,
        iteration_budget_remaining: int,
        iteration_budget_max: int,
        reason: str = "",
    ) -> Optional[Path]:
        """
        Take a snapshot at the START of an agent iteration.

        Returns the path to the snapshot file, or None if disabled / error.
        """
        if not self.enabled:
            return None

        snapshot = {
            "turn_id": turn_id,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "api_call_count": api_call_count,
            "reason": reason or f"pre_api_call_{api_call_count}",
            "messages": _deep_copy_messages(messages),
            "iteration_budget": {
                "used": api_call_count,
                "remaining": iteration_budget_remaining,
                "max_total": iteration_budget_max,
            },
            "tool_results_this_turn": [],   # filled by snapshot_after_tools
        }

        return self._write(turn_id, session_id, snapshot)

    def snapshot_after_tools(
        self,
        turn_id: int,
        session_id: str,
        tool_results: List[Dict],
    ) -> None:
        """
        Record which tools ran in the just-completed iteration.

        Loads the pre-snapshot written by ``snapshot()`` and appends
        tool_results to it, then rewrites.
        """
        if not self.enabled:
            return

        path = self._path_for(turn_id, session_id)
        if not path or not path.exists():
            logger.debug("TurnCheckpoint: no pre-snapshot found at %s", path)
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                snap = json.load(f)
            snap["tool_results_this_turn"] = tool_results
            snap["timestamp_completed"] = datetime.now(timezone.utc).isoformat()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(snap, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug("TurnCheckpoint: failed to record tool results: %s", e)

    def list_checkpoints(self, session_id: str) -> List[Dict]:
        """
        List all snapshots for a session.  Most recent first.
        """
        session_dir = Path(TURN_CHECKPOINT_BASE) / session_id
        if not session_dir.is_dir():
            return []

        checkpoints = []
        for path in sorted(session_dir.glob("*.json"), reverse=True):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    snap = json.load(f)
                checkpoints.append({
                    "turn_id": snap.get("turn_id"),
                    "timestamp": snap.get("timestamp"),
                    "reason": snap.get("reason"),
                    "api_call_count": snap.get("api_call_count"),
                    "tool_count": len(snap.get("tool_results_this_turn", [])),
                    "path": str(path),
                })
            except Exception:
                continue
        return checkpoints

    def get_checkpoint(self, session_id: str, turn_id: int) -> Optional[Dict]:
        """Load a specific checkpoint snapshot."""
        path = self._path_for(turn_id, session_id)
        if not path or not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def prune(self, session_id: str) -> int:
        """
        Remove snapshots beyond self.max_turns for the session.

        Returns number of files deleted.
        """
        session_dir = Path(TURN_CHECKPOINT_BASE) / session_id
        if not session_dir.is_dir():
            return 0

        snapshots = sorted(
            [p for p in session_dir.glob("*.json")],
            key=lambda p: int(p.stem),
            reverse=True,
        )

        deleted = 0
        for path in snapshots[self.max_turns:]:
            try:
                path.unlink()
                deleted += 1
            except Exception:
                pass

        if deleted:
            logger.debug("TurnCheckpoint: pruned %d old snapshots for session %s", deleted, session_id)

        return deleted

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write(self, turn_id: int, session_id: str, snapshot: Dict) -> Optional[Path]:
        try:
            session_dir = Path(TURN_CHECKPOINT_BASE) / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            path = session_dir / f"{turn_id:04d}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
            logger.debug("TurnCheckpoint: saved turn %d for session %s", turn_id, session_id)
            return path
        except Exception as e:
            logger.debug("TurnCheckpoint: failed to write snapshot: %s", e)
            return None

    def _path_for(self, turn_id: int, session_id: str) -> Optional[Path]:
        path = Path(TURN_CHECKPOINT_BASE) / session_id / f"{turn_id:04d}.json"
        return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_copy_messages(messages: List[Dict]) -> List[Dict]:
    """
    Deep-copy the messages list, keeping only the fields needed for recovery.

    Drops transient / oversized fields:
      - reasoning_content (can be re-derived)
      - streaming deltas (not needed post-turn)
    """
    import copy
    safe_fields = {"role", "content", "name", "tool_calls", "tool_call_id",
                   "finish_reason", "reasoning", "author"}
    result = []
    for msg in messages:
        if isinstance(msg, dict):
            copied = {k: v for k, v in msg.items() if k in safe_fields and v is not None}
            # Deep-copy tool_calls
            if "tool_calls" in copied:
                copied["tool_calls"] = copy.deepcopy(copied["tool_calls"])
            result.append(copied)
        else:
            result.append(msg)
    return result


# ---------------------------------------------------------------------------
# CLI / tool interface
# ---------------------------------------------------------------------------

def format_turn_checkpoint_list(session_id: str) -> str:
    """Format a session's checkpoints as a readable string."""
    tc = TurnCheckpoint()
    checkpoints = tc.list_checkpoints(session_id)
    if not checkpoints:
        return f"No turn checkpoints found for session {session_id}"

    lines = [f"Turn checkpoints for session {session_id}:", ""]
    for cp in checkpoints:
        lines.append(
            f"  Turn {cp['turn_id']:4d}  |  {cp['timestamp'][:19]}  |  "
            f"tools={cp['tool_count']}  |  {cp['reason']}"
        )
    return "\n".join(lines)
