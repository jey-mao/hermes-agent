#!/usr/bin/env python3
"""
Video Generation Tool Module

Uses MiniMax Hailuo-2.3 via the plugins/video_gen/minimax/ provider.
Sends a generation request, polls for completion, returns the video URL.

Available models:
- MiniMax-Hailuo-2.3 (default): Smooth motion, highest quality
- MiniMax-Hailuo-02: Faster generation
- T2V-01: Text-to-video SOTA

Usage:
    from tools.video_generation_tool import video_generate_tool, check_video_generation_requirements

    result = video_generate_tool(
        prompt="A beautiful sunset over the ocean",
        model="MiniMax-Hailuo-2.3",
        duration=6,
        resolution="1080P",
    )
"""

import logging
import os
import time
from typing import Any, Dict, Optional

import httpx

from tools.debug_helpers import DebugSession
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

_debug = DebugSession("video_generation", env_var="VIDEO_GENERATION_DEBUG")

BASE_URL = "https://api.minimaxi.com/v1"

_MODELS: Dict[str, Dict[str, Any]] = {
    "MiniMax-Hailuo-2.3": {
        "display": "Hailuo 2.3",
        "speed": "~60s",
        "strengths": "Smooth motion, best quality",
    },
    "MiniMax-Hailuo-02": {
        "display": "Hailuo 02",
        "speed": "~30s",
        "strengths": "Faster generation",
    },
    "T2V-01": {
        "display": "T2V-01",
        "speed": "~60s",
        "strengths": "Text-to-video SOTA",
    },
}

DEFAULT_MODEL = "MiniMax-Hailuo-2.3"
DEFAULT_DURATION = 6
DEFAULT_RESOLUTION = "1080P"
VALID_RESOLUTIONS = ["360P", "480P", "720P", "1080P"]


def _get_api_key() -> str:
    return os.environ.get("MINIMAX_API_KEY", "")


def check_video_generation_requirements() -> bool:
    """True when MINIMAX_API_KEY is set."""
    return bool(_get_api_key())


# ---------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------

def video_generate_tool(
    prompt: str,
    model: str = DEFAULT_MODEL,
    duration: int = DEFAULT_DURATION,
    resolution: str = DEFAULT_RESOLUTION,
    poll_timeout: int = 120,
) -> Dict[str, Any]:
    """Generate a video from a text prompt using MiniMax Hailuo.

    Submits the task, polls until complete (or timeout), and returns
    the video URL on success.

    Returns a dict with keys:
        success       bool
        video         str | None   URL of the generated video
        model         str
        prompt        str
        duration      int
        resolution    str
        task_id       str
        error         str (only when success=False)
        error_type    str (only when success=False)
    """
    if _debug.active:
        logger.debug("video_generate_tool called with prompt=%r model=%s", prompt, model)

    if not prompt or not prompt.strip():
        return _error_dict("prompt is required", model=model)

    api_key = _get_api_key()
    if not api_key:
        return _error_dict(
            "MINIMAX_API_KEY is not set",
            error_type="missing_api_key",
            model=model,
            prompt=prompt,
        )

    if model not in _MODELS:
        return _error_dict(
            f"Unknown model: {model}. Available: {list(_MODELS.keys())}",
            error_type="invalid_model",
            model=model,
            prompt=prompt,
        )

    if resolution not in VALID_RESOLUTIONS:
        resolution = DEFAULT_RESOLUTION

    try:
        # 1. Submit generation task
        submit_resp = httpx.post(
            f"{BASE_URL}/video_generation",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "prompt": prompt,
                "duration": duration,
                "resolution": resolution,
            },
            timeout=30.0,
        )
        submit_resp.raise_for_status()
        task_data = submit_resp.json()

        task_id = task_data.get("task_id")
        if not task_id:
            return _error_dict(
                f"No task_id in response: {task_data}",
                error_type="api_error",
                model=model,
                prompt=prompt,
            )

        logger.info("Video task submitted: task_id=%s", task_id)

        # 2. Poll for result
        poll_url = f"{BASE_URL}/video_generation/result?task_id={task_id}"
        headers = {"Authorization": f"Bearer {api_key}"}
        elapsed = 0
        interval = 3

        while elapsed < poll_timeout:
            time.sleep(interval)
            elapsed += interval

            poll_resp = httpx.get(poll_url, headers=headers, timeout=30.0)
            poll_resp.raise_for_status()
            result = poll_resp.json()

            status = str(result.get("status", "")).lower()
            if status == "success":
                video_url = (
                    result.get("data", {})
                    .get("video_url")
                    or result.get("video_url", "")
                )
                if not video_url:
                    return _error_dict(
                        f"Status success but no video_url: {result}",
                        error_type="api_error",
                        model=model,
                        prompt=prompt,
                        task_id=task_id,
                    )
                return {
                    "success": True,
                    "video": video_url,
                    "model": model,
                    "prompt": prompt,
                    "duration": duration,
                    "resolution": resolution,
                    "task_id": task_id,
                }
            elif status in ("pending", "processing", "in_progress"):
                logger.debug("Task %s status=%s, polling again...", task_id, status)
                continue
            else:
                return _error_dict(
                    f"Generation failed: {result}",
                    error_type="generation_error",
                    model=model,
                    prompt=prompt,
                    task_id=task_id,
                )

        return _error_dict(
            f"Polling timeout after {poll_timeout}s (task_id={task_id})",
            error_type="timeout",
            model=model,
            prompt=prompt,
            task_id=task_id,
        )

    except httpx.HTTPStatusError as e:
        return _error_dict(
            f"HTTP {e.response.status_code}: {e.response.text[:300]}",
            error_type="http_error",
            model=model,
            prompt=prompt,
        )
    except Exception as e:
        logger.exception("video_generate_tool failed")
        return _error_dict(str(e), model=model, prompt=prompt)


def _error_dict(
    error: str,
    error_type: str = "provider_error",
    model: str = DEFAULT_MODEL,
    prompt: str = "",
    task_id: str = "",
) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "success": False,
        "video": None,
        "error": error,
        "error_type": error_type,
        "model": model,
        "prompt": prompt,
    }
    if task_id:
        d["task_id"] = task_id
    return d


# -------------------------------------------------------------------------
# Schema + handler
# -------------------------------------------------------------------------

VIDEO_GENERATE_SCHEMA = {
    "name": "video_generate",
    "description": (
        "Generate a video from a text prompt using MiniMax Hailuo 2.3. "
        "Sends the prompt, waits for generation (~30-60s), returns a video URL. "
        "Display it with MEDIA:<url> and the platform will deliver it as a video."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "Text description of the video to generate. "
                    "Max 2000 characters. Use [运镜指令] for camera control "
                    "(e.g. [推进], [左移], [固定])."
                ),
            },
            "model": {
                "type": "string",
                "enum": list(_MODELS.keys()),
                "description": f"Video model. Defaults to {DEFAULT_MODEL}.",
                "default": DEFAULT_MODEL,
            },
            "duration": {
                "type": "integer",
                "description": f"Video duration in seconds. Defaults to {DEFAULT_DURATION}.",
                "default": DEFAULT_DURATION,
            },
            "resolution": {
                "type": "string",
                "enum": VALID_RESOLUTIONS,
                "description": f"Video resolution. Defaults to {DEFAULT_RESOLUTION}.",
                "default": DEFAULT_RESOLUTION,
            },
        },
        "required": ["prompt"],
    },
}


def _handle_video_generate(args, **kw) -> str:
    prompt = args.get("prompt", "")
    if not prompt:
        return tool_error("prompt is required for video generation")

    result = video_generate_tool(
        prompt=prompt,
        model=args.get("model", DEFAULT_MODEL),
        duration=args.get("duration", DEFAULT_DURATION),
        resolution=args.get("resolution", DEFAULT_RESOLUTION),
    )
    import json
    return json.dumps(result)


registry.register(
    name="video_generate",
    toolset="video_gen",
    schema=VIDEO_GENERATE_SCHEMA,
    handler=_handle_video_generate,
    check_fn=check_video_generation_requirements,
    requires_env=["MINIMAX_API_KEY"],
    is_async=False,
    emoji="🎬",
)


# -------------------------------------------------------------------------
# CLI / diagnostics
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("🎬 Video Generation Tool — MiniMax Hailuo")
    print("=" * 50)

    if not check_video_generation_requirements():
        print("❌ MINIMAX_API_KEY not set")
        print("   Set via: export MINIMAX_API_KEY='your-key-here'")
        print("   Get a key: https://platform.minimaxi.com/")
        raise SystemExit(1)

    print("✅ MINIMAX_API_KEY is set")
    print("\nAvailable models:")
    for mid, meta in _MODELS.items():
        marker = " ← default" if mid == DEFAULT_MODEL else ""
        print(f"  {mid:<25} {meta['speed']:<10} {meta['strengths']}{marker}")
