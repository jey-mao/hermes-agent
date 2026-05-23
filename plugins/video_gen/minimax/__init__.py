"""MiniMax video generation backend.

Uses MiniMax Hailuo-2.3 via ``POST /v1/video_generation``.
Returns a task_id; caller must poll /v1/video_generation/result/<task_id>
to retrieve the video URL once ready.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.minimaxi.com/v1"

_MODELS: Dict[str, Dict[str, Any]] = {
    "MiniMax-Hailuo-2.3": {
        "display": "Hailuo 2.3",
        "speed": "~60s",
        "strengths": "Smooth motion, best quality",
        "price": "MiniMax pricing",
    },
    "MiniMax-Hailuo-02": {
        "display": "Hailuo 02",
        "speed": "~30s",
        "strengths": "Faster generation",
        "price": "MiniMax pricing",
    },
    "T2V-01": {
        "display": "T2V-01",
        "speed": "~60s",
        "strengths": "Text-to-video SOTA",
        "price": "MiniMax pricing",
    },
}

DEFAULT_MODEL = "MiniMax-Hailuo-2.3"

# Duration options per model
_DURATIONS: Dict[str, List[int]] = {
    "MiniMax-Hailuo-2.3": [6],
    "MiniMax-Hailuo-02": [6],
    "T2V-01": [5, 10],
}

_RESOLUTIONS = ["360P", "480P", "720P", "1080P"]


class MiniMaxVideoProvider:
    """Video generation provider (separate from ImageGenProvider ABC)."""

    name = "minimax_video"
    display_name = "MiniMax Video"

    def __init__(self):
        self._api_key: Optional[str] = None

    def is_available(self) -> bool:
        return bool(self._get_api_key())

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": mid,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
            }
            for mid, meta in _MODELS.items()
        ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "MiniMax Video",
            "badge": "API",
            "tag": "Hailuo 2.3 video generation via MiniMax API",
            "env_vars": [
                {
                    "key": "MINIMAX_API_KEY",
                    "prompt": "MiniMax API Key",
                    "url": "https://platform.minimaxi.com/docs/api-reference/api-overview",
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        duration: int = 6,
        resolution: str = "1080P",
        poll_timeout: int = 120,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        api_key = self._get_api_key()
        if not api_key:
            return {
                "success": False,
                "video": None,
                "error": "MINIMAX_API_KEY not set",
                "error_type": "missing_api_key",
                "provider": self.name,
                "model": model,
                "prompt": prompt,
            }

        if model not in _MODELS:
            return {
                "success": False,
                "video": None,
                "error": f"Unknown model: {model}. Available: {list(_MODELS.keys())}",
                "error_type": "invalid_model",
                "provider": self.name,
                "model": model,
                "prompt": prompt,
            }

        try:
            # 1. Submit generation task
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
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
                )
                resp.raise_for_status()
                task_data = resp.json()

            task_id = task_data.get("task_id")
            if not task_id:
                return {
                    "success": False,
                    "video": None,
                    "error": f"No task_id in response: {task_data}",
                    "error_type": "api_error",
                    "provider": self.name,
                    "model": model,
                    "prompt": prompt,
                }

            # 2. Poll for result
            poll_url = f"{BASE_URL}/video_generation/result?task_id={task_id}"
            headers = {"Authorization": f"Bearer {api_key}"}
            elapsed = 0
            interval = 3

            while elapsed < poll_timeout:
                time.sleep(interval)
                elapsed += interval

                with httpx.Client(timeout=30.0) as client:
                    resp = client.get(poll_url, headers=headers)
                    resp.raise_for_status()
                    result = resp.json()

                status = result.get("status", "").lower()
                if status == "success":
                    video_url = (
                        result.get("data", {}).get("video_url")
                        or result.get("video_url", "")
                    )
                    if video_url:
                        return {
                            "success": True,
                            "video": video_url,
                            "model": model,
                            "prompt": prompt,
                            "duration": duration,
                            "resolution": resolution,
                            "provider": self.name,
                            "task_id": task_id,
                        }
                    return {
                        "success": False,
                        "video": None,
                        "error": f"Status success but no video_url: {result}",
                        "error_type": "api_error",
                        "provider": self.name,
                        "model": model,
                        "prompt": prompt,
                    }
                elif status in ("pending", "processing", "in_progress"):
                    logger.debug("Video task %s still %s, polling...", task_id, status)
                    continue
                else:
                    return {
                        "success": False,
                        "video": None,
                        "error": f"Task failed: {result}",
                        "error_type": "generation_error",
                        "provider": self.name,
                        "model": model,
                        "prompt": prompt,
                    }

            return {
                "success": False,
                "video": None,
                "error": f"Polling timeout after {poll_timeout}s",
                "error_type": "timeout",
                "provider": self.name,
                "model": model,
                "prompt": prompt,
            }

        except httpx.HTTPStatusError as e:
            return {
                "success": False,
                "video": None,
                "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}",
                "error_type": "http_error",
                "provider": self.name,
                "model": model,
                "prompt": prompt,
            }
        except Exception as e:
            logger.exception("MiniMax video generation failed")
            return {
                "success": False,
                "video": None,
                "error": str(e),
                "error_type": "provider_error",
                "provider": self.name,
                "model": model,
                "prompt": prompt,
            }

    def _get_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        return os.environ.get("MINIMAX_API_KEY", "")


def register(ctx) -> None:
    """Plugin entry point — video generation is handled by tools/video_generation_tool.py."""
    pass
