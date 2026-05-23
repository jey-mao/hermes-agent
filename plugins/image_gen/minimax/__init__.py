"""MiniMax image generation backend.

Uses MiniMax Image-01 model via ``POST /v1/image_generation``.
Saves returned image URLs; the tool wrapper handles display.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import httpx

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    success_response,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.minimaxi.com/v1"

_MODELS: Dict[str, Dict[str, Any]] = {
    "image-01": {
        "display": "Image-01",
        "speed": "~10s",
        "strengths": "High fidelity, photorealistic",
        "price": "MiniMax pricing",
    },
}

DEFAULT_MODEL = "image-01"

_ASPECT_RATIO_MAP = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}


class MiniMaxImageProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "minimax"

    @property
    def display_name(self) -> str:
        return "MiniMax"

    def is_available(self) -> bool:
        return bool(self._get_api_key())

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": mid,
                "display": meta["display"],
                "speed": meta["speed"],
                "strengths": meta["strengths"],
                "price": meta["price"],
            }
            for mid, meta in _MODELS.items()
        ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "MiniMax",
            "badge": "API",
            "tag": "Image-01 model via MiniMax API",
            "env_vars": [
                {
                    "key": "MINIMAX_API_KEY",
                    "prompt": "MiniMax API Key",
                    "url": "https://platform.minimaxi.com/docs/api-reference/api-overview",
                },
            ],
        }

    def default_model(self) -> str:
        return DEFAULT_MODEL

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        api_key = self._get_api_key()
        if not api_key:
            return error_response(
                error="MINIMAX_API_KEY not set",
                error_type="missing_api_key",
                provider=self.name,
                model=DEFAULT_MODEL,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
            )

        resolved_ar = resolve_aspect_ratio(aspect_ratio)
        mm_ar = _ASPECT_RATIO_MAP.get(resolved_ar, "1:1")

        # Resolve model from config or default
        model = os.environ.get(
            "MINIMAX_IMAGE_MODEL",
            os.environ.get("MINIMAX_API_KEY", ""),  # will be overridden by actual model
        )
        # Actually just use default
        model = DEFAULT_MODEL

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{BASE_URL}/image_generation",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "prompt": prompt,
                        "aspect_ratio": mm_ar,
                        "response_format": "url",
                        "n": 1,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            if data.get("base_resp", {}).get("status_code") != 0:
                return error_response(
                    error=data.get("base_resp", {}).get("status_msg", "Unknown error"),
                    error_type="api_error",
                    provider=self.name,
                    model=model,
                    prompt=prompt,
                    aspect_ratio=resolved_ar,
                )

            image_urls = data.get("data", {}).get("image_urls", [])
            if not image_urls:
                return error_response(
                    error="No image URLs in response",
                    error_type="api_error",
                    provider=self.name,
                    model=model,
                    prompt=prompt,
                    aspect_ratio=resolved_ar,
                )

            return success_response(
                image=image_urls[0],
                model=model,
                prompt=prompt,
                aspect_ratio=resolved_ar,
                provider=self.name,
                extra={"image_urls": image_urls},
            )

        except httpx.HTTPStatusError as e:
            return error_response(
                error=f"HTTP {e.response.status_code}: {e.response.text[:200]}",
                error_type="http_error",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=resolved_ar,
            )
        except Exception as e:
            logger.exception("MiniMax image generation failed")
            return error_response(
                error=str(e),
                error_type="provider_error",
                provider=self.name,
                model=model,
                prompt=prompt,
                aspect_ratio=resolved_ar,
            )

    def _get_api_key(self) -> str:
        return os.environ.get("MINIMAX_API_KEY", "")


def register(ctx) -> None:
    """Plugin entry point — wire MiniMaxImageProvider into the image gen registry."""
    ctx.register_image_gen_provider(MiniMaxImageProvider())
