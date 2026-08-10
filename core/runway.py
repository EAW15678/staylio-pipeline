"""
Shared Runway ML configuration and helpers.

Centralises model constants, cost tables, and clip generation used by
Agent 3 (async video pipeline) and Agent 8 (sync motion rendering).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import httpx
import runwayml

from core.retry import async_retry_with_backoff

logger = logging.getLogger(__name__)

# ── API Configuration ─────────────────────────────────────────────────────
RUNWAYML_API_SECRET = os.environ.get("RUNWAYML_API_SECRET", "")

# ── Model Constants ───────────────────────────────────────────────────────
RUNWAY_MODEL_TURBO = "gen4_turbo"       # 5 credits/sec, $0.05/sec
RUNWAY_MODEL_QUALITY = "gen4.5"         # 12 credits/sec, $0.12/sec

RUNWAY_COST_PER_SEC: dict[str, float] = {
    "gen4_turbo": 0.05,
    "gen4.5": 0.12,
}


async def generate_clip(
    image_url: str,
    prompt_text: str,
    *,
    model: str = "gen4_turbo",
    duration: int = 10,
    ratio: str = "1280:720",
    seed: int | None = None,
    timeout: int = 900,
) -> tuple[bytes | None, str]:
    """
    Generate an image-to-video clip via Runway ML.
    Returns (video_bytes, model_used) or (None, model_used) on failure.

    Caller is responsible for R2 upload and cost attribution.
    """
    if not RUNWAYML_API_SECRET:
        logger.warning("[Runway] RUNWAYML_API_SECRET not set")
        return None, model

    try:
        async def _attempt():
            async with runwayml.AsyncRunwayML(api_key=RUNWAYML_API_SECRET) as client:
                create_kwargs: dict = {
                    "model": model,
                    "prompt_image": image_url,
                    "prompt_text": prompt_text,
                    "duration": duration,
                    "ratio": ratio,
                }
                if seed is not None:
                    create_kwargs["seed"] = seed

                task = await client.image_to_video.create(**create_kwargs)
                result = await task.wait_for_task_output(timeout=timeout)

                out_urls = getattr(result, "output", None) or []
                if not out_urls:
                    raise RuntimeError(
                        f"Runway returned no output for model={model}"
                    )

                async with httpx.AsyncClient(timeout=120) as dl_client:
                    dl = await dl_client.get(out_urls[0])
                    dl.raise_for_status()

                return dl.content

        video_bytes = await async_retry_with_backoff(
            fn=_attempt,
            max_retries=3,
            backoff_base=2.0,
            retryable=(
                httpx.HTTPStatusError,
                httpx.ConnectError,
                httpx.TimeoutException,
                RuntimeError,
            ),
            label="Runway clip generation",
        )
        return video_bytes, model

    except Exception as exc:
        logger.error("[Runway] Clip generation failed: %s", exc)
        return None, model
