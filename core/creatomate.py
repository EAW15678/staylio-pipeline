"""
Shared Creatomate configuration and helpers.

Centralises API constants, cost tables, and render polling used by
Agent 3 (async video pipeline) and Agent 8 (sync assembly rendering).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import httpx

from core.retry import async_retry_with_backoff

logger = logging.getLogger(__name__)

# ── API Configuration ─────────────────────────────────────────────────────
CREATOMATE_API_KEY = os.environ.get("CREATOMATE_API_KEY", "")
CREATOMATE_API_BASE = "https://api.creatomate.com/v2"

# ── Cost ──────────────────────────────────────────────────────────────────
# Essential tier, 1080p renders: $0.84 per render-minute.
CREATOMATE_COST_PER_RENDER_MINUTE = 0.84


async def render_video(
    template_id: str,
    modifications: dict,
    *,
    timeout_polls: int = 60,
    poll_interval: int = 5,
) -> tuple[bytes | None, str | None]:
    """
    Submit a render to Creatomate and poll for completion.
    Returns (video_bytes, render_id) or (None, None) on failure.
    Downloads the output immediately — vendor URLs expire after 30 days.
    """
    if not CREATOMATE_API_KEY:
        logger.warning("[Creatomate] CREATOMATE_API_KEY not set")
        return None, None

    headers = {
        "Authorization": f"Bearer {CREATOMATE_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=300) as client:

            async def _submit_and_poll():
                # Submit render
                render_resp = await client.post(
                    f"{CREATOMATE_API_BASE}/renders",
                    headers=headers,
                    json={
                        "template_id": template_id,
                        "modifications": modifications,
                    },
                )
                render_resp.raise_for_status()
                renders = render_resp.json()
                render = renders[0] if isinstance(renders, list) else renders
                render_id = render.get("id")
                if not render_id:
                    return None, None

                # Poll for completion
                for _ in range(timeout_polls):
                    await asyncio.sleep(poll_interval)
                    try:
                        resp = await client.get(
                            f"{CREATOMATE_API_BASE}/renders/{render_id}",
                            headers={"Authorization": f"Bearer {CREATOMATE_API_KEY}"},
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        status = data.get("status")
                        if status == "succeeded":
                            output_url = data.get("url")
                            if not output_url:
                                return None, render_id
                            # Download immediately — vendor URLs expire
                            dl = await client.get(output_url, timeout=120)
                            dl.raise_for_status()
                            return dl.content, render_id
                        elif status == "failed":
                            logger.error(
                                "[Creatomate] Render %s failed: %s",
                                render_id, data.get("error_message"),
                            )
                            return None, render_id
                    except Exception as exc:
                        logger.warning("[Creatomate] Poll error: %s", exc)

                logger.error("[Creatomate] Render %s timed out", render_id)
                return None, render_id

            return await _submit_and_poll()

    except Exception as exc:
        logger.error("[Creatomate] Render failed: %s", exc)
        return None, None
