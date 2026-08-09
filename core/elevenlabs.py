"""
Shared ElevenLabs TTS configuration and helpers.

Centralises voice IDs, voice settings, and TTS generation functions
used by Agent 3 (async video narration) and Agent 8 (sync narration assets).
"""

import logging
import os
from typing import Optional

import httpx

from pipeline_emitter import emit_media_cost

logger = logging.getLogger(__name__)

# ── API Configuration ─────────────────────────────────────────────────────
ELEVENLABS_API_KEY  = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"
ELEVENLABS_MODEL    = "eleven_multilingual_v2"

# ── Voice Configuration ──────────────────────────────────────────────────
# All 7 vibes from models/property.py VibeProfile enum.
VIBE_VOICE_IDS: dict[str, str] = {
    "romantic_escape":           os.environ.get("VOICE_ROMANTIC_ESCAPE", ""),
    "family_adventure":          os.environ.get("VOICE_FAMILY_ADVENTURE", ""),
    "multigenerational_retreat": os.environ.get("VOICE_MULTIGENERATIONAL", ""),
    "wellness_retreat":          os.environ.get("VOICE_WELLNESS", ""),
    "adventure_base_camp":       os.environ.get("VOICE_ADVENTURE", ""),
    "social_celebrations":       os.environ.get("VOICE_SOCIAL", ""),
    "creative_remote_work":      os.environ.get("VOICE_CREATIVE", ""),
}

# Human-readable labels for review surfaces
VIBE_VOICE_LABELS: dict[str, str] = {
    "romantic_escape":           "Romantic Escape voice",
    "family_adventure":          "Family Adventure voice",
    "multigenerational_retreat": "Multi-Generational voice",
    "wellness_retreat":          "Wellness Retreat voice",
    "adventure_base_camp":       "Adventure Base Camp voice",
    "social_celebrations":       "Social & Celebrations voice",
    "creative_remote_work":      "Creative & Remote Work voice",
}

# Shared voice settings used by both async and sync TTS calls.
DEFAULT_VOICE_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.3,
}


async def async_generate_tts(
    script: str,
    voice_id: str,
    property_id: str,
    label: str,
) -> Optional[bytes]:
    """
    Generate TTS audio via ElevenLabs API (async).
    Returns raw audio bytes on success, None on failure.
    Caller is responsible for uploading to R2.
    """
    from core.retry import async_retry_with_backoff

    if not ELEVENLABS_API_KEY or not voice_id:
        logger.warning(
            "[ElevenLabs] Not configured (key=%s, voice=%s) for property %s",
            bool(ELEVENLABS_API_KEY), bool(voice_id), property_id,
        )
        return None

    try:
        async def _attempt():
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{ELEVENLABS_API_BASE}/text-to-speech/{voice_id}",
                    headers={
                        "xi-api-key": ELEVENLABS_API_KEY,
                        "Content-Type": "application/json",
                    },
                    json={
                        "text": script,
                        "model_id": ELEVENLABS_MODEL,
                        "voice_settings": DEFAULT_VOICE_SETTINGS,
                    },
                )
                resp.raise_for_status()
                return resp.content

        audio_bytes = await async_retry_with_backoff(
            fn=_attempt,
            max_retries=3,
            backoff_base=2.0,
            retryable=(httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException),
            label="ElevenLabs TTS",
        )

        emit_media_cost(
            vendor="elevenlabs",
            service="tts",
            units=len(script),
            unit_name="characters",
            property_id=property_id,
            workflow_name="video_generation",
            generation_reason="narration",
            discriminator=label,
        )

        return audio_bytes

    except Exception as exc:
        logger.error("[ElevenLabs] TTS failed for property %s: %s", property_id, exc)
        return None


def sync_generate_tts(
    script: str,
    voice_id: str,
) -> Optional[bytes]:
    """
    Generate TTS audio via ElevenLabs API (synchronous).
    Returns raw audio bytes on success, None on failure.
    Caller is responsible for uploading to R2.

    Does NOT emit cost — caller handles attribution.
    Does NOT retry — caller wraps with its own retry strategy.
    """
    if not ELEVENLABS_API_KEY or not voice_id:
        logger.warning("[ElevenLabs] Not configured for sync TTS")
        return None

    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{ELEVENLABS_API_BASE}/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": ELEVENLABS_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "text": script,
                    "model_id": ELEVENLABS_MODEL,
                    "voice_settings": DEFAULT_VOICE_SETTINGS,
                },
            )
            resp.raise_for_status()
            return resp.content
    except Exception as exc:
        logger.error("[ElevenLabs] Sync TTS failed: %s", exc)
        return None
