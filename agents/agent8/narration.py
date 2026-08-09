"""
Agent 8 — Stage 4: Narration

Decouples narration from the landing-page video into standalone,
addressable, reusable assets in public.narration_assets. The ElevenLabs
call already works in production (agents/agent3/video_generator.py) —
this is a decoupling, not new construction.

The existing implementation is async, private, and coupled to Agent 3's
R2 upload helpers. Rather than modifying Agent 3 (Agents 1-7 are not to
be modified), this module replicates the same API call pattern — same
endpoint, same model (eleven_multilingual_v2), same voice settings, same
env vars — in a synchronous wrapper.

Standalone module. NOT wired into the pipeline graph.
Call directly: render_narration(spec_id)
"""

import hashlib
import json
import logging
import os
from typing import Optional

import httpx

from agents.agent8.retry import retry_with_backoff
from agents.agent8.creative_director import validate_no_guest_names, _extract_guest_book_entries

logger = logging.getLogger(__name__)

from core.elevenlabs import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_API_BASE,
    ELEVENLABS_MODEL,
    VIBE_VOICE_IDS,
    VIBE_VOICE_LABELS,
    DEFAULT_VOICE_SETTINGS,
)


def _compute_input_hash(script: str, voice_id: str, model: str, language: str) -> str:
    """Cache key: hash of script_text + voice_id + model + language."""
    blob = json.dumps(
        {"script": script, "voice_id": voice_id, "model": model, "language": language},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _detect_verbatim_source(spec: dict) -> tuple[bool, list[dict]]:
    """
    Determine if narration contains reproduced source material.
    Returns (contains_verbatim, source_references).
    """
    prov = spec.get("narration_provenance") or ""
    if "guest_book" in prov:
        # Build source_reference from the spec's source material
        refs = []
        for mat in spec.get("source_material") or []:
            if isinstance(mat, str) and mat.startswith("guest_book"):
                refs.append({"kind": "guest_book", "ref": mat})
        return True, refs
    return False, []


def render_narration(
    spec_id: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> list[dict]:
    """
    Render narration assets from a shot_spec's narration_brief.

    NOT every concept carries narration — that is a directorial choice,
    not a defect. No brief means no assets and a clean return.

    Parameter semantics and cost:

        dry_run=False, force=False (default)
            If cached narration exists (input_hash match): returns it. $0.
            If not: renders via ElevenLabs, writes to DB + R2. ~$0.01-0.05.

        dry_run=True, force=False
            If cached: returns it. NO vendor call. $0.
            If not: returns what WOULD be rendered, writes nothing. $0.

        dry_run=False, force=True
            Ignores cache, re-renders, writes (supersedes prior). ~$0.01-0.05.

        dry_run=True, force=True
            Ignores cache, returns spec without rendering. $0.

    Cost: ElevenLabs TTS is billed per character. ~$0.01 per 500 chars.

    Args:
        spec_id: UUID of the shot_spec row.
        dry_run: If True, return asset specs without rendering or writing.
        force: If True, ignore cached narration and re-render.

    Returns:
        List of narration asset dicts. Empty list if no narration_brief.
    """
    logger.info(
        "[Agent 8 Stage 4] render_narration: spec_id=%s dry_run=%s force=%s",
        spec_id, dry_run, force,
    )

    # ── Load shot_spec ───────────────────────────────────────────────────
    spec = _load_spec(spec_id)
    if not spec:
        raise ValueError(f"Shot spec {spec_id} not found or not active")

    narration_brief = spec.get("narration_brief") or ""
    if not narration_brief.strip():
        logger.info("[Agent 8 Stage 4] No narration_brief on spec %s — clean return", spec_id)
        return []

    property_id = spec["property_id"]
    concept_id = spec.get("concept_id")
    vibe = spec.get("vibe_drift") or ""

    # Load KB for vibe and guest name validation
    kb = _load_kb(property_id)
    vibe_profile = (kb or {}).get("vibe_profile") or "unknown" if kb else "unknown"

    # ── Governance: no guest names ───────────────────────────────────────
    if kb:
        name_violations = validate_no_guest_names(
            {"narration_brief": narration_brief, "overlay_register": []},
            kb,
        )
        if name_violations:
            logger.error(
                "[Agent 8 Stage 4] Guest name found in narration_brief — refusing to render: %s",
                name_violations,
            )
            return [{
                "status": "failed",
                "failure_reason": f"Guest name in narration: {name_violations[0]['detail']}",
                "spec_id": spec_id,
            }]

    # ── Voice selection ──────────────────────────────────────────────────
    voice_id = VIBE_VOICE_IDS.get(vibe_profile, "")
    voice_label = VIBE_VOICE_LABELS.get(vibe_profile, f"{vibe_profile} voice")

    if not voice_id:
        logger.warning("[Agent 8 Stage 4] No voice configured for vibe %s", vibe_profile)

    # ── Verbatim source detection ────────────────────────────────────────
    contains_verbatim, source_refs = _detect_verbatim_source(spec)

    # ── Build asset spec ─────────────────────────────────────────────────
    language = "en"
    input_hash = _compute_input_hash(narration_brief, voice_id, ELEVENLABS_MODEL, language)

    asset = {
        "property_id": property_id,
        "spec_id": spec_id,
        "concept_id": concept_id,
        "script_text": narration_brief,
        "provenance": spec.get("narration_provenance") or "original",
        "source_reference": source_refs if contains_verbatim else [],
        "voice_id": voice_id,
        "voice_label": voice_label,
        "vibe_prior": vibe_profile,
        "language": language,
        "model": ELEVENLABS_MODEL,
        "input_hash": input_hash,
        "character_count": len(narration_brief),
        "contains_verbatim_source": contains_verbatim,
        "created_by_agent": "agent8_stage4",
    }

    # ── Check cache ──────────────────────────────────────────────────────
    if not force:
        cached = _check_cached_narration(property_id, input_hash)
        if cached:
            logger.info("[Agent 8 Stage 4] Cache hit for input_hash %s", input_hash[:12])
            return [cached]

    if dry_run:
        asset["status"] = "pending"
        asset["r2_url"] = None
        asset["duration_seconds"] = None
        asset["alignment"] = None
        return [asset]

    # ── Render ───────────────────────────────────────────────────────────
    render_result = _render_elevenlabs(narration_brief, voice_id, property_id)

    if render_result is None:
        asset["status"] = "failed"
        asset["failure_reason"] = "ElevenLabs TTS call failed after retries"
        asset["r2_url"] = None
        asset["duration_seconds"] = None
        asset["alignment"] = None
        _persist_narration(asset, property_id, input_hash)
        return [asset]

    asset["r2_url"] = render_result["r2_url"]
    asset["duration_seconds"] = render_result.get("duration_seconds")
    asset["alignment"] = render_result.get("alignment")
    asset["status"] = "ready"

    _persist_narration(asset, property_id, input_hash)

    logger.info(
        "[Agent 8 Stage 4] Narration rendered: %d chars, voice=%s, r2=%s",
        len(narration_brief), voice_label, asset["r2_url"],
    )

    return [asset]


# ── Internal helpers ─────────────────────────────────────────────────────

def _load_spec(spec_id: str) -> Optional[dict]:
    """Load active shot_spec by ID."""
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("shot_spec")
            .select("*")
            .eq("spec_id", spec_id)
            .is_("superseded_at", "null")
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None
    except Exception as exc:
        logger.error("[Agent 8 Stage 4] Failed to load spec %s: %s", spec_id, exc)
        return None


def _load_kb(property_id: str) -> Optional[dict]:
    """Load KB — Redis first, Supabase fallback."""
    from core.pipeline_status import get_cached_knowledge_base
    kb = get_cached_knowledge_base(property_id)
    if kb:
        return kb
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("property_knowledge_bases")
            .select("knowledge_base")
            .eq("property_id", property_id)
            .limit(1)
            .execute()
        )
        if result.data and result.data[0].get("knowledge_base"):
            return result.data[0]["knowledge_base"]
    except Exception as exc:
        logger.warning("[Agent 8 Stage 4] KB load failed: %s", exc)
    return None


def _check_cached_narration(property_id: str, input_hash: str) -> Optional[dict]:
    """Check narration_assets for an active row with this input_hash."""
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("narration_assets")
            .select("*")
            .eq("property_id", property_id)
            .eq("input_hash", input_hash)
            .eq("status", "ready")
            .is_("superseded_at", "null")
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None
    except Exception as exc:
        logger.warning("[Agent 8 Stage 4] Cache check failed: %s", exc)
        return None


def _render_elevenlabs(script: str, voice_id: str, property_id: str) -> Optional[dict]:
    """
    Call ElevenLabs TTS API, upload audio to R2, return result.

    Uses the with-timestamps endpoint when available for forced alignment.
    Falls back to the standard endpoint if timestamps fail.

    Replicates the same API pattern as agents/agent3/video_generator.py:401
    (same endpoint, model eleven_multilingual_v2, same voice_settings) but
    synchronous and with retry via agents.agent8.retry.

    NOTE on forced alignment: ElevenLabs offers word-level timestamps via
    the /text-to-speech/{voice_id}/with-timestamps endpoint. This returns
    JSON with audio_base64 + alignment data. If this endpoint is
    unavailable or fails, we fall back to the standard endpoint (audio
    bytes only, alignment=None). SOC-23b: alignment is what lets Stage 7
    pin a phrase to a beat boundary instead of laying narration flat.
    """
    if not ELEVENLABS_API_KEY:
        logger.warning("[Agent 8 Stage 4] ELEVENLABS_API_KEY not set")
        return None

    if not voice_id:
        logger.warning("[Agent 8 Stage 4] No voice_id provided")
        return None

    payload = {
        "text": script,
        "model_id": ELEVENLABS_MODEL,
        "voice_settings": DEFAULT_VOICE_SETTINGS,
    }
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }

    # Try with-timestamps first for forced alignment
    alignment = None
    audio_bytes = None

    try:
        def _call_with_timestamps():
            with httpx.Client(timeout=60) as client:
                resp = client.post(
                    f"{ELEVENLABS_API_BASE}/text-to-speech/{voice_id}/with-timestamps",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                return resp.json()

        result = retry_with_backoff(
            fn=_call_with_timestamps,
            max_retries=2,
            backoff_base=2.0,
            retryable=(httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException),
            label="ElevenLabs TTS with timestamps",
        )

        import base64
        audio_bytes = base64.b64decode(result.get("audio_base64", ""))
        alignment = result.get("alignment")

    except Exception as exc:
        logger.warning(
            "[Agent 8 Stage 4] With-timestamps failed, falling back to standard: %s", exc
        )

        # Fallback: standard TTS (no alignment)
        try:
            def _call_standard():
                with httpx.Client(timeout=60) as client:
                    resp = client.post(
                        f"{ELEVENLABS_API_BASE}/text-to-speech/{voice_id}",
                        headers=headers,
                        json=payload,
                    )
                    resp.raise_for_status()
                    return resp.content

            audio_bytes = retry_with_backoff(
                fn=_call_standard,
                max_retries=2,
                backoff_base=2.0,
                retryable=(httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException),
                label="ElevenLabs TTS standard",
            )
        except Exception as exc2:
            logger.error("[Agent 8 Stage 4] ElevenLabs TTS failed: %s", exc2)
            return None

    if not audio_bytes:
        return None

    # Upload to R2
    try:
        from core.r2_storage import upload_video
        r2_url = upload_video(
            property_id=property_id,
            video_bytes=audio_bytes,
            video_type="narration",
            format_label="mp3",
            extension="mp3",
        )
    except Exception as exc:
        logger.error("[Agent 8 Stage 4] R2 upload failed: %s", exc)
        return None

    # Estimate duration from audio size (rough: MP3 at ~128kbps)
    duration_seconds = round(len(audio_bytes) / (128 * 1024 / 8), 1) if audio_bytes else None

    return {
        "r2_url": r2_url,
        "duration_seconds": duration_seconds,
        "alignment": alignment,
    }


def _persist_narration(asset: dict, property_id: str, input_hash: str) -> None:
    """Write narration asset to Supabase, superseding any prior active row with same input_hash."""
    try:
        from core.supabase_store import get_supabase
        sb = get_supabase()

        # Supersede any prior active row with same input_hash
        sb.table("narration_assets").update(
            {"superseded_at": "now()"}
        ).eq("property_id", property_id).eq("input_hash", input_hash).is_(
            "superseded_at", "null"
        ).execute()

        # Insert new row
        sb.table("narration_assets").insert(asset).execute()

    except Exception as exc:
        logger.error("[Agent 8 Stage 4] Persist failed: %s", exc)
