"""
Agent 8 — Stage 5: Music

Generates background music tracks from a shot_spec's music_brief via
ElevenLabs Music API. One track per concept. Follows the narration.py
(Stage 4) pattern exactly.

Standalone module. NOT wired into the pipeline graph.
Call directly: render_music(spec_id)
"""

import hashlib
import json
import logging
import re
import os
from typing import Optional

from agents.agent8.creative_director import validate_music_brief_prohibited

logger = logging.getLogger(__name__)

# ── Prohibited artist names (common in music prompts) ───────────────────
_PROHIBITED_ARTIST_NAMES = [
    "bon iver", "hans zimmer", "radiohead", "billie eilish", "norah jones",
    "tycho", "brian eno", "sigur ros", "olafur arnalds", "ludovico einaudi",
    "max richter", "boards of canada", "aphex twin", "john williams",
    "ennio morricone", "explosions in the sky", "mogwai", "sufjan stevens",
    "fleet foxes", "iron and wine", "tame impala", "khruangbin",
    "bonobo", "four tet", "caribou", "james blake", "frank ocean",
    "taylor swift", "beyonce", "drake",
]

# Patterns that reference specific artists/works
_REFERENCE_PATTERNS = [
    re.compile(r"\blike\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", re.UNICODE),
    re.compile(r"\binspired\s+by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", re.UNICODE),
    re.compile(r"\bin\s+the\s+style\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", re.UNICODE),
]


def _compute_input_hash(
    prompt_text: Optional[str],
    composition_plan: Optional[dict],
    duration_ms: Optional[int],
    model: str,
) -> str:
    """Cache key: hash of (prompt_text or composition_plan) + duration + model."""
    blob = json.dumps(
        {
            "prompt_text": prompt_text,
            "composition_plan": composition_plan,
            "duration_ms": duration_ms,
            "model": model,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _run_prohibited_input_check(music_brief: dict) -> tuple[str, Optional[str]]:
    """
    Two-layer prohibited input check:
    1. Reuse validate_music_brief_prohibited from creative_director.py
    2. Explicit pattern + artist name scan

    Returns (status, failure_reason) where status is 'passed' | 'failed'.
    """
    # Layer 1: creative_director heuristic
    violations = validate_music_brief_prohibited({"music_brief": music_brief})
    if violations:
        detail = violations[0]["detail"]
        return "failed", f"Prohibited input detected: {detail}"

    # Layer 2: explicit artist name + reference pattern check
    brief_text = json.dumps(music_brief, default=str)
    brief_lower = brief_text.lower()

    # Check against known artist names
    for name in _PROHIBITED_ARTIST_NAMES:
        if name in brief_lower:
            return "failed", f"Prohibited input detected: artist name '{name}' found in music brief"

    # Check for "like [Proper Noun]", "inspired by [Name]", "in the style of [Name]"
    for pattern in _REFERENCE_PATTERNS:
        match = pattern.search(brief_text)
        if match:
            return "failed", f"Prohibited input detected: reference pattern '{match.group(0)}' found in music brief"

    return "passed", None


def _has_structured_sections(music_brief: dict) -> bool:
    """Determine if brief has structured sections/beats suitable for composition_plan."""
    sections = music_brief.get("sections") or music_brief.get("beats") or []
    return isinstance(sections, list) and len(sections) > 0


def _build_composition_plan(music_brief: dict) -> dict:
    """Convert structured music_brief sections into an ElevenLabs composition_plan."""
    sections = music_brief.get("sections") or music_brief.get("beats") or []
    chunks = []
    for section in sections:
        chunk = {
            "text": section.get("description") or section.get("text") or "",
            "duration_ms": int(section.get("duration_ms", 15000)),
        }
        if section.get("positive_styles"):
            chunk["positive_styles"] = section["positive_styles"]
        if section.get("negative_styles"):
            chunk["negative_styles"] = section["negative_styles"]
        chunks.append(chunk)
    return {"chunks": chunks}


def _build_prompt_text(music_brief: dict) -> str:
    """Extract prompt text from an unstructured music_brief."""
    if isinstance(music_brief.get("prompt"), str):
        return music_brief["prompt"]
    if isinstance(music_brief.get("description"), str):
        return music_brief["description"]
    # Fallback: serialize the brief minus any non-prompt keys
    parts = []
    for key in ("mood", "genre", "tempo", "instruments", "description", "prompt"):
        val = music_brief.get(key)
        if val:
            parts.append(f"{key}: {val}" if not isinstance(val, str) else val)
    return "; ".join(parts) if parts else json.dumps(music_brief, default=str)


def render_music(
    spec_id: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> Optional[dict]:
    """
    Render a music track from a shot_spec's music_brief.

    NOT every concept carries music — that is a directorial choice,
    not a defect. No brief means no asset and a clean return (None).

    Parameter semantics and cost:

        dry_run=False, force=False (default)
            If cached music exists (input_hash match): returns it. $0.
            If not: renders via ElevenLabs, writes to DB + R2. ~$0.05-0.20.

        dry_run=True, force=False
            If cached: returns it. NO vendor call. $0.
            If not: returns what WOULD be rendered, writes nothing. $0.

        dry_run=False, force=True
            Ignores cache, re-renders, writes (supersedes prior). ~$0.05-0.20.

        dry_run=True, force=True
            Ignores cache, returns spec without rendering. $0.

    Args:
        spec_id: UUID of the shot_spec row.
        dry_run: If True, return asset spec without rendering or writing.
        force: If True, ignore cached music and re-render.

    Returns:
        Music asset dict, or None if no music_brief.
    """
    logger.info(
        "[Agent 8 Stage 5] render_music: spec_id=%s dry_run=%s force=%s",
        spec_id, dry_run, force,
    )

    # ── Load shot_spec ───────────────────────────────────────────────────
    spec = _load_spec(spec_id)
    if not spec:
        raise ValueError(f"Shot spec {spec_id} not found or not active")

    music_brief = spec.get("music_brief")
    if not music_brief:
        logger.info("[Agent 8 Stage 5] No music_brief on spec %s — clean return", spec_id)
        return None

    property_id = spec["property_id"]
    concept_id = spec.get("concept_id")

    # ── Governance: prohibited input check ──────────────────────────────
    check_status, failure_reason = _run_prohibited_input_check(music_brief)

    if check_status == "failed":
        logger.error(
            "[Agent 8 Stage 5] Prohibited input in music_brief — refusing to render: %s",
            failure_reason,
        )
        return {
            "property_id": property_id,
            "spec_id": spec_id,
            "concept_id": concept_id,
            "status": "failed",
            "failure_reason": failure_reason,
            "prohibited_input_check": "failed",
            "created_by_agent": "agent8_stage5",
        }

    # ── Determine path: composition_plan vs prompt_text ─────────────────
    from core.elevenlabs import ELEVENLABS_MUSIC_MODEL

    composition_plan = None
    prompt_text = None

    if _has_structured_sections(music_brief):
        composition_plan = _build_composition_plan(music_brief)
    else:
        prompt_text = _build_prompt_text(music_brief)

    # Duration
    duration_ms = music_brief.get("duration_ms")
    if not duration_ms and music_brief.get("duration_seconds"):
        duration_ms = int(music_brief["duration_seconds"] * 1000)

    model = ELEVENLABS_MUSIC_MODEL
    input_hash = _compute_input_hash(prompt_text, composition_plan, duration_ms, model)

    # ── Build asset spec ─────────────────────────────────────────────────
    asset = {
        "property_id": property_id,
        "spec_id": spec_id,
        "concept_id": concept_id,
        "model": model,
        "input_hash": input_hash,
        "duration_seconds": (duration_ms / 1000.0) if duration_ms else None,
        "sample_rate": None,
        "license_basis": "self_serve_commercial",
        "prohibited_input_check": check_status,
        "created_by_agent": "agent8_stage5",
    }

    if prompt_text:
        asset["prompt_text"] = prompt_text
    if composition_plan:
        asset["composition_plan"] = composition_plan

    # ── Check cache ──────────────────────────────────────────────────────
    if not force:
        cached = _check_cached_music(property_id, input_hash)
        if cached:
            logger.info("[Agent 8 Stage 5] Cache hit for input_hash %s", input_hash[:12])
            return cached

    if dry_run:
        asset["status"] = "pending"
        asset["r2_url"] = None
        return asset

    # ── Render ───────────────────────────────────────────────────────────
    from core.elevenlabs import generate_music

    audio_bytes, model_used = generate_music(
        prompt_text=prompt_text,
        composition_plan=composition_plan,
        duration_ms=duration_ms,
        force_instrumental=True,
    )
    asset["model"] = model_used

    if audio_bytes is None:
        asset["status"] = "failed"
        asset["failure_reason"] = "ElevenLabs Music API call failed after retries"
        asset["r2_url"] = None
        _persist_music(asset, property_id, input_hash)
        return asset

    # Upload to R2
    try:
        from core.r2_storage import upload_video
        r2_url = upload_video(
            property_id=property_id,
            video_bytes=audio_bytes,
            video_type="music",
            format_label="bg_track",
            extension="mp3",
        )
    except Exception as exc:
        logger.error("[Agent 8 Stage 5] R2 upload failed: %s", exc)
        asset["status"] = "failed"
        asset["failure_reason"] = f"R2 upload failed: {exc}"
        asset["r2_url"] = None
        _persist_music(asset, property_id, input_hash)
        return asset

    asset["r2_url"] = r2_url
    asset["status"] = "ready"
    # Estimate duration from audio size (MP3 ~128kbps)
    if not asset["duration_seconds"] and audio_bytes:
        asset["duration_seconds"] = round(len(audio_bytes) / (128 * 1024 / 8), 1)

    _persist_music(asset, property_id, input_hash)

    logger.info(
        "[Agent 8 Stage 5] Music rendered: model=%s, r2=%s",
        model_used, r2_url,
    )

    return asset


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
        logger.error("[Agent 8 Stage 5] Failed to load spec %s: %s", spec_id, exc)
        return None


def _check_cached_music(property_id: str, input_hash: str) -> Optional[dict]:
    """Check music_assets for an active row with this input_hash."""
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("music_assets")
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
        logger.warning("[Agent 8 Stage 5] Cache check failed: %s", exc)
        return None


def _persist_music(asset: dict, property_id: str, input_hash: str) -> None:
    """Write music asset to Supabase, superseding any prior active row with same input_hash."""
    try:
        from core.supabase_store import get_supabase
        sb = get_supabase()

        # Supersede any prior active row with same input_hash
        sb.table("music_assets").update(
            {"superseded_at": "now()"}
        ).eq("property_id", property_id).eq("input_hash", input_hash).is_(
            "superseded_at", "null"
        ).execute()

        # Insert new row
        sb.table("music_assets").insert(asset).execute()

    except Exception as exc:
        logger.error("[Agent 8 Stage 5] Persist failed: %s", exc)
