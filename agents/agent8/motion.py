"""
Agent 8 — Stage 6: Motion

Generates image-to-video motion clips from a shot_spec's beats via
Runway ML. One clip per beat. Follows the narration.py (Stage 4) and
music.py (Stage 5) patterns.

Model routing: gen4.5 for reflective surfaces (water, mirrors),
gen4_turbo for everything else. Hero flag alone does NOT justify gen4.5.

Frame-exit constraint: pull_back and other frame-exiting moves are
rejected BEFORE any vendor call (Guidelines v1.2 section 3.1).

Standalone module. NOT wired into the pipeline graph.
Call directly: render_beats(spec_id)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from typing import Optional

from pipeline_emitter import emit_media_cost

from core.runway import (
    RUNWAYML_API_SECRET,
    RUNWAY_MODEL_TURBO,
    RUNWAY_MODEL_QUALITY,
    RUNWAY_COST_PER_SEC,
    generate_clip,
)

logger = logging.getLogger(__name__)

# ── Frame-exit constraint (Guidelines v1.2 section 3.1) ──────────────────
FRAME_EXITING_MOVES = {"pull_back"}


def _select_model(beat: dict, inventory_row: dict | None) -> str:
    """
    gen4.5 for reflective surfaces, gen4_turbo for everything else.
    Hero does NOT get gen4.5 — reflectivity is the ONLY justification.
    """
    motion_risk = (inventory_row or {}).get("motion_risk") or []
    if "reflections" in motion_risk or "water_surface" in motion_risk:
        return RUNWAY_MODEL_QUALITY
    return RUNWAY_MODEL_TURBO


def _compute_input_hash(
    source_image_url: str,
    technique: str,
    motion: str,
    params: dict,
    duration: int,
    model: str,
) -> str:
    """Cache key: hash of all inputs that affect the rendered clip."""
    blob = json.dumps(
        {
            "source_image_url": source_image_url,
            "technique": technique,
            "motion": motion,
            "params": params,
            "duration": duration,
            "model": model,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _build_prompt_text(beat: dict) -> str:
    """Build a Runway prompt from beat intent + motion."""
    intent = beat.get("intent") or ""
    motion = beat.get("motion") or ""
    parts = []
    if motion:
        parts.append(f"Slow {motion}")
    parts.append("cinematic")
    if intent:
        parts.append(intent)
    return ", ".join(parts)


def render_beats(
    spec_id: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> list[dict]:
    """
    Render motion clips from a shot_spec's beats array.

    NOT every concept carries beats — that is a directorial choice,
    not a defect. No beats means no clips and a clean return.

    Parameter semantics and cost:

        dry_run=False, force=False (default)
            If cached clip exists (input_hash match): returns it. $0.
            If not: renders via Runway ML, writes to DB + R2. ~$0.50-1.20/clip.

        dry_run=True, force=False
            If cached: returns it. NO vendor call. $0.
            If not: returns what WOULD be rendered, writes nothing. $0.

        dry_run=False, force=True
            Ignores cache, re-renders, writes (supersedes prior). ~$0.50-1.20/clip.

        dry_run=True, force=True
            Ignores cache, returns spec without rendering. $0.

    Args:
        spec_id: UUID of the shot_spec row.
        dry_run: If True, return clip specs without rendering or writing.
        force: If True, ignore cached clips and re-render.

    Returns:
        List of motion clip dicts. Empty list if no beats.
    """
    logger.info(
        "[Agent 8 Stage 6] render_beats: spec_id=%s dry_run=%s force=%s",
        spec_id, dry_run, force,
    )

    # ── Load shot_spec ───────────────────────────────────────────────────
    spec = _load_spec(spec_id)
    if not spec:
        raise ValueError(f"Shot spec {spec_id} not found or not active")

    beats = spec.get("beats") or []
    if not beats:
        logger.info("[Agent 8 Stage 6] No beats on spec %s — clean return", spec_id)
        return []

    property_id = spec["property_id"]
    concept_id = spec.get("concept_id")

    # Load shot_inventory for model routing (motion_risk)
    inventory = _load_shot_inventory(property_id)
    inventory_by_url = {}
    if inventory:
        for row in inventory:
            url = row.get("image_url") or row.get("source_image_url") or ""
            if url:
                inventory_by_url[url] = row

    clips: list[dict] = []

    for idx, beat in enumerate(beats):
        source_image_url = beat.get("source_image_url") or ""
        technique = "generative"
        requested_motion = beat.get("motion") or ""
        params = beat.get("params") or {}
        duration = beat.get("duration") or 10

        # Model selection
        inv_row = inventory_by_url.get(source_image_url)
        model = _select_model(beat, inv_row)

        input_hash = _compute_input_hash(
            source_image_url, technique, requested_motion, params, duration, model,
        )

        clip: dict = {
            "property_id": property_id,
            "spec_id": spec_id,
            "concept_id": concept_id,
            "beat_index": idx,
            "source_image_url": source_image_url,
            "technique": technique,
            "requested_motion": requested_motion,
            "vendor": "runway",
            "model": model,
            "duration": duration,
            "input_hash": input_hash,
            "cost_estimate_usd": round(duration * RUNWAY_COST_PER_SEC[model], 4),
            "persistence_check": "unchecked",
            "persistence_findings": None,
            "created_by_agent": "agent8_stage6",
            "attempt_count": 0,
        }

        # ── Frame-exit constraint (DEFENSIVE) ───────────────────────────
        if requested_motion in FRAME_EXITING_MOVES:
            clip["status"] = "rejected"
            clip["failure_reason"] = (
                f"Frame-exiting move '{requested_motion}' prohibited "
                f"(Guidelines v1.2 §3.1)"
            )
            clip["persistence_check"] = "not_applicable"
            clip["r2_url"] = None
            clips.append(clip)
            continue

        # ── Check cache ─────────────────────────────────────────────────
        if not force:
            cached = _check_cached_clip(property_id, input_hash)
            if cached:
                logger.info(
                    "[Agent 8 Stage 6] Cache hit for beat %d, input_hash %s",
                    idx, input_hash[:12],
                )
                clips.append(cached)
                continue

        if dry_run:
            clip["status"] = "pending"
            clip["r2_url"] = None
            clips.append(clip)
            continue

        # ── Render via Runway ───────────────────────────────────────────
        clip["status"] = "queued"
        prompt_text = _build_prompt_text(beat)

        try:
            clip["status"] = "rendering"
            clip["attempt_count"] = 1

            video_bytes, model_used = asyncio.run(
                generate_clip(
                    image_url=source_image_url,
                    prompt_text=prompt_text,
                    model=model,
                    duration=duration,
                )
            )
            clip["model"] = model_used

            if video_bytes is None:
                clip["status"] = "failed"
                clip["failure_reason"] = "Runway ML returned no video after retries"
                clip["r2_url"] = None
                _persist_clip(clip, property_id, input_hash)
                clips.append(clip)
                continue

            # Upload to R2
            from core.r2_storage import upload_video
            r2_url = upload_video(
                property_id=property_id,
                video_bytes=video_bytes,
                video_type=f"motion_beat_{idx:02d}",
                format_label="mp4",
            )

            clip["r2_url"] = r2_url
            clip["status"] = "ready"
            clip["cost_estimate_usd"] = round(
                duration * RUNWAY_COST_PER_SEC[model_used], 4
            )

            emit_media_cost(
                vendor="runway",
                service=model_used,
                units=duration,
                unit_name="seconds",
                property_id=property_id,
                workflow_name="agent8_stage6",
                generation_reason="motion_clip",
                discriminator=f"beat_{idx:02d}",
            )

            _persist_clip(clip, property_id, input_hash)

            logger.info(
                "[Agent 8 Stage 6] Beat %d rendered: model=%s, r2=%s",
                idx, model_used, r2_url,
            )

        except Exception as exc:
            logger.error(
                "[Agent 8 Stage 6] Beat %d failed: %s", idx, exc,
            )
            clip["status"] = "failed"
            clip["failure_reason"] = str(exc)
            clip["r2_url"] = None
            _persist_clip(clip, property_id, input_hash)

        clips.append(clip)

    return clips


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
        logger.error("[Agent 8 Stage 6] Failed to load spec %s: %s", spec_id, exc)
        return None


def _load_shot_inventory(property_id: str) -> Optional[list[dict]]:
    """Load shot_inventory rows for model routing (motion_risk field)."""
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("shot_inventory")
            .select("image_url, source_image_url, motion_risk")
            .eq("property_id", property_id)
            .execute()
        )
        return result.data or []
    except Exception as exc:
        logger.warning("[Agent 8 Stage 6] shot_inventory load failed: %s", exc)
        return None


def _check_cached_clip(property_id: str, input_hash: str) -> Optional[dict]:
    """Check motion_clips for an active row with this input_hash."""
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("motion_clips")
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
        logger.warning("[Agent 8 Stage 6] Cache check failed: %s", exc)
        return None


def _persist_clip(clip: dict, property_id: str, input_hash: str) -> None:
    """Write motion clip to Supabase, superseding any prior active row with same input_hash."""
    try:
        from core.supabase_store import get_supabase
        sb = get_supabase()

        # Supersede any prior active row with same input_hash
        sb.table("motion_clips").update(
            {"superseded_at": "now()"}
        ).eq("property_id", property_id).eq("input_hash", input_hash).is_(
            "superseded_at", "null"
        ).execute()

        # Insert new row
        sb.table("motion_clips").insert(clip).execute()

    except Exception as exc:
        logger.error("[Agent 8 Stage 6] Persist failed: %s", exc)
