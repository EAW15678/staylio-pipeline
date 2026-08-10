"""
Agent 8 — Stage 7: Assembly

Assembles motion clips, narration, music, and overlays into a final
9:16 master video via Creatomate, then applies post-processing (audio
ducking) and uploads to R2.

Standalone module. NOT wired into the pipeline graph.
Call directly: assemble_master(spec_id)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Optional

from pipeline_emitter import emit_media_cost

from core.creatomate import (
    CREATOMATE_API_KEY,
    CREATOMATE_API_BASE,
    CREATOMATE_COST_PER_RENDER_MINUTE,
    render_video,
)

logger = logging.getLogger(__name__)

# ── Master canvas dimensions ──────────────────────────────────────────────
MASTER_W, MASTER_H = 1080, 1920

# ── Universal safe zone: 900x1400 centered in 1080x1920 ──────────────────
SAFE_W, SAFE_H = 900, 1400
SAFE_LEFT = (MASTER_W - SAFE_W) // 2     # 90
SAFE_TOP = (MASTER_H - SAFE_H) // 2      # 260
SAFE_RIGHT = SAFE_LEFT + SAFE_W          # 990
SAFE_BOTTOM = SAFE_TOP + SAFE_H          # 1660

# ── 9-cell grid mapping for landscape -> 9:16 master ─────────────────────
# The source negative_space grid was computed on a LANDSCAPE image.
# After 9:16 crop, the top/bottom thirds shift significantly.
GRID_REGIONS_IN_MASTER = {
    "top_left":      (SAFE_LEFT, SAFE_TOP, SAFE_LEFT + SAFE_W // 3, SAFE_TOP + SAFE_H // 3),
    "top_center":    (SAFE_LEFT + SAFE_W // 3, SAFE_TOP, SAFE_LEFT + 2 * SAFE_W // 3, SAFE_TOP + SAFE_H // 3),
    "top_right":     (SAFE_LEFT + 2 * SAFE_W // 3, SAFE_TOP, SAFE_RIGHT, SAFE_TOP + SAFE_H // 3),
    "middle_left":   (SAFE_LEFT, SAFE_TOP + SAFE_H // 3, SAFE_LEFT + SAFE_W // 3, SAFE_TOP + 2 * SAFE_H // 3),
    "middle_center": (SAFE_LEFT + SAFE_W // 3, SAFE_TOP + SAFE_H // 3, SAFE_LEFT + 2 * SAFE_W // 3, SAFE_TOP + 2 * SAFE_H // 3),
    "middle_right":  (SAFE_LEFT + 2 * SAFE_W // 3, SAFE_TOP + SAFE_H // 3, SAFE_RIGHT, SAFE_TOP + 2 * SAFE_H // 3),
    "bottom_left":   (SAFE_LEFT, SAFE_TOP + 2 * SAFE_H // 3, SAFE_LEFT + SAFE_W // 3, SAFE_BOTTOM),
    "bottom_center": (SAFE_LEFT + SAFE_W // 3, SAFE_TOP + 2 * SAFE_H // 3, SAFE_LEFT + 2 * SAFE_W // 3, SAFE_BOTTOM),
    "bottom_right":  (SAFE_LEFT + 2 * SAFE_W // 3, SAFE_TOP + 2 * SAFE_H // 3, SAFE_RIGHT, SAFE_BOTTOM),
}

# Ordered list of safe-zone-compliant regions for fallback repositioning
_SAFE_REGIONS_PREFERENCE = [
    "middle_center", "top_center", "bottom_center",
    "middle_left", "middle_right",
    "top_left", "top_right",
    "bottom_left", "bottom_right",
]


def _region_inside_safe_zone(region_name: str) -> bool:
    """Check if a named grid region falls within the safe zone."""
    return region_name in GRID_REGIONS_IN_MASTER


def _check_overlay_safe_zone(overlay: dict) -> tuple[dict, bool, str | None]:
    """
    Check if an overlay's grid_region falls inside the 900x1400 safe zone.
    Returns (overlay, was_moved, original_region_or_None).
    If it doesn't fit, reposition to nearest compliant region.
    If no compliant region exists, mark as held.
    """
    region = overlay.get("grid_region", "")

    # Already in safe zone
    if _region_inside_safe_zone(region):
        return overlay, False, None

    # Try to find a compliant region
    for candidate in _SAFE_REGIONS_PREFERENCE:
        if _region_inside_safe_zone(candidate):
            moved = dict(overlay)
            moved["grid_region"] = candidate
            moved["_repositioned_from"] = region
            return moved, True, region

    # No compliant region — hold the overlay
    held = dict(overlay)
    held["_held"] = True
    held["_held_reason"] = f"No compliant safe-zone region for '{region}'"
    return held, False, region


def _compute_input_hash(
    clip_ids: list[str],
    narration_id: str | None,
    music_id: str | None,
    overlay_payload: list[dict],
    template_id: str,
) -> str:
    """Cache key: hash of all inputs that affect the rendered master."""
    blob = json.dumps(
        {
            "clip_ids": clip_ids,
            "narration_id": narration_id,
            "music_id": music_id,
            "overlay_payload": overlay_payload,
            "template_id": template_id,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _build_render_payload(
    clips: list[dict],
    narration: dict | None,
    music: dict | None,
    overlays: list[dict],
    template_id: str,
) -> dict:
    """
    Build Creatomate render modifications from assembly inputs.
    NO loop flags. Explicit durations on everything.
    """
    mods: dict = {}

    # Clips in beat order — each gets a numbered slot
    for i, clip in enumerate(clips):
        r2_url = clip.get("r2_url", "")
        if r2_url:
            mods[f"clip_{i + 1}.source"] = r2_url
        duration = clip.get("duration")
        if duration:
            mods[f"clip_{i + 1}.duration"] = str(duration)

    # Narration audio
    if narration and narration.get("r2_url"):
        mods["narration.source"] = narration["r2_url"]
        if narration.get("duration_seconds"):
            mods["narration.duration"] = str(narration["duration_seconds"])

        # Alignment-based timing: pin narration to beat boundaries
        alignment = narration.get("alignment")
        if alignment:
            mods["narration.alignment"] = json.dumps(alignment)

    # Music audio
    if music and music.get("r2_url"):
        mods["music.source"] = music["r2_url"]
        if music.get("duration_seconds"):
            mods["music.duration"] = str(music["duration_seconds"])

    # Overlays with safe-zone-verified positions
    for i, ov in enumerate(overlays):
        if ov.get("_held"):
            continue  # Skip held overlays
        region = ov.get("grid_region", "middle_center")
        coords = GRID_REGIONS_IN_MASTER.get(region)
        if coords:
            x1, y1, x2, y2 = coords
            mods[f"overlay_{i + 1}.x"] = str(x1)
            mods[f"overlay_{i + 1}.y"] = str(y1)
            mods[f"overlay_{i + 1}.width"] = str(x2 - x1)
            mods[f"overlay_{i + 1}.height"] = str(y2 - y1)
        if ov.get("text"):
            mods[f"overlay_{i + 1}.text"] = ov["text"]
        if ov.get("source"):
            mods[f"overlay_{i + 1}.source"] = ov["source"]

    return mods


def _compute_template_checksum(template_id: str) -> str:
    """Simple checksum of template_id for tracking template versions."""
    return hashlib.md5(template_id.encode()).hexdigest()[:12]


def _apply_audio_ducking(
    master_bytes: bytes, narration_present: bool
) -> tuple[bytes, bool]:
    """
    Apply sidechain compression (narration ducks music) via FFmpeg.
    Returns (processed_bytes, ducking_applied).
    If FFmpeg not available or narration not present, returns original.
    """
    if not narration_present:
        return master_bytes, False

    # Check FFmpeg availability
    if not shutil.which("ffmpeg"):
        logger.warning("[Agent 8 Stage 7] FFmpeg not available — skipping audio ducking")
        return master_bytes, False

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_in:
            f_in.write(master_bytes)
            in_path = f_in.name

        out_path = in_path.replace(".mp4", "_ducked.mp4")

        # Sidechain compress: narration stream ducks the music stream
        # Filter: extract both audio streams, apply sidechaincompress, remix
        cmd = [
            "ffmpeg", "-y",
            "-i", in_path,
            "-filter_complex",
            "[0:a]asplit=2[sc][mix];"
            "[sc]aformat=fltp[sc_fmt];"
            "[mix][sc_fmt]sidechaincompress="
            "threshold=0.02:ratio=6:attack=10:release=200[ducked]",
            "-map", "0:v",
            "-map", "[ducked]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            out_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=120,
        )

        if result.returncode != 0:
            logger.warning(
                "[Agent 8 Stage 7] FFmpeg ducking failed (rc=%d): %s",
                result.returncode,
                result.stderr.decode(errors="replace")[:500],
            )
            return master_bytes, False

        with open(out_path, "rb") as f_out:
            ducked_bytes = f_out.read()

        return ducked_bytes, True

    except subprocess.TimeoutExpired:
        logger.warning("[Agent 8 Stage 7] FFmpeg ducking timed out")
        return master_bytes, False
    except Exception as exc:
        logger.warning("[Agent 8 Stage 7] FFmpeg ducking error: %s", exc)
        return master_bytes, False
    finally:
        # Clean up temp files
        for p in [in_path, out_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


def assemble_master(
    spec_id: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> dict | None:
    """
    Assemble a final 9:16 master video from a shot_spec's motion clips,
    narration, music, and overlays via Creatomate.

    NOT every concept reaches assembly — earlier stages may have produced
    no clips. No clips means no assembly and a clean return (None).

    Parameter semantics and cost:

        dry_run=False, force=False (default)
            If cached assembly exists (input_hash match): returns it. $0.
            If not: renders via Creatomate, writes to DB + R2. ~$0.84/min.

        dry_run=True, force=False
            If cached: returns it. NO vendor call. $0.
            If not: returns what WOULD be rendered, writes nothing. $0.

        dry_run=False, force=True
            Ignores cache, re-renders, writes (supersedes prior). ~$0.84/min.

        dry_run=True, force=True
            Ignores cache, returns spec without rendering. $0.

    Args:
        spec_id: UUID of the shot_spec row.
        dry_run: If True, return assembly spec without rendering or writing.
        force: If True, ignore cached assembly and re-render.

    Returns:
        Assembled video dict, or None if no clips available.
    """
    logger.info(
        "[Agent 8 Stage 7] assemble_master: spec_id=%s dry_run=%s force=%s",
        spec_id, dry_run, force,
    )

    # ── Load shot_spec ───────────────────────────────────────────────────
    spec = _load_spec(spec_id)
    if not spec:
        raise ValueError(f"Shot spec {spec_id} not found or not active")

    property_id = spec["property_id"]
    concept_id = spec.get("concept_id")

    # ── Load motion clips (ordered by beat_ordinal) ──────────────────────
    clips = _load_motion_clips(spec_id)
    if not clips:
        logger.info("[Agent 8 Stage 7] No motion clips for spec %s — clean return", spec_id)
        return None

    # Sort by beat_index to guarantee beat order
    clips.sort(key=lambda c: c.get("beat_index", 0))
    clip_ids = [c.get("clip_id") or c.get("input_hash", "") for c in clips]

    # ── Load narration ──────────────────────────────────────────────────
    narration = _load_narration(spec_id)
    narration_id = narration.get("narration_id") or narration.get("input_hash") if narration else None

    # ── Load music ──────────────────────────────────────────────────────
    music = _load_music(spec_id)
    music_id = music.get("music_id") or music.get("input_hash") if music else None

    # ── Load overlays and check safe zones ──────────────────────────────
    raw_overlays = spec.get("overlay_register") or []
    overlays: list[dict] = []
    for ov in raw_overlays:
        checked, was_moved, orig_region = _check_overlay_safe_zone(ov)
        if was_moved:
            logger.info(
                "[Agent 8 Stage 7] Overlay repositioned from '%s' to '%s'",
                orig_region, checked.get("grid_region"),
            )
        overlays.append(checked)

    # ── Template selection ──────────────────────────────────────────────
    template_id = spec.get("creatomate_template_id") or os.environ.get(
        "CREATOMATE_MASTER_TEMPLATE_ID", ""
    )
    template_checksum = _compute_template_checksum(template_id)

    # ── Input hash for caching ──────────────────────────────────────────
    overlay_payload = [
        {k: v for k, v in ov.items() if not k.startswith("_")}
        for ov in overlays
        if not ov.get("_held")
    ]
    input_hash = _compute_input_hash(
        clip_ids, narration_id, music_id, overlay_payload, template_id,
    )

    # ── Estimate duration from clips ────────────────────────────────────
    total_duration = sum(c.get("duration", 0) for c in clips)
    cost_estimate = round(
        (total_duration / 60.0) * CREATOMATE_COST_PER_RENDER_MINUTE, 4
    )

    # ── Build assembly record ───────────────────────────────────────────
    assembly: dict = {
        "property_id": property_id,
        "spec_id": spec_id,
        "concept_id": concept_id,
        "clip_ids": clip_ids,
        "narration_id": narration_id,
        "music_id": music_id,
        "overlay_payload": overlay_payload,
        "renderer": "creatomate",
        "template_id_used": template_id,
        "template_checksum": template_checksum,
        "render_payload": None,
        "post_process": [],
        "r2_url": None,
        "duration_seconds": total_duration,
        "aspect_ratio": "9:16",
        "resolution": "1080x1920",
        "has_audio_ducking": False,
        "input_hash": input_hash,
        "cost_estimate_usd": cost_estimate,
        "attempt_count": 0,
        "status": "pending",
        "failure_reason": None,
        "created_by_agent": "agent8_stage7",
    }

    # ── Check cache ─────────────────────────────────────────────────────
    if not force:
        cached = _check_cached_assembly(property_id, input_hash)
        if cached:
            logger.info("[Agent 8 Stage 7] Cache hit for input_hash %s", input_hash[:12])
            return cached

    if dry_run:
        assembly["status"] = "pending"
        return assembly

    # ── Build render payload (NO loop flags) ────────────────────────────
    render_mods = _build_render_payload(
        clips, narration, music, overlays, template_id,
    )
    assembly["render_payload"] = render_mods
    assembly["status"] = "rendering"
    assembly["attempt_count"] = 1

    # ── Render via Creatomate ───────────────────────────────────────────
    try:
        video_bytes, render_id = asyncio.run(
            render_video(
                template_id=template_id,
                modifications=render_mods,
            )
        )
    except Exception as exc:
        logger.error("[Agent 8 Stage 7] Creatomate render failed: %s", exc)
        assembly["status"] = "failed"
        assembly["failure_reason"] = str(exc)
        _persist_assembly(assembly, property_id, input_hash)
        return assembly

    if video_bytes is None:
        assembly["status"] = "failed"
        assembly["failure_reason"] = "Creatomate returned no video after polling"
        _persist_assembly(assembly, property_id, input_hash)
        return assembly

    # ── Post-process: audio ducking ─────────────────────────────────────
    narration_present = narration is not None and narration.get("r2_url") is not None
    processed_bytes, ducking_applied = _apply_audio_ducking(
        video_bytes, narration_present,
    )
    assembly["has_audio_ducking"] = ducking_applied
    if ducking_applied:
        assembly["post_process"] = [{
            "step": "audio_ducking",
            "params": {
                "filter": "sidechaincompress",
                "threshold": 0.02,
                "ratio": 6,
                "attack": 10,
                "release": 200,
            },
        }]

    # ── Upload to R2 ────────────────────────────────────────────────────
    try:
        from core.r2_storage import upload_video
        r2_url = upload_video(
            property_id=property_id,
            video_bytes=processed_bytes,
            video_type="master_assembly",
            format_label="9x16",
            extension="mp4",
        )
    except Exception as exc:
        logger.error("[Agent 8 Stage 7] R2 upload failed: %s", exc)
        assembly["status"] = "failed"
        assembly["failure_reason"] = f"R2 upload failed: {exc}"
        _persist_assembly(assembly, property_id, input_hash)
        return assembly

    assembly["r2_url"] = r2_url
    assembly["status"] = "ready"

    emit_media_cost(
        vendor="creatomate",
        service="video_render",
        units=round(total_duration / 60.0, 2),
        unit_name="minutes",
        property_id=property_id,
        workflow_name="agent8_stage7",
        generation_reason="master_assembly",
        discriminator=spec_id[:8],
    )

    _persist_assembly(assembly, property_id, input_hash)

    logger.info(
        "[Agent 8 Stage 7] Master assembled: r2=%s, duration=%.1fs, ducking=%s",
        r2_url, total_duration, ducking_applied,
    )

    return assembly


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
        logger.error("[Agent 8 Stage 7] Failed to load spec %s: %s", spec_id, exc)
        return None


def _load_motion_clips(spec_id: str) -> list[dict]:
    """Load ready motion_clips for this spec, ordered by beat_index."""
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("motion_clips")
            .select("*")
            .eq("spec_id", spec_id)
            .eq("status", "ready")
            .is_("superseded_at", "null")
            .order("beat_index")
            .execute()
        )
        return result.data or []
    except Exception as exc:
        logger.warning("[Agent 8 Stage 7] motion_clips load failed: %s", exc)
        return []


def _load_narration(spec_id: str) -> Optional[dict]:
    """Load ready narration asset for this spec."""
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("narration_assets")
            .select("*")
            .eq("spec_id", spec_id)
            .eq("status", "ready")
            .is_("superseded_at", "null")
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None
    except Exception as exc:
        logger.warning("[Agent 8 Stage 7] narration load failed: %s", exc)
        return None


def _load_music(spec_id: str) -> Optional[dict]:
    """Load ready music asset for this spec."""
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("music_assets")
            .select("*")
            .eq("spec_id", spec_id)
            .eq("status", "ready")
            .is_("superseded_at", "null")
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None
    except Exception as exc:
        logger.warning("[Agent 8 Stage 7] music load failed: %s", exc)
        return None


def _check_cached_assembly(property_id: str, input_hash: str) -> Optional[dict]:
    """Check assembled_videos for an active row with this input_hash."""
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("assembled_videos")
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
        logger.warning("[Agent 8 Stage 7] Cache check failed: %s", exc)
        return None


def _persist_assembly(assembly: dict, property_id: str, input_hash: str) -> None:
    """Write assembled video to Supabase, superseding any prior active row with same input_hash."""
    try:
        from core.supabase_store import get_supabase
        sb = get_supabase()

        # Supersede any prior active row with same input_hash
        sb.table("assembled_videos").update(
            {"superseded_at": "now()"}
        ).eq("property_id", property_id).eq("input_hash", input_hash).is_(
            "superseded_at", "null"
        ).execute()

        # Insert new row
        sb.table("assembled_videos").insert(assembly).execute()

    except Exception as exc:
        logger.error("[Agent 8 Stage 7] Persist failed: %s", exc)
