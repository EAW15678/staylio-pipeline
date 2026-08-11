"""
Skill: observe — run LLM vision curation on photographs and write observations.

Ports llm_curator.py's 1,954 lines intact. The curation engine is unchanged —
contact-sheet batching, v12 prompt, 640x480 grid, validation, section
assignment all move as-is. Only the edges change:
  - Input: photographs + renditions from substrate (not media_assets)
  - Output: observations keyed on photo_id with superseded_at supersession
  - Vendor calls through the contract (run_step + cost_event recorded)

Usage:
    from skills.observe import observe
    result = observe("a1b2c3d4-...", photo_ids=["uuid1", "uuid2"])
"""

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional
from types import SimpleNamespace

from skills.contract import (
    SkillResult, get_substrate, require_env,
    record_run, record_step, complete_step, complete_run, emit_cost,
)

logger = logging.getLogger(__name__)


def observe(
    property_id: str,
    photo_ids: list = None,
    *,
    max_photos: int = 120,
    force: bool = False,
) -> SkillResult:
    """Run LLM vision curation on photographs and write observations.

    Calls llm_curator.py's run_llm_vision_curation with substrate-sourced
    inputs. The curation engine (contact-sheet batching, v12 prompt, 640x480
    grid, validation, section assignment) runs unchanged. Output is written
    to the observations table keyed on photo_id with superseded_at lifecycle.

    Args:
        property_id: UUID of the property.
        photo_ids: Optional subset of photo_ids to observe. Default: all.
        max_photos: Maximum photos (cost control). Default 120 (full curation).
        force: If True, re-observe even if observations exist.

    Returns:
        SkillResult.ok({"observations_written": N, "cost_usd": X})
    """
    try:
        sb = get_substrate()
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    try:
        api_key = require_env("ANTHROPIC_API_KEY", "Claude vision curation")
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    # ── Load photographs + renditions from substrate ───────────────────
    query = sb.table("photographs").select(
        "photo_id, property_id, content_hash, image_width, image_height"
    ).eq("property_id", property_id)
    if photo_ids:
        query = query.in_("photo_id", photo_ids)
    photos_resp = query.limit(max_photos).execute()
    photos = photos_resp.data or []

    if not photos:
        return SkillResult.failed(
            reason=f"No photographs found for property {property_id}",
            attempted=0, succeeded=0, failed_count=0,
        )

    # Build photo_id → rendition URLs
    photo_id_to_urls = {}  # photo_id → {original, enhanced}
    for photo in photos:
        pid = photo["photo_id"]
        rends = sb.table("renditions").select("kind, storage_url").eq("photo_id", pid).execute()
        urls = {}
        for r in (rends.data or []):
            urls[r["kind"]] = r["storage_url"]
        photo_id_to_urls[pid] = urls

    # Check for existing observations — Ruling 6: nothing-to-do = ok(noop)
    if not force:
        existing = sb.table("observations").select("observation_id", count="exact").eq(
            "property_id", property_id
        ).is_("superseded_at", "null").limit(0).execute()
        if existing.count > 0 and not photo_ids:
            return SkillResult.noop(
                f"Observations already exist ({existing.count} active). Use force=True to re-observe.",
                {"observations_existing": existing.count},
            )

    # ── Record the run ───────────────────────────────────────────────────
    run_id = record_run(sb, property_id, "onboard")
    step_id = record_step(sb, run_id, "observe")

    # ── Build assets list for llm_curator ────────────────────────────────
    # llm_curator.run_llm_vision_curation expects objects with attribute access
    assets = []
    enhanced_bytes_map = {}

    for photo in photos:
        pid = photo["photo_id"]
        urls = photo_id_to_urls.get(pid, {})
        original_url = urls.get("original", "")
        enhanced_url = urls.get("enhanced", "")

        if not original_url and not enhanced_url:
            continue

        asset = SimpleNamespace(
            asset_url_original=original_url,
            asset_url_enhanced=enhanced_url,
            subject_category="uncategorised",
            composition_score=0.0,
            labels_enhanced=[],
            brightness=None,
            dominant_colors=None,
            hero_rank=None,
            is_hero=False,
            source="substrate",
            _photo_id=pid,
        )
        assets.append(asset)

    if not assets:
        complete_step(sb, step_id, status="failed", error_message="No assets with rendition URLs")
        complete_run(sb, run_id, status="failed")
        return SkillResult.failed(reason="No assets with rendition URLs", attempted=0, succeeded=0, failed_count=0)

    # ── Call the curation engine ─────────────────────────────────────────
    # Set the ANTHROPIC_API_KEY for the curator (it reads from env)
    os.environ["ANTHROPIC_API_KEY"] = api_key

    try:
        from agents.agent3.llm_curator import run_llm_vision_curation
        kb = {"vibe_profile": "", "name": {"value": ""}}  # Minimal KB
        prop = sb.table("properties").select("vibe_profile, name").eq("id", property_id).limit(1).execute()
        if prop.data:
            kb["vibe_profile"] = prop.data[0].get("vibe_profile") or ""
            kb["name"] = {"value": prop.data[0].get("name") or ""}

        curation = run_llm_vision_curation(
            assets=assets,
            enhanced_bytes_map=enhanced_bytes_map,  # Empty — curator downloads from URLs
            property_id=property_id,
            kb=kb,
        )
    except Exception as exc:
        error_str = str(exc)
        if "credit balance" in error_str or "authentication" in error_str.lower():
            from skills.contract import escalate_billing
            escalate_billing(sb, property_id, "anthropic", error_str)
            complete_step(sb, step_id, status="failed", error_message=f"Billing: {error_str[:200]}")
            complete_run(sb, run_id, status="failed")
            return SkillResult.failed(
                reason=f"Anthropic billing error: {error_str[:200]}",
                attempted=len(assets), succeeded=0, failed_count=1,
                error_class="billing", human_required=True,
            )
        logger.error("[observe] Curation failed: %s", error_str[:200])
        complete_step(sb, step_id, status="failed", error_message=error_str[:200])
        complete_run(sb, run_id, status="failed")
        return SkillResult.failed(
            reason=f"Curation failed: {error_str[:200]}",
            attempted=len(assets), succeeded=0, failed_count=1,
            error_class="vendor",
        )

    if not curation:
        complete_step(sb, step_id, status="failed", error_message="Curation returned None")
        complete_run(sb, run_id, status="failed")
        return SkillResult.failed(
            reason="Curation returned None — threshold not met or all batches failed",
            attempted=len(assets), succeeded=0, failed_count=1,
            error_class="vendor",
        )

    # ── Write observations from curation results ─────────────────────────
    # Build asset_url → photo_id lookup
    url_to_photo_id = {a.asset_url_original: a._photo_id for a in assets}

    per_image = curation.get("images") or []
    logger.info("[observe] Curation returned %d image results", len(per_image))

    # Supersede ALL existing observations for this property
    now_iso = datetime.now(timezone.utc).isoformat()
    sb.table("observations").update(
        {"superseded_at": now_iso}
    ).eq("property_id", property_id).is_("superseded_at", "null").execute()

    observations_written = 0
    for img in per_image:
        asset_id = img.get("asset_id", "")
        pid = url_to_photo_id.get(asset_id)
        if not pid:
            continue

        try:
            sb.table("observations").insert({
                "property_id": property_id,
                "photo_id": pid,
                "observation_version": 2,  # v2 = substrate-era curation
                "model": "claude-sonnet-4-6",
                "read_basis": "enhanced" if photo_id_to_urls.get(pid, {}).get("enhanced") else "original",
                "depth_structure": img.get("depth_structure"),
                "depth_tier": img.get("depth_tier"),
                "space_direction": img.get("space_direction"),
                "light_direction": img.get("light_direction"),
                "light_quality": img.get("light_quality"),
                "time_of_day_read": img.get("time_of_day_read"),
                "negative_space": img.get("negative_space") or [],
                "foreground_elements": img.get("foreground_elements") or [],
                "frame_element": img.get("frame_element"),
                "beyond_frame_element": img.get("beyond_frame_element"),
                "subject_singularity": img.get("subject_singularity"),
                "focal_point": img.get("visual_summary"),
                "tonal_signature": None,
                "located_amenities": [],
                "motion_affordance": img.get("motion_affordance") or [],
                "motion_risk": img.get("motion_risk") or [],
                "persistence_manifest": [],
                "role": img.get("role"),
                "curated_section": img.get("curated_section"),
                "quality_score": img.get("quality_score"),
                "alt_text": img.get("alt"),
                "duplicate_group": img.get("duplicate_group"),
                "is_best_in_group": img.get("is_best_in_duplicate_group", True),
                "gallery_visible": img.get("gallery_visible", True),
                "tour_eligible": img.get("tour_eligible", True),
                "superseded_at": None,
            }).execute()
            observations_written += 1
        except Exception as exc:
            logger.warning("[observe] Observation write failed for %s: %s", pid[:8], str(exc)[:80])

    # ── Record cost ──────────────────────────────────────────────────────
    # The curator logs cost internally; we also record it in the substrate
    # TODO: extract actual cost from curator's logging (it emits to pipeline_emitter)
    # For now, estimate: ~$0.008 per image (Sonnet vision, 640x480 contact sheets)
    estimated_cost = len(per_image) * 0.008
    emit_cost(sb, run_id, property_id,
              vendor="anthropic", service="claude_vision_curation",
              units=len(per_image), unit_name="images",
              unit_cost=0.008, total_cost=round(estimated_cost, 4),
              workflow_name="observe", generation_reason="llm_curation_v12")

    # ── Complete ─────────────────────────────────────────────────────────
    # Measure final observation count
    final_count = sb.table("observations").select("observation_id", count="exact").eq(
        "property_id", property_id
    ).is_("superseded_at", "null").limit(0).execute().count

    complete_step(sb, step_id, status="complete", metadata={
        "observations_written": observations_written,
        "curation_images": len(per_image),
        "active_observations": final_count,
        "estimated_cost_usd": round(estimated_cost, 4),
    })
    complete_run(sb, run_id, status="complete")

    return SkillResult.ok({
        "observations_written": observations_written,
        "curation_images": len(per_image),
        "active_observations": final_count,
        "estimated_cost_usd": round(estimated_cost, 4),
        "run_id": run_id,
    })
