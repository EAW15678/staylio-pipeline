"""
Skill: observe — run LLM vision curation on photographs and write observations.

Ports llm_curator.py's curation logic. The curation engine itself is unchanged —
only the edges change:
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

from skills.contract import (
    SkillResult, get_substrate, require_env,
    record_run, record_step, complete_step, complete_run, emit_cost,
)

logger = logging.getLogger(__name__)


def observe(
    property_id: str,
    photo_ids: list[str] = None,
    *,
    max_photos: int = 5,
    force: bool = False,
) -> SkillResult:
    """Run LLM vision curation on photographs and write observations.

    Args:
        property_id: UUID of the property.
        photo_ids: Optional subset of photo_ids to observe. Default: all for the property.
        max_photos: Maximum photos to observe (cost control). Default 5 for smoke tests.
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

    # ── Load photographs + renditions ────────────────────────────────────
    query = sb.table("photographs").select(
        "photo_id, property_id, content_hash, image_width, image_height"
    ).eq("property_id", property_id)

    if photo_ids:
        query = query.in_("photo_id", photo_ids)

    photos_resp = query.limit(max_photos).execute()
    photos = photos_resp.data or []

    if not photos:
        return SkillResult.failed(f"No photographs found for property {property_id}")

    # Get enhanced rendition URLs
    photo_url_map = {}  # photo_id → enhanced_url
    for photo in photos:
        rend = sb.table("renditions").select("storage_url").eq(
            "photo_id", photo["photo_id"]
        ).eq("kind", "enhanced").limit(1).execute()
        if rend.data:
            photo_url_map[photo["photo_id"]] = rend.data[0]["storage_url"]
        else:
            # Fall back to original
            rend_orig = sb.table("renditions").select("storage_url").eq(
                "photo_id", photo["photo_id"]
            ).eq("kind", "original").limit(1).execute()
            if rend_orig.data:
                photo_url_map[photo["photo_id"]] = rend_orig.data[0]["storage_url"]

    if not photo_url_map:
        return SkillResult.failed("No rendition URLs found for photographs")

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

    # ── Call Claude vision ───────────────────────────────────────────────
    import anthropic
    import httpx
    import base64
    import io

    client = anthropic.Anthropic(api_key=api_key)

    # Build a simple per-image observation prompt (not the full contact sheet
    # curation — that's the 1,954-line llm_curator.py which ports intact later.
    # This smoke test demonstrates the contract: vendor call → observation → cost_event.)
    observations_written = 0
    failed_count = 0
    total_cost = 0.0

    for photo in photos:
        pid = photo["photo_id"]
        url = photo_url_map.get(pid)
        if not url:
            continue

        try:
            # Download image
            img_resp = httpx.get(url, timeout=30, follow_redirects=True)
            img_resp.raise_for_status()
            img_bytes = img_resp.content
            img_b64 = base64.b64encode(img_bytes).decode()

            # Determine media type
            media_type = "image/jpeg"
            if url.lower().endswith(".png"):
                media_type = "image/png"

            # Call Claude vision
            resp = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_b64}},
                        {"type": "text", "text": (
                            "Analyze this vacation rental photo. Return JSON only:\n"
                            '{"depth_structure":"flat|layered|deep",'
                            '"depth_tier":"close|mid|wide",'
                            '"space_direction":"left|right|center|symmetrical|into_frame",'
                            '"light_direction":"front|side_left|side_right|back|overhead|diffuse",'
                            '"time_of_day_read":"morning|midday|golden_hour|night",'
                            '"motion_risk":["list of risks: reflections, water_surface, straight_architectural_lines, etc"],'
                            '"motion_affordance":["list: push_in, pan_left, pan_right, etc"],'
                            '"focal_point":"what the eye is drawn to",'
                            '"alt_text":"accessibility alt text for this image"}'
                        )},
                    ],
                }],
            )

            # Parse response
            import re
            raw = resp.content[0].text.strip()
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            obs_data = json.loads(raw)

            # Cost
            usage = resp.usage
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)
            cost_usd = (input_tokens * 3 + output_tokens * 15) / 1_000_000
            total_cost += cost_usd

            emit_cost(sb, run_id, property_id,
                      vendor="anthropic", service="claude_vision_observe",
                      units=input_tokens + output_tokens, unit_name="tokens",
                      unit_cost=None, total_cost=round(cost_usd, 6),
                      workflow_name="observe", generation_reason="photo_observation",
                      discriminator=pid[:8])

            # Supersede existing observations for this photo
            now_iso = datetime.now(timezone.utc).isoformat()
            sb.table("observations").update(
                {"superseded_at": now_iso}
            ).eq("property_id", property_id).eq("photo_id", pid).is_(
                "superseded_at", "null"
            ).execute()

            # Write new observation
            sb.table("observations").insert({
                "property_id": property_id,
                "photo_id": pid,
                "observation_version": 1,
                "model": "claude-sonnet-4-6",
                "read_basis": "enhanced" if "enhanced" in url else "original",
                "depth_structure": obs_data.get("depth_structure"),
                "depth_tier": obs_data.get("depth_tier"),
                "space_direction": obs_data.get("space_direction"),
                "light_direction": obs_data.get("light_direction"),
                "time_of_day_read": obs_data.get("time_of_day_read"),
                "motion_risk": obs_data.get("motion_risk") or [],
                "motion_affordance": obs_data.get("motion_affordance") or [],
                "focal_point": obs_data.get("focal_point"),
                "alt_text": obs_data.get("alt_text"),
                "negative_space": [],
                "foreground_elements": [],
                "located_amenities": [],
                "persistence_manifest": [],
                "superseded_at": None,
            }).execute()
            observations_written += 1

            logger.info(
                "[observe] Photo %s: depth=%s motion_risk=%s cost=$%.4f",
                pid[:8], obs_data.get("depth_structure"), obs_data.get("motion_risk"), cost_usd,
            )

        except Exception as exc:
            error_str = str(exc)
            failed_count += 1

            # Ruling 6: billing/auth errors → escalate, no retry, stop
            if "credit balance" in error_str or "authentication" in error_str.lower():
                logger.error("[observe] BILLING ERROR on photo %s: %s", pid[:8], error_str[:120])
                from skills.contract import escalate_billing
                escalate_billing(sb, property_id, "anthropic", error_str)
                complete_step(sb, step_id, status="failed",
                              error_message=f"Billing error: {error_str[:200]}")
                complete_run(sb, run_id, status="failed",
                             error_summary=f"Anthropic billing error after {observations_written} observations")
                return SkillResult.failed(
                    reason=f"Anthropic billing error: {error_str[:200]}",
                    attempted=len(photos),
                    succeeded=observations_written,
                    failed_count=failed_count,
                    error_class="billing",
                    human_required=True,
                )

            logger.error("[observe] Photo %s failed: %s", pid[:8], error_str[:120])
            continue

    # ── Complete the run ─────────────────────────────────────────────────
    status = "complete" if observations_written > 0 else "failed"
    complete_step(sb, step_id, status=status, metadata={
        "observations_written": observations_written,
        "photos_processed": len(photos),
        "cost_usd": round(total_cost, 4),
    })
    complete_run(sb, run_id, status=status)

    if observations_written == 0 and failed_count > 0:
        return SkillResult.failed(
            reason=f"All {failed_count} vendor calls failed",
            attempted=len(photos),
            succeeded=0,
            failed_count=failed_count,
            error_class="vendor",
        )

    return SkillResult.ok({
        "observations_written": observations_written,
        "photos_processed": len(photos),
        "failed_count": failed_count,
        "cost_usd": round(total_cost, 4),
        "run_id": run_id,
    })
