"""
Skill: enhance — Claid.ai photo enhancement + pHash computation.

Reads photographs + renditions('original') from substrate.
Writes renditions('enhanced') for each photograph that lacks one.
Also computes pHash for any photograph missing it (G13).

Input-addressed idempotency: skips any photograph whose original
already has an enhanced rendition. Re-runs converge.

Resolution guard: does not pay Claid to upscale sub-200px thumbnails.

Usage:
    from skills.enhance import enhance
    result = enhance("82cb9d7e-...")
"""

import hashlib
import io
import logging
import os
from datetime import datetime, timezone

import httpx

from skills.contract import (
    SkillResult, get_substrate, require_env,
    record_run, record_step, complete_step, complete_run, emit_cost,
    skills_r2_upload,
)

logger = logging.getLogger(__name__)

# Minimum dimension for Claid enhancement — below this, enhancement
# produces upscaling artifacts and wastes money
MIN_ENHANCE_DIMENSION = 200


def enhance(
    property_id: str,
    photo_ids: list = None,
    *,
    force: bool = False,
) -> SkillResult:
    """Enhance photographs via Claid.ai and compute pHash.

    Returns SkillResult.ok({enhanced, skipped_existing, skipped_low_res,
    phash_computed, cost_usd})
    """
    try:
        sb = get_substrate()
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    try:
        claid_key = require_env("CLAID_API_KEY", "Claid.ai photo enhancement")
        require_env("R2_ENDPOINT_URL", "R2 storage for enhanced renditions")
        require_env("R2_ACCESS_KEY_ID", "R2 storage for enhanced renditions")
        require_env("R2_SECRET_ACCESS_KEY", "R2 storage for enhanced renditions")
        require_env("STAGING_R2_BUCKET", "R2 bucket for skill outputs")
        require_env("STAGING_R2_PUBLIC_URL", "R2 public URL for skill outputs")
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    # ── Load photographs needing enhancement ─────────────────────────────
    query = sb.table("photographs").select(
        "photo_id, image_width, image_height, quality_tier, phash"
    ).eq("property_id", property_id)
    if photo_ids:
        query = query.in_("photo_id", photo_ids)
    photos_resp = query.execute()
    photos = photos_resp.data or []

    if not photos:
        return SkillResult.failed(
            reason=f"No photographs for property {property_id}",
            attempted=0, succeeded=0, failed_count=0,
        )

    # Find which already have enhanced renditions
    needs_enhancement = []
    already_enhanced = 0
    for photo in photos:
        pid = photo["photo_id"]
        enh = sb.table("renditions").select("rendition_id", count="exact").eq(
            "photo_id", pid
        ).eq("kind", "enhanced").limit(0).execute()
        if enh.count > 0 and not force:
            already_enhanced += 1
        else:
            needs_enhancement.append(photo)

    if not needs_enhancement:
        return SkillResult.noop(
            f"All {already_enhanced} photographs already have enhanced renditions.",
            {"already_enhanced": already_enhanced},
        )

    # ── Record run ───────────────────────────────────────────────────────
    run_id = record_run(sb, property_id, "onboard")
    step_id = record_step(sb, run_id, "enhance")

    enhanced_count = 0
    skipped_low_res = 0
    phash_computed = 0
    failed_count = 0
    total_cost = 0.0

    for photo in needs_enhancement:
        pid = photo["photo_id"]
        w = photo.get("image_width") or 0
        h = photo.get("image_height") or 0

        # Resolution guard: skip sub-200px thumbnails
        if w > 0 and h > 0 and (w < MIN_ENHANCE_DIMENSION or h < MIN_ENHANCE_DIMENSION):
            logger.info("[enhance] Skipping %s: %dx%d below minimum %dpx", pid[:8], w, h, MIN_ENHANCE_DIMENSION)
            skipped_low_res += 1
            continue

        # Get original rendition URL
        orig = sb.table("renditions").select("storage_url").eq(
            "photo_id", pid
        ).eq("kind", "original").limit(1).execute()
        if not orig.data:
            logger.warning("[enhance] No original rendition for %s", pid[:8])
            failed_count += 1
            continue

        original_url = orig.data[0]["storage_url"]

        # Download the original
        try:
            img_resp = httpx.get(original_url, timeout=30, follow_redirects=True)
            img_resp.raise_for_status()
            img_bytes = img_resp.content
        except Exception as exc:
            logger.warning("[enhance] Download failed for %s: %s", pid[:8], str(exc)[:60])
            failed_count += 1
            continue

        # ── Compute pHash if missing (G13) ───────────────────────────────
        if not photo.get("phash"):
            try:
                import imagehash
                from PIL import Image
                img = Image.open(io.BytesIO(img_bytes))
                phash_val = str(imagehash.phash(img))
                sb.table("photographs").update({"phash": phash_val}).eq("photo_id", pid).execute()
                phash_computed += 1
                logger.info("[enhance] pHash computed for %s: %s", pid[:8], phash_val)
            except ImportError:
                logger.warning("[enhance] imagehash not installed — cannot compute pHash")
            except Exception as exc:
                logger.warning("[enhance] pHash failed for %s: %s", pid[:8], str(exc)[:60])

        # ── Enhance via Claid ────────────────────────────────────────────
        try:
            import asyncio
            from agents.agent3.claid_enhancer import enhance_photo_async

            async def _enhance():
                async with httpx.AsyncClient(timeout=60) as session:
                    return await enhance_photo_async(session, original_url, 0)

            enhanced_bytes = asyncio.run(_enhance())

            if enhanced_bytes:
                # Upload to skills R2 bucket — env-driven, no hardcoded bucket
                content_hash = hashlib.sha256(img_bytes).hexdigest()
                key = f"{property_id}/enhanced/{pid}.jpg"
                r2_url = skills_r2_upload(key, enhanced_bytes, "image/jpeg")

                # Measure enhanced dimensions
                try:
                    from PIL import Image
                    enh_img = Image.open(io.BytesIO(enhanced_bytes))
                    enh_w, enh_h = enh_img.width, enh_img.height
                except Exception:
                    enh_w, enh_h = w, h

                sb.table("renditions").upsert({
                    "photo_id": pid,
                    "kind": "enhanced",
                    "storage_url": r2_url,
                    "format": "jpg",
                    "width": enh_w,
                    "height": enh_h,
                    "byte_size": len(enhanced_bytes),
                }, on_conflict="photo_id,kind,format").execute()

                claid_cost = 0.012
                total_cost += claid_cost
                emit_cost(sb, run_id, property_id,
                          vendor="claid", service="enhance",
                          units=1, unit_name="images",
                          unit_cost=0.012, total_cost=round(claid_cost, 4),
                          workflow_name="enhance", generation_reason="photo_enhancement",
                          discriminator=pid[:8])

                enhanced_count += 1
                logger.info("[enhance] Enhanced %s: %dx%d → %dx%d", pid[:8], w, h, enh_w, enh_h)
            else:
                logger.warning("[enhance] Claid returned no bytes for %s", pid[:8])
                failed_count += 1

        except Exception as exc:
            error_str = str(exc)
            if "402" in error_str or "billing" in error_str.lower():
                from skills.contract import escalate_billing
                escalate_billing(sb, property_id, "claid", error_str)
                complete_step(sb, step_id, status="failed", error_message=f"Claid billing: {error_str[:200]}")
                complete_run(sb, run_id, status="failed")
                return SkillResult.failed(
                    reason=f"Claid billing error: {error_str[:200]}",
                    attempted=len(needs_enhancement), succeeded=enhanced_count,
                    failed_count=failed_count + 1, error_class="billing", human_required=True,
                )
            logger.warning("[enhance] Claid failed for %s: %s", pid[:8], error_str[:80])
            failed_count += 1

    # ── Complete ─────────────────────────────────────────────────────────
    complete_step(sb, step_id, status="complete", metadata={
        "enhanced": enhanced_count,
        "skipped_existing": already_enhanced,
        "skipped_low_res": skipped_low_res,
        "phash_computed": phash_computed,
        "failed": failed_count,
        "cost_usd": round(total_cost, 4),
    })
    complete_run(sb, run_id, status="complete")

    return SkillResult.ok({
        "enhanced": enhanced_count,
        "skipped_existing": already_enhanced,
        "skipped_low_res": skipped_low_res,
        "phash_computed": phash_computed,
        "failed": failed_count,
        "cost_usd": round(total_cost, 4),
        "run_id": run_id,
    })
