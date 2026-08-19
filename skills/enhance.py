"""
Skill: enhance — per-photograph Claid.ai enhancement via recipes.

Reads photographs + renditions('original') + active observations from
substrate. Selects a recipe per photograph based on observation fields
(light_quality, placement, contains_text, quality_score, depth_tier).
Writes renditions('enhanced') for each canonical photograph.

Recipes (ENHANCE-2/3, ruled by Erick 2026-08-19):
  text_bearing  — gentlest, protects letterforms
  bright_exterior — tames blown highlights
  flat_light    — lifts flat indoor lighting with contrast + polish
  small_weak    — upscales and polishes physically small photos (< 2MP)
  large_weak    — polishes large but soft/poorly-lit photos (no upscale)
  large_clean   — baseline HDR + sharpness, no upscale

Precedence: gentler wins. text_bearing > bright_exterior > flat_light >
small_weak > large_clean. Restraint beats punch when they conflict.

Universal: every recipe includes decompress ("auto") — scraped photos
have been recompressed by every site they passed through.

Also computes pHash for any photograph missing it (G13, legacy heal).

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

CLAID_API_BASE = "https://api.claid.ai/v1-beta1"

# ── Enhancement recipes ─────────────────────────────────────────────
# Each recipe serves a specific outcome (G28: consistent gallery).
# Precedence: gentler wins. text_bearing > bright_exterior > flat_light
# > small_weak > large_weak > large_clean. Restraint beats punch.
#
# polish is restricted to small_weak and flat_light groups — never on
# text-bearing or already-sharp frames. It redraws image parts while
# preserving structure. Ruled by Erick 2026-08-19. Has a 16MP vendor
# ceiling: above that, silently omit (don't fail).
#
# resizing is paired with upscale in small_weak — upscale selects the
# AI model, resizing sets target dimensions. Whether upscale alone
# changes dimensions is unresolved; the pairing is the safe choice.

_POLISH_MAX_MP = 16.0  # vendor ceiling: polish fails above 16MP


def _select_recipe(photo, obs):
    """Select enhancement recipe from observation fields.

    Returns (recipe_name, operations_dict).
    obs may be None (no active observation) — falls back by resolution.
    """
    w = photo.get("image_width") or 0
    h = photo.get("image_height") or 0
    mp = (w * h) / 1_000_000 if w and h else 0

    if obs is None:
        # Fallback: no observation available
        if mp >= 2.0:
            return _build_recipe("large_clean", mp)
        return _build_recipe("small_weak", mp)

    # Precedence: gentler wins
    if obs.get("contains_text"):
        # "Small" means physically small — MP < 2.0. A low quality score
        # means soft/poorly lit, not small, and never qualifies for upscale.
        # Must stay consistent with the small_weak/large_weak split below.
        is_small = mp < 2.0
        return _build_recipe("text_bearing", mp, is_small=is_small)

    placement = obs.get("placement") or "unknown"
    tod = obs.get("time_of_day_read") or ""
    lq = obs.get("light_quality") or ""

    if (placement == "outdoor" and tod in ("midday", "morning")) or lq == "hard":
        return _build_recipe("bright_exterior", mp)

    if lq == "flat":
        return _build_recipe("flat_light", mp)

    # Size and score split — mutually exclusive, exhaustive:
    #   MP < 2.0 → small_weak (upscale + polish)
    #   MP >= 2.0, score < 0.6 → large_weak (polish only, no upscale)
    #   MP >= 2.0, score >= 0.6 → large_clean (baseline)
    if mp < 2.0:
        return _build_recipe("small_weak", mp)

    qs = obs.get("quality_score") or 1.0
    if qs < 0.6:
        return _build_recipe("large_weak", mp)

    return _build_recipe("large_clean", mp)


def _build_recipe(name, mp, is_small=False):
    """Build the Claid operations dict for a named recipe.

    Every recipe includes decompress: auto (removes recompression damage).
    """
    can_polish = mp <= _POLISH_MAX_MP

    if name == "small_weak":
        ops = {
            "restorations": {"decompress": "auto", "upscale": "smart_enhance"},
            "adjustments": {"hdr": 100, "sharpness": 20},
            "resizing": {"width": "150%", "height": "150%"},
        }
        if can_polish:
            ops["restorations"]["polish"] = True
        else:
            logger.debug("[enhance] polish skipped: %.1fMP > %.1fMP ceiling", mp, _POLISH_MAX_MP)

    elif name == "large_weak":
        # A low quality score on a large photograph means soft, poorly
        # lit or badly composed — none of which is fixed by adding
        # pixels. Polish sharpens; upscaling a soft photograph only
        # makes it bigger and softer. No upscale, no resizing.
        ops = {
            "restorations": {"decompress": "auto"},
            "adjustments": {"hdr": 100, "sharpness": 20},
        }
        if can_polish:
            ops["restorations"]["polish"] = True
        else:
            logger.debug("[enhance] polish skipped: %.1fMP > %.1fMP ceiling", mp, _POLISH_MAX_MP)

    elif name == "large_clean":
        # No upscale, no resizing — the photo is already high-res.
        # Baseline HDR + sharpness brings it into the gallery look.
        ops = {
            "restorations": {"decompress": "auto"},
            "adjustments": {"hdr": 100, "sharpness": 15},
        }

    elif name == "flat_light":
        # As large_clean plus contrast lift and polish to compensate
        # for flat indoor lighting.
        ops = {
            "restorations": {"decompress": "auto"},
            "adjustments": {"hdr": 100, "sharpness": 15, "contrast": 15},
        }
        if can_polish:
            ops["restorations"]["polish"] = True
        else:
            logger.debug("[enhance] polish skipped: %.1fMP > %.1fMP ceiling", mp, _POLISH_MAX_MP)

    elif name == "bright_exterior":
        # Tames blown highlights on midday/morning exteriors and hard light.
        # Lower HDR to avoid over-processing; negative exposure pulls back.
        # No polish — these are typically sharp already.
        ops = {
            "restorations": {"decompress": "auto"},
            "adjustments": {"hdr": 70, "exposure": -10, "sharpness": 15},
        }

    elif name == "text_bearing":
        # Gentlest recipe — protects letterforms from redrawing.
        # No polish ever. If the photo is also small/weak, use
        # smart_resize (preserves text better than smart_enhance).
        ops = {
            "restorations": {"decompress": "auto"},
            "adjustments": {"hdr": 60, "sharpness": 10},
        }
        if is_small:
            ops["restorations"]["upscale"] = "smart_resize"
            ops["resizing"] = {"width": "150%", "height": "150%"}

    else:
        # Unknown recipe name — fall back to large_clean
        ops = {
            "restorations": {"decompress": "auto"},
            "adjustments": {"hdr": 100, "sharpness": 15},
        }

    return name, ops


def _apply_edge_stitching(ops, photo, obs):
    """Apply HDR edge stitching modifier for wide/panoramic frames."""
    if obs is None:
        return
    w = photo.get("image_width") or 0
    h = photo.get("image_height") or 0
    depth = obs.get("depth_tier") or ""
    is_wide_aspect = w > 0 and h > 0 and (w / h) >= 2.0
    is_wide_depth = depth in ("wide", "panoramic")

    if (is_wide_aspect or is_wide_depth) and "hdr" in ops.get("adjustments", {}):
        hdr_val = ops["adjustments"]["hdr"]
        if isinstance(hdr_val, (int, float)):
            ops["adjustments"]["hdr"] = {"intensity": hdr_val, "stitching": True}


def _count_credits(operations, input_mp):
    """Count Claid credits for a recipe's operations.

    Rate card (Claid published pricing, 2026-08-19):
      adjustments block (any combination of dials): 1 credit
      polish: 1 credit
      upscale: tiered — 1 (<4MP), 2 (4–9MP), 3 (>=9MP)
      resizing: 0 (when combined with another operation)
      decompress: 0 (not on the rate card)
    """
    credits = 0
    restorations = operations.get("restorations", {})
    adjustments = operations.get("adjustments", {})

    if adjustments:
        credits += 1

    if restorations.get("polish"):
        credits += 1

    if "upscale" in restorations:
        if input_mp >= 9:
            credits += 3
        elif input_mp >= 4:
            credits += 2
        else:
            credits += 1

    # resizing: 0 credits when combined (always combined here)
    # decompress: 0 credits (not on the rate card)

    return max(credits, 1)  # at least 1 credit per call


def _get_credit_rate(sb):
    """Look up the current Claid credit rate from vendor_rates.

    Fails loudly if no rate is found — a missing rate is a config
    error, not something to paper over (G16).
    """
    resp = sb.table("vendor_rates").select("unit_cost").eq(
        "vendor", "claid"
    ).eq("unit_name", "credits").is_(
        "effective_to", "null"
    ).limit(1).execute()

    if not resp.data:
        raise EnvironmentError(
            "No current Claid credit rate found in vendor_rates. "
            "Seed one row: vendor='claid', unit_name='credits', "
            "unit_cost=0.059, effective_from='2026-08-01'."
        )
    return float(resp.data[0]["unit_cost"])


async def _call_claid(session, url, operations):
    """Call Claid v1-beta1 image/edit and return enhanced bytes.

    Modelled on agents/agent3/claid_enhancer.enhance_photo_async —
    same endpoint, same auth, same retry-on-429, same download flow.
    Governance (validate_operations) must be called BEFORE this.
    """
    claid_key = os.environ.get("CLAID_API_KEY", "")
    if not claid_key:
        return None

    payload = {
        "input": url,
        "operations": operations,
        "output": {"format": {"type": "jpeg", "quality": 92}},
    }

    for attempt in range(1, 4):
        try:
            resp = await session.post(
                f"{CLAID_API_BASE}/image/edit",
                headers={
                    "Authorization": f"Bearer {claid_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60.0,
            )
            if resp.status_code == 429:
                import asyncio
                await asyncio.sleep(2 ** attempt)
                continue
            resp.raise_for_status()

            data = resp.json()
            output_url = (
                data.get("output", {}).get("tmp_url")
                or data.get("data", {}).get("output", {}).get("tmp_url")
            )
            if not output_url:
                return None

            dl_resp = await session.get(output_url, timeout=30.0)
            dl_resp.raise_for_status()
            return dl_resp.content

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 402:
                raise  # billing error — let caller handle
            if attempt == 3:
                raise
        except Exception:
            if attempt == 3:
                raise


def enhance(
    property_id: str,
    photo_ids: list = None,
    *,
    force: bool = False,
) -> SkillResult:
    """Enhance photographs via Claid.ai with per-photograph recipes.

    Loads the active observation for each canonical photograph, selects
    a recipe based on observation fields, and dispatches to Claid.
    Every recipe passes validate_operations() before dispatch.

    Returns SkillResult.ok({enhanced, skipped_existing, skipped_low_res,
    recipe counts, cost_usd})
    """
    try:
        sb = get_substrate()
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    try:
        require_env("CLAID_API_KEY", "Claid.ai photo enhancement")
        require_env("R2_ENDPOINT_URL", "R2 storage for enhanced renditions")
        require_env("R2_ACCESS_KEY_ID", "R2 storage for enhanced renditions")
        require_env("R2_SECRET_ACCESS_KEY", "R2 storage for enhanced renditions")
        require_env("STAGING_R2_BUCKET", "R2 bucket for skill outputs")
        require_env("STAGING_R2_PUBLIC_URL", "R2 public URL for skill outputs")
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    # ── Load photographs needing enhancement ─────────────────────────────
    # enhance operates on CANONICALS ONLY, after dedupe and observe.
    query = sb.table("photographs").select(
        "photo_id, image_width, image_height, quality_tier, phash"
    ).eq("property_id", property_id).eq("is_canonical", True)
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

    # ── Load active observations for recipe selection ────────────────────
    obs_resp = sb.table("observations").select(
        "photo_id, light_quality, time_of_day_read, placement, "
        "quality_score, contains_text, depth_tier"
    ).eq("property_id", property_id).is_("superseded_at", "null").execute()
    obs_by_photo = {o["photo_id"]: o for o in (obs_resp.data or [])}

    # ── Record run ───────────────────────────────────────────────────────
    run_id = record_run(sb, property_id, "onboard")
    step_id = record_step(sb, run_id, "enhance")

    # Import governance validator — single source in agents/agent3
    from agents.agent3.claid_enhancer import validate_operations

    # Look up credit rate — fails loudly if not configured (G16)
    credit_rate = _get_credit_rate(sb)

    enhanced_count = 0
    skipped_low_res = 0
    phash_computed = 0
    failed_count = 0
    total_cost = 0.0
    recipe_counts = {}
    polish_skipped = 0

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

        # ── Compute pHash if missing (G13, legacy heal) ──────────────────
        if not photo.get("phash"):
            try:
                img_resp_ph = httpx.get(original_url, timeout=30, follow_redirects=True)
                img_resp_ph.raise_for_status()
                import imagehash
                from PIL import Image
                img = Image.open(io.BytesIO(img_resp_ph.content))
                phash_val = str(imagehash.phash(img))
                sb.table("photographs").update({"phash": phash_val}).eq("photo_id", pid).execute()
                phash_computed += 1
                logger.info("[enhance] pHash computed for %s: %s", pid[:8], phash_val)
            except Exception as exc:
                logger.warning("[enhance] pHash failed for %s: %s", pid[:8], str(exc)[:60])

        # ── Select recipe from observation ───────────────────────────────
        obs = obs_by_photo.get(pid)
        recipe_name, operations = _select_recipe(photo, obs)
        _apply_edge_stitching(operations, photo, obs)

        # Track polish skips
        mp = (w * h) / 1_000_000 if w and h else 0
        if recipe_name in ("small_weak", "flat_light") and mp > _POLISH_MAX_MP:
            polish_skipped += 1

        recipe_counts[recipe_name] = recipe_counts.get(recipe_name, 0) + 1

        # ── Governance check — always before dispatch ────────────────────
        try:
            validate_operations(operations)
        except ValueError as gov_err:
            logger.error("[enhance] Governance violation for %s: %s", pid[:8], str(gov_err)[:120])
            failed_count += 1
            continue

        logger.info("[enhance] %s recipe=%s ops=%s", pid[:8], recipe_name,
                    list(operations.keys()))

        # ── Enhance via Claid ────────────────────────────────────────────
        try:
            import asyncio

            async def _enhance():
                async with httpx.AsyncClient(timeout=60) as session:
                    return await _call_claid(session, original_url, operations)

            enhanced_bytes = asyncio.run(_enhance())

            if enhanced_bytes:
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

                # Claid cost: count credits by operation, look up rate
                input_mp = (w * h) / 1_000_000 if w and h else 1.0
                credits = _count_credits(operations, input_mp)
                claid_cost = round(credits * credit_rate, 4)
                total_cost += claid_cost
                emit_cost(sb, run_id, property_id,
                          vendor="claid", service="enhance",
                          units=credits, unit_name="credits",
                          unit_cost=credit_rate, total_cost=claid_cost,
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
        "recipe_counts": recipe_counts,
        "polish_skipped": polish_skipped,
    })
    complete_run(sb, run_id, status="complete")

    return SkillResult.ok({
        "enhanced": enhanced_count,
        "skipped_existing": already_enhanced,
        "skipped_low_res": skipped_low_res,
        "phash_computed": phash_computed,
        "failed": failed_count,
        "cost_usd": round(total_cost, 4),
        "recipe_counts": recipe_counts,
        "polish_skipped": polish_skipped,
        "run_id": run_id,
    })
