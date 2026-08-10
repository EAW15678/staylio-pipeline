"""
Shot Inventory Builder — Agent 3, post-curation transformer

Pure data transformation: reads property_image_curations.per_image_results
and writes shot_inventory rows. NO LLM call.

Entry point:
    populate_shot_inventory(property_id) -> int   (count of rows written)

Supersession: new rows are inserted with superseded_at = NULL.  Previous
rows for the same property are marked superseded (superseded_at = now()).
"""

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-6"
_INVENTORY_VERSION = 1


def populate_shot_inventory(property_id: str) -> int:
    """
    Read property_image_curations.per_image_results and write shot_inventory rows.
    NO LLM call -- pure data transformation.
    Returns count of rows written.
    """
    from core.supabase_store import get_supabase

    sb = get_supabase()

    # ── Load latest complete curation ────────────────────────────────────
    curation_resp = (
        sb.table("property_image_curations")
        .select("per_image_results")
        .eq("property_id", property_id)
        .eq("status", "complete")
        .is_("superseded_at", "null")
        .order("completed_at", desc=True)
        .limit(1)
        .execute()
    )

    if not curation_resp.data or not curation_resp.data[0].get("per_image_results"):
        logger.warning(
            "[ShotInventory] No complete curation found for property %s — skipping",
            property_id,
        )
        return 0

    per_image_results = curation_resp.data[0]["per_image_results"]
    if not isinstance(per_image_results, list):
        logger.warning("[ShotInventory] per_image_results is not a list — skipping")
        return 0

    # ── Load media_assets to determine read_basis per image ──────────────
    assets_resp = (
        sb.table("media_assets")
        .select("asset_url_original,asset_url_enhanced")
        .eq("property_id", property_id)
        .execute()
    )
    enhanced_lookup: set[str] = set()
    for row in (assets_resp.data or []):
        if row.get("asset_url_enhanced"):
            enhanced_lookup.add(row.get("asset_url_original", ""))

    # ── Supersede old inventory rows ─────────────────────────────────────
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        sb.table("shot_inventory").update(
            {"superseded_at": now_iso}
        ).eq(
            "property_id", property_id
        ).is_(
            "superseded_at", "null"
        ).execute()
    except Exception as exc:
        logger.warning("[ShotInventory] Supersession update failed (non-fatal): %s", exc)

    # ── Build inventory rows ─────────────────────────────────────────────
    rows = []
    for img in per_image_results:
        if not isinstance(img, dict):
            continue
        asset_id = img.get("asset_id")
        if not asset_id:
            continue

        # Derive located_amenities from has_* booleans
        located_amenities = _derive_located_amenities(img)

        # Derive subject_singularity
        has_count = sum(1 for k, v in img.items() if k.startswith("has_") and v is True)
        if has_count == 0:
            subject_singularity = "single"
        elif has_count == 1:
            subject_singularity = "single"
        elif has_count == 2:
            subject_singularity = "dual"
        else:
            subject_singularity = "cluttered"

        # Focal point from visual_summary
        focal_point = img.get("visual_summary") or None

        # Tonal signature — STUB for GCV color derivation
        # TODO: derive from GCV dominant color annotations
        tonal_signature = {
            "hue": "neutral",
            "brightness": "medium",
            "contrast": "medium",
        }

        # Determine read_basis: did the curation read enhanced or original?
        read_basis = "enhanced" if asset_id in enhanced_lookup else "original"

        row = {
            "property_id":        property_id,
            "asset_url":          asset_id,
            "inventory_version":  _INVENTORY_VERSION,
            "model":              _MODEL,
            "read_basis":         read_basis,
            "superseded_at":      None,
            # ── Pass-through fields from curation (v10) ──────────────────
            "physical_room_id":    img.get("physical_room_id"),
            "visual_duplicate_of": img.get("visual_duplicate_of"),
            "depth_structure":     img.get("depth_structure"),
            "foreground_elements": img.get("foreground_elements") or [],
            "frame_element":       img.get("frame_element"),
            "beyond_frame_element": img.get("beyond_frame_element"),
            "space_direction":     img.get("space_direction"),
            "light_direction":     img.get("light_direction"),
            "light_quality":       img.get("light_quality"),
            "time_of_day_read":    img.get("time_of_day_read"),
            "negative_space":      img.get("negative_space") or [],
            "depth_tier":          img.get("depth_tier"),
            "motion_affordance":   img.get("motion_affordance") or [],
            "motion_risk":         img.get("motion_risk") or [],
            # ── Derived fields ───────────────────────────────────────────
            "located_amenities":   located_amenities,
            "subject_singularity": subject_singularity,
            "focal_point":         focal_point,
            "tonal_signature":     tonal_signature,
        }
        rows.append(row)

    if not rows:
        logger.info("[ShotInventory] No rows to write for property %s", property_id)
        return 0

    # ── Upsert to shot_inventory ─────────────────────────────────────────
    try:
        sb.table("shot_inventory").upsert(
            rows,
            on_conflict="property_id,asset_url,superseded_at",
        ).execute()
        logger.info(
            "[ShotInventory] Wrote %d shot_inventory rows for property %s",
            len(rows), property_id,
        )
        return len(rows)
    except Exception as exc:
        logger.error("[ShotInventory] Upsert failed for property %s: %s", property_id, exc)
        return 0


def _derive_located_amenities(img: dict) -> list[dict]:
    """
    Derive located_amenities from has_* boolean flags.
    Maps each True has_* flag to a label.
    """
    _HAS_MAP = {
        "has_bed":                    "bed",
        "has_bathroom_fixture":       "bathroom",
        "has_pool":                   "pool",
        "has_kitchen":                "kitchen",
        "has_living_area":            "living_area",
        "has_exterior":               "exterior",
        "has_hot_tub":                "hot_tub",
        "has_outdoor_lounge":         "outdoor_lounge",
        "has_outdoor_kitchen_grill":  "outdoor_kitchen_grill",
    }
    amenities = []
    for key, label in _HAS_MAP.items():
        if img.get(key) is True:
            amenities.append({"label": label})
    return amenities
