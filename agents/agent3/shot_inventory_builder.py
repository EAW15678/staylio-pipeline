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
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-6"
_INVENTORY_VERSION = 1

# Namespace for deterministic shot_id generation.  uuid5(NAMESPACE, source_asset_id)
# produces the same UUID for the same photograph across re-runs, making the
# superseded_at lifecycle meaningful.
_SHOT_ID_NAMESPACE = uuid.UUID("b7e8c4a1-9f2d-4e3b-a1c6-5d8f7e2b4a9c")


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
        .order("created_at", desc=True)
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

    # ── Load media_assets for read_basis and tonal_signature derivation ──
    assets_resp = (
        sb.table("media_assets")
        .select("asset_url_original,asset_url_enhanced,brightness,dominant_colors,source_phash")
        .eq("property_id", property_id)
        .execute()
    )
    enhanced_lookup: set[str] = set()
    media_asset_map: dict[str, dict] = {}  # asset_url_original → row
    for row in (assets_resp.data or []):
        url = row.get("asset_url_original", "")
        if row.get("asset_url_enhanced"):
            enhanced_lookup.add(url)
        media_asset_map[url] = row

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

        # Tonal signature — derived from GCV data on media_assets.
        # brightness and dominant_colors are set by vision_tagger.py from
        # Google Vision's image_properties_annotation.
        tonal_signature = _derive_tonal_signature(asset_id, media_asset_map)

        # Determine read_basis: did the curation read enhanced or original?
        read_basis = "enhanced" if asset_id in enhanced_lookup else "original"

        # Deterministic shot_id: same photograph → same UUID across re-runs.
        # source_asset_id is the R2 URL from curation (asset_id).
        # ⚠️ DEBT: R2 URLs change every pipeline run.  If photos are re-uploaded,
        # shot_id changes and supersession breaks.  The production fix is
        # content-hash-based identity (see production-architecture.md Q1).
        shot_id = str(uuid.uuid5(_SHOT_ID_NAMESPACE, asset_id))

        # Carry phash from media_assets if available
        media_row = media_asset_map.get(asset_id, {})
        phash_val = media_row.get("source_phash") if media_row else None

        row = {
            "property_id":        property_id,
            "shot_id":            shot_id,
            "source_asset_id":    asset_id,
            "inventory_version":  _INVENTORY_VERSION,
            "model":              _MODEL,
            "read_basis":         read_basis,
            "superseded_at":      None,
            "phash":              phash_val,
            "persistence_manifest": [],
            # ── Pass-through fields from curation ────────────────────────
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
            on_conflict="property_id,shot_id,superseded_at",
        ).execute()
        logger.info(
            "[ShotInventory] Wrote %d shot_inventory rows for property %s",
            len(rows), property_id,
        )
        return len(rows)
    except Exception as exc:
        logger.error("[ShotInventory] Upsert failed for property %s: %s", property_id, exc)
        return 0


def _derive_tonal_signature(asset_id: str, media_asset_map: dict) -> dict:
    """
    Derive tonal_signature from GCV data on media_assets.

    brightness: from media_assets.brightness (float 0-1, set by vision_tagger
    from Google Vision's image_properties_annotation).

    hue: colour-temperature heuristic from dominant_colors hex values.
    Warm = red/orange/yellow bias, cool = blue/cyan bias, neutral = balanced.

    contrast: NOT reliably derivable from GCV output. GCV returns dominant
    colors and pixel fractions, but not luminance range or histogram data.
    Set to None rather than a constant — an explicit null is honest.
    """
    asset = media_asset_map.get(asset_id) or {}
    brightness_val = asset.get("brightness")
    dominant_colors = asset.get("dominant_colors") or []

    # Brightness: map 0-1 float to categorical
    if brightness_val is not None:
        if brightness_val < 0.3:
            brightness = "low"
        elif brightness_val < 0.6:
            brightness = "medium"
        else:
            brightness = "high"
    else:
        brightness = None

    # Hue: colour-temperature from dominant colors
    hue = _classify_hue(dominant_colors) if dominant_colors else None

    return {
        "hue": hue,
        "brightness": brightness,
        "contrast": None,  # Not derivable from GCV — see docstring
    }


def _classify_hue(hex_colors: list[str]) -> str:
    """
    Classify overall hue as warm / cool / neutral from hex color list.
    Uses average colour temperature across the dominant colors.
    """
    if not hex_colors:
        return "neutral"

    warm_score = 0.0
    cool_score = 0.0

    for hex_str in hex_colors:
        hex_clean = hex_str.lstrip("#")
        if len(hex_clean) != 6:
            continue
        try:
            r = int(hex_clean[0:2], 16) / 255.0
            g = int(hex_clean[2:4], 16) / 255.0
            b = int(hex_clean[4:6], 16) / 255.0
        except ValueError:
            continue

        # Warm bias: red and yellow channels dominate
        # Cool bias: blue channel dominates
        warm_score += (r + g * 0.5) - b
        cool_score += b - (r * 0.3 + g * 0.3)

    n = len(hex_colors) or 1
    avg_warm = warm_score / n
    avg_cool = cool_score / n

    if avg_warm > 0.3 and avg_warm > avg_cool:
        return "warm"
    elif avg_cool > 0.1 and avg_cool > avg_warm:
        return "cool"
    return "neutral"


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
