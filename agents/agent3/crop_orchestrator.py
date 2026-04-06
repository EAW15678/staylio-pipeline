"""
TS-08 — Social Format Crop Orchestrator
Pure Python/Pillow implementation — no Node.js, no subprocess.

Called after Vision API tagging identifies category winners.
Only category winners receive social crops — approximately
8-12 images per property rather than all 100.

Crops are generated in-process using Pillow ImageOps.fit,
then uploaded directly to Cloudflare R2 via upload_social_crop().
"""

import io
import logging
from typing import Optional

import httpx
from PIL import Image, ImageOps

from agents.agent3.models import SocialCrop, SubjectCategory
from agents.agent3.r2_storage import upload_social_crop

logger = logging.getLogger(__name__)

# Output pixel dimensions per format label
CROP_DIMENSIONS: dict[str, tuple[int, int]] = {
    "1_1":  (1080, 1080),
    "9_16": (1080, 1920),
    "16_9": (1920, 1080),
}

# Default formats per category — preserved exactly from original
DEFAULT_FORMATS_BY_CATEGORY: dict[SubjectCategory, list[str]] = {
    SubjectCategory.VIEW:                 ["9_16", "1_1", "16_9"],
    SubjectCategory.POOL_HOT_TUB:         ["9_16", "1_1"],
    SubjectCategory.EXTERIOR:             ["16_9", "1_1"],
    SubjectCategory.LIVING_ROOM:          ["1_1", "9_16"],
    SubjectCategory.OUTDOOR_ENTERTAINING: ["9_16", "1_1"],
    SubjectCategory.MASTER_BEDROOM:       ["1_1", "9_16"],
    SubjectCategory.KITCHEN:              ["1_1"],
    SubjectCategory.BATHROOM:             ["1_1"],
    SubjectCategory.STANDARD_BEDROOM:     ["1_1"],
    SubjectCategory.GAME_ENTERTAINMENT:   ["1_1", "9_16"],
    SubjectCategory.LOCAL_AREA:           ["1_1"],
}


def generate_social_crops(
    property_id: str,
    vibe_profile: str,
    property_name: str,
    category_winners: dict[str, str],   # {category_str: enhanced_url}
    apply_overlays: bool = True,
    bannerbear_template_id: Optional[str] = None,
    enhanced_bytes_map: Optional[dict[str, bytes]] = None,
) -> list[SocialCrop]:
    """
    Generate social format crops for all category winner photos.
    Crops with Pillow ImageOps.fit, uploads results directly to R2.

    Args:
        property_id:           Property UUID
        vibe_profile:          Vibe string (reserved for future overlay use)
        property_name:         Property name (reserved for future overlay use)
        category_winners:      Dict of category → enhanced photo URL
        apply_overlays:        Reserved — overlays not yet implemented in Python path
        bannerbear_template_id: Reserved — no longer used
        enhanced_bytes_map:    {asset_url_original: enhanced_bytes} to skip re-downloading

    Returns:
        List of SocialCrop records for Supabase storage
    """
    if not category_winners:
        logger.info(f"[TS-08] No category winners to crop for property {property_id}")
        return []

    bmap = enhanced_bytes_map or {}
    social_crops: list[SocialCrop] = []

    for category_str, photo_url in category_winners.items():
        try:
            category = SubjectCategory(category_str)
        except ValueError:
            category = SubjectCategory.UNCATEGORISED

        formats = DEFAULT_FORMATS_BY_CATEGORY.get(category, ["1_1", "9_16"])

        image_bytes = _get_image_bytes(photo_url, bmap)
        if not image_bytes:
            logger.warning(f"[TS-08] Could not obtain bytes for {photo_url} — skipping")
            continue

        for fmt in formats:
            dims = CROP_DIMENSIONS.get(fmt)
            if not dims:
                continue

            try:
                crop_bytes = _crop_image(image_bytes, dims)
            except Exception as exc:
                logger.error(f"[TS-08] Crop failed for {photo_url} fmt={fmt}: {exc}")
                continue

            try:
                filename = f"{category_str}_{fmt}.jpg"
                r2_url = upload_social_crop(
                    property_id=property_id,
                    crop_bytes=crop_bytes,
                    filename=filename,
                    format_label=fmt,
                )
            except Exception as exc:
                logger.error(f"[TS-08] R2 upload failed for crop {category_str}/{fmt}: {exc}")
                continue

            social_crops.append(SocialCrop(
                property_id=property_id,
                source_asset_url=photo_url,
                crop_url=r2_url,
                format=fmt,
                subject_category=category_str,
                has_overlay=False,
                overlay_template_id=None,
            ))

    logger.info(
        f"[TS-08] Social crops complete for property {property_id}: "
        f"{len(social_crops)} crops generated from {len(category_winners)} category winners"
    )
    return social_crops


# ── Internal helpers ──────────────────────────────────────────────────────

def _get_image_bytes(url: str, enhanced_bytes_map: dict[str, bytes]) -> Optional[bytes]:
    """
    Return image bytes for a URL.
    Checks enhanced_bytes_map first (keyed by asset_url_original) to avoid
    a redundant HTTP download; falls back to fetching the URL directly.
    """
    # The map is keyed by asset_url_original; the url here is asset_url_enhanced.
    # Try a direct key hit first (handles cases where original == enhanced URL).
    if url in enhanced_bytes_map:
        return enhanced_bytes_map[url]

    # Fall back to an HTTP fetch (enhanced URL is publicly accessible via CDN)
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(url, follow_redirects=True)
            resp.raise_for_status()
            return resp.content
    except Exception as exc:
        logger.error(f"[TS-08] Failed to download {url}: {exc}")
        return None


def _crop_image(image_bytes: bytes, dims: tuple[int, int]) -> bytes:
    """
    Centre-crop image to the target dimensions using Pillow ImageOps.fit.
    Returns JPEG bytes at quality 90.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    cropped = ImageOps.fit(img, dims, method=Image.LANCZOS, centering=(0.5, 0.5))
    buf = io.BytesIO()
    cropped.save(buf, format="JPEG", quality=90, optimize=True)
    return buf.getvalue()
