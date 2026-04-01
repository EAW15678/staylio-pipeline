"""
TS-08 — Social Format Crop Orchestrator
Python wrapper around crop_worker.js (Sharp + Bannerbear)

Called after Vision API tagging identifies category winners.
Only category winners receive social crops — approximately
8-12 images per property rather than all 100.

Invokes the Node.js worker via subprocess, passes JSON payload,
collects cropped temp files, uploads to Cloudflare R2.
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from agents.agent3.models import SocialCrop, SubjectCategory
from agents.agent3.r2_storage import upload_social_crop

logger = logging.getLogger(__name__)

# Path to the Node.js crop worker
CROP_WORKER_PATH = Path(__file__).parent / "crop_worker.js"

# Default formats per category when no specific override
DEFAULT_FORMATS_BY_CATEGORY = {
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
) -> list[SocialCrop]:
    """
    Generate social format crops for all category winner photos.
    Calls the Node.js Sharp worker, uploads results to R2.

    Args:
        property_id:           Property UUID
        vibe_profile:          Vibe string for overlay badge
        property_name:         Used in Bannerbear overlay text
        category_winners:      Dict of category → enhanced photo URL
        apply_overlays:        Whether to apply Bannerbear overlays
        bannerbear_template_id: Bannerbear template UID for overlays

    Returns:
        List of SocialCrop records for Supabase storage
    """
    if not category_winners:
        logger.info(f"[TS-08] No category winners to crop for property {property_id}")
        return []

    # Build the crop payload for the Node.js worker
    crop_requests = []
    for category_str, photo_url in category_winners.items():
        try:
            category = SubjectCategory(category_str)
        except ValueError:
            category = SubjectCategory.UNCATEGORISED

        formats = DEFAULT_FORMATS_BY_CATEGORY.get(category, ["1_1", "9_16"])
        crop_requests.append({
            "source_url": photo_url,
            "formats": formats,
            "category": category_str,
            "apply_overlay": apply_overlays and bool(bannerbear_template_id),
            "overlay_template_id": bannerbear_template_id,
        })

    payload = {
        "property_id": property_id,
        "vibe_profile": vibe_profile,
        "property_name": property_name or "",
        "crops": crop_requests,
    }

    # Call Node.js worker
    result = _call_crop_worker(payload)
    if not result or not result.get("success"):
        logger.error(
            f"[TS-08] Crop worker failed for property {property_id}: "
            f"{result.get('error') if result else 'no response'}"
        )
        return []

    # Upload crops to R2 and build SocialCrop records
    social_crops = []
    for item in result.get("results", []):
        if item.get("error"):
            logger.warning(f"[TS-08] Crop failed for {item['source_url']}: {item['error']}")
            continue

        for crop_info in item.get("crops", []):
            tmp_file = crop_info.get("tmp_file")
            if not tmp_file or not os.path.exists(tmp_file):
                continue

            try:
                with open(tmp_file, "rb") as f:
                    crop_bytes = f.read()
                os.unlink(tmp_file)   # Clean up temp file

                filename = f"{crop_info['category']}_{crop_info['format']}.jpg"
                r2_url = upload_social_crop(
                    property_id=property_id,
                    crop_bytes=crop_bytes,
                    filename=filename,
                    format_label=crop_info["format"],
                )

                social_crops.append(SocialCrop(
                    property_id=property_id,
                    source_asset_url=item["source_url"],
                    crop_url=r2_url,
                    format=crop_info["format"],
                    subject_category=crop_info["category"],
                    has_overlay=crop_info.get("has_overlay", False),
                    overlay_template_id=bannerbear_template_id,
                ))

            except Exception as exc:
                logger.error(f"[TS-08] R2 upload failed for crop: {exc}")
                if tmp_file and os.path.exists(tmp_file):
                    os.unlink(tmp_file)

    logger.info(
        f"[TS-08] Social crops complete for property {property_id}: "
        f"{len(social_crops)} crops generated from {len(category_winners)} category winners"
    )
    return social_crops


def _call_crop_worker(payload: dict) -> Optional[dict]:
    """
    Invoke the Node.js crop worker via subprocess.
    Passes JSON payload on stdin, reads JSON result from stdout.
    """
    try:
        result = subprocess.run(
            ["node", str(CROP_WORKER_PATH)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=300,   # 5 min timeout for a full batch
        )
        if result.returncode != 0:
            logger.error(f"Crop worker non-zero exit: {result.stderr[:500]}")
            return None
        if not result.stdout.strip():
            logger.error("Crop worker produced no output")
            return None
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        logger.error("Crop worker timed out after 5 minutes")
        return None
    except Exception as exc:
        logger.error(f"Crop worker subprocess error: {exc}")
        return None
