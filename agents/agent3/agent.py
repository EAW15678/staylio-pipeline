"""
Agent 3 — Visual Media Agent
LangGraph Node

Runs in PARALLEL with Agents 2 and 4 after Agent 1 completes.
Agent 5 (Website Builder) waits for Agent 3 to complete before
assembling the landing page.

Pipeline:
  1. Load knowledge base from Redis (Agent 1 cache)
  2. Download all photos from their source URLs
  3. Run provenance baseline on originals (Vision API)
  4. Upload originals to R2 (staylio-originals)
  5. Enhance all photos via Claid.ai (TS-07) — governance enforced
  6. Upload enhanced photos to R2 (staylio-enhanced)
  7. Run Vision API tagging + scoring on enhanced photos (TS-07b)
  8. Select category winners and hero image
  9. Generate social format crops for category winners (TS-08)
  10. Generate all 8 property videos (TS-09)
  11. Save all MediaAsset and VideoAsset records to Supabase
  12. Cache VisualMediaPackage in Redis for Agent 5
  13. Update pipeline status
"""

import asyncio
import hashlib
import io
import logging
import os
from typing import Optional

import httpx
from PIL import Image

from agents.agent3.claid_enhancer import enhance_photo_batch_sync
from agents.agent3.crop_orchestrator import generate_social_crops
from agents.agent3.models import (
    MediaAsset,
    SubjectCategory,
    VisualMediaPackage,
)
from agents.agent3.r2_storage import (
    BUCKET_ENHANCED,
    BUCKET_ORIGINALS,
    upload_photo_enhanced,
    upload_photo_original,
)
from agents.agent3.video_generator import generate_all_videos
from agents.agent3.vision_tagger import (
    tag_and_score_photos,
    tag_original_for_provenance,
)
from core.pipeline_status import (
    PipelineStepStatus,
    cache_knowledge_base,
    get_cached_knowledge_base,
    update_pipeline_status,
)

logger = logging.getLogger(__name__)

AGENT_NUMBER = 3
DOWNLOAD_CONCURRENCY = 10
DOWNLOAD_TIMEOUT     = 30


def agent3_node(state: dict) -> dict:
    """
    LangGraph node for Agent 3 — Visual Media.
    Runs synchronously; async operations wrapped with asyncio.run().
    """
    property_id = state["property_id"]
    logger.info(f"[Agent 3] Starting visual media processing for property {property_id}")

    update_pipeline_status(property_id, AGENT_NUMBER, PipelineStepStatus.RUNNING)

    # ── Step 1: Load knowledge base ───────────────────────────────────────
    kb = get_cached_knowledge_base(property_id) or state.get("knowledge_base", {})
    if not kb:
        error = f"Agent 3: Knowledge base not found for property {property_id}"
        update_pipeline_status(property_id, AGENT_NUMBER, PipelineStepStatus.FAILED, error)
        return {**state, "errors": state.get("errors", []) + [error], "agent3_complete": False}

    vibe_profile  = kb.get("vibe_profile") or "family_adventure"
    property_name = (kb.get("name") or {}).get("value", "")
    guest_reviews = kb.get("guest_reviews") or []
    seasonal_notes = kb.get("seasonal_notes")

    # ── Step 2: Collect all photo URLs ────────────────────────────────────
    photo_entries = kb.get("photos") or []
    photo_urls    = [p["url"] for p in photo_entries if p.get("url")]

    if not photo_urls:
        logger.warning(f"[Agent 3] No photos found for property {property_id}")
        update_pipeline_status(
            property_id, AGENT_NUMBER, PipelineStepStatus.COMPLETE,
            metadata={"warning": "No photos — visual media skipped"}
        )
        return {**state, "agent3_complete": True, "visual_media_package": {}}

    pkg = VisualMediaPackage(property_id=property_id)

    # ── Step 3: Download + tag originals + upload to R2 ──────────────────
    logger.info(f"[Agent 3] Downloading {len(photo_urls)} photos for property {property_id}")
    downloaded = asyncio.run(_download_photos(photo_urls))

    assets: list[MediaAsset] = []
    for i, (url, photo_bytes) in enumerate(downloaded):
        if not photo_bytes:
            pkg.processing_errors.append(f"Photo download failed: {url}")
            continue

        filename = _stable_filename(url, i)
        source_label = _detect_source(url)

        # Resize if over 9MB to stay within Claid.ai input limits
        photo_bytes = _resize_if_oversized(photo_bytes, max_bytes=9 * 1024 * 1024)

        # Upload original to R2
        try:
            original_r2_url = upload_photo_original(property_id, photo_bytes, filename)
        except Exception as exc:
            pkg.processing_errors.append(f"Original R2 upload failed for photo {i}: {exc}")
            original_r2_url = url   # Fall back to source URL

        asset = MediaAsset(
            property_id=property_id,
            asset_url_original=original_r2_url,
            source=source_label,
        )

        # Baseline Vision API labels for provenance check
        asset = tag_original_for_provenance(asset)
        assets.append(asset)

    # ── Step 4: Claid.ai enhancement (TS-07) ─────────────────────────────
    logger.info(f"[Agent 3] Enhancing {len(assets)} photos via Claid.ai")
    original_r2_urls = [a.asset_url_original for a in assets]
    enhancement_results = enhance_photo_batch_sync(original_r2_urls, property_id)

    for asset, (original_url, enhanced_bytes) in zip(assets, enhancement_results):
        if not enhanced_bytes:
            pkg.processing_errors.append(f"Enhancement failed for {original_url}")
            asset.asset_url_enhanced = None
            continue

        filename = os.path.basename(asset.asset_url_original.split("?")[0])
        try:
            enhanced_r2_url = upload_photo_enhanced(property_id, enhanced_bytes, filename)
            asset.asset_url_enhanced = enhanced_r2_url
        except Exception as exc:
            pkg.processing_errors.append(f"Enhanced R2 upload failed: {exc}")
            asset.asset_url_enhanced = None

    # ── Step 5: Vision API tagging, scoring, hero selection (TS-07b) ─────
    logger.info(f"[Agent 3] Running Vision API tagging for {len(assets)} photos")
    assets, hero_url = tag_and_score_photos(assets, vibe_profile, property_id)

    # Identify category winners
    category_winners = {
        asset.subject_category.value: asset.asset_url_enhanced or asset.asset_url_original
        for asset in assets
        if asset.social_crop_queued and (asset.asset_url_enhanced or asset.asset_url_original)
    }

    pkg.media_assets    = assets
    pkg.hero_photo_url  = hero_url
    pkg.category_winners = category_winners

    # Flag any provenance violations for AM review
    flagged = [a for a in assets if a.provenance_flag]
    if flagged:
        logger.warning(
            f"[Agent 3] {len(flagged)} photos flagged for provenance review — "
            f"labels diverged between original and enhanced"
        )
        for a in flagged:
            pkg.processing_errors.append(
                f"Provenance flag: {a.asset_url_enhanced} — "
                f"subject labels changed after enhancement. Human review required."
            )

    # ── Step 6: Social format crops (TS-08) ───────────────────────────────
    logger.info(f"[Agent 3] Generating social crops for {len(category_winners)} category winners")
    social_crops = generate_social_crops(
        property_id=property_id,
        vibe_profile=vibe_profile,
        property_name=property_name,
        category_winners=category_winners,
        apply_overlays=True,
        bannerbear_template_id=os.environ.get("BANNERBEAR_TEMPLATE_ID"),
    )
    pkg.social_crops = social_crops

    # ── Step 7: Video generation (TS-09) ─────────────────────────────────
    # Content package from Agent 2 (may not be ready yet if Agent 2 is still running)
    # Videos that need Agent 2 output (hero headline, tagline) wait for it
    # Videos that don't need it (walk-through, feature close-up) can proceed
    content_package = state.get("content_package") or {}

    # Get first local recommendation for Video 5
    local_guide = (kb.get("dont_miss_picks") or [])
    location_highlight = local_guide[0] if local_guide else None

    logger.info(f"[Agent 3] Generating 8-video launch library for property {property_id}")
    video_assets = asyncio.run(generate_all_videos(
        property_id=property_id,
        vibe_profile=vibe_profile,
        category_winners=category_winners,
        hero_photo_url=hero_url,
        guest_reviews=guest_reviews,
        content_package=content_package,
        seasonal_notes=seasonal_notes,
        location_highlight=location_highlight,
    ))

    pkg.video_assets  = video_assets
    pkg.videos_queued = len(video_assets) > 0

    review_videos_generated = sum(
        1 for v in video_assets
        if v.video_type.value.startswith("guest_review")
    )
    pkg.review_videos_pending = review_videos_generated < 3

    # ── Step 8: Save to Supabase ──────────────────────────────────────────
    _save_media_assets(assets)
    _save_video_assets(video_assets)
    _save_social_crops(social_crops)
    _flag_provenance_violations(property_id, flagged)

    # ── Step 9: Cache in Redis for Agent 5 ───────────────────────────────
    pkg_dict = pkg.to_dict()
    cache_knowledge_base(f"{property_id}:visual_media", pkg_dict, ttl_seconds=86400)

    # ── Step 10: Pipeline status ──────────────────────────────────────────
    update_pipeline_status(
        property_id, AGENT_NUMBER, PipelineStepStatus.COMPLETE,
        metadata={
            "photos_enhanced": len([a for a in assets if a.asset_url_enhanced]),
            "photos_flagged": len(flagged),
            "category_winners": len(category_winners),
            "social_crops": len(social_crops),
            "videos_generated": len(video_assets),
            "hero_url": hero_url,
            "review_videos_pending": pkg.review_videos_pending,
            "errors": pkg.processing_errors[:5],
        },
    )

    logger.info(
        f"[Agent 3] Complete for property {property_id}. "
        f"Enhanced: {len([a for a in assets if a.asset_url_enhanced])}, "
        f"Crops: {len(social_crops)}, Videos: {len(video_assets)}, "
        f"Hero: {'yes' if hero_url else 'none'}, "
        f"Errors: {len(pkg.processing_errors)}"
    )

    return {
        **state,
        "visual_media_package": pkg_dict,
        "agent3_complete": True,
    }


# ── Internal helpers ──────────────────────────────────────────────────────

def _resize_if_oversized(photo_bytes: bytes, max_bytes: int) -> bytes:
    """Reduce JPEG quality iteratively until the image is under max_bytes."""
    if len(photo_bytes) <= max_bytes:
        return photo_bytes
    try:
        img = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
        quality = 85
        while quality >= 20:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            resized = buf.getvalue()
            if len(resized) <= max_bytes:
                logger.info(
                    f"[Agent 3] Resized photo from {len(photo_bytes) / 1024 / 1024:.1f}MB "
                    f"to {len(resized) / 1024 / 1024:.1f}MB at quality={quality}"
                )
                return resized
            quality -= 10
        logger.warning(
            f"[Agent 3] Could not reduce photo below {max_bytes / 1024 / 1024:.0f}MB "
            f"(final size {len(resized) / 1024 / 1024:.1f}MB at quality={quality + 10})"
        )
        return resized
    except Exception as exc:
        logger.warning(f"[Agent 3] Photo resize failed, using original: {exc}")
        return photo_bytes


async def _download_photos(urls: list[str]) -> list[tuple[str, Optional[bytes]]]:
    """Download all photos concurrently with bounded concurrency."""
    semaphore = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)

    async def _download_one(url: str) -> tuple[str, Optional[bytes]]:
        async with semaphore:
            try:
                async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT) as client:
                    resp = await client.get(url, follow_redirects=True)
                    resp.raise_for_status()
                    return url, resp.content
            except Exception as exc:
                logger.warning(f"Photo download failed: {url}: {exc}")
                return url, None

    tasks = [_download_one(url) for url in urls]
    return await asyncio.gather(*tasks)


def _stable_filename(url: str, index: int) -> str:
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"photo_{index:03d}_{url_hash}.jpg"


def _detect_source(url: str) -> str:
    """Detect whether a photo came from Airbnb, VRBO, or was uploaded."""
    if "airbnb" in url or "muscache" in url:
        return "airbnb_scraped"
    if "vrbo" in url or "homeaway" in url:
        return "vrbo_scraped"
    if "getbooked" in url or "r2.cloudflarestorage" in url:
        return "intake_upload"
    return "unknown"


def _save_media_assets(assets: list[MediaAsset]) -> None:
    """Bulk upsert all media asset records to Supabase."""
    if not assets:
        return
    try:
        from core.supabase_store import get_supabase
        records = [a.to_db_record() for a in assets]
        get_supabase().table("media_assets").upsert(
            records,
            on_conflict="property_id,asset_url_original",
        ).execute()
    except Exception as exc:
        logger.error(f"Supabase media_assets save failed: {exc}")


def _save_video_assets(video_assets: list) -> None:
    """Save video asset records to Supabase."""
    if not video_assets:
        return
    try:
        from core.supabase_store import get_supabase
        records = [v.to_db_record() for v in video_assets]
        get_supabase().table("video_assets").upsert(
            records,
            on_conflict="property_id,video_type,format",
        ).execute()
    except Exception as exc:
        logger.error(f"Supabase video_assets save failed: {exc}")


def _save_social_crops(social_crops: list) -> None:
    """Save social crop records to Supabase."""
    if not social_crops:
        return
    try:
        from core.supabase_store import get_supabase
        records = [
            {
                "property_id": c.property_id,
                "source_asset_url": c.source_asset_url,
                "crop_url": c.crop_url,
                "format": c.format,
                "subject_category": c.subject_category,
                "has_overlay": c.has_overlay,
            }
            for c in social_crops
        ]
        get_supabase().table("social_crops").upsert(
            records,
            on_conflict="property_id,source_asset_url,format",
        ).execute()
    except Exception as exc:
        logger.error(f"Supabase social_crops save failed: {exc}")


def _flag_provenance_violations(property_id: str, flagged_assets: list[MediaAsset]) -> None:
    """Write provenance violation flags to the AM review queue."""
    if not flagged_assets:
        return
    try:
        from core.supabase_store import get_supabase
        from datetime import datetime, timezone
        get_supabase().table("am_review_queue").upsert(
            {
                "property_id": property_id,
                "review_type": "photo_provenance",
                "status": "pending",
                "failure_reasons": [
                    f"Photo provenance flag: {a.asset_url_enhanced}"
                    for a in flagged_assets
                ],
                "reviewer_notes": (
                    f"{len(flagged_assets)} photos have divergent labels between "
                    "original and enhanced versions. Verify no prohibited Claid.ai "
                    "operations were applied."
                ),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="property_id,review_type",
        ).execute()
    except Exception as exc:
        logger.error(f"AM review queue write failed for provenance flags: {exc}")
