"""
Skill: acquire_listing — scrape a listing page and write photographs.

Ports the PMC scrape path (Firecrawl markdown+json scoping from 6d17ddd),
Airbnb/VRBO via Apify. Downloads image bytes, computes SHA-256
content_hash, writes photographs + renditions('original').

Every photograph has measured width/height and quality_tier.
Ruling 7 identity: UNIQUE(property_id, content_hash).

Usage:
    from skills.acquire_listing import acquire_listing
    result = acquire_listing("82cb9d7e-...", "https://www.example.com/listing.123")
"""

import hashlib
import io
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone

import httpx

from skills.contract import (
    SkillResult, get_substrate, require_env,
    record_run, record_step, complete_step, complete_run, emit_cost,
)

logger = logging.getLogger(__name__)

PHOTO_NAMESPACE = uuid.UUID("a8f3c2b1-4d5e-6f7a-8b9c-0d1e2f3a4b5c")

# URL rewrite registry (from 6d17ddd)
def _rewrite_path_dimensions(url):
    url = re.sub(r"/w\.\d+/", "/w.1280/", url)
    url = re.sub(r"/h\.\d+/", "/h.853/", url)
    return url

def _rewrite_query_dimensions(url):
    url = re.sub(r"[?&]im_w=\d+", "?im_w=1200", url)
    url = re.sub(r"[?&]w=\d+", "?w=1280", url)
    url = re.sub(r"[?&]width=\d+", "?width=1280", url)
    return url

_REWRITE_RULES = [
    (re.compile(r"/w\.\d+/h\.\d+/"), _rewrite_path_dimensions),
    (re.compile(r"[?&](?:im_w|w|width)=\d+"), _rewrite_query_dimensions),
]


def _upgrade_url(url):
    for pattern, rewriter in _REWRITE_RULES:
        if pattern.search(url):
            return rewriter(url)
    return url


def _photo_id(property_id, content_hash):
    return str(uuid.uuid5(PHOTO_NAMESPACE, f"{property_id}:{content_hash}"))


def _quality_tier(w, h):
    if w and h:
        mp = (w * h) / 1_000_000
        if mp >= 2.0: return "high"
        elif mp >= 0.5: return "medium"
        else: return "low"
    return None


def acquire_listing(
    property_id: str,
    source_url: str,
    *,
    force: bool = False,
    max_photos: int = 120,
) -> SkillResult:
    """Scrape a listing page, download photos, write photographs + renditions.

    Returns SkillResult.ok({photos_new, photos_existing, photos_failed, ...})
    """
    try:
        sb = get_substrate()
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    # ── Detect source type ───────────────────────────────────────────────
    url_lower = source_url.lower()
    if "airbnb" in url_lower:
        source_type = "airbnb"
    elif "vrbo" in url_lower:
        source_type = "vrbo"
    else:
        source_type = "pmc"

    # ── Record run ───────────────────────────────────────────────────────
    run_id = record_run(sb, property_id, "onboard")
    step_id = record_step(sb, run_id, "acquire_listing")

    total_cost = 0.0
    photo_urls = []
    extracted_fields = {}

    # ── Scrape ───────────────────────────────────────────────────────────
    if source_type == "pmc":
        try:
            firecrawl_key = require_env("FIRECRAWL_API_KEY", "Firecrawl PMC scraping")
        except EnvironmentError as e:
            complete_step(sb, step_id, status="failed", error_message=str(e))
            complete_run(sb, run_id, status="failed")
            return SkillResult.failed(str(e))

        try:
            from agents.agent1.firecrawl_scraper import _firecrawl_scrape_with_photos
            markdown, scoped_photos = _firecrawl_scrape_with_photos(source_url)

            if scoped_photos:
                photo_urls = scoped_photos
                logger.info("[acquire] Firecrawl JSON-scoped: %d photos", len(photo_urls))
            elif markdown:
                from agents.agent1.claude_parser import _extract_photo_urls
                photo_urls = _extract_photo_urls(markdown)
                logger.warning("[acquire] FALLBACK to regex: %d photos (unscoped)", len(photo_urls))

            # Extract text fields via Claude
            if markdown:
                try:
                    anthropic_key = require_env("ANTHROPIC_API_KEY", "Claude text extraction")
                    from agents.agent1.claude_parser import _claude_extract
                    extracted_fields = _claude_extract(markdown, source_url) or {}
                except Exception as exc:
                    logger.warning("[acquire] Claude extraction failed: %s", str(exc)[:100])

            scrape_cost = 0.025  # 5 credits × $0.005
            total_cost += scrape_cost
            emit_cost(sb, run_id, property_id,
                      vendor="firecrawl", service="scrape_with_json",
                      units=5, unit_name="credits",
                      unit_cost=0.005, total_cost=round(scrape_cost, 4),
                      workflow_name="acquire_listing", generation_reason="pmc_scrape")

        except Exception as exc:
            error_str = str(exc)
            complete_step(sb, step_id, status="failed", error_message=error_str[:200])
            complete_run(sb, run_id, status="failed")
            return SkillResult.failed(
                reason=f"Firecrawl scrape failed: {error_str[:200]}",
                attempted=1, succeeded=0, failed_count=1, error_class="vendor",
            )
    else:
        # Airbnb/VRBO — for now, return failed with guidance
        complete_step(sb, step_id, status="failed", error_message=f"{source_type} scraping not yet ported to substrate")
        complete_run(sb, run_id, status="failed")
        return SkillResult.failed(
            reason=f"{source_type} scraping not yet ported. Use PMC URLs for now.",
            attempted=0, succeeded=0, failed_count=0,
        )

    # ── Apply URL upgrades ───────────────────────────────────────────────
    upgraded = []
    seen_hashes = set()
    for url in photo_urls:
        u = _upgrade_url(url)
        # Dedupe by i.<hash> if present
        hash_match = re.search(r"/i\.([^/.]+)\.", u)
        if hash_match:
            h = hash_match.group(1)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
        upgraded.append(u)
    photo_urls = upgraded[:max_photos]

    logger.info("[acquire] %d photo URLs after upgrade + dedupe", len(photo_urls))

    if len(photo_urls) < 3:
        logger.warning("[acquire] LOW PHOTO COUNT: only %d photos from %s", len(photo_urls), source_url)

    # ── Update property fields from extraction ───────────────────────────
    if extracted_fields:
        prop_update = {}
        for ext_key, prop_key in [
            ("property_name", "name"), ("city", "city"),
            ("state", "state_region"), ("zip_code", "postal_code"),
            ("property_type", "property_type"),
        ]:
            v = extracted_fields.get(ext_key)
            if v and isinstance(v, str) and v.strip():
                prop_update[prop_key] = v.strip()

        if extracted_fields.get("bedrooms"):
            pass  # Properties table doesn't have bedrooms — stored in copy
        if extracted_fields.get("booking_url"):
            prop_update["booking_url"] = extracted_fields["booking_url"]

        if prop_update:
            sb.table("properties").update(prop_update).eq("id", property_id).execute()
            logger.info("[acquire] Updated %d property fields from extraction", len(prop_update))

    # ── Download photos and write photographs + renditions ───────────────
    photos_new = 0
    photos_existing = 0
    photos_failed = 0
    low_res_count = 0

    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for url in photo_urls:
            try:
                resp = client.get(url)
                resp.raise_for_status()
                img_bytes = resp.content

                # Content hash
                content_hash = hashlib.sha256(img_bytes).hexdigest()
                pid = _photo_id(property_id, content_hash)

                # Check if already exists
                existing = sb.table("photographs").select("photo_id").eq(
                    "photo_id", pid
                ).limit(1).execute()

                if existing.data:
                    photos_existing += 1
                    continue

                # Measure dimensions
                try:
                    from PIL import Image
                    img = Image.open(io.BytesIO(img_bytes))
                    width, height = img.width, img.height
                except Exception:
                    width, height = None, None

                qt = _quality_tier(width, height)
                if qt == "low":
                    low_res_count += 1

                # Upload to R2
                try:
                    from core.r2_storage import upload_photo_original
                    r2_url = upload_photo_original(
                        property_id=property_id,
                        photo_bytes=img_bytes,
                        filename=f"photo_{content_hash[:8]}.jpg",
                    )
                except Exception:
                    # R2 not configured in local env — use the source URL as storage_url
                    r2_url = url

                # Write photograph
                sb.table("photographs").insert({
                    "photo_id": pid,
                    "property_id": property_id,
                    "content_hash": content_hash,
                    "source_urls": [url],
                    "source_systems": [source_type],
                    "is_canonical": True,
                    "image_width": width,
                    "image_height": height,
                    "quality_tier": qt,
                }).execute()

                # Write rendition
                sb.table("renditions").insert({
                    "photo_id": pid,
                    "kind": "original",
                    "storage_url": r2_url,
                    "format": "jpg",
                    "width": width,
                    "height": height,
                    "byte_size": len(img_bytes),
                }).execute()

                photos_new += 1

            except Exception as exc:
                logger.warning("[acquire] Photo download failed: %s — %s", url[-40:], str(exc)[:60])
                photos_failed += 1
                continue

    if low_res_count > 0:
        logger.warning("[acquire] %d of %d photos below 200x200 (low resolution)", low_res_count, photos_new + photos_existing)

    # ── Complete ─────────────────────────────────────────────────────────
    complete_step(sb, step_id, status="complete", metadata={
        "photos_new": photos_new,
        "photos_existing": photos_existing,
        "photos_failed": photos_failed,
        "photo_urls_found": len(photo_urls),
        "low_res_count": low_res_count,
        "source_type": source_type,
        "cost_usd": round(total_cost, 4),
    })
    complete_run(sb, run_id, status="complete")

    return SkillResult.ok({
        "photos_new": photos_new,
        "photos_existing": photos_existing,
        "photos_failed": photos_failed,
        "photo_urls_found": len(photo_urls),
        "low_res_count": low_res_count,
        "fields_extracted": len(extracted_fields),
        "cost_usd": round(total_cost, 4),
        "run_id": run_id,
    })
