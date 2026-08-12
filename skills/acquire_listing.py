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
    skills_r2_upload,
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


def acquire_owner_photos(
    property_id: str,
    photo_urls: list,
    *,
    run_id: str = None,
) -> dict:
    """Download owner-uploaded photos and write as photographs + renditions.

    Owner photos have source_systems=['intake_portal'] and are treated
    with owner precedence (Ruling 1: owner outranks scraped).
    Returns {photos_new, photos_existing, photos_failed}.
    """
    try:
        sb = get_substrate()
    except EnvironmentError:
        return {"photos_new": 0, "photos_existing": 0, "photos_failed": 0}

    photos_new = 0
    photos_existing = 0
    photos_failed = 0

    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for url_entry in photo_urls:
            url = url_entry if isinstance(url_entry, str) else (url_entry.get("url") if isinstance(url_entry, dict) else None)
            if not url:
                continue

            try:
                resp = client.get(url)
                resp.raise_for_status()
                img_bytes = resp.content
                content_hash = hashlib.sha256(img_bytes).hexdigest()
                pid = _photo_id(property_id, content_hash)

                existing = sb.table("photographs").select("photo_id").eq("photo_id", pid).limit(1).execute()
                if existing.data:
                    photos_existing += 1
                    continue

                try:
                    from PIL import Image
                    img = Image.open(io.BytesIO(img_bytes))
                    width, height = img.width, img.height
                except Exception:
                    width, height = None, None

                # Upload to staging R2
                try:
                    key = f"{property_id}/original/owner_{content_hash[:8]}.jpg"
                    r2_url = skills_r2_upload(key, img_bytes, "image/jpeg")
                except Exception:
                    r2_url = url

                sb.table("photographs").insert({
                    "photo_id": pid,
                    "property_id": property_id,
                    "content_hash": content_hash,
                    "source_urls": [url],
                    "source_systems": ["intake_portal"],
                    "is_canonical": True,
                    "image_width": width,
                    "image_height": height,
                    "quality_tier": _quality_tier(width, height),
                }).execute()

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
                logger.info("[acquire_owner] Owner photo %s: %dx%d", content_hash[:8], width or 0, height or 0)

            except Exception as exc:
                logger.warning("[acquire_owner] Failed: %s — %s", str(url)[-40:], str(exc)[:60])
                photos_failed += 1

    if run_id and (photos_new > 0 or photos_failed > 0):
        emit_cost(sb, run_id, property_id,
                  vendor="supabase_storage", service="owner_photo_download",
                  units=photos_new, unit_name="photos",
                  unit_cost=0, total_cost=0,
                  workflow_name="acquire_listing", generation_reason="owner_upload")

    logger.info("[acquire_owner] Owner photos: %d new, %d existing, %d failed", photos_new, photos_existing, photos_failed)
    return {"photos_new": photos_new, "photos_existing": photos_existing, "photos_failed": photos_failed}


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
    elif source_type == "airbnb":
        try:
            apify_token = require_env("APIFY_API_TOKEN", "Apify Airbnb scraping")
            from agents.agent1.apify_scraper import _scrape_airbnb
            from models.property import PropertyKnowledgeBase, DataSource
            # Build minimal KB for the scraper
            temp_kb = PropertyKnowledgeBase(property_id=property_id)
            temp_kb = _scrape_airbnb(source_url, temp_kb)
            # Extract photos and reviews
            for p in temp_kb.photos:
                photo_urls.append(p.url)
            for r in temp_kb.guest_reviews:
                if r.text:
                    sb.table("guest_evidence").insert({
                        "property_id": property_id,
                        "written_text": r.text or "",
                        "verbal_text": "",
                        "reviewer_name": r.reviewer_name,
                        "stay_date": r.stay_date,
                        "source": "airbnb",
                        "is_guest_book": False,
                        "star_rating": r.star_rating,
                    }).execute()
            # Extract property fields
            extracted_fields = {}
            if temp_kb.name and temp_kb.name.value:
                extracted_fields["property_name"] = temp_kb.name.value
            if temp_kb.bedrooms and temp_kb.bedrooms.value:
                extracted_fields["bedrooms"] = temp_kb.bedrooms.value
            if temp_kb.bathrooms and temp_kb.bathrooms.value:
                extracted_fields["bathrooms"] = temp_kb.bathrooms.value

            apify_cost = 0.05
            total_cost += apify_cost
            emit_cost(sb, run_id, property_id,
                      vendor="apify", service="airbnb_listing",
                      units=1, unit_name="runs",
                      unit_cost=0.05, total_cost=round(apify_cost, 4),
                      workflow_name="acquire_listing", generation_reason="airbnb_scrape")
            logger.info("[acquire] Airbnb: %d photos, %d reviews", len(photo_urls), len(temp_kb.guest_reviews))
        except Exception as exc:
            error_str = str(exc)
            logger.error("[acquire] Airbnb scrape failed: %s", error_str[:120])
            complete_step(sb, step_id, status="failed", error_message=error_str[:200])
            complete_run(sb, run_id, status="failed")
            return SkillResult.failed(
                reason=f"Airbnb scrape failed: {error_str[:200]}",
                attempted=1, succeeded=0, failed_count=1, error_class="vendor",
            )

    elif source_type == "vrbo":
        try:
            apify_token = require_env("APIFY_API_TOKEN", "Apify VRBO scraping")
            from agents.agent1.apify_scraper import _scrape_vrbo
            from models.property import PropertyKnowledgeBase, DataSource
            temp_kb = PropertyKnowledgeBase(property_id=property_id)
            temp_kb = _scrape_vrbo(source_url, temp_kb)
            for p in temp_kb.photos:
                photo_urls.append(p.url)
            for r in temp_kb.guest_reviews:
                if r.text:
                    sb.table("guest_evidence").insert({
                        "property_id": property_id,
                        "written_text": r.text or "",
                        "verbal_text": "",
                        "reviewer_name": r.reviewer_name,
                        "stay_date": r.stay_date,
                        "source": "vrbo",
                        "is_guest_book": False,
                    }).execute()
            extracted_fields = {}
            if temp_kb.name and temp_kb.name.value:
                extracted_fields["property_name"] = temp_kb.name.value

            apify_cost = 0.05
            total_cost += apify_cost
            emit_cost(sb, run_id, property_id,
                      vendor="apify", service="vrbo_listing",
                      units=1, unit_name="runs",
                      unit_cost=0.05, total_cost=round(apify_cost, 4),
                      workflow_name="acquire_listing", generation_reason="vrbo_scrape")
            logger.info("[acquire] VRBO: %d photos, %d reviews", len(photo_urls), len(temp_kb.guest_reviews))
        except Exception as exc:
            error_str = str(exc)
            logger.error("[acquire] VRBO scrape failed: %s", error_str[:120])
            complete_step(sb, step_id, status="failed", error_message=error_str[:200])
            complete_run(sb, run_id, status="failed")
            return SkillResult.failed(
                reason=f"VRBO scrape failed: {error_str[:200]}",
                attempted=1, succeeded=0, failed_count=1, error_class="vendor",
            )

    else:
        complete_step(sb, step_id, status="failed", error_message=f"Unknown source type: {source_type}")
        complete_run(sb, run_id, status="failed")
        return SkillResult.failed(
            reason=f"Unknown source type: {source_type}",
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
    # Ruling 1: Owner-supplied intake data OUTRANKS scraped listing text.
    # Owner-territory fields (name, city, state, postal_code, booking_url)
    # are set by ingest_intake from the owner's portal answers. The scrape
    # fills them ONLY when null — never overwrites owner-supplied values.
    #
    # Fields classified as owner-territory:
    #   name          — owner named it at intake
    #   city          — owner entered it
    #   state_region  — owner entered it
    #   postal_code   — owner entered it
    #   booking_url   — owner designated their booking page
    #
    # Fields the scrape may set freely:
    #   property_type — inferred from listing, owner rarely sets this
    if extracted_fields:
        # Re-read current property to check which fields are null
        current = sb.table("properties").select(
            "name, city, state_region, postal_code, booking_url, property_type"
        ).eq("id", property_id).limit(1).execute()
        current_vals = current.data[0] if current.data else {}

        prop_update = {}

        # Owner-territory: fill only when null
        for ext_key, prop_key in [
            ("property_name", "name"), ("city", "city"),
            ("state", "state_region"), ("zip_code", "postal_code"),
            ("booking_url", "booking_url"),
        ]:
            v = extracted_fields.get(ext_key)
            if v and isinstance(v, str) and v.strip() and not current_vals.get(prop_key):
                prop_update[prop_key] = v.strip()

        # Scrape-territory: always update
        v = extracted_fields.get("property_type")
        if v and isinstance(v, str) and v.strip():
            prop_update["property_type"] = v.strip()

        if prop_update:
            sb.table("properties").update(prop_update).eq("id", property_id).execute()
            logger.info("[acquire] Updated %d property fields (owner-territory: only nulls filled)", len(prop_update))

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

                # Upload to skills R2 bucket — env-driven, no hardcoded bucket
                try:
                    key = f"{property_id}/original/photo_{content_hash[:8]}.jpg"
                    r2_url = skills_r2_upload(key, img_bytes, "image/jpeg")
                except Exception:
                    # Use source URL as storage_url if R2 not configured
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
