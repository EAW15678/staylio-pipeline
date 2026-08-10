"""
TS-01 — PMC Website Scraper
Tool: Firecrawl

Requests both markdown (for Claude text extraction) and json (for LLM-scoped
listing photo URLs) in a single Firecrawl call.  The json format uses
Firecrawl's native LLM extraction with a prompt that scopes to photos of THIS
listing only — no nearby listings, guides, or sidebar content.

FIX NOTES (v1 → v2):
  Switched from /extract (paid-only, 402 error) to /scrape (free tier).
  Returns Markdown passed to Claude parser for structured extraction.

FIX NOTES (v2 → v3):
  Added json format for scoped photo extraction (5 credits vs 1 for
  markdown-only).  Resolves cross-property contamination where sidebar
  "Nearby listings" images were ingested as this property's photos.
  Falls back to regex extraction if json is missing/empty.
"""

import os
import logging
import re
from typing import Optional

import httpx

from pipeline_emitter import emit_media_cost

from models.property import (
    DataSource,
    PropertyKnowledgeBase,
)

logger = logging.getLogger(__name__)

FIRECRAWL_API_BASE = "https://api.firecrawl.dev/v1"
FIRECRAWL_API_KEY  = os.environ["FIRECRAWL_API_KEY"]
REQUEST_TIMEOUT    = 90  # Raised from 60 — json extraction takes longer

# ── Firecrawl JSON extraction config ─────────────────────────────────────
# Generic scoping prompt — no CMS-specific knowledge.  Tested empirically
# on vacationrentalslbi.com: returned 23 listing photos, 0 sidebar photos.
_PHOTO_SCOPING_PROMPT = (
    "Extract only photos that belong to THIS specific vacation rental listing. "
    "Exclude photos from nearby/similar property listings, neighbourhood guides, "
    "advertisements, icons, and UI elements."
)

_PHOTO_SCOPING_SCHEMA = {
    "type": "object",
    "properties": {
        "listing_photos": {
            "type": "array",
            "items": {"type": "object", "properties": {"url": {"type": "string"}}},
            "description": "Photos of THIS listing only",
        },
    },
}

# ── URL rewrite registry ─────────────────────────────────────────────────
# Each rule: (compiled_regex, replacement_function).
# Applied in order after scoping.  First match wins.
# Adding a CMS = adding a rule, not editing logic.

def _rewrite_path_dimensions(url: str) -> str:
    """Rewrite /w.NNN/h.NNN/ path segments to full size.
    Covers PMC sites using dynamic image resizing (e.g. vacationrentalslbi.com).
    Verified: server returns the original image when size exceeds actual; requesting
    w.9999 returns the same bytes as w.1280 (251,962 bytes vs 2,340 for w.84).
    """
    url = re.sub(r"/w\.\d+/", "/w.1280/", url)
    url = re.sub(r"/h\.\d+/", "/h.853/", url)
    return url


def _rewrite_query_dimensions(url: str) -> str:
    """Rewrite ?w=NNN or ?width=NNN query params to request a large size.
    Covers Airbnb CDN (im_w=) and generic image proxy patterns.
    """
    url = re.sub(r"[?&]im_w=\d+", "?im_w=1200", url)
    url = re.sub(r"[?&]w=\d+", "?w=1280", url)
    url = re.sub(r"[?&]width=\d+", "?width=1280", url)
    return url


_REWRITE_RULES = [
    (re.compile(r"/w\.\d+/h\.\d+/"), _rewrite_path_dimensions),
    (re.compile(r"[?&](?:im_w|w|width)=\d+"), _rewrite_query_dimensions),
]


def _upgrade_photo_urls(urls: list[str]) -> list[str]:
    """Apply rewrite rules to upgrade photo URLs to full resolution,
    then deduplicate by image identity.

    Dedup strategy: after rewriting, two URLs that differ only in their
    size parameters will become identical and collapse.  For PMC URLs
    with an i.<hash> identity segment, dedupe on the hash to catch any
    remaining variants.
    """
    upgraded: list[str] = []
    for url in urls:
        new_url = url
        for pattern, rewriter in _REWRITE_RULES:
            if pattern.search(url):
                new_url = rewriter(url)
                break  # first match wins
        upgraded.append(new_url)

    # Deduplicate — prefer first occurrence (preserves Firecrawl ordering)
    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()
    result: list[str] = []
    for url in upgraded:
        if url in seen_urls:
            continue
        # PMC identity: /i.<hash>.ext — collapse size variants of the same image
        hash_match = re.search(r"/i\.([^/.]+)\.", url)
        if hash_match:
            img_hash = hash_match.group(1)
            if img_hash in seen_hashes:
                continue
            seen_hashes.add(img_hash)
        seen_urls.add(url)
        result.append(url)

    if len(result) < len(urls):
        logger.info(
            "[TS-01] URL upgrade: %d → %d (deduplicated %d size variants)",
            len(urls), len(result), len(urls) - len(result),
        )
    return result


# ── Main scraper ─────────────────────────────────────────────────────────

def scrape_pmc_website(
    pmc_website_url: str,
    knowledge_base: PropertyKnowledgeBase,
) -> PropertyKnowledgeBase:
    logger.info(f"[TS-01] Firecrawl scraping PMC website: {pmc_website_url}")

    try:
        markdown, scoped_photos = _firecrawl_scrape_with_photos(pmc_website_url)
    except Exception as exc:
        error_msg = f"Firecrawl scrape failed for {pmc_website_url}: {exc}"
        logger.error(error_msg)
        knowledge_base.ingestion_errors.append(error_msg)
        return knowledge_base

    if not markdown:
        knowledge_base.ingestion_errors.append(
            f"Firecrawl returned empty markdown for {pmc_website_url}"
        )
        return knowledge_base

    # Determine photo source: Firecrawl json (scoped) or regex fallback (unscoped)
    if scoped_photos:
        photo_urls = scoped_photos
        logger.info(
            "[TS-01] Using Firecrawl JSON-scoped photos: %d listing photos",
            len(photo_urls),
        )
    else:
        # LOUD FALLBACK — this is the unscoped path that caused cross-property
        # contamination.  Log at WARNING so it is visible in production logs.
        from agents.agent1.claude_parser import _extract_photo_urls
        photo_urls = _extract_photo_urls(markdown)
        logger.warning(
            "[TS-01] ⚠️ FALLBACK: Firecrawl json scoping returned no photos — "
            "using unscoped regex extraction (%d URLs). Photos may include "
            "sidebar/nearby listings. Source: %s",
            len(photo_urls), pmc_website_url,
        )

    # Upgrade resolution and deduplicate size variants
    photo_urls = _upgrade_photo_urls(photo_urls)

    # Flag low photo counts
    if len(photo_urls) < 3:
        logger.warning(
            "[TS-01] ⚠️ LOW PHOTO COUNT: only %d listing photos extracted from %s",
            len(photo_urls), pmc_website_url,
        )

    emit_media_cost(
        vendor="firecrawl",
        service="scrape_with_json",
        units=5,
        unit_name="credits",
        property_id=str(knowledge_base.property_id),
        workflow_name="listing_generation",
        slot_name="pmc_website_scrape",
        generation_reason="pmc_website_scrape",
    )

    from agents.agent1.claude_parser import _claude_extract, _apply_extraction

    extraction = _claude_extract(markdown, pmc_website_url)
    if not extraction:
        knowledge_base.ingestion_errors.append(
            f"Claude parser returned no data for {pmc_website_url}"
        )
        return knowledge_base

    # Override photo_urls with the scoped + upgraded list
    extraction["photo_urls"] = photo_urls

    knowledge_base = _apply_extraction(extraction, knowledge_base, DataSource.PMC_WEBSITE)

    if DataSource.PMC_WEBSITE not in knowledge_base.ingestion_sources:
        knowledge_base.ingestion_sources.append(DataSource.PMC_WEBSITE)

    logger.info(
        f"[TS-01] PMC website scrape complete. "
        f"Photos: {len(knowledge_base.photos)}, "
        f"Amenities: {len(knowledge_base.amenities)}"
    )
    return knowledge_base


def firecrawl_scrape_to_markdown(url: str) -> Optional[str]:
    """
    Public wrapper for the /preview endpoint.
    Scrapes a URL via Firecrawl and returns the raw markdown.
    Uses markdown-only (1 credit) — no json extraction needed for previews.
    """
    return _firecrawl_scrape_markdown_only(url)


# ── Firecrawl API calls ──────────────────────────────────────────────────

def _firecrawl_scrape_with_photos(url: str) -> tuple[Optional[str], list[str]]:
    """Scrape a PMC listing page for both markdown text and scoped photos.

    Returns (markdown, photo_urls).  photo_urls may be empty if json
    extraction fails — caller must handle the fallback.

    Credit cost: 5 per call (1 base + 4 for json extraction).
    """
    headers = {
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "url": url,
        "formats": ["markdown", "json"],
        "onlyMainContent": True,
        "waitFor": 3000,
        "jsonOptions": {
            "prompt": _PHOTO_SCOPING_PROMPT,
            "schema": _PHOTO_SCOPING_SCHEMA,
        },
    }

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.post(
            f"{FIRECRAWL_API_BASE}/scrape",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    if not data.get("success"):
        logger.warning(f"[TS-01] Firecrawl scrape returned success=false for {url}: {data}")
        return None, []

    result_data = data.get("data", {})
    markdown = result_data.get("markdown") or ""
    if markdown:
        logger.info(f"[TS-01] Firecrawl scrape returned {len(markdown)} chars of markdown")

    # Extract scoped photos from json result
    json_result = result_data.get("json") or {}
    listing_photos = json_result.get("listing_photos") or []
    photo_urls = [
        p.get("url") or p
        for p in listing_photos
        if (p.get("url") if isinstance(p, dict) else p)
    ]
    photo_urls = [u for u in photo_urls if u and isinstance(u, str) and u.startswith("http")]

    logger.info(
        "[TS-01] Firecrawl json extraction returned %d scoped listing photos",
        len(photo_urls),
    )

    return markdown or None, photo_urls


def _firecrawl_scrape_markdown_only(url: str) -> Optional[str]:
    """Markdown-only scrape (1 credit).  Used by the /preview endpoint."""
    headers = {
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": True,
        "waitFor": 3000,
    }

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.post(
            f"{FIRECRAWL_API_BASE}/scrape",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    if not data.get("success"):
        logger.warning(f"[TS-01] Firecrawl scrape returned success=false for {url}: {data}")
        return None

    markdown = data.get("data", {}).get("markdown") or data.get("markdown")
    if not markdown:
        logger.warning(f"[TS-01] Firecrawl scrape returned no markdown for {url}")
        return None

    logger.info(f"[TS-01] Firecrawl scrape returned {len(markdown)} chars of markdown")
    return markdown
