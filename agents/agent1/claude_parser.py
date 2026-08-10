"""
TS-04c — Claude Fallback Parser
Tool: Claude API (claude-sonnet-4-6)

When a URL cannot be handled by Firecrawl or Apify (unrecognised OTA,
or a scraper failure), this module fetches the raw page HTML and sends
it to Claude with a structured extraction prompt.

Also used as the general-purpose normalisation layer: takes raw Markdown
from Firecrawl or raw JSON from Apify and asks Claude to identify gaps,
inconsistencies, and fill any missing fields it can infer.
"""

import os
import json
import logging
import re
from typing import Optional

import anthropic
import httpx

from models.property import (
    DataSource,
    PhotoAsset,
    PropertyField,
    PropertyKnowledgeBase,
)

logger = logging.getLogger(__name__)

ANTHROPIC_CLIENT = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = "claude-sonnet-4-6"
MAX_PAGE_CHARS = 40_000   # Trim large pages to control token cost

# Output token budget for the extraction call.  Two constraints:
#   - Too low → response truncated mid-JSON, json.loads raises (the original bug:
#     2,000 tokens cut Casa de Lubitz at char 5,982 → 0 photos, 0 amenities).
#   - Above 21,333 → Anthropic SDK refuses to send (ValueError: "Streaming is
#     required"); threshold is 3600 * max_tokens / 128_000 > 600.
# With photo_urls extracted by regex before the LLM call, the response contains
# only judgment fields (name, description, amenities, etc.) — comfortably under
# 8,000 tokens.
EXTRACT_MAX_TOKENS = 8_000

# ── Photo URL extraction ─────────────────────────────────────────────────────
# Image file extensions considered valid photo URLs.
_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".avif"})

# Substrings in URLs that indicate non-photo images (logos, icons, tracking pixels).
_NON_PHOTO_PATTERNS = (
    "/logo", "/icon", "/favicon", "/badge", "/sprite", "/pixel",
    "/tracking", "/analytics", "/beacon", "/spacer", "/blank",
    "/widget", "/button", "/arrow", "/spinner",
    "gravatar.com", "facebook.com", "instagram.com", "twitter.com",
    "googletagmanager.com", "google-analytics.com", "doubleclick.net",
)


def _extract_photo_urls(text: str) -> list[str]:
    """Extract photo URLs from markdown or plain text.

    Handles:
      - Markdown images: ![alt](url)
      - Bare image URLs on their own or in markdown links: [text](url.jpg)
      - Bare URLs in text matching image extensions

    Filters out logos, icons, tracking pixels, and data URIs.
    Returns deduplicated list preserving first-seen order.
    """
    urls: list[str] = []
    seen: set[str] = set()

    # 1. Markdown image syntax: ![...](url)
    for match in re.finditer(r'!\[[^\]]*\]\((https?://[^\s)]+)\)', text):
        urls.append(match.group(1))

    # 2. Markdown link syntax where the URL has an image extension: [...](url.jpg)
    for match in re.finditer(r'\[[^\]]*\]\((https?://[^\s)]+)\)', text):
        urls.append(match.group(1))

    # 3. Bare URLs with image extensions (catches URLs not in markdown syntax)
    for match in re.finditer(r'(https?://[^\s"\'<>)\]]+)', text):
        urls.append(match.group(1))

    # Deduplicate, filter, and validate
    result: list[str] = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)

        # Must have a recognisable image extension (strip query params for check)
        path = url.split("?")[0].split("#")[0].lower()
        if not any(path.endswith(ext) for ext in _IMAGE_EXTENSIONS):
            continue

        # Filter out non-photo URLs
        url_lower = url.lower()
        if any(pat in url_lower for pat in _NON_PHOTO_PATTERNS):
            continue

        # Skip tiny images (common pattern: dimensions encoded in URL)
        # e.g. /images/100x100/logo.jpg or ?w=50&h=50
        tiny = re.search(r'[/_ -](\d+)x(\d+)[/_ .-]', url_lower)
        if tiny:
            w, h = int(tiny.group(1)), int(tiny.group(2))
            if w < 100 or h < 100:
                continue

        # Skip data URIs (shouldn't appear but defensive)
        if url.startswith("data:"):
            continue

        result.append(url)

    return result


def parse_unknown_ota(
    url: str,
    knowledge_base: PropertyKnowledgeBase,
) -> PropertyKnowledgeBase:
    """
    TS-04c fallback: fetches the page, strips it to text, sends to Claude.
    Used when: unknown OTA platform, or Apify/Firecrawl returned no results.
    """
    logger.info(f"[TS-04c] Claude fallback parser for: {url}")

    page_text = _fetch_page_text(url)
    if not page_text:
        knowledge_base.ingestion_errors.append(
            f"Claude fallback: could not fetch page content from {url}"
        )
        return knowledge_base

    extraction = _claude_extract(page_text, url)
    if not extraction:
        knowledge_base.ingestion_errors.append(
            f"Claude fallback: extraction returned no data for {url}"
        )
        return knowledge_base

    return _apply_extraction(extraction, knowledge_base, DataSource.CLAUDE_PARSED)


def normalise_and_fill_gaps(
    knowledge_base: PropertyKnowledgeBase,
    raw_markdown: Optional[str] = None,
) -> PropertyKnowledgeBase:
    """
    Post-scrape normalisation pass.
    Claude reviews the assembled knowledge base and:
    1. Infers missing fields where possible (e.g. property type from description)
    2. Generates a slug if name is available but slug is missing
    3. Cleans up description if it contains HTML artefacts
    4. Deduplicates amenity list

    This is a low-cost Haiku call — not the full Sonnet extraction.
    """
    logger.info(f"[TS-04c] Normalisation pass for property {knowledge_base.property_id}")

    # Build a compact summary of current KB state for the prompt
    kb_summary = {
        "name": knowledge_base.name.value if knowledge_base.name else None,
        "description_excerpt": (
            (knowledge_base.description.value or "")[:500]
            if knowledge_base.description else None
        ),
        "bedrooms": knowledge_base.bedrooms.value if knowledge_base.bedrooms else None,
        "bathrooms": knowledge_base.bathrooms.value if knowledge_base.bathrooms else None,
        "max_occupancy": knowledge_base.max_occupancy.value if knowledge_base.max_occupancy else None,
        "property_type": knowledge_base.property_type.value if knowledge_base.property_type else None,
        "city": knowledge_base.city.value if knowledge_base.city else None,
        "state": knowledge_base.state.value if knowledge_base.state else None,
        "amenities_sample": [
            a.value for a in knowledge_base.amenities[:20] if a.value
        ],
        "photo_count": len(knowledge_base.photos),
        "review_count": len(knowledge_base.guest_reviews),
        "sources": knowledge_base.ingestion_sources,
    }

    prompt = f"""You are reviewing scraped vacation rental property data to identify and fix gaps.

Current property data:
{json.dumps(kb_summary, indent=2)}

Tasks:
1. If property_type is null and the description or name suggests it (cabin, condo, beach house, etc.), infer it
2. Generate a URL-safe slug from the property name (lowercase, hyphens, no special chars, max 50 chars)
3. If the amenity list has duplicates or near-duplicates (e.g. "WiFi" and "Wi-Fi"), note the canonical version
4. If description excerpt contains obvious HTML artifacts (tags, escaped entities), note that it needs cleaning

Respond ONLY with a JSON object containing these fields (null if you cannot determine):
{{
  "inferred_property_type": "cabin|condo|villa|beach_house|cottage|townhouse|apartment|house|null",
  "suggested_slug": "url-safe-slug-here-or-null",
  "amenity_duplicates": ["list", "of", "duplicate", "amenity", "strings", "to", "remove"],
  "description_needs_cleaning": true|false,
  "inferred_min_stay": null_or_integer
}}"""

    try:
        resp = ANTHROPIC_CLIENT.messages.create(
            model="claude-haiku-4-5-20251001",   # Use Haiku for this low-stakes normalisation
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        # Use raw_decode to take only the first valid JSON object,
        # ignoring any trailing text or extra objects in the response
        result, _ = json.JSONDecoder().raw_decode(raw.strip())
    except Exception as exc:
        logger.warning(f"[TS-04c] Normalisation pass failed: {exc}")
        return knowledge_base

    # Apply normalisation results
    if result.get("inferred_property_type") and knowledge_base.property_type is None:
        knowledge_base.property_type = PropertyField(
            value=result["inferred_property_type"],
            source=DataSource.CLAUDE_PARSED,
            confidence=0.70,
        )

    if result.get("suggested_slug") and not knowledge_base.slug:
        knowledge_base.slug = result["suggested_slug"]

    if result.get("amenity_duplicates"):
        dupes = {d.lower() for d in result["amenity_duplicates"]}
        knowledge_base.amenities = [
            a for a in knowledge_base.amenities
            if (a.value or "").lower() not in dupes
        ]

    return knowledge_base


# ── Internal helpers ──────────────────────────────────────────────────────

def _fetch_page_text(url: str) -> Optional[str]:
    """Fetch a URL and return plain text content, stripped of HTML."""
    try:
        with httpx.Client(
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0 (compatible; StaylioBot/1.0)"},
            follow_redirects=True,
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception as exc:
        logger.warning(f"Page fetch failed for {url}: {exc}")
        return None

    # Strip HTML tags for a clean text representation
    text = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Trim to control token cost
    return text[:MAX_PAGE_CHARS]


def _claude_extract(page_text: str, source_url: str) -> Optional[dict]:
    """
    Ask Claude to extract structured property data from raw page text.
    Returns a dict matching the PMC extract schema, or None.

    Photo URLs are extracted by regex before the LLM call — they are
    mechanically extractable from markdown image syntax and don't need
    LLM judgment. This keeps the LLM response small (judgment fields
    only) and avoids the token-limit truncation that caused the Casa de
    Lubitz failure (2026-08-10).
    """
    # Extract photo URLs mechanically before the LLM call
    photo_urls = _extract_photo_urls(page_text)
    logger.info("[TS-04c] Regex extracted %d photo URLs from page text", len(photo_urls))

    prompt = f"""You are extracting structured vacation rental property data from a web page.

Source URL: {source_url}
Page content:
---
{page_text}
---

Extract all available property information and respond ONLY with a JSON object.
If a field is not found, use null. Do not invent data.
Do NOT include photo URLs — those are extracted separately.

{{
  "property_name": "string or null",
  "headline": "string or null",
  "description": "full property description text or null",
  "bedrooms": integer_or_null,
  "bathrooms": number_or_null,
  "max_occupancy": integer_or_null,
  "property_type": "cabin|condo|villa|beach_house|cottage|apartment|house|other|null",
  "amenities": ["list", "of", "amenity", "strings"],
  "city": "string or null",
  "state": "two-letter state code or null",
  "zip_code": "string or null",
  "avg_nightly_rate": number_or_null,
  "neighborhood_description": "string or null",
  "booking_url": "direct booking URL if found or null"
}}"""

    try:
        resp = ANTHROPIC_CLIENT.messages.create(
            model=MODEL,
            max_tokens=EXTRACT_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )

        # Check for truncation before attempting JSON parse.
        # Pattern from agents/agent3/llm_curator.py:916-927.
        stop_reason = getattr(resp, "stop_reason", None)
        if stop_reason == "max_tokens":
            output_tokens = getattr(getattr(resp, "usage", None), "output_tokens", None)
            logger.error(
                "[TS-04c] Claude extraction TRUNCATED — response hit max_tokens=%d "
                "(output_tokens=%s). Increase EXTRACT_MAX_TOKENS. Source: %s",
                EXTRACT_MAX_TOKENS, output_tokens, source_url,
            )
            return None

        raw = resp.content[0].text.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)
    except Exception as exc:
        logger.error(f"[TS-04c] Claude extraction failed: {exc}")
        return None

    # Merge regex-extracted photo URLs into the result so _apply_extraction
    # sees the same shape it always has (dict with "photo_urls" key).
    result["photo_urls"] = photo_urls
    return result


def _apply_extraction(
    extraction: dict,
    knowledge_base: PropertyKnowledgeBase,
    source: DataSource,
) -> PropertyKnowledgeBase:
    """Apply Claude-extracted fields into the knowledge base."""

    def _f(val) -> Optional[PropertyField]:
        if val is None:
            return None
        if isinstance(val, str) and not val.strip():
            return None
        return PropertyField(value=val, source=source, confidence=0.70)

    knowledge_base.name        = knowledge_base.merge_field(knowledge_base.name,        _f(extraction.get("property_name")))
    knowledge_base.headline    = knowledge_base.merge_field(knowledge_base.headline,    _f(extraction.get("headline")))
    knowledge_base.description = knowledge_base.merge_field(knowledge_base.description, _f(extraction.get("description")))
    knowledge_base.bedrooms    = knowledge_base.merge_field(knowledge_base.bedrooms,    _f(extraction.get("bedrooms")))
    knowledge_base.bathrooms   = knowledge_base.merge_field(knowledge_base.bathrooms,   _f(extraction.get("bathrooms")))
    knowledge_base.max_occupancy = knowledge_base.merge_field(knowledge_base.max_occupancy, _f(extraction.get("max_occupancy")))
    knowledge_base.property_type = knowledge_base.merge_field(knowledge_base.property_type, _f(extraction.get("property_type")))
    knowledge_base.city        = knowledge_base.merge_field(knowledge_base.city,        _f(extraction.get("city")))
    knowledge_base.state       = knowledge_base.merge_field(knowledge_base.state,       _f(extraction.get("state")))
    knowledge_base.zip_code    = knowledge_base.merge_field(knowledge_base.zip_code,    _f(extraction.get("zip_code")))
    knowledge_base.avg_nightly_rate = knowledge_base.merge_field(
        knowledge_base.avg_nightly_rate, _f(extraction.get("avg_nightly_rate"))
    )
    knowledge_base.neighborhood_description = knowledge_base.merge_field(
        knowledge_base.neighborhood_description,
        _f(extraction.get("neighborhood_description")),
    )
    if not knowledge_base.booking_url:
        knowledge_base.booking_url = extraction.get("booking_url")

    # Amenities
    existing = {a.value.lower() for a in knowledge_base.amenities if a.value}
    for amenity in extraction.get("amenities", []):
        if amenity and amenity.lower() not in existing:
            knowledge_base.amenities.append(PropertyField(value=amenity, source=source, confidence=0.70))
            existing.add(amenity.lower())

    # Photos
    existing_urls = {p.url for p in knowledge_base.photos}
    for url in extraction.get("photo_urls", []):
        if url and url not in existing_urls and url.startswith("http"):
            knowledge_base.photos.append(PhotoAsset(url=url, source=source))
            existing_urls.add(url)

    if source not in knowledge_base.ingestion_sources:
        knowledge_base.ingestion_sources.append(source)

    return knowledge_base


# ── Standalone text-only extraction (PREVIEW-1) ─────────────────────────
# Public wrapper for the /preview endpoint. Calls _claude_extract but
# strips photo_urls from the result — preview is text confirmation only.

def extract_text_from_page(url: str) -> Optional[dict]:
    """
    Fetch a URL, extract structured TEXT fields via Claude. No photos.
    Used by the /preview endpoint for PMC/unknown URLs.
    """
    page_text = _fetch_page_text(url)
    if not page_text:
        return None

    extraction = _claude_extract(page_text, url)
    if not extraction:
        return None

    description = extraction.get("description") or ""
    city = extraction.get("city")
    state = extraction.get("state")
    zip_code = extraction.get("zip_code")

    return {
        "property_name": extraction.get("property_name"),
        "address_hint": ", ".join(filter(None, [city, state, zip_code])) or None,
        "bedrooms": extraction.get("bedrooms"),
        "bathrooms": extraction.get("bathrooms"),
        "max_occupancy": extraction.get("max_occupancy"),
        "property_type": extraction.get("property_type"),
        "amenities": extraction.get("amenities", []),
        "description_lead": description[:200] if description else None,
        "rating": None,
        "avg_nightly_rate": extraction.get("avg_nightly_rate"),
        "source_platform": "pmc",
    }
