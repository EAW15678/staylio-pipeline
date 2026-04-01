"""
TS-01 — PMC Website Scraper
Tool: Firecrawl

Scrapes PMC property pages from their direct booking websites.
Returns LLM-ready Markdown for Claude processing.
Uses Firecrawl's AI Extract endpoint for structured data extraction —
no CSS selectors, handles any PMC site structure automatically.
"""

import os
import re
import logging
from typing import Optional

import httpx

from models.property import (
    DataSource,
    GuestReview,
    PhotoAsset,
    PropertyField,
    PropertyKnowledgeBase,
)

logger = logging.getLogger(__name__)

FIRECRAWL_API_BASE = "https://api.firecrawl.dev/v1"
FIRECRAWL_API_KEY  = os.environ["FIRECRAWL_API_KEY"]
REQUEST_TIMEOUT    = 60  # seconds — Firecrawl scrapes can be slow on first hit


# ── Extraction schema for Firecrawl AI Extract endpoint ────────────────────
# Natural language instructions tell Firecrawl what to extract.
# Returns structured JSON without fragile CSS selectors.

PMC_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "property_name": {
            "type": "string",
            "description": "The name of the vacation rental property"
        },
        "headline": {
            "type": "string",
            "description": "The main marketing headline or tagline for the property"
        },
        "description": {
            "type": "string",
            "description": "Full property description — all marketing copy about the property"
        },
        "bedrooms": {
            "type": "integer",
            "description": "Number of bedrooms"
        },
        "bathrooms": {
            "type": "number",
            "description": "Number of bathrooms (may be 2.5, 3.5, etc.)"
        },
        "max_occupancy": {
            "type": "integer",
            "description": "Maximum number of guests the property can accommodate"
        },
        "property_type": {
            "type": "string",
            "description": "Type of property: cabin, condo, villa, beach house, cottage, etc."
        },
        "amenities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Complete list of all amenities mentioned: pool, hot tub, WiFi, etc."
        },
        "unique_features": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Standout unique features that differentiate this property"
        },
        "address": {
            "type": "string",
            "description": "Property street address if shown"
        },
        "city": {
            "type": "string",
            "description": "City or town where the property is located"
        },
        "state": {
            "type": "string",
            "description": "State abbreviation (NC, FL, CO, etc.)"
        },
        "zip_code": {
            "type": "string",
            "description": "Zip code if shown"
        },
        "avg_nightly_rate": {
            "type": "number",
            "description": "Average or starting nightly rate if displayed on the page"
        },
        "photo_urls": {
            "type": "array",
            "items": {"type": "string"},
            "description": "All image URLs visible on the property page"
        },
        "neighborhood_description": {
            "type": "string",
            "description": "Any description of the surrounding area, neighborhood, or location highlights"
        },
        "booking_url": {
            "type": "string",
            "description": "The URL to book this property directly if a Book Now or Reserve button exists"
        },
    }
}


def scrape_pmc_website(
    pmc_website_url: str,
    knowledge_base: PropertyKnowledgeBase,
) -> PropertyKnowledgeBase:
    """
    Scrapes a PMC property listing page from their direct booking website
    using Firecrawl's AI Extract endpoint.

    Merges extracted data into the knowledge base using the merge policy:
    intake portal data wins, scraped data fills gaps.

    Args:
        pmc_website_url: Full URL of the PMC property page
        knowledge_base: Existing knowledge base to merge into (may have intake data)

    Returns:
        Updated knowledge base with PMC website data merged in
    """
    logger.info(f"[TS-01] Firecrawl scraping PMC website: {pmc_website_url}")

    try:
        response = _firecrawl_extract(pmc_website_url)
    except Exception as exc:
        error_msg = f"Firecrawl scrape failed for {pmc_website_url}: {exc}"
        logger.error(error_msg)
        knowledge_base.ingestion_errors.append(error_msg)
        return knowledge_base

    if not response:
        knowledge_base.ingestion_errors.append(
            f"Firecrawl returned empty response for {pmc_website_url}"
        )
        return knowledge_base

    # ── Map extracted fields into knowledge base ──────────────────────────
    source = DataSource.PMC_WEBSITE

    knowledge_base.name = knowledge_base.merge_field(
        knowledge_base.name,
        _field(response.get("property_name"), source)
    )
    knowledge_base.headline = knowledge_base.merge_field(
        knowledge_base.headline,
        _field(response.get("headline"), source)
    )
    knowledge_base.description = knowledge_base.merge_field(
        knowledge_base.description,
        _field(response.get("description"), source)
    )
    knowledge_base.bedrooms = knowledge_base.merge_field(
        knowledge_base.bedrooms,
        _field(response.get("bedrooms"), source)
    )
    knowledge_base.bathrooms = knowledge_base.merge_field(
        knowledge_base.bathrooms,
        _field(response.get("bathrooms"), source)
    )
    knowledge_base.max_occupancy = knowledge_base.merge_field(
        knowledge_base.max_occupancy,
        _field(response.get("max_occupancy"), source)
    )
    knowledge_base.property_type = knowledge_base.merge_field(
        knowledge_base.property_type,
        _field(response.get("property_type"), source)
    )
    knowledge_base.city = knowledge_base.merge_field(
        knowledge_base.city,
        _field(response.get("city"), source)
    )
    knowledge_base.state = knowledge_base.merge_field(
        knowledge_base.state,
        _field(response.get("state"), source)
    )
    knowledge_base.zip_code = knowledge_base.merge_field(
        knowledge_base.zip_code,
        _field(response.get("zip_code"), source)
    )
    knowledge_base.avg_nightly_rate = knowledge_base.merge_field(
        knowledge_base.avg_nightly_rate,
        _field(response.get("avg_nightly_rate"), source)
    )
    knowledge_base.neighborhood_description = knowledge_base.merge_field(
        knowledge_base.neighborhood_description,
        _field(response.get("neighborhood_description"), source)
    )

    # Booking URL — only fill if not already provided by intake
    if not knowledge_base.booking_url and response.get("booking_url"):
        knowledge_base.booking_url = response["booking_url"]

    # ── Amenities — append scraped amenities not already in list ─────────
    existing_amenities = {
        a.value.lower() for a in knowledge_base.amenities
        if a.value
    }
    for amenity in response.get("amenities", []):
        if amenity and amenity.lower() not in existing_amenities:
            knowledge_base.amenities.append(
                PropertyField(value=amenity, source=source)
            )
            existing_amenities.add(amenity.lower())

    # ── Unique features — append only if not duplicating amenities ────────
    existing_features = {
        f.value.lower() for f in knowledge_base.unique_features
        if f.value
    }
    for feature in response.get("unique_features", []):
        if feature and feature.lower() not in existing_features:
            knowledge_base.unique_features.append(
                PropertyField(value=feature, source=source)
            )
            existing_features.add(feature.lower())

    # ── Photos — add scraped photo URLs not already in list ──────────────
    existing_photo_urls = {p.url for p in knowledge_base.photos}
    for url in response.get("photo_urls", []):
        if url and url not in existing_photo_urls and _is_valid_photo_url(url):
            knowledge_base.photos.append(
                PhotoAsset(url=url, source=source)
            )
            existing_photo_urls.add(url)

    # ── Record source ─────────────────────────────────────────────────────
    if DataSource.PMC_WEBSITE not in knowledge_base.ingestion_sources:
        knowledge_base.ingestion_sources.append(DataSource.PMC_WEBSITE)

    logger.info(
        f"[TS-01] PMC website scrape complete. "
        f"Photos: {len(knowledge_base.photos)}, "
        f"Amenities: {len(knowledge_base.amenities)}"
    )
    return knowledge_base


def _firecrawl_extract(url: str) -> Optional[dict]:
    """
    Calls Firecrawl's /extract endpoint with the PMC property schema.
    Returns structured JSON dict or None on failure.
    """
    headers = {
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "urls": [url],
        "prompt": (
            "Extract all property information from this vacation rental listing page. "
            "Include the complete description, all amenities, all photo URLs, "
            "bedroom and bathroom counts, maximum occupancy, pricing if shown, "
            "and any location or neighborhood information."
        ),
        "schema": PMC_EXTRACT_SCHEMA,
        "enableWebSearch": False,
    }

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.post(
            f"{FIRECRAWL_API_BASE}/extract",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    # Firecrawl extract returns {"success": true, "data": {...}}
    if not data.get("success"):
        logger.warning(f"Firecrawl extract returned success=false for {url}: {data}")
        return None

    return data.get("data") or {}


def _field(
    value,
    source: DataSource,
    confidence: float = 0.85,
) -> Optional[PropertyField]:
    """Helper — wrap a value as a PropertyField, or return None if empty."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, list) and not value:
        return None
    return PropertyField(value=value, source=source, confidence=confidence)


def _is_valid_photo_url(url: str) -> bool:
    """Basic sanity check — reject obvious non-photo URLs."""
    if not url.startswith(("http://", "https://")):
        return False
    # Reject tiny icons/logos/tracking pixels by checking common patterns
    low = url.lower()
    excluded = (
        "favicon", "logo", "icon", "sprite", "pixel", "tracking",
        "analytics", "beacon", ".svg", ".gif", "1x1", "blank"
    )
    return not any(pattern in low for pattern in excluded)
