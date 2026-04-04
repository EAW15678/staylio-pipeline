"""
TS-02 / TS-03 / TS-04b — OTA Scraper
Tool: Apify (Airbnb Actor, VRBO Actor)

Handles three scraping jobs via the same Apify infrastructure:
  - TS-02: IO Airbnb listing scrape (property data + photos + reviews)
  - TS-03: IO VRBO listing scrape (property data + photos)
  - TS-04b: PMC Airbnb review scrape (supplementary OTA review data)

All three use Apify's pre-built maintained Actors with residential proxy
rotation, handling anti-bot protection without any engineering investment.
"""

import os
import re
import time
import logging
from typing import Optional
from urllib.parse import urlparse

import httpx

from models.property import (
    DataSource,
    GuestReview,
    PhotoAsset,
    PropertyField,
    PropertyKnowledgeBase,
)

logger = logging.getLogger(__name__)

APIFY_API_BASE   = "https://api.apify.com/v2"
APIFY_API_TOKEN  = os.environ["APIFY_API_TOKEN"]

# Actor IDs for each platform
AIRBNB_ACTOR_ID  = os.environ.get("APIFY_AIRBNB_ACTOR_ID", "pIyP4eyT6kBUZ2fHe")
VRBO_ACTOR_ID    = os.environ.get("APIFY_VRBO_ACTOR_ID", "kRRC9n6Rv5lEcE3b3")

# Polling config for sync-style run
POLL_INTERVAL_SEC = 5
MAX_POLL_ATTEMPTS = 60   # 5 min timeout for a single Actor run


def detect_ota_platform(url: str) -> Optional[str]:
    """Detect which OTA platform a URL belongs to."""
    if not url:
        return None
    hostname = urlparse(url.lower()).hostname or ""
    if "airbnb" in hostname:
        return "airbnb"
    if "vrbo" in hostname or "homeaway" in hostname:
        return "vrbo"
    if "booking.com" in hostname:
        return "booking_com"
    return None


def scrape_ota_listing(
    ota_url: str,
    knowledge_base: PropertyKnowledgeBase,
    scrape_reviews: bool = True,
) -> PropertyKnowledgeBase:
    """
    Main entry point for OTA scraping (TS-02, TS-03).
    Detects platform, runs the correct Actor, and merges data.

    Args:
        ota_url: The Airbnb or VRBO property listing URL
        knowledge_base: Existing KB to merge into
        scrape_reviews: If True, also scrape guest reviews (slower, more tokens)
    """
    platform = detect_ota_platform(ota_url)

    if platform == "airbnb":
        return _scrape_airbnb(ota_url, knowledge_base, scrape_reviews)
    elif platform == "vrbo":
        return _scrape_vrbo(ota_url, knowledge_base, scrape_reviews)
    elif platform == "booking_com":
        logger.info(f"[TS-04] Booking.com URL detected — not yet implemented. Skipping: {ota_url}")
        knowledge_base.ingestion_errors.append(
            f"Booking.com scraping not yet implemented (TS-04 deferred). URL: {ota_url}"
        )
        return knowledge_base
    else:
        logger.warning(f"[TS-04c] Unrecognised OTA platform for URL: {ota_url}")
        knowledge_base.ingestion_errors.append(
            f"Unrecognised OTA platform for URL: {ota_url} — will attempt Claude fallback parser"
        )
        return knowledge_base


def scrape_airbnb_reviews_only(
    airbnb_url: str,
    knowledge_base: PropertyKnowledgeBase,
) -> PropertyKnowledgeBase:
    """
    TS-04b: Supplementary review scrape for PMC clients who provide an Airbnb URL.
    Pulls guest reviews only — not the full property data.
    Same Actor as TS-02 but invoked separately at intake.
    """
    logger.info(f"[TS-04b] Scraping Airbnb reviews for PMC property: {airbnb_url}")
    return _scrape_airbnb(airbnb_url, knowledge_base, scrape_reviews=True, data_only=False)


# ── Airbnb ────────────────────────────────────────────────────────────────

def _scrape_airbnb(
    url: str,
    knowledge_base: PropertyKnowledgeBase,
    scrape_reviews: bool,
    data_only: bool = True,
) -> PropertyKnowledgeBase:
    """Run the Apify Airbnb Actor and merge results."""
    logger.info(f"[TS-02] Apify Airbnb Actor: {url}")

    actor_input = {
        "startUrls": [{"url": url}],
        "maxListings": 1,
        "includeReviews": scrape_reviews,
        "maxReviews": 50,               # Cap at 50 — enough for social proof
        "currency": "USD",
        "addMoreHostInfo": True,
    }

    raw = _run_apify_actor(AIRBNB_ACTOR_ID, actor_input)
    if not raw:
        knowledge_base.ingestion_errors.append(
            f"Apify Airbnb Actor returned no results for {url}"
        )
        return knowledge_base

    listing = raw[0] if isinstance(raw, list) else raw
    source = DataSource.AIRBNB

    # ── Property data ─────────────────────────────────────────────────────
    if data_only or knowledge_base.name is None:
        knowledge_base.name = knowledge_base.merge_field(
            knowledge_base.name,
            _field(listing.get("name"), source),
        )
    knowledge_base.description = knowledge_base.merge_field(
        knowledge_base.description,
        _field(listing.get("description"), source),
    )
    knowledge_base.bedrooms = knowledge_base.merge_field(
        knowledge_base.bedrooms,
        _field(listing.get("bedrooms"), source),
    )
    knowledge_base.bathrooms = knowledge_base.merge_field(
        knowledge_base.bathrooms,
        _field(listing.get("bathrooms"), source),
    )
    knowledge_base.max_occupancy = knowledge_base.merge_field(
        knowledge_base.max_occupancy,
        _field(listing.get("personCapacity"), source),
    )
    knowledge_base.property_type = knowledge_base.merge_field(
        knowledge_base.property_type,
        _field(listing.get("roomType"), source),
    )
    knowledge_base.airbnb_rating = knowledge_base.merge_field(
        knowledge_base.airbnb_rating,
        _field(listing.get("stars"), source),
    )
    knowledge_base.airbnb_review_count = knowledge_base.merge_field(
        knowledge_base.airbnb_review_count,
        _field(listing.get("numberOfGuests"), source),
    )
    knowledge_base.avg_nightly_rate = knowledge_base.merge_field(
        knowledge_base.avg_nightly_rate,
        _field(_extract_nightly_rate(listing), source),
    )

    # ── Location ─────────────────────────────────────────────────────────
    location = listing.get("location") or listing.get("address") or {}
    if isinstance(location, dict):
        knowledge_base.city = knowledge_base.merge_field(
            knowledge_base.city,
            _field(location.get("city"), source),
        )
        knowledge_base.state = knowledge_base.merge_field(
            knowledge_base.state,
            _field(location.get("state"), source),
        )
        knowledge_base.zip_code = knowledge_base.merge_field(
            knowledge_base.zip_code,
            _field(location.get("zipCode") or location.get("postalCode"), source),
        )

    lat = listing.get("lat") or (listing.get("coordinates") or {}).get("lat")
    lng = listing.get("lng") or (listing.get("coordinates") or {}).get("lng")
    if lat:
        knowledge_base.latitude = knowledge_base.merge_field(
            knowledge_base.latitude,
            _field(lat, source),
        )
    if lng:
        knowledge_base.longitude = knowledge_base.merge_field(
            knowledge_base.longitude,
            _field(lng, source),
        )

    # ── Amenities ─────────────────────────────────────────────────────────
    existing_amenities = {
        a.value.lower() for a in knowledge_base.amenities if a.value
    }
    for amenity in listing.get("amenities", []):
        amenity_name = amenity if isinstance(amenity, str) else amenity.get("name", "")
        if amenity_name and amenity_name.lower() not in existing_amenities:
            knowledge_base.amenities.append(
                PropertyField(value=amenity_name, source=source)
            )
            existing_amenities.add(amenity_name.lower())

    # ── Photos ────────────────────────────────────────────────────────────
    existing_urls = {p.url for p in knowledge_base.photos}
    photos_raw = listing.get("photos") or listing.get("images") or []
    for photo in photos_raw:
        url_val = photo if isinstance(photo, str) else (
            photo.get("url") or photo.get("picture") or photo.get("baseUrl")
        )
        if url_val and url_val not in existing_urls:
            # Prefer the largest available Airbnb image variant
            url_val = _upgrade_airbnb_photo_url(url_val)
            knowledge_base.photos.append(
                PhotoAsset(url=url_val, source=source)
            )
            existing_urls.add(url_val)

    # ── Reviews ───────────────────────────────────────────────────────────
    if scrape_reviews:
        for review in listing.get("reviews", []):
            text = review.get("comments") or review.get("text") or ""
            if not text.strip():
                continue
            knowledge_base.guest_reviews.append(
                GuestReview(
                    text=text.strip(),
                    source=source,
                    reviewer_name=review.get("reviewer", {}).get("firstName")
                        if isinstance(review.get("reviewer"), dict)
                        else review.get("reviewerName"),
                    stay_date=review.get("createdAt") or review.get("date"),
                    star_rating=review.get("rating"),
                    host_response=review.get("response"),
                    is_guest_book=False,
                )
            )

    if DataSource.AIRBNB not in knowledge_base.ingestion_sources:
        knowledge_base.ingestion_sources.append(DataSource.AIRBNB)

    logger.info(
        f"[TS-02] Airbnb scrape complete — "
        f"Photos: {len(knowledge_base.photos)}, "
        f"Reviews: {len(knowledge_base.guest_reviews)}, "
        f"Amenities: {len(knowledge_base.amenities)}"
    )
    return knowledge_base


# ── VRBO ──────────────────────────────────────────────────────────────────

def _scrape_vrbo(
    url: str,
    knowledge_base: PropertyKnowledgeBase,
    scrape_reviews: bool,
) -> PropertyKnowledgeBase:
    """Run the Apify VRBO Actor and merge results."""
    logger.info(f"[TS-03] Apify VRBO Actor: {url}")

    actor_input = {
        "startUrls": [{"url": url}],
        "maxItems": 1,
    }

    raw = _run_apify_actor(VRBO_ACTOR_ID, actor_input)
    if not raw:
        knowledge_base.ingestion_errors.append(
            f"Apify VRBO Actor returned no results for {url}"
        )
        return knowledge_base

    listing = raw[0] if isinstance(raw, list) else raw
    source = DataSource.VRBO

    # ── Property data ─────────────────────────────────────────────────────
    knowledge_base.name = knowledge_base.merge_field(
        knowledge_base.name,
        _field(listing.get("name") or listing.get("headline"), source),
    )
    knowledge_base.description = knowledge_base.merge_field(
        knowledge_base.description,
        _field(listing.get("description"), source),
    )
    knowledge_base.bedrooms = knowledge_base.merge_field(
        knowledge_base.bedrooms,
        _field(
            listing.get("bedrooms") or listing.get("bedroom_count"),
            source,
        ),
    )
    knowledge_base.bathrooms = knowledge_base.merge_field(
        knowledge_base.bathrooms,
        _field(
            listing.get("bathrooms") or listing.get("bathroom_count"),
            source,
        ),
    )
    knowledge_base.max_occupancy = knowledge_base.merge_field(
        knowledge_base.max_occupancy,
        _field(
            listing.get("sleeps") or listing.get("max_occupancy"),
            source,
        ),
    )

    # ── Location ─────────────────────────────────────────────────────────
    knowledge_base.city = knowledge_base.merge_field(
        knowledge_base.city,
        _field(listing.get("city"), source),
    )
    knowledge_base.state = knowledge_base.merge_field(
        knowledge_base.state,
        _field(listing.get("state") or listing.get("stateProvince"), source),
    )
    knowledge_base.zip_code = knowledge_base.merge_field(
        knowledge_base.zip_code,
        _field(listing.get("zip") or listing.get("postalCode"), source),
    )
    if listing.get("latitude"):
        knowledge_base.latitude = knowledge_base.merge_field(
            knowledge_base.latitude,
            _field(listing["latitude"], source),
        )
    if listing.get("longitude"):
        knowledge_base.longitude = knowledge_base.merge_field(
            knowledge_base.longitude,
            _field(listing["longitude"], source),
        )

    # ── Amenities ─────────────────────────────────────────────────────────
    existing_amenities = {
        a.value.lower() for a in knowledge_base.amenities if a.value
    }
    for amenity in listing.get("amenities", []):
        name = amenity if isinstance(amenity, str) else amenity.get("text", "")
        if name and name.lower() not in existing_amenities:
            knowledge_base.amenities.append(
                PropertyField(value=name, source=source)
            )
            existing_amenities.add(name.lower())

    # ── Photos ────────────────────────────────────────────────────────────
    existing_urls = {p.url for p in knowledge_base.photos}
    for photo in listing.get("photos") or listing.get("images") or []:
        url_val = photo if isinstance(photo, str) else (
            photo.get("url") or photo.get("uri")
        )
        if url_val and url_val not in existing_urls:
            knowledge_base.photos.append(
                PhotoAsset(url=url_val, source=source)
            )
            existing_urls.add(url_val)

    if DataSource.VRBO not in knowledge_base.ingestion_sources:
        knowledge_base.ingestion_sources.append(DataSource.VRBO)

    logger.info(
        f"[TS-03] VRBO scrape complete — "
        f"Photos: {len(knowledge_base.photos)}, "
        f"Amenities: {len(knowledge_base.amenities)}"
    )
    return knowledge_base


# ── Apify API helpers ─────────────────────────────────────────────────────

def _run_apify_actor(actor_id: str, actor_input: dict) -> Optional[list]:
    """
    Runs an Apify Actor synchronously and returns the dataset items.
    Polls for completion up to MAX_POLL_ATTEMPTS × POLL_INTERVAL_SEC.
    """
    headers = {"Authorization": f"Bearer {APIFY_API_TOKEN}"}

    # Start the Actor run
    with httpx.Client(timeout=30) as client:
        run_resp = client.post(
            f"{APIFY_API_BASE}/acts/{actor_id}/runs",
            headers=headers,
            json={"input": actor_input},
        )
        run_resp.raise_for_status()
        run_data = run_resp.json()

    run_id = run_data.get("data", {}).get("id")
    if not run_id:
        logger.error(f"Apify Actor {actor_id} did not return a run ID: {run_data}")
        return None

    logger.debug(f"Apify run started: {run_id}")

    # Poll until finished
    for attempt in range(MAX_POLL_ATTEMPTS):
        time.sleep(POLL_INTERVAL_SEC)
        with httpx.Client(timeout=30) as client:
            status_resp = client.get(
                f"{APIFY_API_BASE}/actor-runs/{run_id}",
                headers=headers,
            )
            status_resp.raise_for_status()
            status = status_resp.json().get("data", {}).get("status")

        if status == "SUCCEEDED":
            break
        elif status in ("FAILED", "ABORTED", "TIMED-OUT"):
            logger.error(f"Apify run {run_id} ended with status: {status}")
            return None
        else:
            logger.debug(f"Apify run {run_id} status: {status} (attempt {attempt+1})")
    else:
        logger.error(f"Apify run {run_id} timed out after {MAX_POLL_ATTEMPTS} polls")
        return None

    # Fetch dataset items
    with httpx.Client(timeout=30) as client:
        dataset_resp = client.get(
            f"{APIFY_API_BASE}/actor-runs/{run_id}/dataset/items",
            headers=headers,
            params={"format": "json", "clean": True},
        )
        dataset_resp.raise_for_status()
        return dataset_resp.json()


# ── Utilities ─────────────────────────────────────────────────────────────

def _field(
    value,
    source: DataSource,
    confidence: float = 0.80,
) -> Optional[PropertyField]:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return PropertyField(value=value, source=source, confidence=confidence)


def _extract_nightly_rate(listing: dict) -> Optional[float]:
    """Try various Airbnb API field names for nightly rate."""
    price_fields = [
        "price", "basePrice", "priceRate", "nightly_price",
        "pricing_weekly_factor",
    ]
    for field in price_fields:
        val = listing.get(field)
        if val is not None:
            try:
                # Strip currency symbols if string
                cleaned = re.sub(r"[^\d.]", "", str(val))
                return float(cleaned) if cleaned else None
            except (ValueError, TypeError):
                continue
    return None


def _upgrade_airbnb_photo_url(url: str) -> str:
    """
    Airbnb photo URLs often have size suffixes like /im/picture?im_w=480
    Replace with a higher-resolution variant where possible.
    """
    if "im_w=" in url:
        url = re.sub(r"im_w=\d+", "im_w=1200", url)
    return url
