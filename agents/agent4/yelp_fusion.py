"""
TS-11 — Yelp Fusion API Client
Supplementary local discovery source.

Supplements Google Places with deeper food/nightlife coverage,
more consistent price tier data, and richer review sentiment.

Merge policy (when a business appears in both sources):
  - Name: Google (more official)
  - Address/hours: Google
  - Price tier: Yelp (more consistently populated)
  - Review count: Yelp (higher for food/nightlife)
  - Rating: Weighted composite (Google 60%, Yelp 40%)
  - Phone/website: Google preferred, Yelp as fallback

5-10 Yelp API calls per property intake.
Free developer tier: 500 calls/day — sufficient for launch volume.
"""

import os
import logging
from typing import Optional

import httpx

from agents.agent4.models import (
    DataSource,
    Place,
    PlaceCategory,
)

logger = logging.getLogger(__name__)

YELP_API_KEY  = os.environ.get("YELP_API_KEY", "")
YELP_API_BASE = "https://api.yelp.com/v3"

# Max results per Yelp search
MAX_YELP_RESULTS = 10

# Yelp category aliases → our PlaceCategory
YELP_CATEGORY_TO_PLACE: dict[str, PlaceCategory] = {
    "restaurants":    PlaceCategory.EAT_AND_DRINK,
    "food":           PlaceCategory.EAT_AND_DRINK,
    "coffee":         PlaceCategory.COFFEE_CAFES,
    "cafes":          PlaceCategory.COFFEE_CAFES,
    "bars":           PlaceCategory.NIGHTLIFE,
    "nightlife":      PlaceCategory.NIGHTLIFE,
    "arts":           PlaceCategory.ARTS_CULTURE,
    "shopping":       PlaceCategory.SHOPPING,
    "active":         PlaceCategory.ADVENTURE_OUTDOORS,
    "fitness":        PlaceCategory.WELLNESS,
    "spas":           PlaceCategory.WELLNESS,
    "tours":          PlaceCategory.TOURS_EXPERIENCES,
    "hotels":         None,   # Not relevant for local guide
}

# Yelp searches to run per property — focused on Yelp's strengths
YELP_SEARCHES: list[tuple[str, PlaceCategory]] = [
    ("restaurants",  PlaceCategory.EAT_AND_DRINK),
    ("bars",         PlaceCategory.NIGHTLIFE),
    ("coffee",       PlaceCategory.COFFEE_CAFES),
    ("spas",         PlaceCategory.WELLNESS),
    ("tours",        PlaceCategory.TOURS_EXPERIENCES),
]

# Yelp price tier string → our format
YELP_PRICE_MAP = {"$": "$", "$$": "$$", "$$$": "$$$", "$$$$": "$$$$"}


def fetch_yelp_places(
    latitude: float,
    longitude: float,
    radius_meters: int = 15000,
) -> list[Place]:
    """
    Main entry point for Yelp Fusion data collection.
    Runs targeted searches for Yelp's strongest categories.
    Returns Place objects for merging with Google Places data.
    """
    if not YELP_API_KEY:
        logger.warning("[TS-11] Yelp API key not configured — skipping")
        return []

    all_places: list[Place] = []
    seen_yelp_ids: set[str] = set()

    for yelp_category, place_category in YELP_SEARCHES:
        results = _yelp_search(latitude, longitude, radius_meters, yelp_category)
        for biz in results[:MAX_YELP_RESULTS]:
            yelp_id = biz.get("id", "")
            if yelp_id in seen_yelp_ids:
                continue
            seen_yelp_ids.add(yelp_id)

            place = _build_yelp_place(biz, place_category, latitude, longitude)
            if place:
                all_places.append(place)

    logger.info(f"[TS-11] Yelp Fusion: {len(all_places)} places fetched")
    return all_places


def _yelp_search(
    lat: float,
    lng: float,
    radius: int,
    category: str,
) -> list[dict]:
    """Call Yelp Fusion Business Search endpoint."""
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(
                f"{YELP_API_BASE}/businesses/search",
                headers={"Authorization": f"Bearer {YELP_API_KEY}"},
                params={
                    "latitude": lat,
                    "longitude": lng,
                    "radius": min(radius, 40000),   # Yelp max radius is 40km
                    "categories": category,
                    "sort_by": "rating",
                    "limit": MAX_YELP_RESULTS,
                },
            )
            resp.raise_for_status()
            return resp.json().get("businesses", [])
    except Exception as exc:
        logger.error(f"[TS-11] Yelp search failed for category={category}: {exc}")
        return []


def _build_yelp_place(
    biz: dict,
    category: PlaceCategory,
    property_lat: float,
    property_lng: float,
) -> Optional[Place]:
    """Build a Place object from a Yelp business result."""
    name = biz.get("name", "")
    if not name:
        return None

    # Location
    loc = biz.get("location", {})
    coords = biz.get("coordinates", {})
    lat = coords.get("latitude")
    lng = coords.get("longitude")

    address_parts = [
        loc.get("address1", ""),
        loc.get("city", ""),
        loc.get("state", ""),
    ]
    address = ", ".join(p for p in address_parts if p)

    # Distance
    distance_meters = biz.get("distance")
    distance_miles = round(distance_meters / 1609.34, 1) if distance_meters else None

    # Price
    price_str = YELP_PRICE_MAP.get(biz.get("price", ""))

    # Photo
    photo_url = biz.get("image_url")

    return Place(
        name=name,
        category=category,
        primary_source=DataSource.YELP_FUSION,
        address=address or None,
        distance_miles=distance_miles,
        latitude=lat,
        longitude=lng,
        yelp_rating=float(biz["rating"]) if biz.get("rating") else None,
        yelp_review_count=int(biz["review_count"]) if biz.get("review_count") else None,
        price_level=price_str,
        phone=biz.get("phone"),
        website=biz.get("url"),
        photo_url=photo_url,
        yelp_id=biz.get("id"),
    )
