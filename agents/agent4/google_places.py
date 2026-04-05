"""
TS-10 — Google Places API Client
Primary local discovery data source.

Pulls local business and attraction data for a property's surrounding
area. One-time intake cost ~$0.40/property.
$200/month Google Maps free credit covers first ~500 properties.

API calls per property:
  - Nearby Search per category (filtered by radius and type)
  - Place Details for top results (to get hours, phone, website, photos)

Migrated to Places API (New):
  - Nearby Search: POST https://places.googleapis.com/v1/places:searchNearby
  - Place Details: GET  https://places.googleapis.com/v1/places/{id}
  - Auth via X-Goog-Api-Key header + X-Goog-FieldMask header
"""

import os
import logging
import math
from typing import Optional

import httpx

from agents.agent4.models import (
    DataSource,
    Place,
    PlaceCategory,
    PlaceHours,
)

logger = logging.getLogger(__name__)

GOOGLE_PLACES_API_KEY = os.environ.get("GOOGLE_PLACES_API_KEY", "")
GOOGLE_PLACES_BASE    = "https://places.googleapis.com/v1"

# Search radius bands (meters)
RADIUS_URBAN  = 2000    # ~1.25 miles
RADIUS_SUBURB = 5000    # ~3 miles
RADIUS_RURAL  = 15000   # ~9 miles (default)

# Max results per category to keep costs controlled
MAX_RESULTS_PER_CATEGORY = 8

# Places API (New) type → our PlaceCategory
GOOGLE_TYPE_TO_CATEGORY: dict[str, PlaceCategory] = {
    "restaurant":          PlaceCategory.EAT_AND_DRINK,
    "food":                PlaceCategory.EAT_AND_DRINK,
    "cafe":                PlaceCategory.COFFEE_CAFES,
    "bakery":              PlaceCategory.COFFEE_CAFES,
    "bar":                 PlaceCategory.NIGHTLIFE,
    "night_club":          PlaceCategory.NIGHTLIFE,
    "tourist_attraction":  PlaceCategory.ATTRACTIONS,
    "museum":              PlaceCategory.ARTS_CULTURE,
    "art_gallery":         PlaceCategory.ARTS_CULTURE,
    "movie_theater":       PlaceCategory.ARTS_CULTURE,
    "amusement_park":      PlaceCategory.FAMILY_KIDS,
    "zoo":                 PlaceCategory.FAMILY_KIDS,
    "aquarium":            PlaceCategory.FAMILY_KIDS,
    "park":                PlaceCategory.ADVENTURE_OUTDOORS,
    "campground":          PlaceCategory.ADVENTURE_OUTDOORS,
    "natural_feature":     PlaceCategory.ADVENTURE_OUTDOORS,
    "spa":                 PlaceCategory.WELLNESS,
    "gym":                 PlaceCategory.WELLNESS,
    "shopping_mall":       PlaceCategory.SHOPPING,
    "store":               PlaceCategory.SHOPPING,
}

# Categories to search and their Google Places API types
CATEGORY_SEARCH_TYPES: dict[PlaceCategory, list[str]] = {
    PlaceCategory.EAT_AND_DRINK:      ["restaurant"],
    PlaceCategory.COFFEE_CAFES:       ["cafe", "bakery"],
    PlaceCategory.NIGHTLIFE:          ["bar", "night_club"],
    PlaceCategory.ATTRACTIONS:        ["tourist_attraction"],
    PlaceCategory.ARTS_CULTURE:       ["museum", "art_gallery", "movie_theater"],
    PlaceCategory.FAMILY_KIDS:        ["amusement_park", "zoo", "aquarium"],
    PlaceCategory.ADVENTURE_OUTDOORS: ["park", "campground"],
    PlaceCategory.WELLNESS:           ["spa", "gym"],
    PlaceCategory.SHOPPING:           ["shopping_mall"],
    PlaceCategory.TOURS_EXPERIENCES:  ["tourist_attraction"],
}

# Places API (New) price level enum → display string
_PRICE_LEVEL_MAP = {
    "PRICE_LEVEL_FREE":           "$",
    "PRICE_LEVEL_INEXPENSIVE":    "$",
    "PRICE_LEVEL_MODERATE":       "$$",
    "PRICE_LEVEL_EXPENSIVE":      "$$$",
    "PRICE_LEVEL_VERY_EXPENSIVE": "$$$$",
}

# Field masks for each request type
_NEARBY_FIELD_MASK = ",".join([
    "places.id",
    "places.displayName",
    "places.formattedAddress",
    "places.location",
    "places.rating",
    "places.userRatingCount",
    "places.priceLevel",
    "places.photos",
])

_DETAILS_FIELD_MASK = ",".join([
    "id",
    "displayName",
    "formattedAddress",
    "location",
    "rating",
    "userRatingCount",
    "priceLevel",
    "nationalPhoneNumber",
    "websiteUri",
    "currentOpeningHours",
    "photos",
])


def fetch_local_places(
    latitude: float,
    longitude: float,
    radius_meters: int = RADIUS_RURAL,
) -> list[Place]:
    """
    Main entry point for Google Places data collection.
    Searches all categories in sequence and returns Place objects.
    """
    if not GOOGLE_PLACES_API_KEY:
        logger.warning("[TS-10] Google Places API key not configured — skipping")
        return []

    all_places: list[Place] = []
    seen_place_ids: set[str] = set()

    for category, place_types in CATEGORY_SEARCH_TYPES.items():
        for place_type in place_types:
            results = _nearby_search(latitude, longitude, radius_meters, place_type)
            for result in results[:MAX_RESULTS_PER_CATEGORY]:
                place_id = result.get("id", "")
                if place_id in seen_place_ids:
                    continue
                seen_place_ids.add(place_id)

                details = _place_details(place_id) if place_id else {}
                place = _build_place(result, details, category, latitude, longitude)
                if place:
                    all_places.append(place)

    logger.info(f"[TS-10] Google Places: {len(all_places)} places fetched")
    return all_places


def _nearby_search(
    lat: float,
    lng: float,
    radius: int,
    place_type: str,
) -> list[dict]:
    """Call Places API (New) Nearby Search endpoint."""
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.post(
                f"{GOOGLE_PLACES_BASE}/places:searchNearby",
                headers={
                    "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
                    "X-Goog-FieldMask": _NEARBY_FIELD_MASK,
                },
                json={
                    "includedTypes": [place_type],
                    "maxResultCount": MAX_RESULTS_PER_CATEGORY,
                    "locationRestriction": {
                        "circle": {
                            "center": {
                                "latitude": lat,
                                "longitude": lng,
                            },
                            "radius": float(radius),
                        }
                    },
                },
            )
            resp.raise_for_status()
            return resp.json().get("places", [])
    except Exception as exc:
        logger.error(f"[TS-10] Nearby search failed for type={place_type}: {exc}")
        return []


def _place_details(place_id: str) -> dict:
    """Call Places API (New) Place Details endpoint."""
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(
                f"{GOOGLE_PLACES_BASE}/places/{place_id}",
                headers={
                    "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
                    "X-Goog-FieldMask": _DETAILS_FIELD_MASK,
                },
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.error(f"[TS-10] Place details failed for {place_id}: {exc}")
        return {}


def _build_place(
    result: dict,
    details: dict,
    category: PlaceCategory,
    property_lat: float,
    property_lng: float,
) -> Optional[Place]:
    """Merge Nearby Search result with Place Details into a Place object."""
    name = (
        (result.get("displayName") or {}).get("text")
        or (details.get("displayName") or {}).get("text")
        or ""
    )
    if not name:
        return None

    place_id = result.get("id") or details.get("id", "")

    # Location — new API uses "location" with "latitude"/"longitude" keys
    loc = result.get("location") or details.get("location") or {}
    lat = loc.get("latitude")
    lng = loc.get("longitude")
    distance = _haversine_miles(property_lat, property_lng, lat, lng) if lat and lng else None

    rating       = result.get("rating") or details.get("rating")
    review_count = result.get("userRatingCount") or details.get("userRatingCount")

    # Price level — new API uses string enum
    price_raw = result.get("priceLevel") or details.get("priceLevel")
    price_str = _PRICE_LEVEL_MAP.get(price_raw) if price_raw else None

    # Photos — new API uses resource name strings instead of photo_reference
    photos = result.get("photos") or details.get("photos") or []
    photo_url = _build_photo_url(photos[0]["name"]) if photos else None

    # Hours
    hours = None
    oh = details.get("currentOpeningHours") or {}
    if oh:
        hours = PlaceHours(
            open_now=oh.get("openNow"),
            weekday_text=oh.get("weekdayDescriptions", []),
        )

    return Place(
        name=name,
        category=category,
        primary_source=DataSource.GOOGLE_PLACES,
        address=details.get("formattedAddress") or result.get("formattedAddress"),
        distance_miles=round(distance, 1) if distance is not None else None,
        latitude=lat,
        longitude=lng,
        google_rating=float(rating) if rating is not None else None,
        google_review_count=int(review_count) if review_count is not None else None,
        price_level=price_str,
        phone=details.get("nationalPhoneNumber"),
        website=details.get("websiteUri"),
        photo_url=photo_url,
        google_place_id=place_id,
        hours=hours,
    )


def _build_photo_url(photo_name: str, max_width: int = 800) -> str:
    """Build a Places API (New) photo media URL from a photo resource name."""
    return (
        f"{GOOGLE_PLACES_BASE}/{photo_name}/media"
        f"?maxWidthPx={max_width}"
        f"&key={GOOGLE_PLACES_API_KEY}"
    )


def _haversine_miles(
    lat1: float,
    lng1: float,
    lat2: Optional[float],
    lng2: Optional[float],
) -> Optional[float]:
    """Calculate straight-line distance in miles between two coordinates."""
    if lat2 is None or lng2 is None:
        return None
    R = 3958.8   # Earth radius in miles
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))
