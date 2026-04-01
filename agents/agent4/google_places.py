"""
TS-10 — Google Places API Client
Primary local discovery data source.

Pulls local business and attraction data for a property's surrounding
area. One-time intake cost ~$0.40/property.
$200/month Google Maps free credit covers first ~500 properties.

API calls per property:
  - Nearby Search per category (filtered by radius and type)
  - Place Details for top results (to get hours, phone, website, photos)

Google Places is the primary source for:
  - Attractions, parks, beaches, outdoor activities
  - All business categories (broader coverage than Yelp)
  - Hours and location data
  - Photo URLs
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
GOOGLE_PLACES_BASE    = "https://maps.googleapis.com/maps/api/place"

# Search radius bands (meters)
# Tight for urban properties, wider for rural
RADIUS_URBAN  = 2000    # ~1.25 miles
RADIUS_SUBURB = 5000    # ~3 miles
RADIUS_RURAL  = 15000   # ~9 miles (default)

# Max results per category to keep costs controlled
MAX_RESULTS_PER_CATEGORY = 8

# Google Places API type → our PlaceCategory
GOOGLE_TYPE_TO_CATEGORY: dict[str, PlaceCategory] = {
    # Food & drink
    "restaurant":          PlaceCategory.EAT_AND_DRINK,
    "food":                PlaceCategory.EAT_AND_DRINK,
    "cafe":                PlaceCategory.COFFEE_CAFES,
    "bakery":              PlaceCategory.COFFEE_CAFES,
    "bar":                 PlaceCategory.NIGHTLIFE,
    "night_club":          PlaceCategory.NIGHTLIFE,
    # Attractions
    "tourist_attraction":  PlaceCategory.ATTRACTIONS,
    "museum":              PlaceCategory.ARTS_CULTURE,
    "art_gallery":         PlaceCategory.ARTS_CULTURE,
    "movie_theater":       PlaceCategory.ARTS_CULTURE,
    "amusement_park":      PlaceCategory.FAMILY_KIDS,
    "zoo":                 PlaceCategory.FAMILY_KIDS,
    "aquarium":            PlaceCategory.FAMILY_KIDS,
    # Outdoor
    "park":                PlaceCategory.ADVENTURE_OUTDOORS,
    "campground":          PlaceCategory.ADVENTURE_OUTDOORS,
    "natural_feature":     PlaceCategory.ADVENTURE_OUTDOORS,
    # Wellness
    "spa":                 PlaceCategory.WELLNESS,
    "gym":                 PlaceCategory.WELLNESS,
    # Shopping
    "shopping_mall":       PlaceCategory.SHOPPING,
    "store":               PlaceCategory.SHOPPING,
}

# Categories to search and their Google Places API types
CATEGORY_SEARCH_TYPES: dict[PlaceCategory, list[str]] = {
    PlaceCategory.EAT_AND_DRINK:      ["restaurant"],
    PlaceCategory.COFFEE_CAFES:       ["cafe", "bakery"],
    PlaceCategory.NIGHTLIFE:          ["bar", "night_club"],
    PlaceCategory.ATTRACTIONS:        ["tourist_attraction", "natural_feature"],
    PlaceCategory.ARTS_CULTURE:       ["museum", "art_gallery", "movie_theater"],
    PlaceCategory.FAMILY_KIDS:        ["amusement_park", "zoo", "aquarium"],
    PlaceCategory.ADVENTURE_OUTDOORS: ["park", "campground"],
    PlaceCategory.WELLNESS:           ["spa", "gym"],
    PlaceCategory.SHOPPING:           ["shopping_mall"],
    PlaceCategory.TOURS_EXPERIENCES:  ["tourist_attraction"],
}


def fetch_local_places(
    latitude: float,
    longitude: float,
    radius_meters: int = RADIUS_RURAL,
) -> list[Place]:
    """
    Main entry point for Google Places data collection.
    Searches all categories in sequence (rate-limited appropriately).
    Returns a list of Place objects ready for merging with Yelp data.
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
                place_id = result.get("place_id", "")
                if place_id in seen_place_ids:
                    continue
                seen_place_ids.add(place_id)

                # Get detailed info for top results
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
    """Call Google Places Nearby Search API."""
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(
                f"{GOOGLE_PLACES_BASE}/nearbysearch/json",
                params={
                    "key": GOOGLE_PLACES_API_KEY,
                    "location": f"{lat},{lng}",
                    "radius": radius,
                    "type": place_type,
                    "rankby": "prominence",
                },
            )
            resp.raise_for_status()
            data = resp.json()
        return data.get("results", [])
    except Exception as exc:
        logger.error(f"[TS-10] Nearby search failed for type={place_type}: {exc}")
        return []


def _place_details(place_id: str) -> dict:
    """Call Google Places Details API for a single place."""
    fields = "name,formatted_address,geometry,rating,user_ratings_total,price_level,formatted_phone_number,website,opening_hours,photos"
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(
                f"{GOOGLE_PLACES_BASE}/details/json",
                params={
                    "key": GOOGLE_PLACES_API_KEY,
                    "place_id": place_id,
                    "fields": fields,
                },
            )
            resp.raise_for_status()
            return resp.json().get("result", {})
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
    name = result.get("name") or details.get("name", "")
    if not name:
        return None

    place_id = result.get("place_id", "")

    # Location
    geometry = result.get("geometry", {}).get("location", {}) or \
               details.get("geometry", {}).get("location", {})
    lat = geometry.get("lat")
    lng = geometry.get("lng")
    distance = _haversine_miles(property_lat, property_lng, lat, lng) if lat and lng else None

    # Rating
    rating        = result.get("rating") or details.get("rating")
    review_count  = result.get("user_ratings_total") or details.get("user_ratings_total")

    # Price level (Google uses 0-4 integers)
    price_int = result.get("price_level") if result.get("price_level") is not None \
                else details.get("price_level")
    price_str = {0: "$", 1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}.get(price_int)

    # Photo
    photos = result.get("photos") or details.get("photos") or []
    photo_url = _build_photo_url(photos[0]["photo_reference"]) if photos else None

    # Hours
    hours = None
    oh = details.get("opening_hours", {})
    if oh:
        hours = PlaceHours(
            open_now=oh.get("open_now"),
            weekday_text=oh.get("weekday_text", []),
        )

    return Place(
        name=name,
        category=category,
        primary_source=DataSource.GOOGLE_PLACES,
        address=details.get("formatted_address"),
        distance_miles=round(distance, 1) if distance is not None else None,
        latitude=lat,
        longitude=lng,
        google_rating=float(rating) if rating is not None else None,
        google_review_count=int(review_count) if review_count is not None else None,
        price_level=price_str,
        phone=details.get("formatted_phone_number"),
        website=details.get("website"),
        photo_url=photo_url,
        google_place_id=place_id,
        hours=hours,
    )


def _build_photo_url(photo_reference: str, max_width: int = 800) -> str:
    return (
        f"{GOOGLE_PLACES_BASE}/photo"
        f"?maxwidth={max_width}"
        f"&photo_reference={photo_reference}"
        f"&key={GOOGLE_PLACES_API_KEY}"
    )


def _haversine_miles(lat1: float, lng1: float, lat2: Optional[float], lng2: Optional[float]) -> Optional[float]:
    """Calculate straight-line distance in miles between two coordinates."""
    if lat2 is None or lng2 is None:
        return None
    R = 3958.8   # Earth radius in miles
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng/2)**2
    return R * 2 * math.asin(math.sqrt(a))
