"""
Agent 4 — Local Discovery Data Models

LocalGuide is the complete output of Agent 4.
Stored in Supabase as JSONB alongside the knowledge base.
Consumed by Agent 5 (Website Builder) for landing page assembly.

Structure mirrors the landing page layout defined in the blueprint:
  1. Area introduction (3-5 sentences, vibe voice)
  2. Don't Miss picks (owner-curated, featured placement)
  3. Vibe-matched primary recommendations (6-12 businesses)
  4. Category browser (full local guide for all categories)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PlaceCategory(str, Enum):
    """
    Landing page category tabs for the 'Explore the Area' section.
    Maps to both Google Places types and Yelp categories.
    """
    EAT_AND_DRINK        = "eat_and_drink"
    NIGHTLIFE            = "nightlife"
    ADVENTURE_OUTDOORS   = "adventure_outdoors"
    ATTRACTIONS          = "attractions"
    ARTS_CULTURE         = "arts_culture"
    FAMILY_KIDS          = "family_kids"
    WELLNESS             = "wellness"
    TOURS_EXPERIENCES    = "tours_experiences"
    SHOPPING             = "shopping"
    COFFEE_CAFES         = "coffee_cafes"


class DataSource(str, Enum):
    GOOGLE_PLACES  = "google_places"
    YELP_FUSION    = "yelp_fusion"
    OWNER_CURATED  = "owner_curated"     # Don't Miss picks from intake


@dataclass
class PlaceHours:
    """Business operating hours."""
    open_now: Optional[bool] = None
    periods: list[dict] = field(default_factory=list)
    weekday_text: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "open_now": self.open_now,
            "weekday_text": self.weekday_text,
        }


@dataclass
class Place:
    """
    A single business or attraction in the local guide.
    Merged from Google Places + Yelp Fusion data.
    """
    # Identity
    name: str
    category: PlaceCategory
    primary_source: DataSource

    # Location
    address: Optional[str] = None
    distance_miles: Optional[float] = None    # Distance from property
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    # Quality signals
    google_rating: Optional[float] = None
    google_review_count: Optional[int] = None
    yelp_rating: Optional[float] = None
    yelp_review_count: Optional[int] = None
    price_level: Optional[str] = None          # "$" | "$$" | "$$$" | "$$$$"

    # Content
    description: Optional[str] = None          # AI-synthesized from API data
    phone: Optional[str] = None
    website: Optional[str] = None
    photo_url: Optional[str] = None            # Best available photo URL

    # Source IDs (for deduplication)
    google_place_id: Optional[str] = None
    yelp_id: Optional[str] = None

    # Operating hours
    hours: Optional[PlaceHours] = None

    # Guide metadata
    is_dont_miss: bool = False               # True = owner-curated Don't Miss pick
    vibe_match_score: float = 0.0           # 0.0–1.0 relevance to vibe
    display_order: int = 0                  # Final display order on landing page

    def composite_rating(self) -> Optional[float]:
        """Weighted average of Google and Yelp ratings (Google gets 60% weight)."""
        if self.google_rating and self.yelp_rating:
            return round(self.google_rating * 0.6 + self.yelp_rating * 0.4, 1)
        return self.google_rating or self.yelp_rating

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "primary_source": self.primary_source,
            "address": self.address,
            "distance_miles": self.distance_miles,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "google_rating": self.google_rating,
            "google_review_count": self.google_review_count,
            "yelp_rating": self.yelp_rating,
            "yelp_review_count": self.yelp_review_count,
            "price_level": self.price_level,
            "description": self.description,
            "phone": self.phone,
            "website": self.website,
            "photo_url": self.photo_url,
            "google_place_id": self.google_place_id,
            "yelp_id": self.yelp_id,
            "hours": self.hours.to_dict() if self.hours else None,
            "is_dont_miss": self.is_dont_miss,
            "vibe_match_score": self.vibe_match_score,
            "display_order": self.display_order,
            "composite_rating": self.composite_rating(),
        }


@dataclass
class DontMissPick:
    """Owner-curated featured recommendation from intake Step 5."""
    name: str
    description: str             # Owner's own words — preserved verbatim
    category: Optional[PlaceCategory] = None
    place_ref: Optional[Place] = None   # Matched API record if found


@dataclass
class LocalGuide:
    """
    Complete local area guide for a property.
    Built by Agent 4, consumed by Agent 5 for landing page assembly.
    Stored in Supabase JSONB.
    """
    property_id: str

    # Owner-authored area introduction (AI-refined from owner input)
    area_introduction: Optional[str] = None

    # Featured section — owner Don't Miss picks (max 5, displayed prominently)
    dont_miss_picks: list[DontMissPick] = field(default_factory=list)

    # Vibe-matched primary recommendations (6-12 places, displayed as visual cards)
    primary_recommendations: list[Place] = field(default_factory=list)

    # Full category browser (all places organized by category tab)
    places_by_category: dict[str, list[Place]] = field(default_factory=dict)

    # Metadata
    location_name: str = ""       # "Carolina Beach, NC" — used in area intro
    total_places: int = 0
    sources_used: list[str] = field(default_factory=list)
    processing_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        def _pick(p):
            return {
                "name": p.name,
                "description": p.description,
                "category": p.category,
                "place": p.place_ref.to_dict() if p.place_ref else None,
            }

        return {
            "property_id": self.property_id,
            "area_introduction": self.area_introduction,
            "dont_miss_picks": [_pick(p) for p in self.dont_miss_picks],
            "primary_recommendations": [p.to_dict() for p in self.primary_recommendations],
            "places_by_category": {
                cat: [p.to_dict() for p in places]
                for cat, places in self.places_by_category.items()
            },
            "location_name": self.location_name,
            "total_places": self.total_places,
            "sources_used": self.sources_used,
            "processing_errors": self.processing_errors,
        }
