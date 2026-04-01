"""
Property Knowledge Base — Data Models
Agent 1 populates these models from scraped + intake data.
All downstream agents read from these models.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from enum import Enum


class DataSource(str, Enum):
    INTAKE_PORTAL  = "intake_portal"    # Highest priority — client-submitted
    PMC_WEBSITE    = "pmc_website"      # Firecrawl scrape (TS-01)
    AIRBNB         = "airbnb"           # Apify Airbnb Actor (TS-02 / TS-04b)
    VRBO           = "vrbo"             # Apify VRBO Actor (TS-03)
    BOOKING_COM    = "booking_com"      # Apify Booking.com Actor (TS-04)
    CLAUDE_PARSED  = "claude_parsed"    # Claude fallback parser (TS-04c)


class VibeProfile(str, Enum):
    ROMANTIC_ESCAPE         = "romantic_escape"
    FAMILY_ADVENTURE        = "family_adventure"
    MULTIGENERATIONAL       = "multigenerational_retreat"
    WELLNESS_RETREAT        = "wellness_retreat"
    ADVENTURE_BASE_CAMP     = "adventure_base_camp"
    SOCIAL_CELEBRATIONS     = "social_celebrations"
    CREATIVE_REMOTE_WORK    = "creative_remote_work"


class ClientChannel(str, Enum):
    PMC = "pmc"   # Property Management Company
    IO  = "io"    # Individual Owner


@dataclass
class PhotoAsset:
    url: str
    source: DataSource
    category: Optional[str] = None       # exterior, kitchen, bedroom, view, etc.
    caption: Optional[str] = None        # owner-provided caption from intake
    order: int = 0                       # display ordering hint from source

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "source": self.source,
            "category": self.category,
            "caption": self.caption,
            "order": self.order,
        }


@dataclass
class GuestReview:
    text: str
    source: DataSource
    reviewer_name: Optional[str] = None
    stay_date: Optional[str] = None      # "August 2022" or ISO date string
    star_rating: Optional[float] = None  # OTA reviews only
    host_response: Optional[str] = None  # OTA reviews only
    is_guest_book: bool = False          # True = physical guest book entry

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "source": self.source,
            "reviewer_name": self.reviewer_name,
            "stay_date": self.stay_date,
            "star_rating": self.star_rating,
            "host_response": self.host_response,
            "is_guest_book": self.is_guest_book,
        }


@dataclass
class PropertyField:
    """
    Wraps any data field with provenance tracking.
    Intake portal data always wins over scraped data.
    """
    value: Optional[str | int | float | list | bool]
    source: DataSource
    confidence: float = 1.0    # 0.0–1.0; Claude-parsed fields may be lower

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "source": self.source,
            "confidence": self.confidence,
        }


@dataclass
class PropertyKnowledgeBase:
    """
    The canonical property record.
    Built by Agent 1, read by all downstream agents.
    Stored in Supabase PostgreSQL as JSONB.
    Also cached in Redis with TTL during active pipeline execution.
    """

    # ── Identity ─────────────────────────────────────────────
    property_id: str               # UUID, assigned at intake submission
    client_id: str                 # PMC or IO client UUID
    client_channel: ClientChannel

    # ── Core Property Data ───────────────────────────────────
    name: Optional[PropertyField] = None
    slug: Optional[str] = None             # URL-safe slug for subdomain
    headline: Optional[PropertyField] = None
    description: Optional[PropertyField] = None
    bedrooms: Optional[PropertyField] = None
    bathrooms: Optional[PropertyField] = None
    max_occupancy: Optional[PropertyField] = None
    property_type: Optional[PropertyField] = None  # cabin, condo, villa, etc.

    # ── Location ─────────────────────────────────────────────
    address_line1: Optional[PropertyField] = None
    city: Optional[PropertyField] = None
    state: Optional[PropertyField] = None
    zip_code: Optional[PropertyField] = None
    latitude: Optional[PropertyField] = None
    longitude: Optional[PropertyField] = None
    neighborhood_description: Optional[PropertyField] = None

    # ── Amenities ────────────────────────────────────────────
    amenities: list[PropertyField] = field(default_factory=list)
    # Notable individual amenities from intake (used for content highlights)
    unique_features: list[PropertyField] = field(default_factory=list)

    # ── Vibe & AI Configuration ──────────────────────────────
    vibe_profile: Optional[VibeProfile] = None
    ideal_guest_description: Optional[str] = None   # AI-generated from vibe
    booking_url: Optional[str] = None               # PMC's direct booking URL
    ical_url: Optional[str] = None                  # Availability calendar feed

    # ── Pricing (informational only — not used for bookings) ─
    avg_nightly_rate: Optional[PropertyField] = None
    min_stay_nights: Optional[PropertyField] = None

    # ── OTA Metadata ─────────────────────────────────────────
    airbnb_url: Optional[str] = None
    vrbo_url: Optional[str] = None
    booking_com_url: Optional[str] = None
    pmc_website_url: Optional[str] = None
    airbnb_rating: Optional[PropertyField] = None
    airbnb_review_count: Optional[PropertyField] = None

    # ── Owner Content ────────────────────────────────────────
    owner_story: Optional[str] = None       # From intake Step 6 — highest value
    seasonal_notes: Optional[str] = None    # From intake Step 7
    dont_miss_picks: list[str] = field(default_factory=list)  # Step 5 curated picks

    # ── Visual Assets ─────────────────────────────────────────
    photos: list[PhotoAsset] = field(default_factory=list)

    # ── Social Proof ─────────────────────────────────────────
    guest_reviews: list[GuestReview] = field(default_factory=list)
    # guest_book entries are stored separately in Supabase with is_guest_book=True

    # ── Pipeline Metadata ────────────────────────────────────
    ingested_at: Optional[datetime] = None
    ingestion_sources: list[DataSource] = field(default_factory=list)
    ingestion_errors: list[str] = field(default_factory=list)   # Non-fatal warnings
    ingestion_complete: bool = False

    def to_dict(self) -> dict:
        """Serialize to dict for Supabase JSONB storage."""
        def _field(f):
            return f.to_dict() if f is not None else None

        return {
            "property_id": self.property_id,
            "client_id": self.client_id,
            "client_channel": self.client_channel,
            "name": _field(self.name),
            "slug": self.slug,
            "headline": _field(self.headline),
            "description": _field(self.description),
            "bedrooms": _field(self.bedrooms),
            "bathrooms": _field(self.bathrooms),
            "max_occupancy": _field(self.max_occupancy),
            "property_type": _field(self.property_type),
            "address_line1": _field(self.address_line1),
            "city": _field(self.city),
            "state": _field(self.state),
            "zip_code": _field(self.zip_code),
            "latitude": _field(self.latitude),
            "longitude": _field(self.longitude),
            "neighborhood_description": _field(self.neighborhood_description),
            "amenities": [a.to_dict() for a in self.amenities],
            "unique_features": [u.to_dict() for u in self.unique_features],
            "vibe_profile": self.vibe_profile,
            "ideal_guest_description": self.ideal_guest_description,
            "booking_url": self.booking_url,
            "ical_url": self.ical_url,
            "avg_nightly_rate": _field(self.avg_nightly_rate),
            "min_stay_nights": _field(self.min_stay_nights),
            "airbnb_url": self.airbnb_url,
            "vrbo_url": self.vrbo_url,
            "booking_com_url": self.booking_com_url,
            "pmc_website_url": self.pmc_website_url,
            "airbnb_rating": _field(self.airbnb_rating),
            "airbnb_review_count": _field(self.airbnb_review_count),
            "owner_story": self.owner_story,
            "seasonal_notes": self.seasonal_notes,
            "dont_miss_picks": self.dont_miss_picks,
            "photos": [p.to_dict() for p in self.photos],
            "guest_reviews": [r.to_dict() for r in self.guest_reviews],
            "ingested_at": self.ingested_at.isoformat() if self.ingested_at else None,
            "ingestion_sources": [s for s in self.ingestion_sources],
            "ingestion_errors": self.ingestion_errors,
            "ingestion_complete": self.ingestion_complete,
        }

    def merge_field(
        self,
        current: Optional[PropertyField],
        incoming: Optional[PropertyField],
    ) -> Optional[PropertyField]:
        """
        Merge policy: intake portal data always wins.
        Among scraped sources, higher-confidence value wins.
        If confidence is equal, current (earlier) source wins.
        """
        if incoming is None:
            return current
        if current is None:
            return incoming
        if current.source == DataSource.INTAKE_PORTAL:
            return current   # intake always wins
        if incoming.source == DataSource.INTAKE_PORTAL:
            return incoming
        if incoming.confidence > current.confidence:
            return incoming
        return current
