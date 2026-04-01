"""
Supabase Persistence Layer
Reads and writes the property knowledge base to Supabase PostgreSQL.
Also reads intake submission data to seed Agent 1.
"""

import os
import json
import logging
from typing import Optional
from datetime import datetime, timezone

from supabase import create_client, Client

from models.property import (
    ClientChannel,
    DataSource,
    GuestReview,
    PhotoAsset,
    PropertyField,
    PropertyKnowledgeBase,
    VibeProfile,
)

logger = logging.getLogger(__name__)

_supabase: Optional[Client] = None


def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        _supabase = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_ROLE_KEY"],
        )
    return _supabase


def load_intake_submission(property_id: str) -> Optional[PropertyKnowledgeBase]:
    """
    Reads the intake portal submission from Supabase and builds
    the initial knowledge base seeded with intake data.
    Intake portal data has DataSource.INTAKE_PORTAL — highest priority in merge policy.
    """
    try:
        result = (
            get_supabase()
            .table("intake_submissions")
            .select("*")
            .eq("property_id", property_id)
            .single()
            .execute()
        )
        data = result.data
    except Exception as exc:
        logger.error(f"Failed to load intake submission for {property_id}: {exc}")
        return None

    if not data:
        logger.warning(f"No intake submission found for property_id={property_id}")
        return None

    S = DataSource.INTAKE_PORTAL
    C = float("inf")  # Infinite confidence — intake always wins

    def f(val) -> Optional[PropertyField]:
        if val is None:
            return None
        if isinstance(val, str) and not val.strip():
            return None
        return PropertyField(value=val, source=S, confidence=1.0)

    # Determine channel
    channel = (
        ClientChannel.PMC
        if data.get("client_channel") == "pmc"
        else ClientChannel.IO
    )

    kb = PropertyKnowledgeBase(
        property_id=property_id,
        client_id=data["client_id"],
        client_channel=channel,

        # Core fields from intake
        name=f(data.get("property_name")),
        headline=f(data.get("headline")),
        description=f(data.get("description")),
        bedrooms=f(data.get("bedrooms")),
        bathrooms=f(data.get("bathrooms")),
        max_occupancy=f(data.get("max_occupancy")),
        property_type=f(data.get("property_type")),

        # Location
        address_line1=f(data.get("address_line1")),
        city=f(data.get("city")),
        state=f(data.get("state")),
        zip_code=f(data.get("zip_code")),

        # Vibe and booking config
        vibe_profile=VibeProfile(data["vibe_profile"]) if data.get("vibe_profile") else None,
        booking_url=data.get("booking_url"),
        ical_url=data.get("ical_url"),

        # Owner content
        owner_story=data.get("owner_story"),
        seasonal_notes=data.get("seasonal_notes"),
        dont_miss_picks=data.get("dont_miss_picks") or [],

        # OTA URLs from intake
        airbnb_url=data.get("airbnb_url"),
        vrbo_url=data.get("vrbo_url"),
        pmc_website_url=data.get("pmc_website_url"),

        ingestion_sources=[DataSource.INTAKE_PORTAL],
    )

    # Amenities submitted at intake
    for amenity in data.get("amenities") or []:
        kb.amenities.append(PropertyField(value=amenity, source=S, confidence=1.0))

    # Unique features from intake
    for feature in data.get("unique_features") or []:
        kb.unique_features.append(PropertyField(value=feature, source=S, confidence=1.0))

    # Photos uploaded at intake — these are already in R2, URL stored in intake record
    for photo in data.get("uploaded_photos") or []:
        kb.photos.append(PhotoAsset(
            url=photo.get("url", ""),
            source=S,
            category=photo.get("category"),
            caption=photo.get("caption"),
            order=photo.get("order", 0),
        ))

    # Physical guest book entries entered at intake
    for entry in data.get("guest_book_entries") or []:
        kb.guest_reviews.append(GuestReview(
            text=entry.get("review_text", ""),
            source=S,
            reviewer_name=entry.get("guest_name"),
            stay_date=entry.get("stay_date"),
            is_guest_book=True,
        ))

    logger.info(
        f"Intake submission loaded for property {property_id}. "
        f"Channel: {channel}. Photos: {len(kb.photos)}. "
        f"Guest book entries: {len(kb.guest_reviews)}."
    )
    return kb


def save_knowledge_base(kb: PropertyKnowledgeBase) -> bool:
    """
    Upserts the full knowledge base dict to Supabase.
    Called at the end of Agent 1 before passing to parallel agents.
    """
    try:
        get_supabase().table("property_knowledge_bases").upsert(
            {
                "property_id": kb.property_id,
                "client_id": kb.client_id,
                "data": kb.to_dict(),
                "ingestion_complete": kb.ingestion_complete,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="property_id",
        ).execute()
        logger.info(f"Knowledge base saved to Supabase for property {kb.property_id}")
        return True
    except Exception as exc:
        logger.error(f"Failed to save knowledge base for {kb.property_id}: {exc}")
        return False
