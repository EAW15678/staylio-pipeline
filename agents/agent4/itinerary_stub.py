"""
TS-11b — Phase 2 Guest Itinerary Generator
STATUS: DEFERRED — DO NOT BUILD UNTIL PHASE 1 IS COMPLETE

This module defines the interface for the Phase 2 Guest Itinerary
Generator so that:
  (a) Agent 4 can reference it cleanly in the pipeline
  (b) The landing page can include the itinerary CTA button from day one
  (c) The data capture schema is defined before the feature is built

Interface is correct and stable. Implementation is stubbed.

Pre-build decisions required before implementation (from TS-11b):
  1. DATA OWNERSHIP: Who owns guest contact data — Staylio, PMC, or both?
     Must be resolved and documented in PMC client ToS before build.
  2. CONSENT: TCPA (SMS), CAN-SPAM (email), GDPR (European guests)
  3. ITINERARY FORMAT: day-by-day structured text, HTML page, or PDF
  4. REMARKETING: what follow-up sequences trigger after capture?
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class DeliveryMethod(str, Enum):
    EMAIL = "email"
    SMS   = "sms"
    POPUP = "popup"   # No contact capture — fallback


@dataclass
class ItineraryRequest:
    """Data submitted by a guest requesting a personalised itinerary."""
    property_id: str
    travel_dates_start: Optional[str] = None     # ISO date string
    travel_dates_end: Optional[str] = None
    group_size: Optional[int] = None
    guest_email: Optional[str] = None            # Captured if email delivery chosen
    guest_phone: Optional[str] = None            # Captured if SMS delivery chosen
    delivery_method: DeliveryMethod = DeliveryMethod.POPUP
    guest_interests: list[str] = field(default_factory=list)   # Optional tags


@dataclass
class GeneratedItinerary:
    """A generated day-by-day itinerary for a guest's specific travel dates."""
    property_id: str
    request: ItineraryRequest
    days: list[dict] = field(default_factory=list)   # day-by-day structure
    generated_at: Optional[str] = None
    delivery_status: str = "pending"


def generate_itinerary(
    request: ItineraryRequest,
    local_guide: "LocalGuide",  # noqa: F821
) -> GeneratedItinerary:
    """
    PHASE 2 STUB — not implemented.

    When implemented, this function will:
    1. Pull live Eventbrite/PredictHQ events for request.travel_dates_start/end
    2. Build a personalised day-by-day itinerary using Claude
    3. Incorporate local guide content from Phase 1 (places, recommendations)
    4. Deliver via email (SendGrid), SMS (Twilio), or popup
    5. Store guest contact in Supabase if email/SMS was provided

    Implementation is blocked on:
    - DATA OWNERSHIP resolution (see TS-11b)
    - CONSENT language sign-off (legal counsel)
    - REMARKETING workflow design
    """
    logger.info(
        f"[TS-11b] Itinerary request received for property {request.property_id} "
        f"(Phase 2 — not yet implemented)"
    )
    raise NotImplementedError(
        "Guest itinerary generator is a Phase 2 feature. "
        "See TS-11b in the Tech Stack Decision Register."
    )


def get_itinerary_cta_config(property_id: str) -> dict:
    """
    Returns the CTA configuration for the itinerary button on the
    landing page. The button is shown on all property pages from
    Phase 1 launch — it links to the Phase 2 feature endpoint.
    When Phase 2 is not yet live, clicking the button shows a
    'Coming Soon — enter your email for early access' prompt.

    This allows the landing page to include the CTA from day one
    without requiring the full Phase 2 implementation.
    """
    return {
        "property_id": property_id,
        "cta_text": "Get Your Personalised Itinerary",
        "cta_subtext": "Enter your travel dates — we'll plan your perfect stay",
        "phase2_live": False,   # Flip to True when Phase 2 launches
        "coming_soon_email_capture": True,
    }
