"""
TS-25 — Meta Marketing API
3-Phase Paid Launch Campaign

Every new property receives a structured 60-day Meta campaign:

  Phase A — Awareness (Days 1–14, ~$35)
    In-stream overlay ads, travel-intent geo targeting
    Goal: broad awareness, high impressions

  Phase B — In-Feed Video (Days 15–40, ~$70)
    Video 1 (Vibe Match) in 9:16 as in-feed Reel ad
    CTA: "Learn More" → property page
    Goal: 400–600 qualified clicks

  Phase C — Retargeting (Days 41–60, ~$45)
    Guest review videos to page visitors who didn't convert
    CTA: "Book Now" → booking URL
    Goal: recover warm unconverted visitors

All campaigns run through Staylio's centralized Meta Business Manager.
Ad spend tracked per property in Supabase.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx

from agents.agent6.models import (
    CampaignPhase,
    CampaignStatus,
    META_PHASE_BUDGETS,
    MetaCampaign,
)

logger = logging.getLogger(__name__)

META_ACCESS_TOKEN = os.environ.get("META_ACCESS_TOKEN", "")
META_AD_ACCOUNT_ID = os.environ.get("META_AD_ACCOUNT_ID", "")  # act_XXXXXXXX
META_GRAPH_API_BASE = "https://graph.facebook.com/v19.0"


# Vibe → Meta interest targeting keywords
VIBE_TARGETING = {
    "romantic_escape":           ["Romance travel", "Couples vacation", "Honeymoon"],
    "family_adventure":          ["Family vacation", "Family travel", "Beach family"],
    "multigenerational_retreat": ["Family reunion", "Multi-generational travel", "Vacation rental"],
    "wellness_retreat":          ["Wellness travel", "Yoga retreat", "Nature travel"],
    "adventure_base_camp":       ["Outdoor adventure", "Hiking", "Mountain travel"],
    "social_celebrations":       ["Bachelorette party", "Girls trip", "Group travel"],
}


def launch_meta_campaign(
    property_id: str,
    page_url: str,
    booking_url: str,
    vibe_profile: str,
    latitude: Optional[float],
    longitude: Optional[float],
    hero_video_url: Optional[str],
    review_video_urls: list[str],
    slug: str,
) -> list[MetaCampaign]:
    """
    Launch the full 3-phase Meta campaign for a new property.
    Returns list of MetaCampaign records (one per phase).

    Phases are launched sequentially with start/end dates
    based on the current date + phase duration.
    """
    if not META_ACCESS_TOKEN or not META_AD_ACCOUNT_ID:
        logger.warning("[TS-25] Meta API credentials not configured — skipping paid campaign")
        return []

    now = datetime.now(timezone.utc).date()
    campaigns: list[MetaCampaign] = []

    # Phase A: Days 1-14
    phase_a = _create_awareness_campaign(
        property_id=property_id,
        page_url=page_url,
        vibe_profile=vibe_profile,
        latitude=latitude,
        longitude=longitude,
        start_date=now,
        end_date=now + timedelta(days=14),
        slug=slug,
    )
    if phase_a:
        campaigns.append(phase_a)

    # Phase B: Days 15-40 (only if we have hero video)
    if hero_video_url:
        phase_b = _create_infeed_campaign(
            property_id=property_id,
            page_url=page_url,
            vibe_profile=vibe_profile,
            latitude=latitude,
            longitude=longitude,
            hero_video_url=hero_video_url,
            start_date=now + timedelta(days=14),
            end_date=now + timedelta(days=40),
            slug=slug,
        )
        if phase_b:
            campaigns.append(phase_b)

    # Phase C: Days 41-60 (only if we have review videos)
    if review_video_urls:
        phase_c = _create_retargeting_campaign(
            property_id=property_id,
            booking_url=booking_url,
            review_video_urls=review_video_urls,
            start_date=now + timedelta(days=40),
            end_date=now + timedelta(days=60),
            slug=slug,
        )
        if phase_c:
            campaigns.append(phase_c)

    logger.info(
        f"[TS-25] Meta campaign launched for property {property_id}: "
        f"{len(campaigns)} phases active"
    )
    return campaigns


def _create_awareness_campaign(
    property_id: str,
    page_url: str,
    vibe_profile: str,
    latitude: Optional[float],
    longitude: Optional[float],
    start_date,
    end_date,
    slug: str,
) -> Optional[MetaCampaign]:
    """Phase A — Awareness: In-stream overlay ads."""
    campaign = MetaCampaign(
        property_id=property_id,
        phase=CampaignPhase.PHASE_A_AWARENESS,
        budget_usd=META_PHASE_BUDGETS[CampaignPhase.PHASE_A_AWARENESS],
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    try:
        # Create campaign
        campaign_resp = _meta_post(
            f"/act_{META_AD_ACCOUNT_ID.lstrip('act_')}/campaigns",
            {
                "name": f"Staylio-A-{slug}-awareness",
                "objective": "REACH",
                "status": "ACTIVE",
                "special_ad_categories": [],
            }
        )
        if not campaign_resp:
            return None
        campaign.meta_campaign_id = campaign_resp.get("id")

        # Create ad set with geographic targeting
        targeting = _build_targeting(vibe_profile, latitude, longitude, radius_km=300)
        adset_resp = _meta_post(
            f"/act_{META_AD_ACCOUNT_ID.lstrip('act_')}/adsets",
            {
                "name": f"Staylio-A-{slug}-adset",
                "campaign_id": campaign.meta_campaign_id,
                "daily_budget": int(campaign.budget_usd / 14 * 100),  # cents
                "billing_event": "IMPRESSIONS",
                "optimization_goal": "REACH",
                "targeting": targeting,
                "start_time": start_date.isoformat(),
                "end_time": end_date.isoformat(),
                "status": "ACTIVE",
            }
        )
        if adset_resp:
            campaign.meta_adset_id = adset_resp.get("id")

        campaign.status = CampaignStatus.ACTIVE
        return campaign

    except Exception as exc:
        logger.error(f"[TS-25] Phase A campaign creation failed for {property_id}: {exc}")
        campaign.status = CampaignStatus.FAILED
        return campaign


def _create_infeed_campaign(
    property_id: str,
    page_url: str,
    vibe_profile: str,
    latitude: Optional[float],
    longitude: Optional[float],
    hero_video_url: str,
    start_date,
    end_date,
    slug: str,
) -> Optional[MetaCampaign]:
    """Phase B — In-feed video ad using Video 1 (Vibe Match)."""
    campaign = MetaCampaign(
        property_id=property_id,
        phase=CampaignPhase.PHASE_B_INFEED,
        budget_usd=META_PHASE_BUDGETS[CampaignPhase.PHASE_B_INFEED],
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    try:
        campaign_resp = _meta_post(
            f"/act_{META_AD_ACCOUNT_ID.lstrip('act_')}/campaigns",
            {
                "name": f"Staylio-B-{slug}-infeed",
                "objective": "LINK_CLICKS",
                "status": "ACTIVE",
                "special_ad_categories": [],
            }
        )
        if not campaign_resp:
            return None
        campaign.meta_campaign_id = campaign_resp.get("id")

        targeting = _build_targeting(vibe_profile, latitude, longitude, radius_km=400)
        days_running = (end_date - start_date).days
        adset_resp = _meta_post(
            f"/act_{META_AD_ACCOUNT_ID.lstrip('act_')}/adsets",
            {
                "name": f"Staylio-B-{slug}-adset",
                "campaign_id": campaign.meta_campaign_id,
                "daily_budget": int(campaign.budget_usd / days_running * 100),
                "billing_event": "LINK_CLICKS",
                "optimization_goal": "LINK_CLICKS",
                "targeting": targeting,
                "start_time": start_date.isoformat(),
                "end_time": end_date.isoformat(),
                "status": "ACTIVE",
            }
        )
        if adset_resp:
            campaign.meta_adset_id = adset_resp.get("id")

        # Upload video and create ad
        video_id = _upload_meta_video(hero_video_url, slug)
        if video_id and campaign.meta_adset_id:
            ad_resp = _create_video_ad(
                campaign.meta_adset_id,
                video_id,
                page_url + f"?utm_source=facebook&utm_medium=paid&utm_campaign={slug}&utm_content=phase_b",
                "Learn More",
                f"Discover {slug.replace('-', ' ').title()} — book direct",
                slug,
            )
            if ad_resp:
                campaign.meta_ad_id = ad_resp.get("id")

        campaign.status = CampaignStatus.ACTIVE
        return campaign

    except Exception as exc:
        logger.error(f"[TS-25] Phase B campaign creation failed for {property_id}: {exc}")
        campaign.status = CampaignStatus.FAILED
        return campaign


def _create_retargeting_campaign(
    property_id: str,
    booking_url: str,
    review_video_urls: list[str],
    start_date,
    end_date,
    slug: str,
) -> Optional[MetaCampaign]:
    """Phase C — Retargeting: Review videos to warm page visitors."""
    campaign = MetaCampaign(
        property_id=property_id,
        phase=CampaignPhase.PHASE_C_RETARGETING,
        budget_usd=META_PHASE_BUDGETS[CampaignPhase.PHASE_C_RETARGETING],
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    try:
        campaign_resp = _meta_post(
            f"/act_{META_AD_ACCOUNT_ID.lstrip('act_')}/campaigns",
            {
                "name": f"Staylio-C-{slug}-retargeting",
                "objective": "CONVERSIONS",
                "status": "ACTIVE",
                "special_ad_categories": [],
            }
        )
        if not campaign_resp:
            return None
        campaign.meta_campaign_id = campaign_resp.get("id")

        # Retargeting audience: visitors to property page (requires Meta pixel)
        days_running = (end_date - start_date).days
        adset_resp = _meta_post(
            f"/act_{META_AD_ACCOUNT_ID.lstrip('act_')}/adsets",
            {
                "name": f"Staylio-C-{slug}-adset",
                "campaign_id": campaign.meta_campaign_id,
                "daily_budget": int(campaign.budget_usd / days_running * 100),
                "billing_event": "IMPRESSIONS",
                "optimization_goal": "OFFSITE_CONVERSIONS",
                "targeting": {
                    "custom_audiences": [],   # Pixel-based audience added after pixel fires
                    "geo_locations": {"countries": ["US"]},
                },
                "start_time": start_date.isoformat(),
                "end_time": end_date.isoformat(),
                "status": "ACTIVE",
            }
        )
        if adset_resp:
            campaign.meta_adset_id = adset_resp.get("id")

        campaign.status = CampaignStatus.ACTIVE
        return campaign

    except Exception as exc:
        logger.error(f"[TS-25] Phase C campaign creation failed for {property_id}: {exc}")
        campaign.status = CampaignStatus.FAILED
        return campaign


def _build_targeting(
    vibe_profile: str,
    latitude: Optional[float],
    longitude: Optional[float],
    radius_km: int,
) -> dict:
    """Build Meta targeting spec from vibe profile and property location."""
    interests = VIBE_TARGETING.get(vibe_profile, ["Vacation rental", "Travel"])
    targeting: dict = {
        "age_min": 25,
        "age_max": 65,
        "genders": [1, 2],
        "flexible_spec": [{"interests": [{"name": i} for i in interests]}],
        "geo_locations": {"countries": ["US"]},
    }
    if latitude and longitude:
        targeting["geo_locations"] = {
            "custom_locations": [{
                "latitude": latitude,
                "longitude": longitude,
                "radius": radius_km,
                "distance_unit": "kilometer",
            }]
        }
    return targeting


def _upload_meta_video(video_url: str, slug: str) -> Optional[str]:
    """Upload a video to Meta ad library. Returns video ID."""
    try:
        # Download video bytes first
        with httpx.Client(timeout=60) as client:
            dl = client.get(video_url)
            dl.raise_for_status()
            video_bytes = dl.content

        # Upload to Meta
        resp = httpx.post(
            f"https://graph-video.facebook.com/v19.0/act_{META_AD_ACCOUNT_ID.lstrip('act_')}/advideos",
            headers={"Authorization": f"Bearer {META_ACCESS_TOKEN}"},
            files={"source": (f"{slug}-hero.mp4", video_bytes, "video/mp4")},
            data={"title": f"{slug} hero video"},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("id")
    except Exception as exc:
        logger.error(f"[TS-25] Meta video upload failed: {exc}")
        return None


def _create_video_ad(
    adset_id: str,
    video_id: str,
    destination_url: str,
    cta_type: str,
    message: str,
    slug: str,
) -> Optional[dict]:
    """Create a Meta video ad creative and attach to an ad set."""
    try:
        # Create creative
        creative_resp = _meta_post(
            f"/act_{META_AD_ACCOUNT_ID.lstrip('act_')}/adcreatives",
            {
                "name": f"Staylio-{slug}-creative",
                "object_story_spec": {
                    "page_id": os.environ.get("META_PAGE_ID", ""),
                    "video_data": {
                        "video_id": video_id,
                        "call_to_action": {
                            "type": cta_type.upper().replace(" ", "_"),
                            "value": {"link": destination_url},
                        },
                        "message": message,
                    }
                }
            }
        )
        if not creative_resp:
            return None

        # Create ad
        return _meta_post(
            f"/act_{META_AD_ACCOUNT_ID.lstrip('act_')}/ads",
            {
                "name": f"Staylio-{slug}-ad",
                "adset_id": adset_id,
                "creative": {"creative_id": creative_resp.get("id")},
                "status": "ACTIVE",
            }
        )
    except Exception as exc:
        logger.error(f"[TS-25] Ad creation failed: {exc}")
        return None


def _meta_post(endpoint: str, payload: dict) -> Optional[dict]:
    """Make a Meta Graph API POST request."""
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{META_GRAPH_API_BASE}{endpoint}",
                headers={
                    "Authorization": f"Bearer {META_ACCESS_TOKEN}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.error(f"[TS-25] Meta API error for {endpoint}: {exc}")
        return None


import os  # noqa: E402 — needed for META_PAGE_ID env access in _create_video_ad
