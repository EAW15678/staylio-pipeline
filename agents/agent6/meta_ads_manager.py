"""
TS-25 — Meta Marketing API
4-Month Paid Placement Taper

Every new property receives a decaying paid placement schedule:

  Month 1 — Heaviest spend   (default $50, env PAID_MONTH_1_USD)
  Month 2 — Much less        (default $25, env PAID_MONTH_2_USD)
  Month 3 — Nearly none      (default $10, env PAID_MONTH_3_USD)
  Month 4+ — Zero. SEO carries it.

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
    MetaCampaign,
)

logger = logging.getLogger(__name__)

META_ACCESS_TOKEN = os.environ.get("META_ACCESS_TOKEN", "")
META_AD_ACCOUNT_ID = os.environ.get("META_AD_ACCOUNT_ID", "")  # act_XXXXXXXX
META_GRAPH_API_BASE = "https://graph.facebook.com/v19.0"

# Paid placement taper — configurable, Erick sets the amounts.
# Defaults are placeholders proportional to the prior structure.
PAID_TAPER = {
    1: float(os.environ.get("PAID_MONTH_1_USD", "50.0")),   # Heaviest
    2: float(os.environ.get("PAID_MONTH_2_USD", "25.0")),   # Much less
    3: float(os.environ.get("PAID_MONTH_3_USD", "10.0")),   # Nearly none
}
# Month 4+: zero. SEO carries it.


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
    Launch 4-month decaying paid placement for a new property.
    Months 1-3 get budgets from PAID_TAPER. Month 4+: zero.

    Returns list of MetaCampaign records (one per active month).
    """
    if not META_ACCESS_TOKEN or not META_AD_ACCOUNT_ID:
        logger.warning("[TS-25] Meta API credentials not configured — skipping paid campaign")
        return []

    now = datetime.now(timezone.utc).date()
    campaigns: list[MetaCampaign] = []

    for month_num in sorted(PAID_TAPER.keys()):
        budget = PAID_TAPER[month_num]
        if budget <= 0:
            continue
        month_start = now + timedelta(days=30 * (month_num - 1))
        month_end = now + timedelta(days=30 * month_num)

        phase = {
            1: CampaignPhase.PHASE_A_AWARENESS,
            2: CampaignPhase.PHASE_B_INFEED,
            3: CampaignPhase.PHASE_C_RETARGETING,
        }.get(month_num, CampaignPhase.PHASE_A_AWARENESS)

        campaign = _create_taper_campaign(
            property_id=property_id,
            page_url=page_url,
            booking_url=booking_url,
            vibe_profile=vibe_profile,
            latitude=latitude,
            longitude=longitude,
            hero_video_url=hero_video_url,
            start_date=month_start,
            end_date=month_end,
            budget_usd=budget,
            month_num=month_num,
            phase=phase,
            slug=slug,
        )
        if campaign:
            campaigns.append(campaign)

    logger.info(
        f"[TS-25] Meta taper campaign launched for property {property_id}: "
        f"{len(campaigns)} months active, total ${sum(c.budget_usd for c in campaigns):.0f}"
    )
    return campaigns


def _create_taper_campaign(
    property_id: str,
    page_url: str,
    booking_url: str,
    vibe_profile: str,
    latitude: Optional[float],
    longitude: Optional[float],
    hero_video_url: Optional[str],
    start_date,
    end_date,
    budget_usd: float,
    month_num: int,
    phase: CampaignPhase,
    slug: str,
) -> Optional[MetaCampaign]:
    """Create a single month's paid placement campaign."""
    campaign = MetaCampaign(
        property_id=property_id,
        phase=phase,
        budget_usd=budget_usd,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    try:
        days_running = max((end_date - start_date).days, 1)

        campaign_resp = _meta_post(
            f"/act_{META_AD_ACCOUNT_ID.lstrip('act_')}/campaigns",
            {
                "name": f"Staylio-M{month_num}-{slug}",
                "objective": "LINK_CLICKS" if month_num <= 2 else "CONVERSIONS",
                "status": "ACTIVE",
                "special_ad_categories": [],
            }
        )
        if not campaign_resp:
            return None
        campaign.meta_campaign_id = campaign_resp.get("id")

        targeting = _build_targeting(vibe_profile, latitude, longitude, radius_km=300)
        adset_resp = _meta_post(
            f"/act_{META_AD_ACCOUNT_ID.lstrip('act_')}/adsets",
            {
                "name": f"Staylio-M{month_num}-{slug}-adset",
                "campaign_id": campaign.meta_campaign_id,
                "daily_budget": int(budget_usd / days_running * 100),  # cents
                "billing_event": "IMPRESSIONS",
                "optimization_goal": "LINK_CLICKS" if month_num <= 2 else "OFFSITE_CONVERSIONS",
                "targeting": targeting,
                "start_time": start_date.isoformat(),
                "end_time": end_date.isoformat(),
                "status": "ACTIVE",
            }
        )
        if adset_resp:
            campaign.meta_adset_id = adset_resp.get("id")

        # If hero video available, attach as ad creative for month 1
        if month_num == 1 and hero_video_url and campaign.meta_adset_id:
            video_id = _upload_meta_video(hero_video_url, slug)
            if video_id:
                ad_resp = _create_video_ad(
                    campaign.meta_adset_id,
                    video_id,
                    page_url + f"?utm_source=facebook&utm_medium=paid&utm_campaign={slug}&utm_content=m{month_num}",
                    "Learn More",
                    f"Discover {slug.replace('-', ' ').title()} — book direct",
                    slug,
                )
                if ad_resp:
                    campaign.meta_ad_id = ad_resp.get("id")

        campaign.status = CampaignStatus.ACTIVE
        return campaign

    except Exception as exc:
        logger.error(f"[TS-25] Month {month_num} campaign creation failed for {property_id}: {exc}")
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

