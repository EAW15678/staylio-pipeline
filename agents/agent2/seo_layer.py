"""
TS-05c — SEO Keyword Intelligence Layer
Tools: DataForSEO API + Google Search Console API

Runs BEFORE content generation to seed the Claude prompt with
real keyword data — search volume, difficulty, and related terms
for the property's location and vibe type.

Also provides the refresh signal: Google Search Console tells us
which live property pages have underperforming click-through rates
and need a content refresh.
"""

import os
import json
import logging
import base64
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

DATAFORSEO_LOGIN    = os.environ.get("DATAFORSEO_LOGIN", "")
DATAFORSEO_PASSWORD = os.environ.get("DATAFORSEO_PASSWORD", "")
DATAFORSEO_BASE     = "https://api.dataforseo.com/v3"

# Number of keywords to fetch per property (controls API cost)
# At $0.0006 per keyword result, 25 keywords = $0.015 per property
MAX_KEYWORDS = 25


def fetch_seo_keywords(
    city: Optional[str],
    state: Optional[str],
    property_type: Optional[str],
    vibe_profile: str,
) -> list[str]:
    """
    Fetches search volume and related keyword data for a property's
    location and vibe type from DataForSEO.

    Returns a list of keyword strings ranked by search volume,
    ready for injection into the Claude content generation prompt.

    Falls back gracefully to seed keywords if DataForSEO is unavailable.
    """
    if not city and not state:
        logger.info("[TS-05c] No location data — using seed keywords only")
        return _seed_keywords(vibe_profile, property_type)

    location = f"{city or ''} {state or ''}".strip()
    queries = _build_keyword_queries(location, property_type, vibe_profile)

    if not DATAFORSEO_LOGIN:
        logger.info("[TS-05c] DataForSEO credentials not configured — using seed keywords")
        return _seed_keywords(vibe_profile, property_type)

    try:
        keywords = _call_dataforseo(queries)
        if keywords:
            logger.info(f"[TS-05c] DataForSEO returned {len(keywords)} keywords for {location}")
            return keywords[:MAX_KEYWORDS]
    except Exception as exc:
        logger.warning(f"[TS-05c] DataForSEO API call failed: {exc} — falling back to seed keywords")

    return _seed_keywords(vibe_profile, property_type)


def check_gsc_refresh_needed(
    property_id: str,
    page_url: str,
) -> tuple[bool, list[str]]:
    """
    TS-05c refresh signal: checks Google Search Console for a live
    property page to identify underperforming content blocks.

    Returns:
        (refresh_needed: bool, underperforming_fields: list of field names)

    For example: (True, ["hero_headline", "property_description"])
    signals that these fields should be regenerated.

    GSC data is only available for pages that have been live for 28+ days
    and have accumulated measurable impression data.
    """
    # GSC integration requires OAuth service account credentials
    # This is a Phase 1 stub — full implementation in Weeks 7-8 build phase
    # The interface is defined here so Agent 2 can call it without changes
    # when the full implementation lands.
    logger.debug(f"[TS-05c] GSC refresh check for property {property_id} — stub")
    return False, []


# ── DataForSEO API ────────────────────────────────────────────────────────

def _call_dataforseo(queries: list[str]) -> list[str]:
    """
    Calls DataForSEO Keywords Data API to get search volume for a list
    of seed queries, then returns related keywords ranked by search volume.
    """
    credentials = base64.b64encode(
        f"{DATAFORSEO_LOGIN}:{DATAFORSEO_PASSWORD}".encode()
    ).decode()

    payload = [
        {
            "keywords": queries,
            "location_code": 2840,   # United States
            "language_code": "en",
        }
    ]

    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{DATAFORSEO_BASE}/keywords_data/google_ads/search_volume/live",
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    keywords = []
    for task in data.get("tasks", []):
        for item in (task.get("result") or []):
            keyword = item.get("keyword")
            volume  = item.get("search_volume", 0) or 0
            if keyword and volume > 10:
                keywords.append((keyword, volume))

    # Sort by search volume descending
    keywords.sort(key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in keywords]


def _build_keyword_queries(
    location: str,
    property_type: Optional[str],
    vibe_profile: str,
) -> list[str]:
    """
    Build seed keyword queries to send to DataForSEO.
    These are the queries whose related keywords we want to discover.
    """
    pt = property_type or "vacation rental"
    queries = [
        f"vacation rental {location}",
        f"{pt} rental {location}",
        f"short term rental {location}",
        f"book {pt} {location}",
        f"vacation home {location}",
    ]

    # Add vibe-specific queries
    vibe_queries = _vibe_keyword_seeds(vibe_profile, location, pt)
    queries.extend(vibe_queries)

    return list(dict.fromkeys(queries))   # dedupe preserving order


def _vibe_keyword_seeds(
    vibe_profile: str,
    location: str,
    property_type: str,
) -> list[str]:
    """Return vibe-specific keyword seed queries."""
    from models.property import VibeProfile

    seeds = {
        VibeProfile.ROMANTIC_ESCAPE: [
            f"romantic getaway {location}",
            f"couples retreat {location}",
            f"romantic cabin {location}",
            f"honeymoon rental {location}",
        ],
        VibeProfile.FAMILY_ADVENTURE: [
            f"family vacation rental {location}",
            f"family {property_type} {location}",
            f"vacation rental with pool {location}",
            f"large family rental {location}",
        ],
        VibeProfile.MULTIGENERATIONAL: [
            f"large group rental {location}",
            f"family reunion rental {location}",
            f"multi generational vacation {location}",
            f"big house rental {location}",
        ],
        VibeProfile.WELLNESS_RETREAT: [
            f"wellness retreat {location}",
            f"relaxing vacation rental {location}",
            f"peaceful cabin {location}",
            f"mountain retreat {location}",
        ],
        VibeProfile.ADVENTURE_BASE_CAMP: [
            f"outdoor adventure rental {location}",
            f"hiking cabin {location}",
            f"ski chalet rental {location}",
            f"adventure vacation rental {location}",
        ],
        VibeProfile.SOCIAL_CELEBRATIONS: [
            f"bachelorette party house {location}",
            f"party house rental {location}",
            f"group vacation rental {location}",
            f"event rental {location}",
        ],
    }
    return seeds.get(vibe_profile, [])


def _seed_keywords(vibe_profile: str, property_type: Optional[str]) -> list[str]:
    """
    Fallback keyword list when DataForSEO is unavailable.
    Generic but better than nothing — ensures Claude has some SEO context.
    """
    from models.property import VibeProfile

    base = [
        "vacation rental",
        f"{property_type or 'vacation home'} rental",
        "short term rental",
        "book direct",
        "direct booking vacation rental",
    ]

    vibe_specific = {
        VibeProfile.ROMANTIC_ESCAPE: ["romantic getaway", "couples retreat", "honeymoon rental", "romantic cabin"],
        VibeProfile.FAMILY_ADVENTURE: ["family vacation rental", "vacation rental with pool", "large family rental", "kids friendly rental"],
        VibeProfile.MULTIGENERATIONAL: ["large group rental", "family reunion house", "multi-generational vacation", "group house rental"],
        VibeProfile.WELLNESS_RETREAT: ["wellness retreat rental", "peaceful cabin rental", "relaxing vacation home", "mountain retreat"],
        VibeProfile.ADVENTURE_BASE_CAMP: ["adventure vacation rental", "hiking cabin rental", "ski chalet rental", "outdoor adventure stay"],
        VibeProfile.SOCIAL_CELEBRATIONS: ["bachelorette party house", "group celebration rental", "party house rental", "event house rental"],
    }

    return base + vibe_specific.get(vibe_profile, [])
