"""
UTM Link Generator

Every post published by Agent 6 MUST include a UTM-tagged link
to the Staylio property landing page. This is non-negotiable —
it is the foundation of Tier 1 attribution in TS-16.

No post should ever be published without a UTM-tagged link.

UTM parameters:
  utm_source:   platform name (instagram, tiktok, facebook, pinterest)
  utm_medium:   social
  utm_campaign: property slug
  utm_content:  post identifier (video type or content type + sequence)
  utm_term:     vibe profile (for segmentation in GA4)

Additional custom parameter:
  utm_property_id: property UUID (for cross-domain attribution in Segment)
"""

from urllib.parse import urlencode, urlparse, urlunparse, parse_qs

from agents.agent6.models import ContentType, Platform


def build_utm_link(
    page_url: str,
    platform: Platform,
    slug: str,
    content_identifier: str,
    vibe_profile: str = "",
    property_id: str = "",
) -> str:
    """
    Build a UTM-tagged URL for a social media post.

    Args:
        page_url:            Base Staylio property page URL
        platform:            The publishing platform
        slug:                Property URL slug (utm_campaign value)
        content_identifier:  Post-specific identifier (utm_content value)
                             Convention: "{video_type}_{week}_{seq}"
                             e.g. "vibe_match_w1_01" or "photo_pool_w3_04"
        vibe_profile:        Property vibe (utm_term)
        property_id:         UUID for cross-domain Segment attribution

    Returns:
        Full UTM-tagged URL string
    """
    if not page_url:
        return ""

    # Strip any existing query string from page_url before adding UTM
    parsed = urlparse(page_url)
    base_url = urlunparse(parsed._replace(query="", fragment=""))

    params = {
        "utm_source":   platform.value,
        "utm_medium":   "social",
        "utm_campaign": slug,
        "utm_content":  content_identifier,
    }
    if vibe_profile:
        params["utm_term"] = vibe_profile
    if property_id:
        params["utm_property_id"] = property_id

    query_string = urlencode(params)
    return f"{base_url}?{query_string}"


def build_utm_link_for_post(
    page_url: str,
    platform: Platform,
    slug: str,
    video_type: str,
    week: int,
    sequence: int,
    vibe_profile: str = "",
    property_id: str = "",
) -> str:
    """
    Convenience wrapper for video post UTM links.
    Builds the content identifier from video type + week + sequence.
    """
    content_id = f"{video_type}_w{week:02d}_{sequence:02d}"
    return build_utm_link(
        page_url=page_url,
        platform=platform,
        slug=slug,
        content_identifier=content_id,
        vibe_profile=vibe_profile,
        property_id=property_id,
    )


def validate_utm_link(url: str) -> tuple[bool, list[str]]:
    """
    Validate that a URL contains all required UTM parameters.
    Returns (is_valid, list_of_missing_params).

    Used as a pre-publish check — no post without all required UTMs.
    """
    if not url:
        return False, ["url is empty"]

    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False, ["url is not a valid URL"]

    params = parse_qs(parsed.query)
    required = ["utm_source", "utm_medium", "utm_campaign", "utm_content"]
    missing = [p for p in required if p not in params]

    return len(missing) == 0, missing
