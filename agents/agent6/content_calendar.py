"""
Content Calendar Builder

Builds a monthly content calendar for a new property.

Volume: 8 concepts × 4 platforms = 32 posts per month.
Each concept is repurposed across TikTok, Pinterest, Facebook,
and Instagram — one post per platform per concept.

Key constraint: minimum 60-minute gap between posts on same platform.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from agents.agent6.models import (
    ContentCalendar,
    ContentType,
    Platform,
    PostRecord,
    STEADY_STATE_CADENCE_PER_WEEK,
)
from agents.agent6.utm_generator import build_utm_link_for_post

logger = logging.getLogger(__name__)

# ── Volume constants ─────────────────────────────────────────────────────
CONCEPTS_PER_CYCLE = 8
PLATFORMS = ["tiktok", "pinterest", "facebook", "instagram"]
POSTS_PER_CYCLE = CONCEPTS_PER_CYCLE * len(PLATFORMS)  # 32

# Optimal posting times per platform (UTC hour) — based on STR audience research
OPTIMAL_POST_TIMES: dict[Platform, list[int]] = {
    Platform.TIKTOK:    [12, 19],    # Noon and 7pm — peak STR browsing
    Platform.INSTAGRAM: [11],
    Platform.PINTEREST: [21],        # Pinterest peaks in evening
    Platform.FACEBOOK:  [13],
}

# Concept rotation — each concept maps to a video type or photo style.
# These are distributed round-robin across the month's 8 concept slots.
CONCEPT_ROTATION = [
    "vibe_match",        # Concept 1 — hero video
    "walk_through",      # Concept 2 — property tour
    "guest_review_1",    # Concept 3 — guest review
    "guest_review_2",    # Concept 4 — guest review
    "local_highlight",   # Concept 5 — destination / area
    "feature_closeup",   # Concept 6 — feature detail
    "seasonal",          # Concept 7 — seasonal angle
    "guest_review_3",    # Concept 8 — guest review
]


def build_content_calendar(
    property_id: str,
    page_url: str,
    slug: str,
    vibe_profile: str,
    video_assets: list[dict],      # From Agent 3 visual_media_package
    social_captions: list[dict],   # From Agent 2 content_package (legacy — see SEAM below)
    photo_urls: list[dict],        # Category winner photos from Agent 3
    launch_date: Optional[str] = None,
) -> ContentCalendar:
    """
    Build a monthly content calendar: 8 concepts × 4 platforms = 32 posts.

    Args:
        property_id:     Property UUID
        page_url:        Live Staylio property page URL (set by Agent 5)
        slug:            URL slug for UTM parameters
        vibe_profile:    For UTM term and content selection
        video_assets:    List of VideoAsset dicts from Agent 3
        social_captions: List of SocialCaption dicts from Agent 2 (legacy, see SEAM)
        photo_urls:      Category winner photo URLs from Agent 3
        launch_date:     ISO date string; defaults to today

    Returns:
        ContentCalendar with 32 posts (8 concepts × 4 platforms)
    """
    start = datetime.now(timezone.utc).date()
    if launch_date:
        try:
            start = datetime.fromisoformat(launch_date).date()
        except ValueError:
            pass

    calendar = ContentCalendar(
        property_id=property_id,
        page_url=page_url,
        slug=slug,
        launch_date=start.isoformat(),
    )

    # Build lookup structures
    video_map  = _build_video_map(video_assets)
    caption_map = _build_caption_map(social_captions)
    photo_list = [p.get("url") or p if isinstance(p, str) else p.get("asset_url_enhanced") or p.get("url", "") for p in photo_urls[:12]]

    # Track last post time per platform to enforce 60-min gap
    last_post_time: dict[Platform, datetime] = {}
    post_sequence: dict[str, int] = {}   # {platform_video_type: sequence_counter}

    # Distribute 32 posts (8 concepts × 4 platforms) across ~30 days.
    # Each concept gets a "publish day" spread across the month.
    days_in_cycle = 30
    concept_days = [
        int(i * days_in_cycle / CONCEPTS_PER_CYCLE)
        for i in range(CONCEPTS_PER_CYCLE)
    ]

    platform_enum_map = {
        "tiktok":    Platform.TIKTOK,
        "instagram": Platform.INSTAGRAM,
        "pinterest": Platform.PINTEREST,
        "facebook":  Platform.FACEBOOK,
    }

    for concept_idx, day_offset in enumerate(concept_days):
        current_date = start + timedelta(days=day_offset)
        week = day_offset // 7 + 1
        concept_type = CONCEPT_ROTATION[concept_idx % len(CONCEPT_ROTATION)]

        for platform_str in PLATFORMS:
            platform = platform_enum_map[platform_str]

            # Decide media: video (9:16 for TikTok/IG, 1:1 for FB) or photo (Pinterest)
            if platform == Platform.PINTEREST:
                media_url = photo_list[concept_idx % len(photo_list)] if photo_list else ""
                content_type = ContentType.PIN
                video_type_label = None
            elif platform == Platform.FACEBOOK:
                media_url = video_map.get(concept_type, {}).get("1_1", "")
                if not media_url and photo_list:
                    media_url = photo_list[concept_idx % len(photo_list)]
                content_type = ContentType.VIDEO_REEL if media_url and concept_type in video_map else ContentType.FEED_PHOTO
                video_type_label = concept_type if concept_type in video_map else None
            else:
                # TikTok + Instagram — 9:16 vertical video
                media_url = video_map.get(concept_type, {}).get("9_16", "")
                content_type = ContentType.VIDEO_REEL
                video_type_label = concept_type

            if not media_url:
                continue

            post_time = _next_valid_post_time(
                current_date, platform, slot=0, last_post_time=last_post_time,
            )
            seq = _next_seq(post_sequence, f"{platform.value}_{concept_type}")
            utm = build_utm_link_for_post(
                page_url, platform, slug,
                video_type_label or "photo", week, seq, vibe_profile, property_id,
            )

            # SEAM: Agent 8 Stage 9 owns captions via platform_variants.caption.
            # Until Stage 9 is built, this field will be None. Agent 6 must NOT
            # fall back to Agent 2's seed captions silently — that would publish
            # content that was never directed or reviewed.
            platform_variant_caption = None   # TODO: read from platform_variants when Stage 9 exists

            if platform_variant_caption is not None:
                caption_text = platform_variant_caption
                hashtags = []  # platform_variants will include hashtags
            else:
                caption_text, hashtags = _get_caption(
                    caption_map, platform, video_type_label or "photo", seq,
                )

            post = PostRecord(
                property_id=property_id,
                platform=platform,
                content_type=content_type,
                caption=caption_text,
                hashtags=hashtags,
                media_url=media_url,
                video_type=video_type_label,
                page_url=page_url,
                utm_link=utm,
                scheduled_at=post_time.isoformat(),
            )
            calendar.posts.append(post)
            last_post_time[platform] = post_time

    calendar.total_scheduled = len(calendar.posts)
    logger.info(
        f"[Agent 6] Calendar built for property {property_id}: "
        f"{calendar.total_scheduled} posts scheduled ({CONCEPTS_PER_CYCLE} concepts × {len(PLATFORMS)} platforms) "
        f"(TikTok: {sum(1 for p in calendar.posts if p.platform == Platform.TIKTOK)}, "
        f"Instagram: {sum(1 for p in calendar.posts if p.platform == Platform.INSTAGRAM)}, "
        f"Pinterest: {sum(1 for p in calendar.posts if p.platform == Platform.PINTEREST)}, "
        f"Facebook: {sum(1 for p in calendar.posts if p.platform == Platform.FACEBOOK)})"
    )
    return calendar


def build_steady_state_posts(
    property_id: str,
    page_url: str,
    slug: str,
    vibe_profile: str,
    video_assets: list[dict],
    social_captions: list[dict],
    photo_urls: list[dict],
    weeks_ahead: int = 4,
) -> list[PostRecord]:
    """
    Generate steady-state posts for subsequent months.
    Same volume as initial calendar: 8 concepts × 4 platforms = 32 posts.
    Called monthly to keep the queue full.
    """
    # Delegate to the same calendar builder — steady-state uses identical volume.
    calendar = build_content_calendar(
        property_id=property_id,
        page_url=page_url,
        slug=slug,
        vibe_profile=vibe_profile,
        video_assets=video_assets,
        social_captions=social_captions,
        photo_urls=photo_urls,
    )
    return calendar.posts


# ── Internal helpers ──────────────────────────────────────────────────────

def _build_video_map(video_assets: list[dict]) -> dict:
    """
    Build a lookup: {video_type: {format: r2_url}}.
    e.g. {"vibe_match": {"9_16": "https://...", "1_1": "...", "16_9": "..."}}
    """
    vmap: dict = {}
    for v in video_assets:
        if not isinstance(v, dict):
            continue
        vt = v.get("video_type", "")
        fmt = v.get("format", "")
        url = v.get("r2_url", "")
        if vt and fmt and url:
            if vt not in vmap:
                vmap[vt] = {}
            vmap[vt][fmt] = url
    return vmap


def _build_caption_map(social_captions: list[dict]) -> dict:
    """
    Build a lookup: {platform: {video_number: [caption_dict, ...]}}.
    """
    cmap: dict = {}
    for c in social_captions:
        if not isinstance(c, dict):
            continue
        platform = c.get("platform", "")
        video_num = c.get("video_number", "")
        if platform and video_num:
            cmap.setdefault(platform, {}).setdefault(video_num, []).append(c)
    return cmap


def _get_caption(
    caption_map: dict,
    platform: Platform,
    video_type: str,
    seq: int,
) -> tuple[str, list[str]]:
    """Get a caption from the map, cycling through available options."""
    platform_caps = caption_map.get(platform.value, {})
    # Try matching by video number first (1-8 for videos, or 'photo')
    video_num_map = {
        "vibe_match": "1", "walk_through": "2", "guest_review_1": "3",
        "guest_review_2": "4", "local_highlight": "5", "feature_closeup": "6",
        "seasonal": "7", "guest_review_3": "8",
    }
    video_num = video_num_map.get(video_type, "1")
    caps = platform_caps.get(video_num, [])

    if caps:
        cap = caps[seq % len(caps)]
        return cap.get("caption", ""), cap.get("hashtags", [])
    return _fallback_caption(platform, video_type), _default_hashtags(platform)


def _fallback_caption(platform: Platform, video_type: str) -> str:
    """Fallback caption when Agent 2 captions are unavailable."""
    messages = {
        "vibe_match": "This is the one. Link in bio to book direct.",
        "walk_through": "Take a tour 👀 Link in bio.",
        "guest_review_1": "When guests say it better than we ever could.",
        "local_highlight": "Your new favourite spot nearby.",
        "feature_closeup": "This feature. That's it. That's the post.",
        "seasonal": "The timing is perfect. Link in bio.",
    }
    return messages.get(video_type, "Book direct. Link in bio.")


def _default_hashtags(platform: Platform) -> list[str]:
    """Default hashtag set when Agent 2 captions are unavailable."""
    base = ["#vacationrental", "#bookdirect", "#staystaylio"]
    platform_extras = {
        Platform.TIKTOK:    ["#travelTikTok", "#vacationrental", "#vrbo"],
        Platform.INSTAGRAM: ["#vacay", "#airbnb", "#travelgram"],
        Platform.PINTEREST: ["#vacationrental", "#travel", "#booking"],
        Platform.FACEBOOK:  ["#vacation", "#travel"],
    }
    return base + platform_extras.get(platform, [])


def _next_valid_post_time(
    date,
    platform: Platform,
    slot: int,
    last_post_time: dict,
) -> datetime:
    """
    Calculate the next valid post time for a platform,
    respecting the 60-minute minimum gap rule.
    """
    base = _scheduled_time(date, platform, slot)
    last = last_post_time.get(platform)
    if last and (base - last).total_seconds() < 3600:
        base = last + timedelta(hours=1, minutes=5)
    return base


def _scheduled_time(date, platform: Platform, slot: int) -> datetime:
    """Convert a date + platform + slot to a scheduled UTC datetime."""
    hours = OPTIMAL_POST_TIMES.get(platform, [12])
    hour = hours[slot % len(hours)]
    return datetime(
        date.year, date.month, date.day,
        hour, 0, 0,
        tzinfo=timezone.utc,
    )


def _next_seq(seq_counters: dict, key: str) -> int:
    """Increment and return the sequence counter for a key."""
    seq_counters[key] = seq_counters.get(key, 0) + 1
    return seq_counters[key]
