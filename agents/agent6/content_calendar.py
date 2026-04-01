"""
Content Calendar Builder

Builds the 60-day launch sprint content calendar for a new property,
followed by steady-state scheduling.

60-Day Sprint Structure (TS-15):
  Weeks 1–2:  Account launch burst — short high-completion videos first
              TikTok 2x daily (Videos 2+6 → then 1,3,4 rotation)
              Instagram 1x daily, Pinterest 7 pins/week, Facebook 1x daily

  Weeks 3–4:  Engagement build — longer videos, TikTok Spark Ad nomination
              TikTok 2x daily (Videos 1,3,4,5 — drives saves/shares)
              Same cadence on other platforms

  Weeks 5–8:  Sustained + paid amplification
              TikTok 1–2x daily (Videos 7, 8 introduced)
              All platforms continue, Spark Ads running

Total per property over 60 days: ~260 pieces of content
  ~100 TikTok, ~60 Instagram, ~60 Pinterest, ~40 Facebook

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

# Optimal posting times per platform (UTC hour) — based on STR audience research
OPTIMAL_POST_TIMES: dict[Platform, list[int]] = {
    Platform.TIKTOK:    [12, 19],    # Noon and 7pm — peak STR browsing
    Platform.INSTAGRAM: [11],
    Platform.PINTEREST: [21],        # Pinterest peaks in evening
    Platform.FACEBOOK:  [13],
}

# Video sequence for 60-day sprint per week range
# Values are video_type strings matching VideoType enum values
TIKTOK_VIDEO_SEQUENCE = {
    "weeks_1_2": [
        "walk_through",      # Video 2 — short, high completion
        "feature_closeup",   # Video 6 — 12-15s, FYP-optimised
        "vibe_match",        # Video 1 — hero video
        "guest_review_1",    # Video 3
    ],
    "weeks_3_4": [
        "vibe_match",        # Video 1
        "guest_review_1",    # Video 3
        "guest_review_2",    # Video 4
        "local_highlight",   # Video 5 — destination search intent
    ],
    "weeks_5_8": [
        "vibe_match",
        "seasonal",          # Video 7
        "guest_review_3",    # Video 8
        "local_highlight",
        "walk_through",
    ],
}


def build_content_calendar(
    property_id: str,
    page_url: str,
    slug: str,
    vibe_profile: str,
    video_assets: list[dict],      # From Agent 3 visual_media_package
    social_captions: list[dict],   # From Agent 2 content_package
    photo_urls: list[dict],        # Category winner photos from Agent 3
    launch_date: Optional[str] = None,
) -> ContentCalendar:
    """
    Build the complete 60-day content calendar for a property.

    Args:
        property_id:     Property UUID
        page_url:        Live Staylio property page URL (set by Agent 5)
        slug:            URL slug for UTM parameters
        vibe_profile:    For UTM term and content selection
        video_assets:    List of VideoAsset dicts from Agent 3
        social_captions: List of SocialCaption dicts from Agent 2
        photo_urls:      Category winner photo URLs from Agent 3
        launch_date:     ISO date string; defaults to today

    Returns:
        ContentCalendar with all posts scheduled
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

    day = 0
    post_sequence: dict[str, int] = {}   # {platform_video_type: sequence_counter}

    while day < 60:
        current_date = start + timedelta(days=day)
        week = day // 7 + 1

        if week <= 2:
            phase_key = "weeks_1_2"
        elif week <= 4:
            phase_key = "weeks_3_4"
        else:
            phase_key = "weeks_5_8"

        # ── TikTok ────────────────────────────────────────────────────────
        tiktok_videos_today = _get_tiktok_videos_for_day(
            day, phase_key, video_map
        )
        for i, (video_type, video_url) in enumerate(tiktok_videos_today):
            post_time = _next_valid_post_time(
                current_date,
                Platform.TIKTOK,
                slot=i,
                last_post_time=last_post_time,
            )
            seq = _next_seq(post_sequence, f"tiktok_{video_type}")
            utm = build_utm_link_for_post(
                page_url, Platform.TIKTOK, slug,
                video_type, week, seq, vibe_profile, property_id
            )
            caption_text, hashtags = _get_caption(
                caption_map, Platform.TIKTOK, video_type, seq
            )
            post = PostRecord(
                property_id=property_id,
                platform=Platform.TIKTOK,
                content_type=ContentType.VIDEO_REEL,
                caption=caption_text,
                hashtags=hashtags,
                media_url=video_url,
                video_type=video_type,
                page_url=page_url,
                utm_link=utm,
                scheduled_at=post_time.isoformat(),
            )
            calendar.posts.append(post)
            last_post_time[Platform.TIKTOK] = post_time

        # ── Instagram ─────────────────────────────────────────────────────
        ig_post = _build_instagram_post(
            day, week, phase_key, current_date,
            property_id, page_url, slug, vibe_profile,
            video_map, photo_list, caption_map,
            last_post_time, post_sequence,
        )
        if ig_post:
            calendar.posts.append(ig_post)
            last_post_time[Platform.INSTAGRAM] = datetime.fromisoformat(ig_post.scheduled_at)

        # ── Pinterest ─────────────────────────────────────────────────────
        pin_post = _build_pinterest_post(
            day, current_date, property_id, page_url, slug,
            vibe_profile, photo_list, caption_map,
            last_post_time, post_sequence,
        )
        if pin_post:
            calendar.posts.append(pin_post)
            last_post_time[Platform.PINTEREST] = datetime.fromisoformat(pin_post.scheduled_at)

        # ── Facebook ──────────────────────────────────────────────────────
        fb_post = _build_facebook_post(
            day, week, current_date, property_id, page_url, slug,
            vibe_profile, video_map, photo_list, caption_map,
            last_post_time, post_sequence,
        )
        if fb_post:
            calendar.posts.append(fb_post)
            last_post_time[Platform.FACEBOOK] = datetime.fromisoformat(fb_post.scheduled_at)

        day += 1

    calendar.total_scheduled = len(calendar.posts)
    logger.info(
        f"[Agent 6] Calendar built for property {property_id}: "
        f"{calendar.total_scheduled} posts scheduled over 60 days "
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
    Generate steady-state posts for the next N weeks after Day 60.
    Called monthly to keep the queue full.
    """
    video_map   = _build_video_map(video_assets)
    caption_map = _build_caption_map(social_captions)
    photo_list  = [p.get("url", "") for p in photo_urls[:12] if p.get("url")]

    posts: list[PostRecord] = []
    start = datetime.now(timezone.utc).date()
    post_sequence: dict[str, int] = {}

    for week_offset in range(weeks_ahead):
        for day_of_week in range(7):
            current_date = start + timedelta(weeks=week_offset, days=day_of_week)
            week_number = week_offset + 1

            for platform, posts_this_week in STEADY_STATE_CADENCE_PER_WEEK.items():
                # Distribute posts across the week
                posts_per_day_chance = posts_this_week / 7.0
                # Post on this day if it falls within the weekly budget
                if day_of_week < posts_this_week:
                    post_time = _scheduled_time(current_date, platform, slot=0)
                    video_type = _pick_steady_state_video(platform, video_map, post_sequence)
                    media_url = video_map.get(video_type, {}).get("9_16") or (photo_list[day_of_week % len(photo_list)] if photo_list else "")
                    seq = _next_seq(post_sequence, f"{platform.value}_{video_type}_steady")
                    utm = build_utm_link_for_post(
                        page_url, platform, slug,
                        video_type or "photo", week_number, seq, vibe_profile, property_id
                    )
                    caption_text, hashtags = _get_caption(
                        caption_map, platform, video_type or "photo", seq
                    )
                    posts.append(PostRecord(
                        property_id=property_id,
                        platform=platform,
                        content_type=ContentType.VIDEO_REEL if video_type else ContentType.FEED_PHOTO,
                        caption=caption_text,
                        hashtags=hashtags,
                        media_url=media_url,
                        video_type=video_type,
                        page_url=page_url,
                        utm_link=utm,
                        scheduled_at=post_time.isoformat(),
                    ))
    return posts


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


def _get_tiktok_videos_for_day(
    day: int,
    phase_key: str,
    video_map: dict,
) -> list[tuple[str, str]]:
    """
    Return (video_type, url) pairs to post on TikTok for this day.
    """
    sequence = TIKTOK_VIDEO_SEQUENCE.get(phase_key, [])
    if not sequence:
        return []

    posts_per_day = 2 if phase_key in ("weeks_1_2", "weeks_3_4") else 1
    results = []
    for i in range(posts_per_day):
        idx = (day * posts_per_day + i) % len(sequence)
        video_type = sequence[idx]
        url = video_map.get(video_type, {}).get("9_16", "")
        if url:
            results.append((video_type, url))
    return results


def _build_instagram_post(
    day: int, week: int, phase_key: str, current_date,
    property_id, page_url, slug, vibe_profile,
    video_map, photo_list, caption_map,
    last_post_time, post_sequence,
) -> Optional[PostRecord]:
    """Build one Instagram post for the day."""
    # Alternate between Reels and feed photos
    use_video = day % 3 != 2   # 2 out of 3 days use video
    if use_video:
        video_type = _pick_instagram_video(day, video_map)
        url = video_map.get(video_type, {}).get("9_16") if video_type else None
        content_type = ContentType.VIDEO_REEL
    else:
        video_type = None
        url = photo_list[day % len(photo_list)] if photo_list else None
        content_type = ContentType.FEED_PHOTO

    if not url:
        return None

    post_time = _next_valid_post_time(current_date, Platform.INSTAGRAM, 0, last_post_time)
    seq = _next_seq(post_sequence, f"instagram_{video_type or 'photo'}")
    utm = build_utm_link_for_post(
        page_url, Platform.INSTAGRAM, slug,
        video_type or "photo", week, seq, vibe_profile, property_id
    )
    caption_text, hashtags = _get_caption(caption_map, Platform.INSTAGRAM, video_type or "photo", seq)

    return PostRecord(
        property_id=property_id,
        platform=Platform.INSTAGRAM,
        content_type=content_type,
        caption=caption_text,
        hashtags=hashtags,
        media_url=url,
        video_type=video_type,
        page_url=page_url,
        utm_link=utm,
        scheduled_at=post_time.isoformat(),
    )


def _build_pinterest_post(
    day, current_date, property_id, page_url, slug,
    vibe_profile, photo_list, caption_map,
    last_post_time, post_sequence,
) -> Optional[PostRecord]:
    """Build one Pinterest pin for the day (7 pins/week)."""
    if not photo_list:
        return None
    photo_url = photo_list[day % len(photo_list)]
    post_time = _next_valid_post_time(current_date, Platform.PINTEREST, 0, last_post_time)
    week = day // 7 + 1
    seq = _next_seq(post_sequence, "pinterest_pin")
    utm = build_utm_link_for_post(
        page_url, Platform.PINTEREST, slug, "photo", week, seq, vibe_profile, property_id
    )
    caption_text, hashtags = _get_caption(caption_map, Platform.PINTEREST, "photo", seq)
    return PostRecord(
        property_id=property_id,
        platform=Platform.PINTEREST,
        content_type=ContentType.PIN,
        caption=caption_text,
        hashtags=hashtags,
        media_url=photo_url,
        video_type=None,
        page_url=page_url,
        utm_link=utm,
        scheduled_at=post_time.isoformat(),
    )


def _build_facebook_post(
    day, week, current_date, property_id, page_url, slug,
    vibe_profile, video_map, photo_list, caption_map,
    last_post_time, post_sequence,
) -> Optional[PostRecord]:
    """Build one Facebook post for the day."""
    # Facebook gets the 1:1 format of videos or feed photos
    video_type = _pick_instagram_video(day, video_map)
    url = video_map.get(video_type, {}).get("1_1") if video_type else None
    if not url and photo_list:
        url = photo_list[day % len(photo_list)]
        video_type = None

    if not url:
        return None

    post_time = _next_valid_post_time(current_date, Platform.FACEBOOK, 0, last_post_time)
    seq = _next_seq(post_sequence, f"facebook_{video_type or 'photo'}")
    utm = build_utm_link_for_post(
        page_url, Platform.FACEBOOK, slug,
        video_type or "photo", week, seq, vibe_profile, property_id
    )
    caption_text, hashtags = _get_caption(caption_map, Platform.FACEBOOK, video_type or "photo", seq)
    return PostRecord(
        property_id=property_id,
        platform=Platform.FACEBOOK,
        content_type=ContentType.VIDEO_REEL if video_type else ContentType.FEED_PHOTO,
        caption=caption_text,
        hashtags=hashtags,
        media_url=url,
        video_type=video_type,
        page_url=page_url,
        utm_link=utm,
        scheduled_at=post_time.isoformat(),
    )


def _pick_instagram_video(day: int, video_map: dict) -> Optional[str]:
    """Rotate through available video types for Instagram."""
    rotation = ["vibe_match", "guest_review_1", "walk_through", "guest_review_2",
                "local_highlight", "seasonal", "guest_review_3", "feature_closeup"]
    for vt in rotation[day % len(rotation):] + rotation[:day % len(rotation)]:
        if vt in video_map:
            return vt
    return None


def _pick_steady_state_video(platform: Platform, video_map: dict, seq: dict) -> Optional[str]:
    """Pick a video type for steady-state posts."""
    rotation = ["vibe_match", "guest_review_1", "local_highlight",
                "seasonal", "guest_review_2", "walk_through", "guest_review_3"]
    key = f"steady_{platform.value}"
    idx = seq.get(key, 0) % len(rotation)
    for vt in rotation[idx:] + rotation[:idx]:
        if vt in video_map:
            return vt
    return None


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
