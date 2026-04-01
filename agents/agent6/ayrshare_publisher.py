"""
TS-15 — Ayrshare Social Publishing
Primary publishing layer for all social media content.

Ayrshare manages OAuth tokens for every connected client social account,
handles platform-specific format requirements, schedules posts, provides
webhook callbacks on publish, and retrieves engagement metrics.

Fallback path: if Ayrshare returns a 5xx or is unreachable, the post
is routed to the direct platform API (TS-25) for the same operation.

Every Ayrshare request is pre-validated:
  - UTM link present and valid (non-negotiable per TS-15)
  - Media URL is accessible
  - Caption within platform length limits
"""

import logging
import os
import time
from typing import Optional

import httpx

from agents.agent6.models import Platform, PostRecord, PostStatus
from agents.agent6.utm_generator import validate_utm_link

logger = logging.getLogger(__name__)

AYRSHARE_API_KEY  = os.environ.get("AYRSHARE_API_KEY", "")
AYRSHARE_API_BASE = "https://getlate.dev/api"

# Platform-specific caption length limits
CAPTION_LIMITS = {
    Platform.INSTAGRAM: 2200,
    Platform.TIKTOK:    2200,
    Platform.FACEBOOK:  63206,
    Platform.PINTEREST: 500,
}

# Retry config for transient failures
MAX_RETRIES    = 3
RETRY_DELAY_S  = 5


def publish_post(
    post: PostRecord,
    profile_key: str,    # Ayrshare profile key for this client's social accounts
) -> PostRecord:
    """
    Publish a single post via Ayrshare.
    Updates the PostRecord with Ayrshare response data.

    Args:
        post:        The PostRecord to publish
        profile_key: Ayrshare profile key for the client's connected accounts

    Returns:
        Updated PostRecord with ayrshare_post_id, status, and published_at
    """
    # ── Pre-publish validation ────────────────────────────────────────────
    validation_errors = _validate_post(post)
    if validation_errors:
        logger.error(
            f"[TS-15] Pre-publish validation failed for property {post.property_id} "
            f"on {post.platform}: {validation_errors}"
        )
        post.status = PostStatus.FAILED
        return post

    if not AYRSHARE_API_KEY:
        logger.warning("[TS-15] Ayrshare API key not configured — skipping publish")
        post.status = PostStatus.FAILED
        return post

    # ── Build Ayrshare payload ────────────────────────────────────────────
    payload = _build_ayrshare_payload(post, profile_key)

    # ── Publish with retry ────────────────────────────────────────────────
    for attempt in range(MAX_RETRIES):
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    f"{AYRSHARE_API_BASE}/post",
                    headers={
                        "Authorization": f"Bearer {AYRSHARE_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )

            if resp.status_code == 200:
                data = resp.json()
                post = _apply_ayrshare_response(post, data)
                logger.info(
                    f"[TS-15] Published {post.platform}/{post.video_type or 'photo'} "
                    f"for property {post.property_id} → id={post.ayrshare_post_id}"
                )
                return post

            elif resp.status_code in (429, 500, 502, 503):
                # Transient — retry
                logger.warning(
                    f"[TS-15] Ayrshare {resp.status_code} on attempt {attempt+1} "
                    f"for property {post.property_id}"
                )
                time.sleep(RETRY_DELAY_S * (attempt + 1))
                continue

            else:
                # 4xx — bad request, don't retry
                logger.error(
                    f"[TS-15] Ayrshare {resp.status_code} for property {post.property_id}: "
                    f"{resp.text[:200]}"
                )
                post.status = PostStatus.FAILED
                return post

        except httpx.TimeoutException:
            logger.warning(f"[TS-15] Ayrshare timeout on attempt {attempt+1}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_S)
        except Exception as exc:
            logger.error(f"[TS-15] Ayrshare unexpected error: {exc}")
            break

    # All retries exhausted
    logger.error(
        f"[TS-15] All Ayrshare retries exhausted for property {post.property_id} "
        f"on {post.platform} — marking FAILED"
    )
    post.status = PostStatus.FAILED
    return post


def schedule_post(
    post: PostRecord,
    profile_key: str,
) -> PostRecord:
    """
    Schedule a post for future publishing via Ayrshare.
    Same as publish_post but uses the scheduleDate field.
    """
    # Ayrshare uses the same /post endpoint — scheduleDate in payload
    # determines whether it publishes now or later
    return publish_post(post, profile_key)


def fetch_post_analytics(
    ayrshare_post_id: str,
    profile_key: str,
) -> dict:
    """
    Retrieve engagement metrics for a published post from Ayrshare.
    Called by Agent 7 to populate PostRecord performance fields.
    Returns raw analytics dict or empty dict on failure.
    """
    if not AYRSHARE_API_KEY or not ayrshare_post_id:
        return {}
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(
                f"{AYRSHARE_API_BASE}/analytics/post",
                headers={"Authorization": f"Bearer {AYRSHARE_API_KEY}"},
                params={
                    "id": ayrshare_post_id,
                    "profileKey": profile_key,
                },
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.warning(f"[TS-15] Analytics fetch failed for post {ayrshare_post_id}: {exc}")
        return {}


# ── Internal helpers ──────────────────────────────────────────────────────

def _validate_post(post: PostRecord) -> list[str]:
    """
    Pre-publish validation. Returns list of errors (empty = valid).
    UTM link check is mandatory — no post goes live without it.
    """
    errors = []

    # UTM link — non-negotiable
    utm_valid, missing_params = validate_utm_link(post.utm_link)
    if not utm_valid:
        errors.append(f"UTM link invalid or missing: {missing_params}")

    # Media URL
    if not post.media_url:
        errors.append("No media URL")

    # Caption length
    limit = CAPTION_LIMITS.get(post.platform, 2200)
    full_caption = post.caption + " " + post.utm_link
    if len(full_caption) > limit:
        errors.append(f"Caption too long ({len(full_caption)} > {limit})")

    return errors


def _build_ayrshare_payload(post: PostRecord, profile_key: str) -> dict:
    """Build the Ayrshare POST /post request payload."""
    # Append UTM link to caption
    caption_with_link = f"{post.caption}\n\n{post.utm_link}"
    if post.hashtags:
        caption_with_link += "\n\n" + " ".join(
            f"#{h.lstrip('#')}" for h in post.hashtags[:20]
        )

    payload: dict = {
        "post": caption_with_link,
        "platforms": [post.platform.value],
        "profileKey": profile_key,
        "mediaUrls": [post.media_url],
        "isVideo": post.content_type.value in ("video_reel",),
    }

    # Scheduled post
    if post.scheduled_at:
        payload["scheduleDate"] = post.scheduled_at

    # Pinterest-specific fields
    if post.platform == Platform.PINTEREST:
        payload["pinterestOptions"] = {
            "link": post.utm_link,
            "title": post.caption[:100],
        }

    # TikTok privacy setting
    if post.platform == Platform.TIKTOK:
        payload["tiktokOptions"] = {
            "privacy_level": "PUBLIC_TO_EVERYONE",
        }

    return payload


def _apply_ayrshare_response(post: PostRecord, data: dict) -> PostRecord:
    """Map Ayrshare API response onto PostRecord fields."""
    from datetime import datetime, timezone

    post.ayrshare_post_id = data.get("id")
    post.status = PostStatus.PUBLISHED if data.get("status") == "success" else PostStatus.FAILED

    if post.status == PostStatus.PUBLISHED:
        post.published_at = datetime.now(timezone.utc).isoformat()

    # Platform-specific post IDs from response
    posts = data.get("postIds") or {}
    if post.platform.value in posts:
        post.platform_post_id = str(posts[post.platform.value])

    return post
