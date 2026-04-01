"""
Agent 6 — Social Media Marketing Agent
LangGraph Node

Runs AFTER Agent 5 completes (landing page must be live before
social publishing begins — all links in posts point to the live page).

Pipeline:
  1. Load all inputs from Redis/state
  2. Build 60-day content calendar
  3. Schedule all posts via Ayrshare in batches
  4. Launch 3-phase Meta paid campaign
  5. Register property in regional TikTok Spark cluster
  6. Save all records to Supabase
  7. Update pipeline status → signals Agent 7

Ongoing (scheduled job, not pipeline — runs continuously post-launch):
  - Weekly Spark Ad nomination evaluation
  - Steady-state post queue refill (monthly)
  - Publish confirmation webhook processing
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from agents.agent6.ayrshare_publisher import publish_post, schedule_post
from agents.agent6.content_calendar import build_content_calendar
from agents.agent6.meta_ads_manager import launch_meta_campaign
from agents.agent6.models import (
    ContentCalendar,
    MetaCampaign,
    PostRecord,
    PostStatus,
    SparkCluster,
)
from agents.agent6.spark_nominator import (
    activate_spark_ad,
    evaluate_spark_nominations,
    get_cluster_for_property,
)
from core.pipeline_status import (
    PipelineStepStatus,
    get_cached_knowledge_base,
    update_pipeline_status,
)

logger = logging.getLogger(__name__)

AGENT_NUMBER = 6

# Number of posts to schedule in one Ayrshare batch call
# Ayrshare has rate limits — don't flood with 260 requests at once
BATCH_SIZE = 10


def agent6_node(state: dict) -> dict:
    """
    LangGraph node for Agent 6 — Social Media Marketing.
    Runs after Agent 5 deploys the property landing page.
    """
    property_id = state["property_id"]
    logger.info(f"[Agent 6] Starting social media setup for property {property_id}")

    update_pipeline_status(property_id, AGENT_NUMBER, PipelineStepStatus.RUNNING)

    # ── Step 1: Load all inputs ───────────────────────────────────────────
    kb = get_cached_knowledge_base(property_id) or state.get("knowledge_base", {})
    content_pkg  = _load_cache(property_id, "agent2") or state.get("content_package", {})
    visual_media = _load_cache(property_id, "agent3") or state.get("visual_media_package", {})
    page_data    = _load_cache(property_id, "landing_page") or {}

    page_url  = page_data.get("page_url") or state.get("page_url", "")
    slug      = page_data.get("slug") or kb.get("slug", "")

    if not page_url:
        error = f"Agent 6: No page URL available for property {property_id} — social publishing skipped"
        logger.warning(error)
        update_pipeline_status(property_id, AGENT_NUMBER, PipelineStepStatus.FAILED, error)
        return {**state, "errors": state.get("errors", []) + [error], "agent6_complete": False}

    vibe_profile   = kb.get("vibe_profile", "family_adventure")
    video_assets   = visual_media.get("video_assets") or []
    social_captions = content_pkg.get("social_captions") or []
    booking_url    = kb.get("booking_url", "")
    city           = _extract(kb, "city")
    state_abbr     = _extract(kb, "state")
    lat            = _extract_float(kb, "latitude")
    lng            = _extract_float(kb, "longitude")

    # Category winner photos for photo posts
    category_winners = visual_media.get("category_winners") or {}
    photo_urls = [{"url": url, "asset_url_enhanced": url} for url in category_winners.values()]

    # Get Ayrshare profile key for this client
    profile_key = _get_ayrshare_profile_key(kb.get("client_id"))
    if not profile_key:
        logger.warning(f"[Agent 6] No Ayrshare profile key for client — social accounts not connected yet")

    # ── Step 2: Build content calendar ───────────────────────────────────
    logger.info(f"[Agent 6] Building content calendar for property {property_id}")
    calendar = build_content_calendar(
        property_id=property_id,
        page_url=page_url,
        slug=slug,
        vibe_profile=vibe_profile,
        video_assets=video_assets,
        social_captions=social_captions,
        photo_urls=photo_urls,
        launch_date=datetime.now(timezone.utc).date().isoformat(),
    )

    # ── Step 3: Schedule posts via Ayrshare ───────────────────────────────
    scheduled_count = 0
    failed_count    = 0

    if profile_key and calendar.posts:
        logger.info(f"[Agent 6] Scheduling {len(calendar.posts)} posts via Ayrshare")
        for post in calendar.posts:
            # Filter out posts with no media (happens when video library is incomplete)
            if not post.media_url:
                post.status = PostStatus.CANCELLED
                continue
            updated_post = schedule_post(post, profile_key)
            if updated_post.status == PostStatus.PUBLISHED:
                scheduled_count += 1
            else:
                failed_count += 1
    else:
        # No Ayrshare profile — mark all posts as pending for when accounts connect
        logger.info(f"[Agent 6] Ayrshare profile not connected — {len(calendar.posts)} posts queued")
        for post in calendar.posts:
            post.status = PostStatus.SCHEDULED   # Will be picked up when accounts connect

    # ── Step 4: Meta paid campaign launch ────────────────────────────────
    meta_campaigns: list[MetaCampaign] = []
    hero_video_url = _get_hero_video_url(visual_media)
    review_video_urls = _get_review_video_urls(visual_media)

    if booking_url and hero_video_url:
        logger.info(f"[Agent 6] Launching Meta 3-phase campaign for property {property_id}")
        meta_campaigns = launch_meta_campaign(
            property_id=property_id,
            page_url=page_url,
            booking_url=booking_url,
            vibe_profile=vibe_profile,
            latitude=lat,
            longitude=lng,
            hero_video_url=hero_video_url,
            review_video_urls=review_video_urls,
            slug=slug,
        )

    # ── Step 5: Register in TikTok Spark cluster ──────────────────────────
    cluster_id = get_cluster_for_property(property_id, city, state_abbr)
    if cluster_id:
        _register_property_in_cluster(property_id, cluster_id, city, state_abbr)
        logger.info(f"[Agent 6] Property {property_id} registered in cluster {cluster_id}")

    # ── Step 6: Save to Supabase ──────────────────────────────────────────
    _save_calendar(calendar)
    _save_meta_campaigns(meta_campaigns)

    # ── Step 7: Update pipeline status ───────────────────────────────────
    update_pipeline_status(
        property_id, AGENT_NUMBER,
        PipelineStepStatus.COMPLETE,
        metadata={
            "posts_scheduled": scheduled_count,
            "posts_queued": len(calendar.posts) - scheduled_count - failed_count,
            "posts_failed": failed_count,
            "total_posts": len(calendar.posts),
            "meta_phases_launched": len(meta_campaigns),
            "meta_total_budget": sum(c.budget_usd for c in meta_campaigns),
            "tiktok_cluster": cluster_id,
            "profile_key_connected": bool(profile_key),
        },
    )

    logger.info(
        f"[Agent 6] Complete for property {property_id}. "
        f"Posts: {len(calendar.posts)} total ({scheduled_count} scheduled). "
        f"Meta phases: {len(meta_campaigns)}. Cluster: {cluster_id or 'none'}."
    )

    return {
        **state,
        "social_calendar": calendar.to_summary(),
        "meta_campaigns": [c.to_dict() for c in meta_campaigns],
        "spark_cluster_id": cluster_id,
        "agent6_complete": True,
        "agent7_ready": True,  # Signal Agent 7 to start analytics
    }


# ── Scheduled jobs (run outside the pipeline) ─────────────────────────────

def run_weekly_spark_evaluation(cluster: SparkCluster) -> SparkCluster:
    """
    Weekly job: evaluate organic performance across a cluster and
    potentially swap the active Spark Ad to a better performer.

    Called by a cron job every Monday, not during intake pipeline.
    """
    logger.info(f"[Agent 6] Weekly Spark evaluation for cluster {cluster.cluster_id}")

    # Load recent TikTok posts for all properties in this cluster
    recent_posts = _load_cluster_recent_posts(cluster)
    nominated = evaluate_spark_nominations(cluster, recent_posts)

    if nominated:
        # Only swap if the new nominee is meaningfully better
        if (
            not cluster.active_spark_post_id
            or nominated.platform_post_id != cluster.active_spark_post_id
        ):
            advertiser_id = os.environ.get("TIKTOK_ADVERTISER_ID", "")
            cluster = activate_spark_ad(cluster, nominated, advertiser_id)
            _save_cluster(cluster)

    return cluster


def publish_pending_posts(property_id: str, profile_key: str) -> int:
    """
    Publish posts that were queued but not yet sent to Ayrshare
    (e.g. social accounts connected after initial intake).
    Called when a PMC connects their social accounts post-intake.
    Returns count of newly published posts.
    """
    pending = _load_pending_posts(property_id)
    published = 0
    for post in pending[:BATCH_SIZE]:
        updated = publish_post(post, profile_key)
        if updated.status == PostStatus.PUBLISHED:
            published += 1
        _update_post_status(updated)
    return published


# ── Helpers ───────────────────────────────────────────────────────────────

def _extract(kb: dict, key: str) -> Optional[str]:
    f = kb.get(key)
    if isinstance(f, dict):
        return f.get("value")
    return f


def _extract_float(kb: dict, key: str) -> Optional[float]:
    val = _extract(kb, key)
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _load_cache(property_id: str, agent_label: str) -> dict:
    from core.pipeline_status import get_cached_knowledge_base
    return get_cached_knowledge_base(f"{property_id}:{agent_label}") or {}


def _get_hero_video_url(visual_media: dict) -> Optional[str]:
    for v in visual_media.get("video_assets") or []:
        if isinstance(v, dict) and v.get("video_type") == "vibe_match" and v.get("format") == "9_16":
            return v.get("r2_url")
    return None


def _get_review_video_urls(visual_media: dict) -> list[str]:
    urls = []
    for v in (visual_media.get("video_assets") or []):
        if isinstance(v, dict) and v.get("video_type", "").startswith("guest_review") and v.get("format") == "9_16":
            url = v.get("r2_url")
            if url:
                urls.append(url)
    return urls[:3]


def _get_ayrshare_profile_key(client_id: Optional[str]) -> Optional[str]:
    """Look up the Ayrshare profile key for a client from Supabase."""
    if not client_id:
        return None
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("pmc_clients")
            .select("ayrshare_profile_key")
            .eq("client_id", client_id)
            .single()
            .execute()
        )
        return (result.data or {}).get("ayrshare_profile_key")
    except Exception:
        return None


def _register_property_in_cluster(
    property_id: str,
    cluster_id: str,
    city: Optional[str],
    state_abbr: Optional[str],
) -> None:
    """Add a property to its regional Spark cluster in Supabase."""
    try:
        from core.supabase_store import get_supabase
        # Upsert the cluster record
        get_supabase().table("spark_clusters").upsert(
            {
                "cluster_id": cluster_id,
                "region_name": f"{city or ''}, {state_abbr or ''}".strip(", "),
            },
            on_conflict="cluster_id",
        ).execute()
        # Add property to cluster members
        get_supabase().table("spark_cluster_members").upsert(
            {"cluster_id": cluster_id, "property_id": property_id},
            on_conflict="cluster_id,property_id",
        ).execute()
    except Exception as exc:
        logger.error(f"[Agent 6] Failed to register property in cluster: {exc}")


def _save_calendar(calendar: ContentCalendar) -> None:
    """Save calendar summary and all posts to Supabase."""
    try:
        from core.supabase_store import get_supabase
        get_supabase().table("content_calendars").upsert(
            {
                "property_id": calendar.property_id,
                "launch_date": calendar.launch_date,
                "total_scheduled": calendar.total_scheduled,
                "summary": calendar.to_summary(),
            },
            on_conflict="property_id",
        ).execute()
        # Save posts in batches
        if calendar.posts:
            records = [p.to_dict() for p in calendar.posts[:500]]
            get_supabase().table("social_posts").upsert(
                records,
                on_conflict="property_id,platform,scheduled_at",
            ).execute()
    except Exception as exc:
        logger.error(f"[Agent 6] Failed to save calendar: {exc}")


def _save_meta_campaigns(campaigns: list[MetaCampaign]) -> None:
    if not campaigns:
        return
    try:
        from core.supabase_store import get_supabase
        get_supabase().table("meta_campaigns").upsert(
            [c.to_dict() for c in campaigns],
            on_conflict="property_id,phase",
        ).execute()
    except Exception as exc:
        logger.error(f"[Agent 6] Failed to save Meta campaigns: {exc}")


def _save_cluster(cluster: SparkCluster) -> None:
    try:
        from core.supabase_store import get_supabase
        get_supabase().table("spark_clusters").upsert(
            cluster.to_dict(),
            on_conflict="cluster_id",
        ).execute()
    except Exception as exc:
        logger.error(f"[Agent 6] Failed to save cluster: {exc}")


def _load_cluster_recent_posts(cluster: SparkCluster) -> list[PostRecord]:
    """Load recent TikTok posts for cluster properties from Supabase."""
    try:
        from core.supabase_store import get_supabase
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        result = (
            get_supabase()
            .table("social_posts")
            .select("*")
            .in_("property_id", cluster.property_ids)
            .eq("platform", "tiktok")
            .eq("status", "published")
            .gte("published_at", cutoff)
            .execute()
        )
        posts = []
        for row in (result.data or []):
            p = PostRecord(
                property_id=row["property_id"],
                platform=Platform.TIKTOK,
                content_type=row.get("content_type", "video_reel"),
                caption=row.get("caption", ""),
                hashtags=row.get("hashtags", []),
                media_url=row.get("media_url", ""),
                page_url=row.get("page_url", ""),
                utm_link=row.get("utm_link", ""),
            )
            p.ayrshare_post_id = row.get("ayrshare_post_id")
            p.platform_post_id = row.get("platform_post_id")
            p.views = row.get("views", 0)
            p.likes = row.get("likes", 0)
            p.shares = row.get("shares", 0)
            p.completion_rate = row.get("completion_rate", 0.0)
            p.link_clicks = row.get("link_clicks", 0)
            p.status = PostStatus.PUBLISHED
            posts.append(p)
        return posts
    except Exception as exc:
        logger.error(f"[Agent 6] Failed to load cluster posts: {exc}")
        return []


def _load_pending_posts(property_id: str) -> list[PostRecord]:
    """Load SCHEDULED posts not yet sent to Ayrshare."""
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("social_posts")
            .select("*")
            .eq("property_id", property_id)
            .eq("status", "scheduled")
            .is_("ayrshare_post_id", "null")
            .order("scheduled_at")
            .limit(50)
            .execute()
        )
        posts = []
        for row in (result.data or []):
            p = PostRecord(
                property_id=row["property_id"],
                platform=Platform(row["platform"]),
                content_type=row.get("content_type", "video_reel"),
                caption=row.get("caption", ""),
                hashtags=row.get("hashtags", []),
                media_url=row.get("media_url", ""),
                page_url=row.get("page_url", ""),
                utm_link=row.get("utm_link", ""),
                scheduled_at=row.get("scheduled_at"),
            )
            posts.append(p)
        return posts
    except Exception as exc:
        logger.error(f"[Agent 6] Failed to load pending posts: {exc}")
        return []


def _update_post_status(post: PostRecord) -> None:
    """Update a single post's status in Supabase after publish attempt."""
    try:
        from core.supabase_store import get_supabase
        if post.ayrshare_post_id:
            get_supabase().table("social_posts").update({
                "status": post.status,
                "ayrshare_post_id": post.ayrshare_post_id,
                "platform_post_id": post.platform_post_id,
                "published_at": post.published_at,
            }).eq("property_id", post.property_id).eq(
                "scheduled_at", post.scheduled_at
            ).execute()
    except Exception as exc:
        logger.error(f"[Agent 6] Failed to update post status: {exc}")
