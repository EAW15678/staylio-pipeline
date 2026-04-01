"""
TS-25 — TikTok Spark Ad Regional Cluster Manager

TikTok's $500 minimum campaign budget makes per-property ad spend
structurally impractical. The solution is regional market clusters.

One Spark Ad campaign per market cluster (e.g. "Carolina Beach NC").
Shared budget $500–1,000/month per cluster.
Minimum 5 active properties before cluster activates.

Weekly nomination: Agent 6 scans organic post performance across
all properties in the cluster. The top-performing post (by composite
engagement score: completion rate 40%, shares 30%, likes 20%,
link clicks 10%) is nominated as the Spark Ad for that cluster.

Spark Ads amplify existing organic TikTok posts — the boosted post
retains all its organic engagement (likes, comments, shares accumulate
on the original post, building social proof alongside paid reach).

Targeting: TikTok travel-intent audiences, broad (NOT narrow interest
targeting — research confirms broad outperforms narrow for STR content).
Geographic radius around cluster centroid.
"""

import logging
import os
from typing import Optional

import httpx

from agents.agent6.models import (
    Platform,
    PostRecord,
    SPARK_CLUSTER_MIN_PROPERTIES,
    SparkCluster,
)

logger = logging.getLogger(__name__)

TIKTOK_ACCESS_TOKEN = os.environ.get("TIKTOK_ACCESS_TOKEN", "")
TIKTOK_APP_ID       = os.environ.get("TIKTOK_APP_ID", "")
TIKTOK_API_BASE     = "https://business-api.tiktok.com/open_api/v1.3"

# Minimum engagement score for Spark Ad nomination
MIN_SPARK_SCORE = 0.15


def evaluate_spark_nominations(
    cluster: SparkCluster,
    recent_posts: list[PostRecord],
) -> Optional[PostRecord]:
    """
    Evaluate organic posts from all properties in a cluster and
    nominate the top performer for Spark Ad amplification.

    Args:
        cluster:       The regional cluster to evaluate
        recent_posts:  Published TikTok posts from all properties in
                       the cluster from the last 7 days

    Returns:
        The nominated PostRecord or None if no posts qualify
    """
    if not cluster.is_eligible:
        logger.info(
            f"[TS-25] Cluster {cluster.cluster_id} has only "
            f"{len(cluster.property_ids)} properties — need {SPARK_CLUSTER_MIN_PROPERTIES} to activate"
        )
        return None

    tiktok_posts = [
        p for p in recent_posts
        if p.platform == Platform.TIKTOK
        and p.property_id in cluster.property_ids
        and p.platform_post_id   # Must have a live TikTok post ID
    ]

    if not tiktok_posts:
        return None

    # Score each post and pick the top performer
    scored = sorted(tiktok_posts, key=lambda p: p.engagement_score(), reverse=True)
    top_post = scored[0]

    if top_post.engagement_score() < MIN_SPARK_SCORE:
        logger.info(
            f"[TS-25] Top post score {top_post.engagement_score():.3f} is below "
            f"minimum {MIN_SPARK_SCORE} — no nomination this week for {cluster.cluster_id}"
        )
        return None

    logger.info(
        f"[TS-25] Spark Ad nomination for cluster {cluster.cluster_id}: "
        f"post {top_post.ayrshare_post_id} from property {top_post.property_id} "
        f"(score={top_post.engagement_score():.3f})"
    )
    return top_post


def activate_spark_ad(
    cluster: SparkCluster,
    nominated_post: PostRecord,
    tiktok_advertiser_id: str,
) -> SparkCluster:
    """
    Activate or swap the Spark Ad for a cluster.
    If the cluster has an existing Spark Ad, pauses it first.
    Then creates a new Spark Ad campaign boosting the nominated post.

    Returns updated SparkCluster with new campaign ID.
    """
    if not TIKTOK_ACCESS_TOKEN or not tiktok_advertiser_id:
        logger.warning("[TS-25] TikTok credentials not configured — skipping Spark Ad")
        return cluster

    # Pause existing campaign if active
    if cluster.tiktok_campaign_id and cluster.is_active:
        _pause_tiktok_campaign(cluster.tiktok_campaign_id, tiktok_advertiser_id)

    # Create new Spark Ad campaign
    campaign_id = _create_spark_campaign(
        post=nominated_post,
        cluster=cluster,
        advertiser_id=tiktok_advertiser_id,
    )

    if campaign_id:
        cluster.tiktok_campaign_id = campaign_id
        cluster.active_spark_post_id = nominated_post.platform_post_id
        cluster.active_property_id = nominated_post.property_id
        cluster.is_active = True
        nominated_post.nominated_for_spark = True
        nominated_post.spark_cluster_id = cluster.cluster_id

        logger.info(
            f"[TS-25] Spark Ad activated for cluster {cluster.cluster_id}: "
            f"campaign={campaign_id}, post={nominated_post.platform_post_id}"
        )

    return cluster


def get_cluster_for_property(
    property_id: str,
    city: Optional[str],
    state: Optional[str],
) -> Optional[str]:
    """
    Determine the regional cluster ID for a property based on location.
    Returns cluster_id string or None if no cluster defined for this market.

    Cluster ID convention: lowercase-city-state (e.g. "carolina-beach-nc")
    """
    if not city or not state:
        return None
    cluster_id = f"{city.lower().replace(' ', '-')}-{state.lower()}"
    return cluster_id


# ── TikTok API helpers ────────────────────────────────────────────────────

def _create_spark_campaign(
    post: PostRecord,
    cluster: SparkCluster,
    advertiser_id: str,
) -> Optional[str]:
    """
    Create a TikTok Spark Ad campaign for the nominated organic post.
    Returns campaign ID or None on failure.
    """
    try:
        # Monthly budget in micro-dollars (TikTok uses micro-currency)
        daily_budget_micro = int(cluster.monthly_budget_usd / 30 * 1_000_000)

        # Step 1: Create campaign
        campaign_payload = {
            "advertiser_id": advertiser_id,
            "campaign_name": f"Staylio-Spark-{cluster.cluster_id}",
            "objective_type": "TRAFFIC",
            "budget_mode": "BUDGET_MODE_DAY",
            "budget": daily_budget_micro,
            "operation_status": "ENABLE",
        }
        campaign_resp = _tiktok_post("/campaign/create/", campaign_payload)
        if not campaign_resp or campaign_resp.get("code") != 0:
            logger.error(f"[TS-25] TikTok campaign creation failed: {campaign_resp}")
            return None
        campaign_id = campaign_resp["data"]["campaign_id"]

        # Step 2: Create ad group with Spark-specific placement
        adgroup_payload = {
            "advertiser_id": advertiser_id,
            "campaign_id": campaign_id,
            "adgroup_name": f"Spark-{cluster.cluster_id}-adgroup",
            "placements": ["PLACEMENT_TIKTOK"],
            "optimization_goal": "CLICK",
            "billing_event": "CPC",
            "bid_type": "BID_TYPE_NO_BID",
            "operation_status": "ENABLE",
        }
        adgroup_resp = _tiktok_post("/adgroup/create/", adgroup_payload)
        if not adgroup_resp or adgroup_resp.get("code") != 0:
            logger.error(f"[TS-25] TikTok ad group creation failed: {adgroup_resp}")
            return None
        adgroup_id = adgroup_resp["data"]["adgroup_id"]

        # Step 3: Create Spark Ad using the organic post's TikTok post ID
        ad_payload = {
            "advertiser_id": advertiser_id,
            "adgroup_id": adgroup_id,
            "ad_name": f"Spark-{cluster.cluster_id}-ad",
            "identity_type": "TT_USER",
            "identity_id": post.platform_post_id,
            "dark_post_status": "OFF",   # Spark Ads use the organic post
            "operation_status": "ENABLE",
        }
        ad_resp = _tiktok_post("/ad/create/", ad_payload)
        if ad_resp and ad_resp.get("code") == 0:
            return str(campaign_id)

        logger.error(f"[TS-25] TikTok ad creation failed: {ad_resp}")
        return None

    except Exception as exc:
        logger.error(f"[TS-25] Spark campaign creation error: {exc}")
        return None


def _pause_tiktok_campaign(campaign_id: str, advertiser_id: str) -> None:
    """Pause an existing TikTok campaign."""
    try:
        _tiktok_post("/campaign/status/update/", {
            "advertiser_id": advertiser_id,
            "campaign_ids": [campaign_id],
            "operation_status": "DISABLE",
        })
    except Exception as exc:
        logger.warning(f"[TS-25] Could not pause TikTok campaign {campaign_id}: {exc}")


def _tiktok_post(endpoint: str, payload: dict) -> Optional[dict]:
    """Make a TikTok for Business API POST request."""
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{TIKTOK_API_BASE}{endpoint}",
                headers={
                    "Access-Token": TIKTOK_ACCESS_TOKEN,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.error(f"[TS-25] TikTok API error for {endpoint}: {exc}")
        return None
