"""
Agent 6 — Social Media Marketing Data Models

PostRecord:     A single scheduled or published social post
CampaignRecord: A Meta or TikTok paid ad campaign for a property
ContentCalendar: The full 60-day + steady-state publishing schedule
SparkCluster:   A regional TikTok Spark Ad cluster (5+ property minimum)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Platform(str, Enum):
    INSTAGRAM = "instagram"
    TIKTOK    = "tiktok"
    FACEBOOK  = "facebook"
    PINTEREST = "pinterest"


class ContentType(str, Enum):
    VIDEO_REEL = "video_reel"    # Short-form video — TikTok, Instagram Reels, FB Reels
    FEED_PHOTO = "feed_photo"    # Static image — Instagram feed, Facebook feed
    STORY      = "story"         # 9:16 ephemeral — Instagram/Facebook Stories
    PIN        = "pin"           # Pinterest pin — photo + keyword-rich caption


class PostStatus(str, Enum):
    SCHEDULED  = "scheduled"
    PUBLISHED  = "published"
    FAILED     = "failed"
    CANCELLED  = "cancelled"


class CampaignPhase(str, Enum):
    PHASE_A_AWARENESS   = "phase_a_awareness"    # Days 1-14 in-stream overlay
    PHASE_B_INFEED      = "phase_b_infeed"        # Days 15-40 in-feed video
    PHASE_C_RETARGETING = "phase_c_retargeting"   # Days 41-60 retargeting


class CampaignStatus(str, Enum):
    PENDING  = "pending"
    ACTIVE   = "active"
    PAUSED   = "paused"
    COMPLETE = "complete"
    FAILED   = "failed"


# ── Platform cadence config ───────────────────────────────────────────────
# Posts per day per platform during 60-day launch sprint phases
SPRINT_CADENCE = {
    "weeks_1_2": {
        Platform.TIKTOK:    2,   # 2x daily — new account discovery window
        Platform.INSTAGRAM: 1,
        Platform.PINTEREST: 1,   # 7/week = 1/day
        Platform.FACEBOOK:  1,
    },
    "weeks_3_4": {
        Platform.TIKTOK:    2,
        Platform.INSTAGRAM: 1,
        Platform.PINTEREST: 1,
        Platform.FACEBOOK:  1,
    },
    "weeks_5_8": {
        Platform.TIKTOK:    1,   # Shift from volume to quality
        Platform.INSTAGRAM: 1,
        Platform.PINTEREST: 1,
        Platform.FACEBOOK:  1,
    },
}

# Steady-state cadence after Day 60
STEADY_STATE_CADENCE_PER_WEEK = {
    Platform.TIKTOK:    3,
    Platform.INSTAGRAM: 3,
    Platform.PINTEREST: 3,
    Platform.FACEBOOK:  2,
}

# Minimum gap between posts on the same platform (minutes)
MIN_POST_GAP_MINUTES = 60

# TikTok cluster minimum property count before Spark Ads activate
SPARK_CLUSTER_MIN_PROPERTIES = 5

# Meta launch campaign budget per phase
META_PHASE_BUDGETS = {
    CampaignPhase.PHASE_A_AWARENESS:   35.0,
    CampaignPhase.PHASE_B_INFEED:      70.0,
    CampaignPhase.PHASE_C_RETARGETING: 45.0,
}
META_TOTAL_LAUNCH_BUDGET = 150.0


@dataclass
class PostRecord:
    """A single social media post — scheduled or published."""
    property_id: str
    platform: Platform
    content_type: ContentType
    caption: str
    hashtags: list[str]

    # Media
    media_url: str               # R2 URL of the video or photo asset
    video_type: Optional[str] = None   # e.g. "vibe_match", "walk_through"

    # UTM-tagged link — every post must have one (TS-15 notes)
    page_url: str = ""           # The Staylio property page URL
    utm_link: str = ""           # Full UTM-tagged URL

    # Schedule
    scheduled_at: Optional[str] = None   # ISO datetime string
    published_at: Optional[str] = None
    status: PostStatus = PostStatus.SCHEDULED

    # Ayrshare response
    ayrshare_post_id: Optional[str] = None
    platform_post_id: Optional[str] = None

    # Performance (populated by Agent 7 after publish)
    views: int = 0
    likes: int = 0
    shares: int = 0
    comments: int = 0
    completion_rate: float = 0.0   # For video posts
    link_clicks: int = 0

    # Spark Ad nomination (TikTok)
    nominated_for_spark: bool = False
    spark_cluster_id: Optional[str] = None

    def engagement_score(self) -> float:
        """Composite engagement score for Spark Ad nomination ranking."""
        return (
            self.completion_rate * 0.4 +
            (min(self.shares / 100, 1.0)) * 0.3 +
            (min(self.likes / 500, 1.0)) * 0.2 +
            (min(self.link_clicks / 50, 1.0)) * 0.1
        )

    def to_dict(self) -> dict:
        return {
            "property_id": self.property_id,
            "platform": self.platform,
            "content_type": self.content_type,
            "caption": self.caption,
            "hashtags": self.hashtags,
            "media_url": self.media_url,
            "video_type": self.video_type,
            "page_url": self.page_url,
            "utm_link": self.utm_link,
            "scheduled_at": self.scheduled_at,
            "published_at": self.published_at,
            "status": self.status,
            "ayrshare_post_id": self.ayrshare_post_id,
            "platform_post_id": self.platform_post_id,
            "views": self.views,
            "likes": self.likes,
            "shares": self.shares,
            "comments": self.comments,
            "completion_rate": self.completion_rate,
            "link_clicks": self.link_clicks,
            "nominated_for_spark": self.nominated_for_spark,
            "spark_cluster_id": self.spark_cluster_id,
        }


@dataclass
class ContentCalendar:
    """
    The full publishing schedule for a property.
    60-day sprint followed by steady-state cadence.
    """
    property_id: str
    page_url: str
    slug: str
    launch_date: str              # ISO date — first post date
    posts: list[PostRecord] = field(default_factory=list)
    sprint_complete: bool = False  # True after Day 60
    total_scheduled: int = 0

    def to_summary(self) -> dict:
        return {
            "property_id": self.property_id,
            "launch_date": self.launch_date,
            "total_scheduled": self.total_scheduled,
            "sprint_complete": self.sprint_complete,
            "by_platform": {
                p.value: sum(1 for post in self.posts if post.platform == p)
                for p in Platform
            },
        }


@dataclass
class MetaCampaign:
    """A Meta (Instagram + Facebook) paid advertising campaign."""
    property_id: str
    phase: CampaignPhase
    status: CampaignStatus = CampaignStatus.PENDING
    meta_campaign_id: Optional[str] = None
    meta_adset_id: Optional[str] = None
    meta_ad_id: Optional[str] = None
    budget_usd: float = 0.0
    spend_to_date: float = 0.0
    impressions: int = 0
    clicks: int = 0
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "property_id": self.property_id,
            "phase": self.phase,
            "status": self.status,
            "meta_campaign_id": self.meta_campaign_id,
            "budget_usd": self.budget_usd,
            "spend_to_date": self.spend_to_date,
            "impressions": self.impressions,
            "clicks": self.clicks,
            "start_date": self.start_date,
            "end_date": self.end_date,
        }


@dataclass
class SparkCluster:
    """
    A regional TikTok Spark Ad cluster.
    Activated only when cluster has >= SPARK_CLUSTER_MIN_PROPERTIES.
    """
    cluster_id: str              # e.g. "carolina-beach-nc"
    region_name: str             # Human-readable: "Carolina Beach, NC"
    property_ids: list[str] = field(default_factory=list)
    monthly_budget_usd: float = 750.0
    active_spark_post_id: Optional[str] = None   # Current nominated post
    active_property_id: Optional[str] = None      # Which property's post is live
    tiktok_campaign_id: Optional[str] = None
    is_active: bool = False

    @property
    def is_eligible(self) -> bool:
        return len(self.property_ids) >= SPARK_CLUSTER_MIN_PROPERTIES

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "region_name": self.region_name,
            "property_count": len(self.property_ids),
            "is_active": self.is_active,
            "is_eligible": self.is_eligible,
            "active_spark_post_id": self.active_spark_post_id,
            "monthly_budget_usd": self.monthly_budget_usd,
        }
