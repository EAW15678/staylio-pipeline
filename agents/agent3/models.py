"""
Agent 3 — Visual Media Data Models

MediaAsset is the record written to Supabase PostgreSQL (TS-18)
for every photo in the property media library.

The full schema mirrors what was specified in TS-07b:
  property_id, asset_url_enhanced (R2), asset_url_original (R2),
  labels_enhanced (jsonb), labels_original (jsonb),
  provenance_flag (bool), composition_score, brightness, sharpness,
  safe_search_pass (bool), subject_category (enum),
  category_rank (int), hero_rank (int), social_crop_queued (bool)

VideoAsset records the 8 rendered video files per property.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SubjectCategory(str, Enum):
    """
    Subject category taxonomy for the property photo library.
    Maps from Google Cloud Vision API label combinations.
    Required before Agent 3 build per TS-07b open questions.
    """
    EXTERIOR          = "exterior"
    LIVING_ROOM       = "living_room"
    KITCHEN           = "kitchen"
    MASTER_BEDROOM    = "master_bedroom"
    STANDARD_BEDROOM  = "standard_bedroom"
    BATHROOM          = "bathroom"
    POOL_HOT_TUB      = "pool_hot_tub"
    OUTDOOR_ENTERTAINING = "outdoor_entertaining"
    VIEW              = "view"
    GAME_ENTERTAINMENT = "game_entertainment"
    LOCAL_AREA        = "local_area"
    UNCATEGORISED     = "uncategorised"


class VideoType(str, Enum):
    VIBE_MATCH        = "vibe_match"           # Video 1 — 45-60s hero
    WALK_THROUGH      = "walk_through"         # Video 2 — 15-20s silent
    GUEST_REVIEW_1    = "guest_review_1"       # Video 3
    GUEST_REVIEW_2    = "guest_review_2"       # Video 4
    LOCAL_HIGHLIGHT   = "local_highlight"      # Video 5
    FEATURE_CLOSEUP   = "feature_closeup"      # Video 6 — 12-15s
    SEASONAL          = "seasonal"             # Video 7
    GUEST_REVIEW_3    = "guest_review_3"       # Video 8


class VideoFormat(str, Enum):
    VERTICAL   = "9_16"    # TikTok, Instagram Reels, Stories
    SQUARE     = "1_1"     # Instagram feed, Facebook feed
    LANDSCAPE  = "16_9"    # Landing page embed, YouTube


# Which formats each video type gets rendered in
VIDEO_FORMAT_MATRIX: dict[VideoType, list[VideoFormat]] = {
    VideoType.VIBE_MATCH:        [VideoFormat.VERTICAL, VideoFormat.SQUARE, VideoFormat.LANDSCAPE],
    VideoType.WALK_THROUGH:      [VideoFormat.VERTICAL],
    VideoType.GUEST_REVIEW_1:    [VideoFormat.VERTICAL, VideoFormat.SQUARE],
    VideoType.GUEST_REVIEW_2:    [VideoFormat.VERTICAL, VideoFormat.SQUARE],
    VideoType.LOCAL_HIGHLIGHT:   [VideoFormat.VERTICAL, VideoFormat.SQUARE],
    VideoType.FEATURE_CLOSEUP:   [VideoFormat.VERTICAL],
    VideoType.SEASONAL:          [VideoFormat.VERTICAL, VideoFormat.SQUARE],
    VideoType.GUEST_REVIEW_3:    [VideoFormat.VERTICAL, VideoFormat.SQUARE],
}

# Target duration (seconds) per video type — used for Runway ML credit planning
VIDEO_TARGET_DURATION: dict[VideoType, float] = {
    VideoType.VIBE_MATCH:        52.5,   # midpoint of 45-60s
    VideoType.WALK_THROUGH:      17.5,   # midpoint of 15-20s
    VideoType.GUEST_REVIEW_1:    37.5,   # midpoint of 30-45s
    VideoType.GUEST_REVIEW_2:    37.5,
    VideoType.LOCAL_HIGHLIGHT:   25.0,   # midpoint of 20-30s
    VideoType.FEATURE_CLOSEUP:   13.5,   # midpoint of 12-15s
    VideoType.SEASONAL:          25.0,
    VideoType.GUEST_REVIEW_3:    37.5,
}


@dataclass
class MediaAsset:
    """
    One photo in the property media library.
    Written to Supabase after Vision API tagging completes.
    """
    property_id: str
    asset_url_original: str          # R2 URL — pre-enhancement source
    asset_url_enhanced: Optional[str] = None   # R2 URL — Claid.ai output

    # Vision API metadata
    labels_original: list[str] = field(default_factory=list)
    labels_enhanced: list[str] = field(default_factory=list)
    composition_score: float = 0.0   # 0.0–1.0 composite quality score
    brightness: float = 0.0          # 0.0–1.0
    sharpness: float = 0.0           # 0.0–1.0
    dominant_colors: list[str] = field(default_factory=list)  # hex codes

    # Governance
    safe_search_pass: bool = True
    provenance_flag: bool = False    # True if labels diverge between original and enhanced

    # Library organisation
    subject_category: SubjectCategory = SubjectCategory.UNCATEGORISED
    category_rank: int = 0           # Rank within subject category (1 = best)
    hero_rank: Optional[int] = None  # Rank as hero candidate for this vibe; None if not eligible
    social_crop_queued: bool = False  # True if this image is a category winner → gets crops

    # Source tracking
    source: str = ""                 # "intake_upload" | "airbnb_scraped" | "vrbo_scraped" etc.

    def to_db_record(self) -> dict:
        """Serialise for Supabase upsert."""
        return {
            "property_id": self.property_id,
            "asset_url_original": self.asset_url_original,
            "asset_url_enhanced": self.asset_url_enhanced,
            "labels_original": self.labels_original,
            "labels_enhanced": self.labels_enhanced,
            "composition_score": self.composition_score,
            "brightness": self.brightness,
            "sharpness": self.sharpness,
            "dominant_colors": self.dominant_colors,
            "safe_search_pass": self.safe_search_pass,
            "provenance_flag": self.provenance_flag,
            "subject_category": self.subject_category,
            "category_rank": self.category_rank,
            "hero_rank": self.hero_rank,
            "social_crop_queued": self.social_crop_queued,
            "source": self.source,
        }


@dataclass
class SocialCrop:
    """A social-format cropped version of a category-winner photo."""
    property_id: str
    source_asset_url: str        # The enhanced photo this was cropped from
    crop_url: str                # R2 URL of the cropped file
    format: str                  # "1_1" | "9_16" | "16_9"
    subject_category: str
    has_overlay: bool = False    # True if Bannerbear overlay was applied
    overlay_template_id: Optional[str] = None


@dataclass
class VideoAsset:
    """
    One rendered video file for the property.
    ~20 total per property (8 video types × avg 2.5 formats).
    """
    property_id: str
    video_type: VideoType
    format: VideoFormat
    r2_url: str                      # Cloudflare R2 storage URL
    duration_seconds: float
    has_narration: bool = False
    voice_id: Optional[str] = None   # ElevenLabs voice ID used
    script_text: Optional[str] = None
    queued_for_social: bool = True   # Ready for Agent 6 publishing

    def to_db_record(self) -> dict:
        return {
            "property_id": self.property_id,
            "video_type": self.video_type,
            "format": self.format,
            "r2_url": self.r2_url,
            "duration_seconds": self.duration_seconds,
            "has_narration": self.has_narration,
            "voice_id": self.voice_id,
            "script_text": self.script_text,
            "queued_for_social": self.queued_for_social,
        }


@dataclass
class VisualMediaPackage:
    """
    Complete output of Agent 3 for a single property.
    Consumed by Agent 5 (landing page assembly) and Agent 6 (social publishing).
    """
    property_id: str
    hero_photo_url: Optional[str] = None          # Enhanced URL of the selected hero
    hero_category: Optional[SubjectCategory] = None
    media_assets: list[MediaAsset] = field(default_factory=list)
    social_crops: list[SocialCrop] = field(default_factory=list)
    video_assets: list[VideoAsset] = field(default_factory=list)
    videos_queued: bool = False          # True when all 8 videos are generated
    review_videos_pending: bool = False  # True if < 3 guest book entries at intake

    # Category winners for each subject category (URL of best photo per category)
    category_winners: dict[str, str] = field(default_factory=dict)

    processing_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "property_id": self.property_id,
            "hero_photo_url": self.hero_photo_url,
            "hero_category": self.hero_category,
            "videos_queued": self.videos_queued,
            "review_videos_pending": self.review_videos_pending,
            "category_winners": self.category_winners,
            "video_count": len(self.video_assets),
            "crop_count": len(self.social_crops),
            "processing_errors": self.processing_errors,
        }
