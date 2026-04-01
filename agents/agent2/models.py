"""
Agent 2 Content Package — Data Models

ContentPackage is the output contract for Agent 2.
Agent 5 (Website Builder) reads it to assemble the landing page.
Agent 6 (Social Media) reads the social_captions section.
Agent 7 (Analytics) reads the seo fields for reporting.

All fields are Optional — Agent 2 may produce partial packages
if some generation steps fail gracefully.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class QualityResult(str, Enum):
    PASS       = "pass"
    FAIL       = "fail"
    NEEDS_REVIEW = "needs_review"   # Score borderline — AM queue


@dataclass
class FAQ:
    question: str
    answer: str


@dataclass
class FeatureSpotlight:
    """A specific property feature with marketing copy."""
    feature_name: str      # e.g. "Rooftop Pool", "Chef's Kitchen"
    headline: str          # Short punchy headline, 5-10 words
    description: str       # 2-4 sentences of compelling copy


@dataclass
class SocialCaption:
    """A single social media caption with platform metadata."""
    platform: str          # instagram | tiktok | facebook | pinterest
    caption: str
    hashtags: list[str]
    content_type: str      # photo | reel | story | pin | video


@dataclass
class QualityScore:
    """Output of the quality gate (TS-05d)."""
    vibe_consistency: float        # 1.0–5.0
    specificity: float             # 1.0–5.0
    tone_coherence: float          # 1.0–5.0
    completeness_pass: bool        # All required fields populated
    overall_score: float           # Weighted average
    result: QualityResult
    failure_reasons: list[str]     # Populated when result != PASS
    reviewer_notes: str            # Claude's plain-English assessment


@dataclass
class ContentPackage:
    """
    The complete content output for a single property.
    Built by Agent 2, consumed by Agents 5 and 6.
    Stored in Supabase as JSONB alongside the knowledge base.
    """

    property_id: str
    vibe_profile: str

    # ── Landing Page Content (Sonnet-generated) ───────────────────────────
    hero_headline: Optional[str] = None        # 6-12 word hero headline
    vibe_tagline: Optional[str] = None         # 5-8 word sub-headline tagline

    # Full property description — 3-5 paragraphs, landing page quality
    property_description: Optional[str] = None

    # 3-5 spotlight callouts for the most compelling property features
    feature_spotlights: list[FeatureSpotlight] = field(default_factory=list)

    # Top 6-8 amenities with short marketing copy (1-2 sentences each)
    amenity_highlights: dict[str, str] = field(default_factory=dict)

    # 3-5 sentence neighborhood/area introduction in vibe voice
    neighborhood_intro: Optional[str] = None

    # 5-7 FAQs relevant to the vibe and property type
    faqs: list[FAQ] = field(default_factory=list)

    # Refined owner story (if owner provided one) — preserves voice, improves flow
    owner_story_refined: Optional[str] = None

    # ── SEO Fields ────────────────────────────────────────────────────────
    # 150-160 char meta description for search results
    seo_meta_description: Optional[str] = None

    # Title tag for the property page
    seo_page_title: Optional[str] = None

    # Target keywords injected from DataForSEO (TS-05c)
    seo_target_keywords: list[str] = field(default_factory=list)

    # Alt text by photo category for accessibility + image SEO
    seo_alt_texts: dict[str, str] = field(default_factory=dict)

    # ── Social Media Captions (Haiku-generated) ───────────────────────────
    # 60-day content sprint: captions for each of the 8 property videos
    # plus photo post captions organised by category
    social_captions: list[SocialCaption] = field(default_factory=list)

    # ── Quality Gate Result ───────────────────────────────────────────────
    quality_score: Optional[QualityScore] = None
    approved_for_publish: bool = False

    # ── Generation Metadata ───────────────────────────────────────────────
    generated_by_model: str = ""    # claude-sonnet-4-6 | gpt-4o (fallback)
    generation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        def _faq(f): return {"question": f.question, "answer": f.answer}
        def _spot(s): return {
            "feature_name": s.feature_name,
            "headline": s.headline,
            "description": s.description,
        }
        def _cap(c): return {
            "platform": c.platform,
            "caption": c.caption,
            "hashtags": c.hashtags,
            "content_type": c.content_type,
        }
        def _qs(q): return {
            "vibe_consistency": q.vibe_consistency,
            "specificity": q.specificity,
            "tone_coherence": q.tone_coherence,
            "completeness_pass": q.completeness_pass,
            "overall_score": q.overall_score,
            "result": q.result,
            "failure_reasons": q.failure_reasons,
            "reviewer_notes": q.reviewer_notes,
        } if q else None

        return {
            "property_id": self.property_id,
            "vibe_profile": self.vibe_profile,
            "hero_headline": self.hero_headline,
            "vibe_tagline": self.vibe_tagline,
            "property_description": self.property_description,
            "feature_spotlights": [_spot(s) for s in self.feature_spotlights],
            "amenity_highlights": self.amenity_highlights,
            "neighborhood_intro": self.neighborhood_intro,
            "faqs": [_faq(f) for f in self.faqs],
            "owner_story_refined": self.owner_story_refined,
            "seo_meta_description": self.seo_meta_description,
            "seo_page_title": self.seo_page_title,
            "seo_target_keywords": self.seo_target_keywords,
            "seo_alt_texts": self.seo_alt_texts,
            "social_captions": [_cap(c) for c in self.social_captions],
            "quality_score": _qs(self.quality_score),
            "approved_for_publish": self.approved_for_publish,
            "generated_by_model": self.generated_by_model,
            "generation_errors": self.generation_errors,
        }
