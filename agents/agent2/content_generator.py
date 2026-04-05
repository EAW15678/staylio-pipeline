"""
TS-05 / TS-05b / TS-06 — Content Generator

Three-tier LLM routing:
  Tier 1 (Sonnet):  Landing page copy, feature spotlights, FAQs, neighborhood intro,
                    owner story refinement, SEO fields — anything where tone precision
                    and length matter.
  Tier 2 (Haiku):   Social media captions, alt texts — high-volume, shorter outputs
                    where speed matters more than nuance.
  Tier 3 (GPT-4o):  Fallback for Tier 1 when Claude API is rate-limited or down.
                    Uses the same prompts — output varies slightly but follows
                    the same vibe system.
"""

import json
import logging
import re
import time
from typing import Optional

import anthropic
import openai

from agents.agent2.models import (
    FAQ,
    ContentPackage,
    FeatureSpotlight,
    SocialCaption,
)
from agents.agent2.prompts.vibe_templates import (
    get_system_prompt,
    get_user_prompt,
)

logger = logging.getLogger(__name__)

# ── LLM Models ────────────────────────────────────────────────────────────
SONNET_MODEL  = "claude-sonnet-4-6"
HAIKU_MODEL   = "claude-haiku-4-5-20251001"
GPT4O_MODEL   = "gpt-4o"

# Fallback: switch to GPT-4o after this many consecutive Claude failures
FALLBACK_THRESHOLD = 2

# Token budgets (controls cost)
SONNET_MAX_TOKENS = 4000   # Full content package
HAIKU_MAX_TOKENS  = 4000   # Social captions batch
GPT4O_MAX_TOKENS  = 4000


def generate_content_package(
    kb: dict,
    seo_keywords: list[str],
    anthropic_client: anthropic.Anthropic,
    openai_client: Optional[openai.OpenAI] = None,
) -> ContentPackage:
    """
    Entry point for content generation.
    Runs Sonnet for landing page copy, Haiku for social captions.
    Falls back to GPT-4o if Claude is unavailable.

    Args:
        kb:               Property knowledge base dict (from Agent 1 / Redis)
        seo_keywords:     Keywords from DataForSEO (TS-05c)
        anthropic_client: Authenticated Anthropic client
        openai_client:    Authenticated OpenAI client (TS-06 fallback)

    Returns:
        ContentPackage with all generated fields populated
    """
    property_id  = kb.get("property_id", "")
    vibe_profile = kb.get("vibe_profile") or ""

    pkg = ContentPackage(
        property_id=property_id,
        vibe_profile=vibe_profile,
    )

    if not vibe_profile:
        pkg.generation_errors.append("No vibe profile found in knowledge base — cannot generate content")
        return pkg

    # ── Tier 1: Claude Sonnet — landing page content ──────────────────────
    logger.info(f"[Agent 2] Generating landing page content (Sonnet) for property {property_id}")

    sonnet_result, llm_errors = _generate_sonnet(
        kb, seo_keywords, anthropic_client, openai_client
    )
    pkg.generation_errors.extend(llm_errors)

    if sonnet_result:
        pkg = _apply_sonnet_result(pkg, sonnet_result)
        pkg.generated_by_model = SONNET_MODEL
        pkg.seo_target_keywords = seo_keywords[:15]
    else:
        pkg.generation_errors.append("All LLMs failed for landing page content generation")
        return pkg

    # ── Tier 2: Claude Haiku — social media captions ─────────────────────
    logger.info(f"[Agent 2] Generating social captions (Haiku) for property {property_id}")

    captions = _generate_social_captions(kb, pkg, anthropic_client)
    pkg.social_captions = captions

    return pkg


def generate_caption_batch(
    kb: dict,
    content_pkg: ContentPackage,
    anthropic_client: anthropic.Anthropic,
    batch_type: str = "video_launch",
) -> list[SocialCaption]:
    """
    Standalone caption generator for Agent 6's ongoing publishing needs.
    Called when new captions are needed beyond the initial batch.
    Uses Haiku for cost efficiency.
    """
    return _generate_social_captions(kb, content_pkg, anthropic_client, batch_type)


# ── Sonnet Generation ─────────────────────────────────────────────────────

def _generate_sonnet(
    kb: dict,
    keywords: list[str],
    anthropic_client: anthropic.Anthropic,
    openai_client: Optional[openai.OpenAI],
) -> tuple[Optional[dict], list[str]]:
    """
    Try Claude Sonnet first, fall back to GPT-4o on failure.
    Returns (parsed JSON dict or None, list of error detail strings).
    """
    vibe = kb.get("vibe_profile", "")
    system = get_system_prompt(vibe)
    user   = get_user_prompt(vibe, kb, keywords)
    errors: list[str] = []

    # Attempt 1: Claude Sonnet
    for attempt in range(FALLBACK_THRESHOLD):
        try:
            resp = anthropic_client.messages.create(
                model=SONNET_MODEL,
                max_tokens=SONNET_MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            raw = resp.content[0].text
            return _parse_json_response(raw), errors
        except anthropic.RateLimitError as exc:
            msg = f"Claude RateLimitError on attempt {attempt + 1}: {exc}"
            logger.warning(f"[Agent 2] {msg}")
            errors.append(msg)
            time.sleep(2 ** attempt)   # exponential back-off
        except anthropic.APIStatusError as exc:
            msg = f"Claude APIStatusError on attempt {attempt + 1}: status={exc.status_code} body={exc.message}"
            logger.warning(f"[Agent 2] {msg}")
            errors.append(msg)
            if exc.status_code < 500:
                break   # 4xx — bad request, don't retry
            time.sleep(2 ** attempt)
        except Exception as exc:
            msg = f"Claude unexpected error on attempt {attempt + 1}: {type(exc).__name__}: {exc}"
            logger.error(f"[Agent 2] {msg}")
            errors.append(msg)
            break

    # Attempt 2: GPT-4o fallback (TS-06)
    if openai_client:
        logger.warning("[Agent 2] Claude failed — falling back to GPT-4o (TS-06)")
        result, gpt_error = _generate_gpt4o(system, user, openai_client)
        if gpt_error:
            errors.append(gpt_error)
        return result, errors

    return None, errors


def _generate_gpt4o(
    system: str,
    user: str,
    openai_client: openai.OpenAI,
) -> tuple[Optional[dict], Optional[str]]:
    """GPT-4o fallback for landing page content generation (TS-06).
    Returns (parsed JSON dict or None, error string or None).
    """
    try:
        resp = openai_client.chat.completions.create(
            model=GPT4O_MODEL,
            max_tokens=GPT4O_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        return _parse_json_response(raw), None
    except Exception as exc:
        msg = f"GPT-4o fallback failed: {type(exc).__name__}: {exc}"
        logger.error(f"[Agent 2] {msg}")
        return None, msg


# ── Haiku Social Caption Generation ──────────────────────────────────────

_SOCIAL_CAPTION_SYSTEM = """You are a social media copywriter who creates captions for vacation rental property accounts on Instagram, TikTok, Facebook, and Pinterest. Your captions are platform-appropriate, vibe-consistent, and always include a clear call to action linking to the property page. Every caption must feel like it was written specifically for this property — not a template fill-in.

Caption requirements by platform:
- Instagram: 100-150 words, 5-10 relevant hashtags, emotional lead line, CTA at end
- TikTok: 50-80 words, 3-5 hashtags, hook in first line (makes viewer want to watch), destination/experience focused
- Facebook: 80-120 words, 2-3 hashtags, slightly more descriptive than Instagram, include a question to drive comments
- Pinterest: 40-60 words, 8-12 keyword-rich hashtags, describe what makes this property/location bookable, search-optimised language"""

def _generate_social_captions(
    kb: dict,
    pkg: ContentPackage,
    anthropic_client: anthropic.Anthropic,
    batch_type: str = "video_launch",
) -> list[SocialCaption]:
    """
    Generate the 60-day launch sprint caption library using Haiku.
    Produces captions for all 8 videos plus an initial set of photo post captions.
    """
    vibe = kb.get("vibe_profile", "")
    name = (kb.get("name") or {}).get("value", "the property")
    city = (kb.get("city") or {}).get("value", "")
    headline = pkg.hero_headline or ""
    tagline  = pkg.vibe_tagline or ""

    # Video types that need captions
    video_types = [
        ("1", "hero/vibe match video", "the property's vibe and atmosphere"),
        ("2", "silent walk-through", "a visual tour of the spaces"),
        ("3", "guest book story", "an authentic handwritten guest review"),
        ("4", "guest book story", "a second authentic guest review"),
        ("5", "local area highlight", "a local recommendation near the property"),
        ("6", "feature close-up", "the property's most compelling single feature"),
        ("7", "seasonal/occasion content", "the property's peak season experience"),
        ("8", "guest book story", "a third authentic guest review"),
    ]

    prompt = f"""Generate social media captions for a vacation rental property for its 60-day launch content sprint.

Property: {name}
Location: {city}
Vibe: {vibe}
Hero headline: {headline}
Tagline: {tagline}
Standout features: {', '.join([u.get('value', '') for u in (kb.get('unique_features') or [])[:6] if u.get('value')])}
Top amenities: {', '.join([a.get('value', '') for a in (kb.get('amenities') or [])[:10] if a.get('value')])}

Generate captions for all 8 property videos (one set of 3 platform captions per video: Instagram, TikTok, and Pinterest). Each video type and subject is listed below.

Videos:
{chr(10).join(f"Video {num}: {video_type} — shows {subject}" for num, video_type, subject in video_types)}

Respond ONLY with a JSON array. Each element covers one platform caption for one video:
[
  {{
    "video_number": "1",
    "platform": "instagram",
    "caption": "...",
    "hashtags": ["hashtag1", "hashtag2"],
    "content_type": "reel"
  }},
  ...
]

Produce 3 captions per video (Instagram, TikTok, Pinterest) = 24 caption objects total.
All captions must reference this specific property and feel distinct from each other."""

    try:
        resp = anthropic_client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=HAIKU_MAX_TOKENS,
            system=_SOCIAL_CAPTION_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text
        items = _parse_json_response(raw)
        if items is None:
            logger.warning("[Agent 2] Haiku caption JSON parse returned None — returning empty captions")
            return []
        if not isinstance(items, list):
            items = items.get("captions") or items.get("items") or []

        captions = []
        for item in items:
            captions.append(SocialCaption(
                platform=item.get("platform", "instagram"),
                caption=item.get("caption", ""),
                hashtags=item.get("hashtags", []),
                content_type=item.get("content_type", "video"),
            ))
        logger.info(f"[Agent 2] Generated {len(captions)} social captions (Haiku)")
        return captions

    except Exception as exc:
        logger.error(f"[Agent 2] Haiku social caption generation failed: {exc}")
        return []


# ── Response Parsing ──────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> Optional[dict | list]:
    """Strip markdown fences and parse JSON. Returns None on failure."""
    if not raw:
        return None
    # Strip leading/trailing whitespace, then remove ```json / ``` fences
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Remove opening fence — handles ```json, ```JSON, ``` with or without newline
        cleaned = re.sub(r"^```[a-zA-Z]*\s*\n?", "", cleaned)
    if cleaned.endswith("```"):
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error(f"[Agent 2] JSON parse failed: {exc}\nRaw: {raw[:200]}")
        logger.warning(f"[Agent 2] JSON parse failed. Raw (first 500 chars): {raw[:500]!r}")
        return None


# ── Apply Sonnet Result to Package ────────────────────────────────────────

def _apply_sonnet_result(pkg: ContentPackage, result: dict) -> ContentPackage:
    """Map the Sonnet JSON response onto the ContentPackage fields."""
    pkg.hero_headline       = result.get("hero_headline")
    pkg.vibe_tagline        = result.get("vibe_tagline")
    pkg.property_description = result.get("property_description")
    pkg.neighborhood_intro  = result.get("neighborhood_intro")
    pkg.owner_story_refined = result.get("owner_story_refined")
    pkg.seo_meta_description = result.get("seo_meta_description")
    pkg.seo_page_title      = result.get("seo_page_title")
    pkg.seo_alt_texts       = result.get("seo_alt_texts") or {}

    # Feature spotlights
    for s in result.get("feature_spotlights") or []:
        if isinstance(s, dict):
            pkg.feature_spotlights.append(FeatureSpotlight(
                feature_name=s.get("feature_name", ""),
                headline=s.get("headline", ""),
                description=s.get("description", ""),
            ))

    # Amenity highlights
    pkg.amenity_highlights = result.get("amenity_highlights") or {}

    # FAQs
    for f in result.get("faqs") or []:
        if isinstance(f, dict):
            pkg.faqs.append(FAQ(
                question=f.get("question", ""),
                answer=f.get("answer", ""),
            ))

    return pkg
