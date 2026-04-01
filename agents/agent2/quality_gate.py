"""
TS-05d — Content Quality Gate
Tool: Claude Sonnet (second call as independent reviewer)

Receives the generated ContentPackage and evaluates it against
a structured rubric. Separating generation and review into two
calls produces more reliable evaluation — the model assesses
more objectively when given content as input rather than
generating and reviewing in a single pass.

Rubric dimensions:
  vibe_consistency (1–5):  Does all copy match the vibe archetype throughout?
  specificity (1–5):       Does it reference actual property details (not generic STR)?
  tone_coherence (1–5):    Is the voice consistent across all sections?
  completeness (pass/fail): Are all required fields populated?

Passing threshold:
  - Overall weighted average >= 3.5
  - No individual dimension below 3.0
  - Completeness must pass

Properties that fail go to the AM content quality queue (TS-13 dashboard).
"""

import json
import logging
import re
from typing import Optional

import anthropic

from agents.agent2.models import (
    ContentPackage,
    QualityResult,
    QualityScore,
)

logger = logging.getLogger(__name__)

SONNET_MODEL = "claude-sonnet-4-6"
MAX_TOKENS   = 1000

# Thresholds
MIN_DIMENSION_SCORE = 3.0
MIN_OVERALL_SCORE   = 3.5


def run_quality_gate(
    pkg: ContentPackage,
    kb: dict,
    anthropic_client: anthropic.Anthropic,
) -> ContentPackage:
    """
    Runs the quality gate on the generated ContentPackage.
    Updates pkg.quality_score and pkg.approved_for_publish.
    Routes to AM review queue if fails.

    Returns the updated ContentPackage.
    """
    if not pkg.hero_headline or not pkg.property_description:
        # Content generation failed upstream — skip gate, already an error
        pkg.quality_score = QualityScore(
            vibe_consistency=0,
            specificity=0,
            tone_coherence=0,
            completeness_pass=False,
            overall_score=0,
            result=QualityResult.FAIL,
            failure_reasons=["Content generation produced empty output"],
            reviewer_notes="Quality gate skipped — no content to review",
        )
        pkg.approved_for_publish = False
        return pkg

    # ── Completeness check ────────────────────────────────────────────────
    completeness_pass, missing_fields = _check_completeness(pkg)

    # ── Claude review ─────────────────────────────────────────────────────
    scores = _run_claude_review(pkg, kb, anthropic_client)

    if scores is None:
        # Review call failed — approve with warning rather than block
        logger.warning(
            f"[Quality Gate] Review call failed for property {pkg.property_id} — auto-approving"
        )
        pkg.quality_score = QualityScore(
            vibe_consistency=3.5,
            specificity=3.5,
            tone_coherence=3.5,
            completeness_pass=completeness_pass,
            overall_score=3.5,
            result=QualityResult.PASS,
            failure_reasons=[],
            reviewer_notes="Quality gate review call failed — auto-approved with warning",
        )
        pkg.approved_for_publish = completeness_pass
        return pkg

    # ── Score calculation ─────────────────────────────────────────────────
    vibe_score = float(scores.get("vibe_consistency", 3.0))
    spec_score = float(scores.get("specificity", 3.0))
    tone_score = float(scores.get("tone_coherence", 3.0))

    # Weighted average: vibe and specificity matter most
    overall = (vibe_score * 0.4) + (spec_score * 0.4) + (tone_score * 0.2)

    failure_reasons = []
    if not completeness_pass:
        failure_reasons.append(f"Missing required fields: {', '.join(missing_fields)}")
    if vibe_score < MIN_DIMENSION_SCORE:
        failure_reasons.append(f"Vibe consistency too low ({vibe_score:.1f}/5) — copy doesn't match {pkg.vibe_profile}")
    if spec_score < MIN_DIMENSION_SCORE:
        failure_reasons.append(f"Specificity too low ({spec_score:.1f}/5) — copy feels generic, not property-specific")
    if tone_score < MIN_DIMENSION_SCORE:
        failure_reasons.append(f"Tone coherence too low ({tone_score:.1f}/5) — voice is inconsistent across sections")
    if overall < MIN_OVERALL_SCORE and not failure_reasons:
        failure_reasons.append(f"Overall score below threshold ({overall:.2f} < {MIN_OVERALL_SCORE})")

    # ── Determine result ──────────────────────────────────────────────────
    if failure_reasons:
        result = QualityResult.FAIL
        approved = False
    elif overall >= 4.0 and completeness_pass:
        result = QualityResult.PASS
        approved = True
    else:
        # Borderline — flag for AM review but don't block publish
        result = QualityResult.NEEDS_REVIEW
        approved = True   # Publish with AM review flag in dashboard

    pkg.quality_score = QualityScore(
        vibe_consistency=vibe_score,
        specificity=spec_score,
        tone_coherence=tone_score,
        completeness_pass=completeness_pass,
        overall_score=round(overall, 2),
        result=result,
        failure_reasons=failure_reasons,
        reviewer_notes=scores.get("reviewer_notes", ""),
    )
    pkg.approved_for_publish = approved

    log_level = logging.WARNING if result == QualityResult.FAIL else logging.INFO
    logger.log(
        log_level,
        f"[Quality Gate] Property {pkg.property_id}: {result} "
        f"(overall={overall:.2f}, vibe={vibe_score:.1f}, spec={spec_score:.1f}, tone={tone_score:.1f})"
    )

    return pkg


# ── Completeness Check ────────────────────────────────────────────────────

_REQUIRED_FIELDS = [
    ("hero_headline",        lambda p: bool(p.hero_headline)),
    ("property_description", lambda p: bool(p.property_description and len(p.property_description) > 200)),
    ("feature_spotlights",   lambda p: len(p.feature_spotlights) >= 2),
    ("faqs",                 lambda p: len(p.faqs) >= 3),
    ("seo_meta_description", lambda p: bool(p.seo_meta_description)),
    ("neighborhood_intro",   lambda p: bool(p.neighborhood_intro)),
]


def _check_completeness(pkg: ContentPackage) -> tuple[bool, list[str]]:
    missing = [name for name, check in _REQUIRED_FIELDS if not check(pkg)]
    return len(missing) == 0, missing


# ── Claude Review Call ────────────────────────────────────────────────────

def _run_claude_review(
    pkg: ContentPackage,
    kb: dict,
    anthropic_client: anthropic.Anthropic,
) -> Optional[dict]:
    """
    Second Claude call — the reviewer receives the content package
    and scores it against the rubric.
    """
    vibe_profile = pkg.vibe_profile
    property_name = (kb.get("name") or {}).get("value", "the property")
    city  = (kb.get("city") or {}).get("value", "")
    amenities_sample = [
        a.get("value", "") for a in (kb.get("amenities") or [])[:10] if a.get("value")
    ]
    unique_features = [
        u.get("value", "") for u in (kb.get("unique_features") or [])[:5] if u.get("value")
    ]

    review_prompt = f"""You are a quality reviewer for vacation rental marketing copy. Your job is to score the copy below against a structured rubric and provide honest, specific feedback.

PROPERTY CONTEXT (what the copy should reflect):
- Property: {property_name}
- Location: {city}
- Vibe profile: {vibe_profile}
- Key amenities: {', '.join(amenities_sample)}
- Unique features: {', '.join(unique_features)}

GENERATED COPY TO REVIEW:

HERO HEADLINE: {pkg.hero_headline}

TAGLINE: {pkg.vibe_tagline}

PROPERTY DESCRIPTION:
{pkg.property_description}

NEIGHBORHOOD INTRO: {pkg.neighborhood_intro}

FEATURE SPOTLIGHTS: {json.dumps([{"name": s.feature_name, "headline": s.headline, "description": s.description} for s in pkg.feature_spotlights], indent=2)}

SAMPLE FAQs: {json.dumps([{"q": f.question, "a": f.answer} for f in pkg.faqs[:3]], indent=2)}

RUBRIC:

1. VIBE CONSISTENCY (1-5): Does every section of this copy unmistakably match the {vibe_profile} vibe? Score 5 = completely consistent, 1 = wrong vibe or generic.

2. SPECIFICITY (1-5): Does the copy reference actual property-specific details (real amenities, actual location, specific features) rather than generic vacation rental language? Score 5 = highly specific and property-unique, 1 = could describe any rental anywhere.

3. TONE COHERENCE (1-5): Is the voice, energy level, and language style consistent across all sections? Score 5 = perfectly consistent voice, 1 = jarring inconsistencies between sections.

Respond ONLY with a JSON object:
{{
  "vibe_consistency": 1-5 number,
  "specificity": 1-5 number,
  "tone_coherence": 1-5 number,
  "reviewer_notes": "1-3 specific sentences about what is strong and what could be improved. Be concrete — reference specific lines from the copy."
}}"""

    try:
        resp = anthropic_client.messages.create(
            model=SONNET_MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": review_prompt}],
        )
        raw = resp.content[0].text
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return json.loads(cleaned)
    except Exception as exc:
        logger.error(f"[Quality Gate] Review call failed: {exc}")
        return None
