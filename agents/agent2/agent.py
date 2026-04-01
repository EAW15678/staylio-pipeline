"""
Agent 2 — Content Enhancement Agent
LangGraph Node

Reads the property knowledge base from Redis (cached by Agent 1),
generates all marketing content, runs the quality gate, and writes
the content package to Supabase.

Runs in PARALLEL with Agents 3 and 4 after Agent 1 completes.
Agent 5 (Website Builder) waits for Agent 2 to complete before
assembling the landing page.

Pipeline:
  1. Load knowledge base from Redis cache (Agent 1 wrote it)
  2. Fetch SEO keywords from DataForSEO (TS-05c)
  3. Generate landing page content — Claude Sonnet (TS-05)
  4. Generate social media captions — Claude Haiku (TS-05b)
  5. Run quality gate — Claude Sonnet reviewer (TS-05d)
  6. If quality gate PASS or NEEDS_REVIEW: save and proceed
     If quality gate FAIL: flag for AM review, hold publish
  7. Save content package to Supabase
  8. Update pipeline status
"""

import json
import logging
import os
from typing import TypedDict, Optional

import anthropic
import openai

from agents.agent2.content_generator import generate_content_package
from agents.agent2.quality_gate import run_quality_gate
from agents.agent2.seo_layer import fetch_seo_keywords, check_gsc_refresh_needed
from agents.agent2.models import ContentPackage, QualityResult
from core.pipeline_status import (
    PipelineStepStatus,
    get_cached_knowledge_base,
    update_pipeline_status,
)
from agents.agent1.agent import PipelineState

logger = logging.getLogger(__name__)

AGENT_NUMBER = 2

# ── Clients (initialised lazily) ─────────────────────────────────────────
_anthropic_client: Optional[anthropic.Anthropic] = None
_openai_client: Optional[openai.OpenAI] = None


def _get_anthropic() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _anthropic_client


def _get_openai() -> Optional[openai.OpenAI]:
    global _openai_client
    if _openai_client is None:
        key = os.environ.get("OPENAI_API_KEY")
        if key:
            _openai_client = openai.OpenAI(api_key=key)
    return _openai_client


# ── LangGraph Node ────────────────────────────────────────────────────────

def agent2_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node for Agent 2 — Content Enhancement.
    Runs in parallel with Agents 3 and 4 after Agent 1 completes.
    """
    property_id = state["property_id"]
    logger.info(f"[Agent 2] Starting content enhancement for property {property_id}")

    update_pipeline_status(property_id, AGENT_NUMBER, PipelineStepStatus.RUNNING)

    # ── Step 1: Load knowledge base from Redis ────────────────────────────
    kb = get_cached_knowledge_base(property_id)
    if kb is None:
        # Redis miss — try loading from state dict directly
        kb = state.get("knowledge_base")
    if not kb:
        error = f"Agent 2: Knowledge base not found for property {property_id}"
        logger.error(error)
        update_pipeline_status(property_id, AGENT_NUMBER, PipelineStepStatus.FAILED, error)
        return {**state, "errors": state.get("errors", []) + [error], "agent2_complete": False}

    # ── Step 2: SEO keyword fetch ─────────────────────────────────────────
    city         = (kb.get("city") or {}).get("value")
    state_abbr   = (kb.get("state") or {}).get("value")
    property_type = (kb.get("property_type") or {}).get("value")
    vibe_profile = kb.get("vibe_profile")

    seo_keywords = fetch_seo_keywords(city, state_abbr, property_type, vibe_profile)
    logger.info(f"[Agent 2] SEO keywords loaded: {len(seo_keywords)} terms")

    # ── Step 3+4: Generate content package ───────────────────────────────
    anthropic_client = _get_anthropic()
    openai_client    = _get_openai()

    pkg = generate_content_package(
        kb=kb,
        seo_keywords=seo_keywords,
        anthropic_client=anthropic_client,
        openai_client=openai_client,
    )

    if pkg.generation_errors and not pkg.hero_headline:
        # Complete generation failure
        error = f"Agent 2: Content generation failed: {'; '.join(pkg.generation_errors)}"
        update_pipeline_status(property_id, AGENT_NUMBER, PipelineStepStatus.FAILED, error)
        return {**state, "errors": state.get("errors", []) + [error], "agent2_complete": False}

    # ── Step 5: Quality gate ──────────────────────────────────────────────
    pkg = run_quality_gate(pkg, kb, anthropic_client)

    # ── Step 6: Route based on quality result ────────────────────────────
    needs_human_review = (
        pkg.quality_score is not None
        and pkg.quality_score.result == QualityResult.FAIL
    )

    if needs_human_review:
        logger.warning(
            f"[Agent 2] Property {property_id} flagged for AM review. "
            f"Reasons: {pkg.quality_score.failure_reasons}"
        )
        # Flag in Supabase for the AM quality queue (TS-13)
        _flag_for_am_review(property_id, pkg)
        update_pipeline_status(
            property_id, AGENT_NUMBER,
            PipelineStepStatus.FAILED,
            error_message="Content quality gate failed — awaiting AM review",
            metadata={"quality_failure_reasons": pkg.quality_score.failure_reasons},
        )
        return {
            **state,
            "agent2_complete": False,
            "agent2_needs_review": True,
            "errors": state.get("errors", []) + [
                f"Quality gate failed: {'; '.join(pkg.quality_score.failure_reasons)}"
            ],
        }

    # ── Step 7: Save content package to Supabase ─────────────────────────
    _save_content_package(pkg)

    # ── Step 8: Update pipeline status ───────────────────────────────────
    update_pipeline_status(
        property_id, AGENT_NUMBER,
        PipelineStepStatus.COMPLETE,
        metadata={
            "quality_result": pkg.quality_score.result if pkg.quality_score else "unknown",
            "quality_score": pkg.quality_score.overall_score if pkg.quality_score else None,
            "social_captions_generated": len(pkg.social_captions),
            "generated_by": pkg.generated_by_model,
            "used_fallback_llm": "gpt-4o" in pkg.generated_by_model,
        },
    )

    logger.info(
        f"[Agent 2] Complete for property {property_id}. "
        f"Quality: {pkg.quality_score.result if pkg.quality_score else 'N/A'} "
        f"({pkg.quality_score.overall_score if pkg.quality_score else 0:.2f}). "
        f"Captions: {len(pkg.social_captions)}."
    )

    return {
        **state,
        "content_package": pkg.to_dict(),
        "agent2_complete": True,
        "agent2_needs_review": pkg.quality_score.result == QualityResult.NEEDS_REVIEW
            if pkg.quality_score else False,
    }


# ── Supabase Writes ───────────────────────────────────────────────────────

def _save_content_package(pkg: ContentPackage) -> None:
    """Save the content package to Supabase."""
    try:
        from core.supabase_store import get_supabase
        from datetime import datetime, timezone

        get_supabase().table("content_packages").upsert(
            {
                "property_id": pkg.property_id,
                "data": pkg.to_dict(),
                "approved_for_publish": pkg.approved_for_publish,
                "quality_result": pkg.quality_score.result if pkg.quality_score else None,
                "quality_score": pkg.quality_score.overall_score if pkg.quality_score else None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="property_id",
        ).execute()
    except Exception as exc:
        logger.error(f"[Agent 2] Failed to save content package to Supabase: {exc}")


def _flag_for_am_review(property_id: str, pkg: ContentPackage) -> None:
    """Write to the AM quality review queue in Supabase."""
    try:
        from core.supabase_store import get_supabase
        from datetime import datetime, timezone

        get_supabase().table("am_review_queue").upsert(
            {
                "property_id": property_id,
                "review_type": "content_quality",
                "status": "pending",
                "failure_reasons": pkg.quality_score.failure_reasons if pkg.quality_score else [],
                "reviewer_notes": pkg.quality_score.reviewer_notes if pkg.quality_score else "",
                "quality_scores": {
                    "vibe_consistency": pkg.quality_score.vibe_consistency if pkg.quality_score else None,
                    "specificity": pkg.quality_score.specificity if pkg.quality_score else None,
                    "tone_coherence": pkg.quality_score.tone_coherence if pkg.quality_score else None,
                    "overall": pkg.quality_score.overall_score if pkg.quality_score else None,
                },
                "content_snapshot": {
                    "hero_headline": pkg.hero_headline,
                    "description_excerpt": (pkg.property_description or "")[:300],
                },
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="property_id",
        ).execute()
    except Exception as exc:
        logger.error(f"[Agent 2] Failed to write AM review queue entry: {exc}")
