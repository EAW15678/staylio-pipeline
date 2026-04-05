"""
Agent 4 — Local Discovery Agent
LangGraph Node

Runs in PARALLEL with Agents 2 and 3 after Agent 1 completes.
Agent 5 (Website Builder) waits for Agent 4 to complete before
assembling the landing page.

Pipeline:
  1. Load knowledge base from Redis (Agent 1 cache)
  2. Extract property coordinates and location from KB
  3. Run Google Places API fetch (TS-10) + Yelp Fusion fetch (TS-11) in parallel
  4. Merge, deduplicate, and rank results
  5. Apply vibe-profile filter for primary recommendations
  6. Layer in owner Don't Miss picks from intake
  7. Generate area introduction via Claude Haiku
  8. Save LocalGuide to Supabase
  9. Cache in Redis for Agent 5
  10. Update pipeline status
"""

import asyncio
import concurrent.futures
import logging
import os
from typing import Optional

import anthropic

from agents.agent4.google_places import fetch_local_places, RADIUS_RURAL
from agents.agent4.guide_assembler import assemble_local_guide
from agents.agent4.models import LocalGuide
from agents.agent4.yelp_fusion import fetch_yelp_places
from core.pipeline_status import (
    PipelineStepStatus,
    cache_knowledge_base,
    get_cached_knowledge_base,
    update_pipeline_status,
)

logger = logging.getLogger(__name__)

AGENT_NUMBER = 4

_anthropic_client: Optional[anthropic.Anthropic] = None


def _get_anthropic() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _anthropic_client


def agent4_node(state: dict) -> dict:
    """
    LangGraph node for Agent 4 — Local Discovery.
    Runs in parallel with Agents 2 and 3.
    """
    property_id = state["property_id"]
    logger.info(f"[Agent 4] Starting local discovery for property {property_id}")

    update_pipeline_status(property_id, AGENT_NUMBER, PipelineStepStatus.RUNNING)

    # ── Step 1: Load knowledge base ───────────────────────────────────────
    kb = get_cached_knowledge_base(property_id) or state.get("knowledge_base", {})
    if not kb:
        error = f"Agent 4: Knowledge base not found for property {property_id}"
        update_pipeline_status(property_id, AGENT_NUMBER, PipelineStepStatus.FAILED, error)
        return {**state, "errors": state.get("errors", []) + [error], "agent4_complete": False}

    # ── Step 2: Extract coordinates and location ──────────────────────────
    lat = _extract_float(kb, "latitude")
    lng = _extract_float(kb, "longitude")
    city  = _extract_str(kb, "city")
    state_abbr = _extract_str(kb, "state")
    zip_code   = _extract_str(kb, "zip_code")
    vibe_profile = kb.get("vibe_profile") or "family_adventure"
    dont_miss    = kb.get("dont_miss_picks") or []
    neighborhood_desc = _extract_str(kb, "neighborhood_description")

    if not lat or not lng:
        # Try to geocode from city/state — graceful degradation
        lat, lng = _geocode_fallback(city, state_abbr, zip_code)

    if not lat or not lng:
        error = f"Agent 4: No coordinates available for property {property_id} — local guide skipped"
        logger.warning(error)
        update_pipeline_status(
            property_id, AGENT_NUMBER,
            PipelineStepStatus.COMPLETE,
            metadata={"warning": "No coordinates — local guide not generated"},
        )
        return {**state, "agent4_complete": True, "local_guide": {}}

    # ── Step 3: Google Places fetch (Yelp removed — too expensive) ────────
    logger.info(f"[Agent 4] Fetching local places for {city}, {state_abbr} ({lat:.4f}, {lng:.4f})")

    google_places_result = []
    yelp_places_result   = []  # Kept for guide_assembler compatibility — always empty

    try:
        google_places_result = fetch_local_places(lat, lng, RADIUS_RURAL)
    except Exception as exc:
        logger.error(f"[Agent 4] Google Places fetch failed: {exc}")

    if not google_places_result:
        error = f"Agent 4: Google Places fetch failed for property {property_id}"
        update_pipeline_status(property_id, AGENT_NUMBER, PipelineStepStatus.FAILED, error)
        return {**state, "errors": state.get("errors", []) + [error], "agent4_complete": False}

    # ── Step 4-7: Assemble local guide ────────────────────────────────────
    guide = assemble_local_guide(
        property_id=property_id,
        google_places=google_places_result,
        yelp_places=yelp_places_result,
        vibe_profile=vibe_profile,
        owner_dont_miss=dont_miss,
        neighborhood_description=neighborhood_desc,
        city=city,
        state=state_abbr,
        anthropic_client=_get_anthropic(),
    )

    # ── Step 8: Save to Supabase ──────────────────────────────────────────
    _save_local_guide(guide)

    # ── Step 9: Cache in Redis for Agent 5 ───────────────────────────────
    guide_dict = guide.to_dict()
    cache_knowledge_base(f"{property_id}:local_guide", guide_dict, ttl_seconds=86400)

    # ── Step 10: Update pipeline status ──────────────────────────────────
    update_pipeline_status(
        property_id, AGENT_NUMBER,
        PipelineStepStatus.COMPLETE,
        metadata={
            "total_places": guide.total_places,
            "primary_recommendations": len(guide.primary_recommendations),
            "dont_miss_picks": len(guide.dont_miss_picks),
            "sources": guide.sources_used,
            "location": f"{city}, {state_abbr}",
        },
    )

    logger.info(
        f"[Agent 4] Complete for property {property_id}. "
        f"Places: {guide.total_places}, Primary: {len(guide.primary_recommendations)}, "
        f"Don't Miss: {len(guide.dont_miss_picks)}"
    )

    return {
        **state,
        "local_guide": guide_dict,
        "agent4_complete": True,
    }


# ── Helpers ───────────────────────────────────────────────────────────────

def _extract_float(kb: dict, key: str) -> Optional[float]:
    field = kb.get(key)
    if field is None:
        return None
    val = field.get("value") if isinstance(field, dict) else field
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _extract_str(kb: dict, key: str) -> Optional[str]:
    field = kb.get(key)
    if field is None:
        return None
    val = field.get("value") if isinstance(field, dict) else field
    return str(val).strip() if val else None


def _geocode_fallback(
    city: Optional[str],
    state: Optional[str],
    zip_code: Optional[str],
) -> tuple[Optional[float], Optional[float]]:
    """
    Simple geocoding fallback using Google Geocoding API.
    Used when the property knowledge base lacks explicit coordinates.
    Returns (lat, lng) or (None, None).
    """
    api_key = os.environ.get("GOOGLE_PLACES_API_KEY", "")
    if not api_key:
        return None, None

    address = ", ".join(filter(None, [zip_code, city, state]))
    if not address:
        return None, None

    try:
        import httpx
        with httpx.Client(timeout=10) as client:
            resp = client.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params={"address": address, "key": api_key},
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if results:
                loc = results[0]["geometry"]["location"]
                return loc["lat"], loc["lng"]
    except Exception as exc:
        logger.warning(f"[Agent 4] Geocoding fallback failed for '{address}': {exc}")

    return None, None


def _save_local_guide(guide: LocalGuide) -> None:
    """Save the local guide to Supabase."""
    try:
        from core.supabase_store import get_supabase
        from datetime import datetime, timezone
        get_supabase().table("property_local_guides").upsert(
            {
                "property_id": guide.property_id,
                "data": guide.to_dict(),
                "total_places": guide.total_places,
                "location_name": guide.location_name,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="property_id",
        ).execute()
    except Exception as exc:
        logger.error(f"[Agent 4] Failed to save local guide to Supabase: {exc}")
