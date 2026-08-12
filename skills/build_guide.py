"""
Skill: build_guide — discover nearby venues and build a local area guide.

Ports Agent 4's guide_assembler intact. Edges only:
  - Input: properties (location fields) from substrate
  - Output: local_guides (upsert on property_id) + cost_events
  - Ruling 6 semantics: noop if guide exists, failed on vendor error

Usage:
    from skills.build_guide import build_guide
    result = build_guide("a1b2c3d4-...")
"""

import json
import logging
import os
import re

import httpx

from skills.contract import (
    SkillResult, get_substrate, require_env,
    record_run, record_step, complete_step, complete_run, emit_cost,
)

logger = logging.getLogger(__name__)


def parse_search_anchors(surround_areas: str) -> list:
    """Parse owner's surround_areas into named search anchors.

    Splits on NEWLINES and SLASHES only — NOT commas.
    "Carolina Beach, NC" is ONE place; "Wilmington / Southport" is TWO.

    Returns a list of non-empty stripped strings.
    """
    if not surround_areas:
        return []
    # Split on newlines and slashes, preserve commas (city, state pairs)
    parts = re.split(r"[\n/]+", surround_areas)
    return [p.strip() for p in parts if p.strip()]


def geocode_anchors(anchors: list, api_key: str) -> list:
    """Geocode each named anchor to (lat, lng, name).

    Returns list of {"name": str, "lat": float, "lng": float} dicts.
    Anchors that fail geocoding are silently dropped.
    """
    results = []
    if not api_key:
        return results
    with httpx.Client(timeout=10) as client:
        for anchor in anchors:
            try:
                resp = client.get(
                    "https://maps.googleapis.com/maps/api/geocode/json",
                    params={"address": anchor, "key": api_key},
                )
                data = resp.json()
                if data.get("results"):
                    loc = data["results"][0]["geometry"]["location"]
                    results.append({
                        "name": anchor,
                        "lat": loc["lat"],
                        "lng": loc["lng"],
                    })
            except Exception as exc:
                logger.warning("[build_guide] Geocode failed for '%s': %s", anchor, str(exc)[:60])
    return results


def merge_venues_by_place_id(venue_lists: list) -> list:
    """Merge multiple venue lists, deduplicating by place_id.

    Each venue_list is a list of dicts with at least a 'place_id' key.
    First occurrence wins. Each venue gets an 'anchor_area' label from
    whichever list contributed it.
    """
    seen = set()
    merged = []
    for area_name, venues in venue_lists:
        for venue in venues:
            pid = venue.get("place_id") or venue.get("name", "")
            if pid in seen:
                continue
            seen.add(pid)
            venue["anchor_area"] = area_name
            merged.append(venue)
    return merged


def build_guide(
    property_id: str,
    *,
    force: bool = False,
) -> SkillResult:
    """Discover nearby venues and write local_guides.

    Returns SkillResult.ok({places_count, area_intro_length})
    or SkillResult.noop / SkillResult.failed.
    """
    try:
        sb = get_substrate()
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    # ── Check existing guide — noop if not forcing ───────────────────────
    if not force:
        existing = sb.table("local_guides").select("guide_id", count="exact").eq(
            "property_id", property_id
        ).limit(0).execute()
        if existing.count > 0:
            return SkillResult.noop(
                f"Local guide already exists. Use force=True to regenerate.",
                {"guides_existing": existing.count},
            )

    # ── Load property ────────────────────────────────────────────────────
    prop_resp = sb.table("properties").select("*").eq("id", property_id).limit(1).execute()
    if not prop_resp.data:
        return SkillResult.failed(reason=f"Property {property_id} not found", attempted=0, succeeded=0, failed_count=0)
    prop = prop_resp.data[0]

    lat = prop.get("latitude")
    lng = prop.get("longitude")
    city = prop.get("city")
    state = prop.get("state_region")
    vibe = prop.get("vibe_profile") or ""

    if not lat or not lng:
        if not city:
            return SkillResult.failed(
                reason="No coordinates or city — cannot discover venues",
                attempted=0, succeeded=0, failed_count=0,
            )

    # ── Check vendor requirements ────────────────────────────────────────
    places_key = os.environ.get("GOOGLE_PLACES_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    # ── Record the run ───────────────────────────────────────────────────
    run_id = record_run(sb, property_id, "onboard")
    step_id = record_step(sb, run_id, "build_guide")

    total_cost = 0.0

    try:
        # Build KB dict for Agent 4
        kb = {
            "property_id": property_id,
            "latitude": {"value": lat},
            "longitude": {"value": lng},
            "city": {"value": city},
            "state": {"value": state},
            "zip_code": {"value": prop.get("postal_code")},
            "vibe_profile": vibe,
            "dont_miss_picks": [],
            "neighborhood_description": {"value": None},
        }

        # Load guest_evidence for don't-miss picks (owner tips)
        ev_resp = sb.table("guest_evidence").select("written_text, verbal_text").eq(
            "property_id", property_id
        ).eq("is_guest_book", True).execute()

        import anthropic as anthropic_mod
        from agents.agent4.google_places import fetch_local_places
        from agents.agent4.guide_assembler import assemble_local_guide

        # ── Load surround_areas for additional search anchors ────────────
        ctx_resp = sb.table("owner_context").select("surround_areas").eq(
            "property_id", property_id
        ).is_("superseded_at", "null").order("version", desc=True).limit(1).execute()
        surround_areas = ""
        if ctx_resp.data:
            surround_areas = ctx_resp.data[0].get("surround_areas") or ""

        anchors = parse_search_anchors(surround_areas)
        if anchors:
            logger.info("[build_guide] Search anchors from surround_areas: %s", anchors)

        # ── Fetch places from Google — primary coordinate + anchor coordinates
        google_places = []
        venue_lists = []

        if places_key and lat and lng:
            try:
                primary_venues = fetch_local_places(float(lat), float(lng))
                venue_lists.append((city or "Primary", primary_venues))
                logger.info("[build_guide] Google Places (primary): %d venues", len(primary_venues))
            except Exception as exc:
                logger.warning("[build_guide] Google Places (primary) failed: %s", str(exc)[:120])

            # Search around each anchor coordinate
            if anchors:
                geocoded = geocode_anchors(anchors, places_key)
                for anchor in geocoded:
                    try:
                        anchor_venues = fetch_local_places(anchor["lat"], anchor["lng"])
                        venue_lists.append((anchor["name"], anchor_venues))
                        logger.info("[build_guide] Google Places (%s): %d venues", anchor["name"], len(anchor_venues))
                    except Exception as exc:
                        logger.warning("[build_guide] Google Places (%s) failed: %s", anchor["name"], str(exc)[:80])

            # Merge + dedupe by place_id
            if venue_lists:
                google_places = merge_venues_by_place_id(venue_lists)
                logger.info("[build_guide] Merged venues: %d total (deduped by place_id)", len(google_places))

            places_cost = 0.04 * max(1, len(venue_lists))
            total_cost += places_cost
            emit_cost(sb, run_id, property_id,
                      vendor="google_places", service="nearby_search",
                      units=len(google_places), unit_name="venues",
                      unit_cost=None, total_cost=round(places_cost, 4),
                      workflow_name="build_guide", generation_reason="venue_discovery")
        else:
            logger.info("[build_guide] No Google Places API key or no coordinates — skipping venue discovery")

        # Assemble guide with correct signature
        ant_client = anthropic_mod.Anthropic(api_key=anthropic_key) if anthropic_key else None
        guide_obj = assemble_local_guide(
            property_id=property_id,
            google_places=google_places,
            yelp_places=[],
            vibe_profile=vibe or "romantic_escape",
            owner_dont_miss=[],
            neighborhood_description=None,
            city=city,
            state=state,
            anthropic_client=ant_client,
        )

        # Convert LocalGuide object to JSON-safe dict
        def _to_json_safe(obj):
            """Recursively convert dataclass/object to JSON-serializable dict."""
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, list):
                return [_to_json_safe(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _to_json_safe(v) for k, v in obj.items()}
            if hasattr(obj, '__dict__'):
                return {k: _to_json_safe(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
            return str(obj)

        if guide_obj and hasattr(guide_obj, '__dict__'):
            guide_data = {
                "area_introduction": getattr(guide_obj, 'area_introduction', '') or '',
                "dont_miss_picks": _to_json_safe(getattr(guide_obj, 'dont_miss_picks', []) or []),
                "primary_recommendations": _to_json_safe(getattr(guide_obj, 'primary_recommendations', []) or []),
                "places_by_category": _to_json_safe(getattr(guide_obj, 'places_by_category', {}) or {}),
                "total_places": getattr(guide_obj, 'total_places', 0) or 0,
                "location_name": getattr(guide_obj, 'location_name', '') or f"{city or ''}, {state or ''}".strip(", "),
            }
        else:
            guide_data = {
                "area_introduction": "",
                "dont_miss_picks": [],
                "primary_recommendations": [],
                "places_by_category": {},
                "total_places": 0,
                "location_name": f"{city or ''}, {state or ''}".strip(", "),
            }

        # LLM area intro cost
        if guide_data.get("area_introduction") and anthropic_key:
            intro_cost = 0.005
            total_cost += intro_cost
            emit_cost(sb, run_id, property_id,
                      vendor="anthropic", service="claude_area_intro",
                      units=1, unit_name="calls",
                      unit_cost=0.005, total_cost=round(intro_cost, 4),
                      workflow_name="build_guide", generation_reason="area_introduction")

    except Exception as exc:
        error_str = str(exc)
        if "credit balance" in error_str:
            from skills.contract import escalate_billing
            escalate_billing(sb, property_id, "anthropic", error_str)
            complete_step(sb, step_id, status="failed", error_message=f"Billing: {error_str[:200]}")
            complete_run(sb, run_id, status="failed")
            return SkillResult.failed(
                reason=f"Billing error: {error_str[:200]}",
                attempted=1, succeeded=0, failed_count=1,
                error_class="billing", human_required=True,
            )
        complete_step(sb, step_id, status="failed", error_message=error_str[:200])
        complete_run(sb, run_id, status="failed")
        return SkillResult.failed(
            reason=f"Guide assembly failed: {error_str[:200]}",
            attempted=1, succeeded=0, failed_count=1, error_class="vendor",
        )

    # ── Write to local_guides ────────────────────────────────────────────
    location_name = guide_data.get("location_name") or f"{city or ''}, {state or ''}".strip(", ")
    total_places = guide_data.get("total_places") or len(places)

    sb.table("local_guides").upsert({
        "property_id": property_id,
        "area_introduction": guide_data.get("area_introduction") or "",
        "dont_miss_picks": guide_data.get("dont_miss_picks") or [],
        "primary_recommendations": guide_data.get("primary_recommendations") or [],
        "places_by_category": guide_data.get("places_by_category") or {},
        "total_places": total_places,
        "location_name": location_name,
    }, on_conflict="property_id").execute()

    # ── Complete ─────────────────────────────────────────────────────────
    complete_step(sb, step_id, status="complete", metadata={
        "total_places": total_places,
        "area_intro_length": len(guide_data.get("area_introduction") or ""),
        "cost_usd": round(total_cost, 4),
    })
    complete_run(sb, run_id, status="complete")

    return SkillResult.ok({
        "total_places": total_places,
        "area_intro_length": len(guide_data.get("area_introduction") or ""),
        "location_name": location_name,
        "cost_usd": round(total_cost, 4),
        "run_id": run_id,
    })
