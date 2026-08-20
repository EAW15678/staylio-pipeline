"""
Skill: conceive — generate a creative concept (premise) for a property's hero video.

The premise is the seed of the entire video. It is written by the model,
not templated, because templating from vibe and property name produces
the sameness the product exists to fix. Two properties sharing a vibe
diverge over time — this is where that divergence begins.

Ruled by Erick, 2026-08-19: the model writes the premise, grounded in
the owner's own words, real guest-book text, and the property's actual
character. The truth line applies — the premise must not assert an
amenity, feature or setting the property does not have.

Usage:
    from skills.conceive import conceive
    result = conceive("property_id")
"""

import json
import logging
import uuid
from datetime import datetime, timezone

from skills.contract import (
    SkillResult, get_substrate, require_env,
    record_run, record_step, complete_step, complete_run, emit_cost,
)

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-6"


def conceive(
    property_id: str,
    *,
    force: bool = False,
) -> SkillResult:
    """Generate a creative concept for a property's hero video.

    Reads owner_context, properties, guest_evidence, local_guides.
    Writes one row to concepts with status='active'.

    Returns SkillResult.ok({concept_id, title, premise})
    """
    try:
        sb = get_substrate()
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    # ── Idempotency: check for existing active concept ─────────────────
    if not force:
        existing = sb.table("concepts").select("concept_id", count="exact").eq(
            "property_id", property_id
        ).eq("status", "active").is_("superseded_at", "null").limit(0).execute()
        if existing.count > 0:
            return SkillResult.noop(
                f"Active concept already exists for property {property_id[:12]}.",
                {"concepts_existing": existing.count},
            )

    try:
        api_key = require_env("ANTHROPIC_API_KEY", "Claude for concept generation")
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    # ── Load context ───────────────────────────────────────────────────
    prop = sb.table("properties").select(
        "name, city, state_region, vibe_profile, amenities"
    ).eq("id", property_id).limit(1).execute()
    prop_data = prop.data[0] if prop.data else {}

    ctx_resp = sb.table("owner_context").select(
        "owner_story, wow_factor, hidden_gems, guest_love"
    ).eq("property_id", property_id).is_(
        "superseded_at", "null"
    ).order("version", desc=True).limit(1).execute()
    owner_ctx = ctx_resp.data[0] if ctx_resp.data else {}

    ev_resp = sb.table("guest_evidence").select(
        "written_text, verbal_text, reviewer_name"
    ).eq("property_id", property_id).eq("is_guest_book", True).limit(5).execute()
    guest_evidence = ev_resp.data or []

    guide_resp = sb.table("local_guides").select(
        "area_introduction, location_name"
    ).eq("property_id", property_id).limit(1).execute()
    guide = guide_resp.data[0] if guide_resp.data else {}

    # ── Build prompt ───────────────────────────────────────────────────
    name = prop_data.get("name") or "the property"
    city = prop_data.get("city") or ""
    state = prop_data.get("state_region") or ""
    vibe = prop_data.get("vibe_profile") or ""
    from core.page_builder.amenity_taxonomy import extract_amenity_names
    amenities = extract_amenity_names(prop_data.get("amenities") or [])

    guest_snippets = []
    for ev in guest_evidence:
        text = ev.get("written_text") or ev.get("verbal_text") or ""
        if text.strip():
            reviewer = ev.get("reviewer_name") or "Guest"
            guest_snippets.append(f'  "{text.strip()[:200]}" — {reviewer}')

    prompt = f"""You are a creative director writing a one-paragraph premise for a 30-second property hero video.

PROPERTY:
  Name: {name}
  Location: {city}, {state}
  Vibe: {vibe}
  Amenities: {', '.join(str(a) for a in amenities[:20]) if amenities else 'Not listed'}

OWNER'S WORDS:
  Story: {owner_ctx.get('owner_story', 'N/A')}
  Wow factor: {owner_ctx.get('wow_factor', 'N/A')}
  Hidden gems: {owner_ctx.get('hidden_gems', 'N/A')}
  What guests love: {owner_ctx.get('guest_love', 'N/A')}

GUEST BOOK:
{chr(10).join(guest_snippets) if guest_snippets else '  No entries available'}

LOCATION CONTEXT:
  {guide.get('area_introduction', 'N/A')[:300]}

CONSTRAINTS — the truth line:
- The premise may ONLY draw on the owner's own words, real guest-book text,
  the wow factor, hidden gems, and the property's actual location and character.
- It must NOT assert an amenity, feature or setting the property does not have.
- It must NOT invent guest reactions or owner statements.
- It must be grounded in what is real about this specific property.

Write:
1. A short title (under 60 characters)
2. A one-paragraph premise that will seed the director, narrator and music.
   It should establish WHY this property, for WHOM, and what makes it unforgettable.
   30 seconds. This is the creative seed, not the script.

Return ONLY valid JSON:
{{"title": "...", "premise": "..."}}"""

    # ── Record run ─────────────────────────────────────────────────────
    run_id = record_run(sb, property_id, "onboard")
    step_id = record_step(sb, run_id, "conceive")

    # ── Call model ─────────────────────────────────────────────────────
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        import re
        json_match = re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            complete_step(sb, step_id, status="failed", error_message="No JSON in model response")
            complete_run(sb, run_id, status="failed")
            return SkillResult.failed(reason="Model returned no JSON")

        # Use raw_decode to handle trailing text (DIRECT-BRANT fix)
        decoder = json.JSONDecoder()
        start = text.find('{')
        result_data, _ = decoder.raw_decode(text, start)

    except Exception as exc:
        error_str = str(exc)
        complete_step(sb, step_id, status="failed", error_message=error_str[:200])
        complete_run(sb, run_id, status="failed")
        return SkillResult.failed(
            reason=f"Concept generation failed: {error_str[:200]}",
            attempted=1, succeeded=0, failed_count=1, error_class="vendor",
        )

    title = result_data.get("title", "Untitled")
    premise = result_data.get("premise", "")

    if not premise:
        complete_step(sb, step_id, status="failed", error_message="Empty premise")
        complete_run(sb, run_id, status="failed")
        return SkillResult.failed(reason="Model returned empty premise")

    # ── Supersede existing concepts ────────────────────────────────────
    now_iso = datetime.now(timezone.utc).isoformat()
    if force:
        sb.table("concepts").update({"superseded_at": now_iso, "status": "superseded"}).eq(
            "property_id", property_id
        ).eq("status", "active").is_("superseded_at", "null").execute()

    # ── Write concept ──────────────────────────────────────────────────
    concept_id = str(uuid.uuid4())
    cycle_month = datetime.now(timezone.utc).replace(day=1).strftime("%Y-%m-%d")

    sb.table("concepts").insert({
        "concept_id": concept_id,
        "property_id": property_id,
        "cycle_month": cycle_month,
        "concept_number": 1,
        "concept_version": 1,
        "title": title,
        "premise": premise,
        "vibe_prior": vibe,
        "evidence_basis": [
            f"Owner story: {owner_ctx.get('owner_story', '')[:100]}",
            f"Wow factor: {owner_ctx.get('wow_factor', '')[:100]}",
        ] + [f"Guest: {s[:80]}" for s in guest_snippets[:3]],
        "status": "active",
        "created_by_agent": "conceive",
    }).execute()

    # ── Cost ───────────────────────────────────────────────────────────
    cost = 0.05  # ~$0.05 per concept call
    emit_cost(sb, run_id, property_id,
              vendor="anthropic", service="claude_concept",
              units=1, unit_name="concepts",
              unit_cost=cost, total_cost=cost,
              workflow_name="conceive", generation_reason="concept_premise")

    complete_step(sb, step_id, status="complete", metadata={
        "concept_id": concept_id,
        "title": title,
        "premise_length": len(premise),
    })
    complete_run(sb, run_id, status="complete")

    return SkillResult.ok({
        "concept_id": concept_id,
        "title": title,
        "premise": premise,
        "run_id": run_id,
    })
