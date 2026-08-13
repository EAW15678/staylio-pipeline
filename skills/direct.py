"""
Skill: direct — the director generates a direction from a concept.

Reads OBSERVATIONS directly (the shot_inventory relay is dead).
Vibe, wow_factor, guest_evidence, owner_context are inputs the
director reasons with — never a switch into a template library.

The brief is encoded as a `directions` row with beats[], narration_brief,
music_brief, overlay_register. Every direction is traceable to its
concept and the observations it drew from.

Usage:
    from skills.direct import direct
    result = direct("property_id", concept_id="uuid")
"""

import hashlib
import json
import logging
import os

from skills.contract import (
    SkillResult, get_substrate, require_env,
    record_run, record_step, complete_step, complete_run, emit_cost,
)

logger = logging.getLogger(__name__)


def direct(
    property_id: str,
    concept_id: str,
    *,
    force: bool = False,
) -> SkillResult:
    """Generate a direction (shot spec) for a concept.

    Reads observations for the property's canonical photographs to get
    motion_affordance, depth, focal_point, motion_risk — all the fields
    the director reasons with. Also reads owner_context, guest_evidence,
    and the concept's premise.

    Returns SkillResult.ok({direction_id, beat_count, ...})
    """
    try:
        sb = get_substrate()
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    try:
        require_env("ANTHROPIC_API_KEY", "Claude for creative direction")
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    # ── Check existing direction ────────────────────────────────────────
    if not force:
        existing = sb.table("directions").select("direction_id", count="exact").eq(
            "concept_id", concept_id
        ).is_("superseded_at", "null").limit(0).execute()
        if existing.count > 0:
            return SkillResult.noop(
                f"Direction already exists for concept {concept_id[:12]}.",
                {"directions_existing": existing.count},
            )

    # ── Load concept ────────────────────────────────────────────────────
    concept_resp = sb.table("concepts").select("*").eq(
        "concept_id", concept_id
    ).is_("superseded_at", "null").limit(1).execute()
    if not concept_resp.data:
        return SkillResult.failed(
            reason=f"Concept {concept_id[:12]} not found or superseded",
            attempted=0, succeeded=0, failed_count=0,
        )
    concept = concept_resp.data[0]

    # ── Load observations (replaces shot_inventory) ─────────────────────
    obs_resp = sb.table("observations").select(
        "observation_id, photo_id, "
        "motion_affordance, motion_risk, depth_structure, depth_tier, "
        "space_direction, light_direction, light_quality, time_of_day_read, "
        "negative_space, foreground_elements, frame_element, "
        "beyond_frame_element, subject_singularity, focal_point, "
        "tonal_signature, located_amenities, "
        "role, curated_section, quality_score, alt_text"
    ).eq("property_id", property_id).is_("superseded_at", "null").execute()
    observations = obs_resp.data or []

    if not observations:
        return SkillResult.failed(
            reason="No observations for this property — run observe first",
            attempted=0, succeeded=0, failed_count=0,
        )

    # Get rendition URLs for each observed photo
    photo_ids = [o["photo_id"] for o in observations]
    renditions_by_photo = {}
    for pid in photo_ids:
        rend = sb.table("renditions").select("kind, storage_url").eq("photo_id", pid).execute()
        for r in (rend.data or []):
            renditions_by_photo.setdefault(pid, {})[r["kind"]] = r["storage_url"]

    # ── Load context inputs ─────────────────────────────────────────────
    prop = sb.table("properties").select(
        "name, vibe_profile, city, state_region"
    ).eq("id", property_id).limit(1).execute()
    prop_data = prop.data[0] if prop.data else {}

    ctx_resp = sb.table("owner_context").select(
        "owner_story, wow_factor, hidden_gems, guest_love"
    ).eq("property_id", property_id).is_(
        "superseded_at", "null"
    ).order("version", desc=True).limit(1).execute()
    owner_ctx = ctx_resp.data[0] if ctx_resp.data else {}

    ev_resp = sb.table("guest_evidence").select(
        "written_text, verbal_text, reviewer_name, is_guest_book"
    ).eq("property_id", property_id).eq("is_guest_book", True).limit(10).execute()
    guest_evidence = ev_resp.data or []

    # ── Build candidate frames for the director ─────────────────────────
    frames = []
    for obs in observations:
        pid = obs["photo_id"]
        urls = renditions_by_photo.get(pid, {})
        display_url = urls.get("enhanced") or urls.get("original", "")
        if not display_url:
            continue
        frames.append({
            "photo_id": pid,
            "url": display_url,
            "motion_affordance": obs.get("motion_affordance") or [],
            "motion_risk": obs.get("motion_risk") or [],
            "depth_structure": obs.get("depth_structure"),
            "depth_tier": obs.get("depth_tier"),
            "space_direction": obs.get("space_direction"),
            "light_direction": obs.get("light_direction"),
            "time_of_day_read": obs.get("time_of_day_read"),
            "negative_space": obs.get("negative_space") or [],
            "focal_point": obs.get("focal_point"),
            "subject_singularity": obs.get("subject_singularity"),
            "foreground_elements": obs.get("foreground_elements") or [],
            "frame_element": obs.get("frame_element"),
            "beyond_frame_element": obs.get("beyond_frame_element"),
            "tonal_signature": obs.get("tonal_signature"),
            "located_amenities": obs.get("located_amenities") or [],
            "curated_section": obs.get("curated_section"),
            "quality_score": obs.get("quality_score"),
            "alt_text": obs.get("alt_text"),
        })

    if not frames:
        return SkillResult.failed(
            reason="No frames with rendition URLs — cannot direct",
            attempted=0, succeeded=0, failed_count=0,
        )

    # ── Record run ──────────────────────────────────────────────────────
    run_id = record_run(sb, property_id, "direct")
    step_id = record_step(sb, run_id, "direct")

    # ── Call the creative director engine ────────────────────────────────
    try:
        from agents.agent8.creative_director import direct_concept

        # The existing direct_concept reads from shot_inventory + concept_ledger.
        # For substrate, we need to adapt the interface. For now, call it
        # via the existing code path which reads from the prototype tables.
        # TODO: refactor direct_concept to accept frames + concept as dicts
        #       instead of reading from DB internally.
        #
        # INTERIM: write the substrate data to the expected prototype tables,
        # or call the Claude prompt directly.
        #
        # For this port, we call Claude directly with the substrate data.

        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        direction_result = _call_director(
            client, concept, frames, prop_data, owner_ctx, guest_evidence
        )

        if not direction_result:
            complete_step(sb, step_id, status="failed", error_message="Director returned no result")
            complete_run(sb, run_id, status="failed")
            return SkillResult.failed(
                reason="Director returned no result",
                attempted=1, succeeded=0, failed_count=1, error_class="vendor",
            )

        # ── Compute input hash for idempotency ──────────────────────────
        hash_input = json.dumps({
            "concept_id": concept_id,
            "frame_ids": sorted([f["photo_id"] for f in frames]),
            "premise": concept.get("premise"),
        }, sort_keys=True)
        input_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        # ── Supersede existing directions ───────────────────────────────
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        sb.table("directions").update(
            {"superseded_at": now_iso}
        ).eq("concept_id", concept_id).is_("superseded_at", "null").execute()

        # ── Write direction ─────────────────────────────────────────────
        import uuid
        direction_id = str(uuid.uuid4())
        sb.table("directions").insert({
            "direction_id": direction_id,
            "property_id": property_id,
            "concept_id": concept_id,
            "beats": direction_result.get("beats", []),
            "beat_count": len(direction_result.get("beats", [])),
            "target_duration_sec": direction_result.get("target_duration_sec", 30),
            "narrative_order": direction_result.get("narrative_order"),
            "continuity_notes": direction_result.get("continuity_notes", []),
            "narration_brief": direction_result.get("narration_brief"),
            "narration_provenance": direction_result.get("narration_provenance"),
            "music_brief": direction_result.get("music_brief", {}),
            "overlay_register": direction_result.get("overlay_register", []),
            "director_rationale": direction_result.get("director_rationale"),
            "director_model": "claude-sonnet-4-6",
            "evidence_used": direction_result.get("evidence_used", []),
            "vibe_drift": direction_result.get("vibe_drift"),
            "input_hash": input_hash,
            "status": "draft",
            "created_by_agent": "skills/direct",
        }).execute()

        # ── Cost ────────────────────────────────────────────────────────
        emit_cost(sb, run_id, property_id,
                  vendor="anthropic", service="claude_creative_direction",
                  units=1, unit_name="directions",
                  unit_cost=0.05, total_cost=0.05,
                  workflow_name="direct", generation_reason="creative_direction")

        complete_step(sb, step_id, status="complete", metadata={
            "direction_id": direction_id,
            "beat_count": len(direction_result.get("beats", [])),
            "frames_considered": len(frames),
        })
        complete_run(sb, run_id, status="complete")

        return SkillResult.ok({
            "direction_id": direction_id,
            "beat_count": len(direction_result.get("beats", [])),
            "frames_considered": len(frames),
            "run_id": run_id,
        })

    except Exception as exc:
        error_str = str(exc)
        if "credit balance" in error_str or "authentication" in error_str.lower():
            from skills.contract import escalate_billing
            escalate_billing(sb, property_id, "anthropic", error_str)
        complete_step(sb, step_id, status="failed", error_message=error_str[:200])
        complete_run(sb, run_id, status="failed")
        return SkillResult.failed(
            reason=f"Direction failed: {error_str[:200]}",
            attempted=1, succeeded=0, failed_count=1, error_class="vendor",
        )


def _call_director(client, concept, frames, prop, owner_ctx, guest_evidence) -> dict:
    """Call Claude to generate a direction from concept + observations.

    This is the creative director's reasoning — driven by the vibe,
    not a template library. The brief encodes Erick's director rules:
    - 30 seconds, mixes guest book snippets with motion
    - Camera moves reveal NOTHING beyond the original frame
    - Life inside may be imagined (people, pets)
    - Director may excerpt guest text for video
    - Photo selection on observations (motion_affordance, motion_risk, depth)
    """
    vibe = prop.get("vibe_profile") or "multigenerational_retreat"
    name = prop.get("name") or "the property"

    guest_snippets = []
    for ev in guest_evidence[:5]:
        text = ev.get("written_text") or ev.get("verbal_text") or ""
        if text.strip():
            reviewer = ev.get("reviewer_name") or "Guest"
            guest_snippets.append(f'"{text.strip()[:200]}" — {reviewer}')

    frame_descriptions = []
    for i, f in enumerate(frames[:30]):  # cap at 30 frames
        desc = (
            f'Frame {i+1} [photo_id={f["photo_id"][:8]}]: '
            f'section={f["curated_section"]}, '
            f'motion_affordance={f["motion_affordance"]}, '
            f'motion_risk={f["motion_risk"]}, '
            f'depth={f["depth_tier"]}, '
            f'focal_point={f["focal_point"]}, '
            f'space_direction={f["space_direction"]}, '
            f'negative_space={f["negative_space"]}'
        )
        frame_descriptions.append(desc)

    prompt = f"""You are a creative director for a 30-second property hero video.

PROPERTY: {name} in {prop.get("city", "")}, {prop.get("state_region", "")}
VIBE: {vibe}
CONCEPT PREMISE: {concept.get("premise", "")}

OWNER STORY: {owner_ctx.get("owner_story", "N/A")}
WOW FACTOR: {owner_ctx.get("wow_factor", "N/A")}
HIDDEN GEMS: {owner_ctx.get("hidden_gems", "N/A")}

GUEST BOOK EXCERPTS (you MAY excerpt these for narration):
{chr(10).join(guest_snippets) if guest_snippets else "None available."}

CANDIDATE FRAMES (from LLM curation observations):
{chr(10).join(frame_descriptions)}

RULES:
1. Select 5-6 frames (beats) that tell the story. Choose on motion_affordance, depth, quality.
2. Camera moves may reveal NOTHING beyond the original frame — no pull-backs or pans that force the model to invent what is outside the edge. If a frame has "pull_back" in motion_risk, reject it.
3. The life inside may be imagined — you MAY describe people, pets in motion prompts where appropriate.
4. For narration: you may excerpt guest text freely. The vibe drives tone.
5. Target 30 seconds total (5-6 beats at 5-6 seconds each).
6. Music brief should match the vibe mood. NO artist name references.

Return JSON:
{{
  "beats": [
    {{
      "ordinal": 1,
      "photo_id": "...",
      "technique": "generative",
      "requested_motion": "push_in",
      "motion_prompt": "Slow push into the pool area as golden light catches the water surface",
      "duration_seconds": 5
    }},
    ...
  ],
  "narration_brief": "The narration script text",
  "narration_provenance": "original" or "guest_book",
  "music_brief": {{"mood": "...", "tempo": "...", "instruments": "..."}},
  "overlay_register": [],
  "narrative_order": "description of the narrative arc",
  "continuity_notes": ["note about visual flow between beats"],
  "director_rationale": "Why these frames and this sequence",
  "evidence_used": ["guest_book", "wow_factor", "observations"],
  "vibe_drift": null,
  "target_duration_sec": 30
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text
    # Extract JSON from response
    import re
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        return json.loads(json_match.group())
    return None
