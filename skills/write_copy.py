"""
Skill: write_copy — generate marketing copy from substrate data.

Ports Agent 2's content_generator + quality_gate intact. Edges only:
  - Input: properties, guest_evidence, observations, local_guides from substrate
  - Output: copy_versions (append, version increments, never overwrite)
  - Quality gate returns held (never auto-approves) — Ruling 6
  - Vendor calls emit cost_events

Usage:
    from skills.write_copy import write_copy
    result = write_copy("a1b2c3d4-...")
"""

import json
import logging
import os
from typing import Optional

from core.page_builder.amenity_taxonomy import extract_amenity_names as _extract_amenity_names
from skills.contract import (
    SkillResult, get_substrate, require_env,
    record_run, record_step, complete_step, complete_run, emit_cost,
)

logger = logging.getLogger(__name__)


def write_copy(
    property_id: str,
    *,
    force: bool = False,
) -> SkillResult:
    """Generate marketing copy and write to copy_versions.

    Returns SkillResult.ok({version, quality_result, cost_usd})
    or SkillResult.noop / SkillResult.held / SkillResult.failed.
    """
    try:
        sb = get_substrate()
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    try:
        api_key = require_env("ANTHROPIC_API_KEY", "Claude content generation")
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    # ── Check for existing copy — noop if not forcing ────────────────────
    if not force:
        existing = sb.table("copy_versions").select("version", count="exact").eq(
            "property_id", property_id
        ).limit(0).execute()
        if existing.count > 0:
            return SkillResult.noop(
                f"Copy already exists ({existing.count} versions). Use force=True to regenerate.",
                {"versions_existing": existing.count},
            )

    # ── Load property ────────────────────────────────────────────────────
    prop_resp = sb.table("properties").select("*").eq("id", property_id).limit(1).execute()
    if not prop_resp.data:
        return SkillResult.failed(reason=f"Property {property_id} not found", attempted=0, succeeded=0, failed_count=0)
    prop = prop_resp.data[0]

    # ── Build KB dict for content_generator ──────────────────────────────
    # Build KB dict — use empty strings (not None) for PropertyField values,
    # because the vibe_templates lambda does .get('value', '')[:800] and
    # None[:800] crashes (key present with None value bypasses the default).
    def _pf(val):
        """PropertyField-shaped dict. None → empty string to prevent slice crash."""
        return {"value": val or ""}

    kb = {
        "property_id": property_id,
        "name": _pf(prop.get("name")),
        "city": _pf(prop.get("city")),
        "state": _pf(prop.get("state_region")),
        "property_type": _pf(prop.get("property_type")),
        "vibe_profile": prop.get("vibe_profile") or "romantic_escape",
        "booking_url": prop.get("booking_url") or "#",
        "bedrooms": _pf(prop.get("bedrooms")),
        "bathrooms": _pf(prop.get("bathrooms")),
        "max_occupancy": _pf(prop.get("max_occupancy")),
        "description": _pf(prop.get("description")),
        "amenities": [{"value": n} for n in _extract_amenity_names(prop.get("amenities") or [])],
        "unique_features": [],
        "neighborhood_description": _pf(None),
        "owner_story": None,
        "wow_factor": None,
        "hidden_gems": None,
        "seasonal_notes": None,
    }

    # Load owner_context
    ctx_resp = sb.table("owner_context").select("*").eq(
        "property_id", property_id
    ).is_("superseded_at", "null").order("version", desc=True).limit(1).execute()
    if ctx_resp.data:
        ctx = ctx_resp.data[0]
        kb["owner_story"] = ctx.get("owner_story") or None
        kb["wow_factor"] = ctx.get("wow_factor") or None
        kb["hidden_gems"] = ctx.get("hidden_gems") or None
        kb["seasonal_notes"] = ctx.get("seasonal_notes") or None
        # Additional context for richer copy
        if ctx.get("guest_love"):
            kb["guest_love"] = ctx["guest_love"]
        if ctx.get("arrival_info"):
            kb["arrival_info"] = ctx["arrival_info"]
        if ctx.get("area_vibe"):
            kb["neighborhood_description"] = _pf(ctx["area_vibe"])
        if ctx.get("dont_miss"):
            kb["dont_miss"] = ctx["dont_miss"]
        if ctx.get("surround_areas"):
            kb["surround_areas"] = ctx["surround_areas"]

    # Load guest evidence
    ev_resp = sb.table("guest_evidence").select("*").eq("property_id", property_id).execute()
    guest_reviews = []
    for ev in (ev_resp.data or []):
        text = ev.get("written_text") or ""
        verbal = ev.get("verbal_text") or ""
        combined = f"{text}\n\n{verbal}".strip() if text and verbal else (text or verbal)
        guest_reviews.append({
            "text": combined,
            "reviewer_name": ev.get("reviewer_name"),
            "stay_date": ev.get("stay_date"),
            "is_guest_book": ev.get("is_guest_book", True),
        })
    kb["guest_reviews"] = guest_reviews

    # Load local guide for neighborhood context
    guide_resp = sb.table("local_guides").select("area_introduction").eq("property_id", property_id).limit(1).execute()
    if guide_resp.data and guide_resp.data[0].get("area_introduction"):
        kb["neighborhood_description"] = {"value": guide_resp.data[0]["area_introduction"]}

    # ── Record the run ───────────────────────────────────────────────────
    run_id = record_run(sb, property_id, "onboard")
    step_id = record_step(sb, run_id, "write_copy")

    # ── Determine next version ───────────────────────────────────────────
    ver_resp = sb.table("copy_versions").select("version").eq(
        "property_id", property_id
    ).order("version", desc=True).limit(1).execute()
    next_version = (ver_resp.data[0]["version"] + 1) if ver_resp.data else 1

    # ── Generate content ─────────────────────────────────────────────────
    os.environ["ANTHROPIC_API_KEY"] = api_key
    total_cost = 0.0

    try:
        import anthropic
        from agents.agent2.prompts.vibe_templates import get_system_prompt, get_user_prompt

        vibe = kb.get("vibe_profile", "romantic_escape")
        system = get_system_prompt(vibe)
        user = get_user_prompt(vibe, kb, [])  # No SEO keywords for now

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        # Check stop_reason
        stop_reason = getattr(resp, "stop_reason", None)
        if stop_reason == "max_tokens":
            logger.warning("[write_copy] Response truncated at max_tokens")

        raw = resp.content[0].text.strip()

        # Parse JSON
        import re
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        content = json.loads(raw)

        # Cost
        usage = resp.usage
        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        cost_usd = (input_tokens * 3 + output_tokens * 15) / 1_000_000
        total_cost += cost_usd

        emit_cost(sb, run_id, property_id,
                  vendor="anthropic", service="claude_sonnet_copy",
                  units=input_tokens + output_tokens, unit_name="tokens",
                  unit_cost=None, total_cost=round(cost_usd, 6),
                  workflow_name="write_copy", generation_reason="landing_page_copy")

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
            reason=f"Content generation failed: {error_str[:200]}",
            attempted=1, succeeded=0, failed_count=1, error_class="vendor",
        )

    # ── Quality gate ─────────────────────────────────────────────────────
    # Ruling 6: held = judgment gates. Never auto-approve.
    quality_score = None
    quality_result = "pass"
    gate_cost = 0.0

    try:
        gate_resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": (
                    f"Review this vacation rental marketing copy for quality.\n\n"
                    f"Property: {prop.get('name')}, {prop.get('city')}\n"
                    f"Vibe: {vibe}\n\n"
                    f"Copy:\n{json.dumps(content, indent=2)[:3000]}\n\n"
                    f"Score each 1-5:\n"
                    f"1. vibe_consistency: does the tone match {vibe}?\n"
                    f"2. specificity: does it reference THIS property's features?\n"
                    f"3. tone_coherence: is the voice consistent throughout?\n\n"
                    f"Return JSON only: {{\"vibe_consistency\": N, \"specificity\": N, "
                    f"\"tone_coherence\": N, \"overall\": N.N, \"issues\": []}}"
                ),
            }],
        )

        gate_raw = gate_resp.content[0].text.strip()
        gate_raw = re.sub(r"^```json\s*", "", gate_raw)
        gate_raw = re.sub(r"\s*```$", "", gate_raw)
        gate_data = json.loads(gate_raw)
        quality_score = gate_data.get("overall", 0)

        # Gate scoring
        if quality_score >= 4.0:
            quality_result = "pass"
        elif quality_score >= 3.5:
            quality_result = "needs_review"
        else:
            quality_result = "fail"

        gate_usage = gate_resp.usage
        gate_input = getattr(gate_usage, "input_tokens", 0)
        gate_output = getattr(gate_usage, "output_tokens", 0)
        gate_cost = (gate_input * 3 + gate_output * 15) / 1_000_000
        total_cost += gate_cost

        emit_cost(sb, run_id, property_id,
                  vendor="anthropic", service="claude_sonnet_quality_gate",
                  units=gate_input + gate_output, unit_name="tokens",
                  unit_cost=None, total_cost=round(gate_cost, 6),
                  workflow_name="write_copy", generation_reason="quality_gate")

    except Exception as exc:
        # Ruling 6: gate failure = HELD, not auto-pass
        logger.warning("[write_copy] Quality gate failed: %s — marking held", exc)
        quality_result = "needs_review"
        quality_score = None

    # ── Write copy_version ───────────────────────────────────────────────
    status = "approved" if quality_result == "pass" else "draft"

    sb.table("copy_versions").insert({
        "property_id": property_id,
        "version": next_version,
        "content": content,
        "quality_score": quality_score,
        "quality_result": quality_result,
        "generated_by_model": "claude-sonnet-4-6",
        "seo_target_keywords": [],
        "status": status,
    }).execute()

    # ── Held path: write hitl_queue_items ─────────────────────────────────
    # Fixed 2026-08-20: three invalid values found by reviewing every
    # hitl_queue_items insert site — "copy_review" → "content_review"
    # (CHECK constraint), "normal" → "p2" (CHECK: p0/p1/p2/p3; p2 = real
    # but not urgent, matching quality-gate semantics), "pending" → "open"
    # (CHECK: open/in_progress/resolved/dismissed). Also added account_id
    # lookup. Historical case (property f2147602, 2026-08-12, score 3.7)
    # found and confirmed — deliberately not backfilled per Erick's ruling.
    if quality_result in ("needs_review", "fail"):
        wc_account_id = None
        try:
            wc_prop = sb.table("properties").select("account_id").eq("id", property_id).limit(1).execute()
            if wc_prop.data:
                wc_account_id = wc_prop.data[0].get("account_id")
        except Exception:
            pass
        sb.table("hitl_queue_items").insert({
            "property_id": property_id,
            "account_id": wc_account_id,
            "queue_type": "content_review",
            "reason_code": f"quality_gate_{quality_result}",
            "title": f"Copy v{next_version} quality gate: {quality_result} (score: {quality_score})",
            "description": f"Quality gate scored {quality_score}/5. Review required before publishing.",
            "priority": "p2",
            "status": "open",
            "payload": json.dumps({"version": next_version, "quality_score": quality_score, "quality_result": quality_result}),
            "created_by_type": "agent",
        }).execute()
        logger.info("[write_copy] Copy held for review: version=%d score=%s", next_version, quality_score)

    # ── Complete ─────────────────────────────────────────────────────────
    complete_step(sb, step_id, status="complete", metadata={
        "version": next_version,
        "quality_result": quality_result,
        "quality_score": quality_score,
        "cost_usd": round(total_cost, 4),
    })
    complete_run(sb, run_id, status="complete")

    if quality_result in ("needs_review", "fail"):
        return SkillResult.held(
            reason=f"Copy v{next_version} quality gate: {quality_result} (score: {quality_score}). Review required.",
            data={
                "version": next_version,
                "quality_result": quality_result,
                "quality_score": quality_score,
                "cost_usd": round(total_cost, 4),
                "run_id": run_id,
            },
        )

    return SkillResult.ok({
        "version": next_version,
        "quality_result": quality_result,
        "quality_score": quality_score,
        "cost_usd": round(total_cost, 4),
        "run_id": run_id,
    })
