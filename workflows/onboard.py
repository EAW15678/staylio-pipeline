"""
Workflow: onboard — sequences skills to onboard a property.

The thin orchestrator per the architecture: sequences skills, records
ONE run with per-skill run_steps, carries NO state (substrate is the
state). Ruling 6 end-to-end: any failed → halt workflow, record,
hitl if human_required; noop → continue.

Sequence:
  ingest_intake → acquire_listing (per source_url) → enhance →
  deduplicate → observe → write_copy ∥ build_guide → publish_page

Usage:
    from workflows.onboard import onboard
    result = onboard("82cb9d7e-...", source_urls=["https://..."])
"""

import logging
from datetime import datetime, timezone

from skills.contract import (
    SkillResult, get_substrate,
    record_run, record_step, complete_step, complete_run,
)

logger = logging.getLogger(__name__)


def onboard(
    property_id: str,
    source_urls: list = None,
    *,
    force: bool = False,
    run_id: str = None,
) -> SkillResult:
    """Run the full onboarding workflow for a property.

    Each skill runs in sequence. Any failed result halts the workflow.
    Noop results continue (the skill's work is already done).
    The substrate is the state — no in-memory passing between skills.

    If run_id is provided, uses that existing run row (for fire-and-poll
    where the caller creates the run before returning to the client).
    """
    try:
        sb = get_substrate()
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    # ── One run for the whole workflow ────────────────────────────────────
    if not run_id:
        run_id = record_run(sb, property_id, "onboard")
    logger.info("[onboard] Starting workflow for property %s, run=%s", property_id[:12], run_id[:12])

    skills_run = []  # [(name, status, data)]

    def _run_skill(name: str, fn, *args, **kwargs) -> SkillResult:
        """Run a skill, record the step, halt on failure."""
        step_id = record_step(sb, run_id, name)
        logger.info("[onboard] Running skill: %s", name)

        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            result = SkillResult.failed(reason=f"Exception: {str(exc)[:200]}")

        status_str = result.status
        is_noop = isinstance(result.data, dict) and result.data.get("noop", False)
        label = f"{status_str}{'(noop)' if is_noop else ''}"

        complete_step(sb, step_id, status="complete" if result.is_ok else "failed",
                      error_message=result.reason if not result.is_ok else None,
                      metadata={"skill_status": status_str, "noop": is_noop})

        skills_run.append((name, label, result.data if result.is_ok else result.reason))
        logger.info("[onboard] %s → %s", name, label)

        if not result.is_ok and result.status != "held":
            # Failed → halt workflow
            complete_run(sb, run_id, status="failed",
                         error_summary=f"Halted at {name}: {result.reason}")
            return result

        return result

    # ── 1. Ingest intake ─────────────────────────────────────────────────
    from skills.ingest_intake import ingest_intake
    r = _run_skill("ingest_intake", ingest_intake, property_id, force=force)
    if not r.is_ok and r.status != "held":
        return SkillResult.failed(
            reason=f"Workflow halted at ingest_intake: {r.reason}",
            attempted=1, succeeded=0, failed_count=1,
        )

    # ── 2. Acquire listing(s) ────────────────────────────────────────────
    # Multi-source: try all URLs, log failures but don't halt on individual sources.
    # At least one source must succeed, or we halt.
    acquire_urls = list(source_urls) if source_urls else []
    if not acquire_urls:
        prop = sb.table("properties").select("primary_listing_url, airbnb_url, vrbo_url").eq("id", property_id).execute()
        if prop.data:
            p = prop.data[0]
            acquire_urls = [u for u in [p.get("primary_listing_url"), p.get("airbnb_url"), p.get("vrbo_url")] if u]

    sources_succeeded = 0
    sources_failed = 0
    for url in acquire_urls:
        from skills.acquire_listing import acquire_listing
        r = _run_skill("acquire_listing", acquire_listing, property_id, url, force=force)
        if r.is_ok:
            sources_succeeded += 1
        else:
            sources_failed += 1
            logger.warning("[onboard] acquire_listing failed for %s — continuing with other sources", url[:40])

    # ── 2b. Owner photos ─────────────────────────────────────────────────
    # Check for uploaded_photo_urls in intake answers
    owner_answers = sb.table("intake_answers").select("answer_json").eq(
        "property_id", property_id
    ).eq("question_key", "uploaded_photo_urls").order("answered_at", desc=True).limit(1).execute()
    if owner_answers.data:
        owner_urls = owner_answers.data[0].get("answer_json") or []
        if owner_urls:
            from skills.acquire_listing import acquire_owner_photos
            step_id = record_step(sb, run_id, "acquire_owner_photos")
            owner_result = acquire_owner_photos(property_id, owner_urls, run_id=run_id)
            complete_step(sb, step_id, status="complete", metadata={
                "skill_status": "ok",
                "photos_new": owner_result.get("photos_new", 0),
                "photos_existing": owner_result.get("photos_existing", 0),
            })
            skills_run.append(("acquire_owner_photos", "ok", owner_result))
            logger.info("[onboard] acquire_owner_photos → %d new, %d existing",
                       owner_result.get("photos_new", 0), owner_result.get("photos_existing", 0))

    # ── 3. Deduplicate ─────────────────────────────────────────────────
    # Runs BEFORE enhance so Claid is only paid for photographs that
    # survived deduplication. pHash is computed at acquisition (PHASH-1).
    from skills.deduplicate import deduplicate
    r = _run_skill("deduplicate", deduplicate, property_id, force=force)
    if not r.is_ok and r.status != "held":
        return SkillResult.failed(reason=f"Workflow halted at deduplicate: {r.reason}")

    # ── 4. Enhance ───────────────────────────────────────────────────────
    # enhance operates on CANONICALS ONLY, after dedupe.
    from skills.enhance import enhance
    r = _run_skill("enhance", enhance, property_id, force=force)
    if not r.is_ok and r.status != "held":
        return SkillResult.failed(reason=f"Workflow halted at enhance: {r.reason}")

    # ── 5. Observe ───────────────────────────────────────────────────────
    from skills.observe import observe
    r = _run_skill("observe", observe, property_id, force=force)
    if not r.is_ok and r.status != "held":
        return SkillResult.failed(reason=f"Workflow halted at observe: {r.reason}")

    # ── 6. Write copy + Build guide (could be parallel, run sequential for now)
    from skills.write_copy import write_copy
    r = _run_skill("write_copy", write_copy, property_id, force=force)
    # held is OK for copy — continue to guide

    from skills.build_guide import build_guide
    r = _run_skill("build_guide", build_guide, property_id, force=force)
    # guide failure is non-fatal — page can render without it

    # ── 7. Publish page ──────────────────────────────────────────────────
    from skills.publish_page import publish_page
    r = _run_skill("publish_page", publish_page, property_id, force=force)
    if not r.is_ok and r.status != "held":
        return SkillResult.failed(reason=f"Workflow halted at publish_page: {r.reason}")

    # ── Complete ─────────────────────────────────────────────────────────
    complete_run(sb, run_id, status="complete")

    # Summarize
    summary = {
        "run_id": run_id,
        "skills": [(name, status) for name, status, _ in skills_run],
        "page_url": r.data.get("page_url") if r.is_ok else None,
        "slug": r.data.get("slug") if r.is_ok else None,
    }

    logger.info(
        "[onboard] Complete: %s — %d skills, page=%s",
        property_id[:12], len(skills_run),
        summary.get("page_url", "none"),
    )

    return SkillResult.ok(summary)
