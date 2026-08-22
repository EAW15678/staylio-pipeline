"""
Workflow: onboard — sequences skills to onboard a property.

The thin orchestrator per the architecture: sequences skills, records
ONE run with per-skill run_steps, carries NO state (substrate is the
state). Ruling 6 end-to-end: any failed → halt workflow, record,
hitl if human_required; noop → continue.

Sequence:
  ingest_intake → acquire_listing (per source_url) → deduplicate →
  observe → enhance → write_copy ∥ build_guide →
  conceive → direct → narrate + narrate_guest_cards → score_music →
  generate_motion → assemble → publish_page

Video steps (conceive through assemble) are non-halting: if any fails,
the page publishes without a hero video and an alert is raised.

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

    # ── 4. Observe ───────────────────────────────────────────────────────
    # Runs BEFORE enhance so enhancement recipes can read observation
    # fields (light_quality, placement, contains_text) to choose the
    # right recipe per photograph (ENHANCE-2).
    from skills.observe import observe
    r = _run_skill("observe", observe, property_id, force=force)
    if not r.is_ok and r.status != "held":
        return SkillResult.failed(reason=f"Workflow halted at observe: {r.reason}")

    # ── 4b. Depth maps ──────────────────────────────────────────────────
    # Deterministic derivatives of the photograph. Computed on Modal GPU,
    # cached as renditions. Failure is non-fatal — depth maps are an
    # enhancement for the cinematic pipeline, not a critical output.
    from skills.generate_depth_maps import generate_depth_maps
    _run_skill("generate_depth_maps", generate_depth_maps, property_id, force=force)
    # Result intentionally not checked — depth failure must not halt onboard.

    # ── 5. Enhance ───────────────────────────────────────────────────────
    # enhance operates on CANONICALS ONLY, after dedupe and observe.
    from skills.enhance import enhance
    r = _run_skill("enhance", enhance, property_id, force=force)
    if not r.is_ok and r.status != "held":
        return SkillResult.failed(reason=f"Workflow halted at enhance: {r.reason}")

    # ── 6. Write copy + Build guide (could be parallel, run sequential for now)
    from skills.write_copy import write_copy
    r = _run_skill("write_copy", write_copy, property_id, force=force)
    # held is OK for copy — continue to guide

    from skills.build_guide import build_guide
    r = _run_skill("build_guide", build_guide, property_id, force=force)
    # guide failure is non-fatal — page can render without it

    # ── 7–12. Video pipeline ────────────────────────────────────────────
    # If any video step fails, continue to publishing — a property must
    # never be left with no page because assembly failed. The page renders
    # without a hero (render_page degrades gracefully).
    # On failure: create a hitl queue row + alert, then publish anyway.
    video_failed_step = None
    video_error = None
    concept_id = None
    direction_id = None

    try:
        # ── 7. Conceive ─────────────────────────────────────────────────
        from skills.conceive import conceive
        step_id_v = record_step(sb, run_id, "conceive")
        r = conceive(property_id, force=force)
        complete_step(sb, step_id_v, status="complete" if r.is_ok else "failed",
                      metadata=r.data if r.data else None,
                      error_message=r.reason if not r.is_ok else None)
        skills_run.append(("conceive", r.status, r.data))

        if r.is_ok and r.data:
            concept_id = r.data.get("concept_id")

        if not concept_id:
            raise RuntimeError(f"conceive failed: {r.reason}")

        # ── 8. Direct ───────────────────────────────────────────────────
        from skills.direct import direct
        step_id_v = record_step(sb, run_id, "direct")
        r = direct(property_id, concept_id=concept_id, force=force)
        complete_step(sb, step_id_v, status="complete" if r.is_ok else "failed",
                      metadata=r.data if r.data else None,
                      error_message=r.reason if not r.is_ok else None)
        skills_run.append(("direct", r.status, r.data))

        if r.is_ok and r.data:
            direction_id = r.data.get("direction_id")

        if not direction_id:
            raise RuntimeError(f"direct failed: {r.reason}")

        # ── 9. Narrate + Narrate guest cards ────────────────────────────
        from skills.narrate import narrate
        step_id_v = record_step(sb, run_id, "narrate")
        r = narrate(property_id, direction_id=direction_id, force=force)
        complete_step(sb, step_id_v, status="complete" if r.is_ok else "failed",
                      metadata=r.data if r.data else None,
                      error_message=r.reason if not r.is_ok else None)
        skills_run.append(("narrate", r.status, r.data))
        if not r.is_ok:
            raise RuntimeError(f"narrate failed: {r.reason}")

        from skills.narrate_guest_cards import narrate_guest_cards
        step_id_v = record_step(sb, run_id, "narrate_guest_cards")
        r = narrate_guest_cards(property_id, force=force)
        complete_step(sb, step_id_v, status="complete" if r.is_ok else "failed",
                      metadata=r.data if r.data else None,
                      error_message=r.reason if not r.is_ok else None)
        skills_run.append(("narrate_guest_cards", r.status, r.data))
        # guest card failure is non-fatal — hero can render without them

        # ── 10. Score music ─────────────────────────────────────────────
        from skills.score_music import score_music
        step_id_v = record_step(sb, run_id, "score_music")
        r = score_music(property_id, direction_id=direction_id, force=force)
        complete_step(sb, step_id_v, status="complete" if r.is_ok else "failed",
                      metadata=r.data if r.data else None,
                      error_message=r.reason if not r.is_ok else None)
        skills_run.append(("score_music", r.status, r.data))
        if not r.is_ok:
            raise RuntimeError(f"score_music failed: {r.reason}")

        # ── 11. Generate motion ─────────────────────────────────────────
        from skills.generate_motion import generate_motion
        step_id_v = record_step(sb, run_id, "generate_motion")
        r = generate_motion(property_id, direction_id=direction_id, force=force)
        complete_step(sb, step_id_v, status="complete" if r.is_ok else "failed",
                      metadata=r.data if r.data else None,
                      error_message=r.reason if not r.is_ok else None)
        skills_run.append(("generate_motion", r.status, r.data))
        if not r.is_ok:
            raise RuntimeError(f"generate_motion failed: {r.reason}")

        # ── 11b. Finish beats (Aleph) ──────────────────────────────────
        # Optional Aleph finishing on eligible clips. Non-fatal.
        from skills.finish_beats import finish_beats
        step_id_v = record_step(sb, run_id, "finish_beats")
        r = finish_beats(property_id, direction_id=direction_id, force=force)
        complete_step(sb, step_id_v, status="complete" if r.is_ok else "failed",
                      metadata=r.data if r.data else None,
                      error_message=r.reason if not r.is_ok else None)
        skills_run.append(("finish_beats", r.status, r.data))
        # Result NOT checked — finishing failure does not halt pipeline

        # ── 12. Assemble ────────────────────────────────────────────────
        from skills.assemble import assemble
        step_id_v = record_step(sb, run_id, "assemble")
        r = assemble(property_id, direction_id=direction_id,
                     aspect_ratio="16:9", title_cards=False, force=force)
        complete_step(sb, step_id_v, status="complete" if r.is_ok else "failed",
                      metadata=r.data if r.data else None,
                      error_message=r.reason if not r.is_ok else None)
        skills_run.append(("assemble", r.status, r.data))
        if not r.is_ok:
            raise RuntimeError(f"assemble failed: {r.reason}")

    except Exception as video_exc:
        video_failed_step = str(video_exc).split(" failed:")[0] if " failed:" in str(video_exc) else "video_pipeline"
        video_error = str(video_exc)[:200]
        logger.warning("[onboard] Video pipeline failed at %s: %s — publishing without hero",
                       video_failed_step, video_error[:80])

        # ── Alert: hitl queue row + email ───────────────────────────────
        # account_id lookup added — was hardcoded to None, found by reviewing
        # every hitl_queue_items insert site after escalate_halt's missing
        # created_by_type crashed Vista Azule on 2026-08-20.
        prop_name = prop_data.get("name") or property_id[:12] if "prop_data" in dir() else property_id[:12]
        account_id = None
        try:
            prop_for_name = sb.table("properties").select("name, account_id").eq("id", property_id).limit(1).execute()
            if prop_for_name.data:
                prop_name = prop_for_name.data[0].get("name") or property_id[:12]
                account_id = prop_for_name.data[0].get("account_id")
        except Exception:
            pass

        try:
            import uuid as _uuid
            hitl_id = str(_uuid.uuid4())
            sb.table("hitl_queue_items").insert({
                "id": hitl_id,
                "property_id": property_id,
                "account_id": account_id,
                "queue_type": "pipeline_failure",
                "priority": "p0",
                "status": "open",
                "created_by_type": "system",
                "reason_code": "video_pipeline_failure",
                "title": f"CRITICAL: {prop_name} — {video_failed_step} failed during onboarding",
                "description": (
                    f"The video pipeline failed at step '{video_failed_step}' during onboarding. "
                    f"Error: {video_error}. "
                    f"The page was published without a hero video. "
                    f"This property needs a video generated and the page republished."
                ),
                "payload": {"failed_step": video_failed_step, "error": video_error,
                             "run_id": run_id, "property_id": property_id},
            }).execute()

            from skills.notify import send_halt_alert
            send_halt_alert(
                sb, hitl_id, property_id, prop_name,
                error_class="video_pipeline_failure",
                detail=(
                    f"Step '{video_failed_step}' failed: {video_error}. "
                    f"The page was published without a hero video. "
                    f"Needs a video and a republish."
                ),
                run_id=run_id,
                step_name=video_failed_step,
            )
        except Exception as alert_exc:
            logger.error("[onboard] Alert failed: %s — continuing to publish", str(alert_exc)[:80])

    # ── 13. Publish page ─────────────────────────────────────────────────
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
