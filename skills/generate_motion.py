"""
Skill: generate_motion — render motion clips from a direction's beats.

Ports Agent 8 Stage 6. Routes on observations.motion_risk:
  - reflections/water_surface → gen4.5 (quality model)
  - all other → gen4_turbo (speed model)

Frame-exit guard: pull_back is ALWAYS rejected before any vendor call.
Camera moves may reveal NOTHING beyond the original frame.

Usage:
    from skills.generate_motion import generate_motion
    result = generate_motion("property_id", direction_id="uuid")
"""

import hashlib
import json
import logging
import os
import uuid

from skills.contract import (
    SkillResult, get_substrate, require_env,
    record_run, record_step, complete_step, complete_run, emit_cost,
    skills_r2_upload,
)

logger = logging.getLogger(__name__)

# Motion types that exit the frame boundary — DOWNGRADED to push_in.
# pull_back: camera moves away from the subject, revealing what lies
#   beyond the frame edge → Runway invents content.
# tilt_up: camera tilts upward, re-projecting the upper frame edge →
#   corrupts text (BEACCH from "BEACH" on Brant's water tower) and
#   reveals sky/content that wasn't in the original frame.
# These are NOT dropped — the shot stays in the film at push_in.
FRAME_EXITING_MOVES = {"pull_back", "tilt_up"}

# Cost per second by model
RUNWAY_COST = {
    "gen4_turbo": 0.05,
    "gen4.5": 0.12,
}


def _select_model(beat: dict, motion_risk: list) -> str:
    """Route to quality model for reflective surfaces, turbo for everything else."""
    if "reflections" in motion_risk or "water_surface" in motion_risk:
        return "gen4.5"
    return "gen4_turbo"


def generate_motion(
    property_id: str,
    direction_id: str,
    *,
    aspect_ratio: str = "16:9",
    force: bool = False,
) -> SkillResult:
    """Render motion clips from a direction's beats array.

    Each beat becomes a video_artifacts(kind='clip') row. Frame-exit
    moves are rejected before any vendor call.

    aspect_ratio: "16:9" for hero (1920x1080), "9:16" for social (1080x1920).
    Passed to Runway at generation time — clips are rendered at this ratio.

    Returns SkillResult.ok({clips_rendered, clips_rejected, clips_cached, cost_usd})
    """
    try:
        sb = get_substrate()
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    try:
        runway_key = require_env("RUNWAYML_API_SECRET", "Runway Gen-4 motion generation")
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    # ── Load direction ──────────────────────────────────────────────────
    dir_resp = sb.table("directions").select("*").eq(
        "direction_id", direction_id
    ).is_("superseded_at", "null").limit(1).execute()
    if not dir_resp.data:
        return SkillResult.failed(reason=f"Direction {direction_id[:12]} not found")
    direction = dir_resp.data[0]
    beats = direction.get("beats") or []

    if not beats:
        return SkillResult.noop("No beats in this direction.", {})

    # ── Load observations for motion_risk routing ───────────────────────
    obs_resp = sb.table("observations").select(
        "photo_id, motion_risk"
    ).eq("property_id", property_id).is_("superseded_at", "null").execute()
    motion_risk_by_photo = {
        o["photo_id"]: o.get("motion_risk") or []
        for o in (obs_resp.data or [])
    }

    # ── Load rendition URLs ─────────────────────────────────────────────
    photo_ids = list({b.get("photo_id") for b in beats if b.get("photo_id")})
    renditions = {}
    for pid in photo_ids:
        rend = sb.table("renditions").select("kind, storage_url").eq("photo_id", pid).execute()
        for r in (rend.data or []):
            renditions.setdefault(pid, {})[r["kind"]] = r["storage_url"]

    # ── Record run ──────────────────────────────────────────────────────
    run_id = record_run(sb, property_id, "monthly_cycle")
    step_id = record_step(sb, run_id, "generate_motion")

    clips_rendered = 0
    clips_rejected = 0
    clips_cached = 0
    total_cost = 0.0

    for beat in beats:
        photo_id = beat.get("photo_id")
        if not photo_id:
            continue

        requested_motion = beat.get("requested_motion", "")
        motion_prompt = beat.get("motion_prompt", "")
        duration = beat.get("duration_seconds", 5)
        ordinal = beat.get("ordinal", 0)

        technique = beat.get("technique", "generative")

        # ── Source image URL (needed for both paths) ───────────────────
        urls = renditions.get(photo_id, {})
        source_url = urls.get("enhanced") or urls.get("original", "")
        if not source_url:
            logger.warning("[motion] No rendition URL for photo %s", photo_id[:8])
            clips_rejected += 1
            continue

        # ── Bounded: no Runway call, write artifact with rendition URL ─
        if technique == "bounded":
            hash_input = json.dumps({
                "source_url": source_url,
                "motion": requested_motion,
                "technique": "bounded",
                "duration": duration,
                "aspect_ratio": aspect_ratio,
            }, sort_keys=True)
            input_hash = hashlib.sha256(hash_input.encode()).hexdigest()

            if not force:
                existing = sb.table("video_artifacts").select("artifact_id", count="exact").eq(
                    "input_hash", input_hash
                ).eq("kind", "clip").eq("status", "ready").is_(
                    "superseded_at", "null"
                ).limit(0).execute()
                if existing.count > 0:
                    clips_cached += 1
                    continue

            artifact_id = str(uuid.uuid4())
            sb.table("video_artifacts").insert({
                "artifact_id": artifact_id,
                "property_id": property_id,
                "kind": "clip",
                "direction_id": direction_id,
                "concept_id": direction.get("concept_id"),
                "photo_id": photo_id,
                "input_hash": input_hash,
                "storage_url": source_url,
                "duration_seconds": duration,
                "model": None,
                "vendor": "creatomate",
                "beat_ordinal": ordinal,
                "requested_motion": requested_motion,
                "technique": "bounded",
                "motion_params": {
                    "prompt": None,
                    "actual_motion": requested_motion,
                    "downgraded": False,
                    "original_motion": None,
                    "original_prompt": None,
                },
                "cost_estimate_usd": 0,
                "status": "ready",
                "created_by_agent": "skills/generate_motion",
            }).execute()

            clips_rendered += 1
            logger.info("[motion] Beat %d: bounded %s %ds cost=$0",
                       ordinal, requested_motion, duration)
            continue

        # ── Frame-exit guard: downgrade, never drop ────────────────────
        # A rejected motion keeps the shot — downgraded to push_in.
        actual_motion = requested_motion
        actual_prompt = motion_prompt
        downgraded = False
        if requested_motion in FRAME_EXITING_MOVES:
            actual_motion = "push_in"
            # Rewrite the prompt to match push_in — never send pull-back
            # or tilt-up language with a push_in label.
            actual_prompt = (
                f"Slow push in toward the centre of the frame. "
                f"Hold the existing composition steady."
            )
            downgraded = True
            logger.warning(
                "[motion] DOWNGRADED beat %d photo %s: '%s' → 'push_in' (frame-exit guard)",
                ordinal, str(photo_id)[:8], requested_motion,
            )

        # ── Model selection ─────────────────────────────────────────────
        risk = motion_risk_by_photo.get(photo_id, [])
        model = _select_model(beat, risk)

        # ── Input hash for idempotency ──────────────────────────────────
        hash_input = json.dumps({
            "source_url": source_url,
            "motion": actual_motion,
            "prompt": actual_prompt,
            "duration": duration,
            "model": model,
            "aspect_ratio": aspect_ratio,
        }, sort_keys=True)
        input_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        if not force:
            existing = sb.table("video_artifacts").select("artifact_id", count="exact").eq(
                "input_hash", input_hash
            ).eq("kind", "clip").eq("status", "ready").is_(
                "superseded_at", "null"
            ).limit(0).execute()
            if existing.count > 0:
                clips_cached += 1
                continue

        # ── Call Runway Gen-4 ───────────────────────────────────────────
        try:
            import runwayml
            runway_client = runwayml.RunwayML(api_key=runway_key)

            task = runway_client.image_to_video.create(
                model=model,
                prompt_image=source_url,
                prompt_text=actual_prompt,
                duration=duration,
                ratio="1280:720" if aspect_ratio == "16:9" else "720:1280",
            )

            # Poll for completion
            import time
            task_id = task.id
            for _ in range(180):  # up to 15 minutes
                time.sleep(5)
                task_status = runway_client.tasks.retrieve(task_id)
                if task_status.status == "SUCCEEDED":
                    break
                elif task_status.status in ("FAILED", "CANCELLED"):
                    raise RuntimeError(f"Runway task {task_id} {task_status.status}")
            else:
                raise RuntimeError(f"Runway task {task_id} timed out")

            # Download output
            output_url = task_status.output[0] if task_status.output else None
            if not output_url:
                raise RuntimeError("Runway returned no output URL")

            import httpx
            video_resp = httpx.get(output_url, timeout=60, follow_redirects=True)
            video_resp.raise_for_status()
            video_bytes = video_resp.content

        except Exception as exc:
            error_str = str(exc)
            logger.warning("[motion] Runway failed for beat %d: %s", ordinal, error_str[:80])
            clips_rejected += 1
            continue

        # ── Upload to R2 ────────────────────────────────────────────────
        key = f"{property_id}/clips/{input_hash[:12]}.mp4"
        r2_url = skills_r2_upload(key, video_bytes, "video/mp4")

        # ── Write video_artifacts ───────────────────────────────────────
        clip_cost = round(duration * RUNWAY_COST.get(model, 0.05), 4)
        total_cost += clip_cost

        artifact_id = str(uuid.uuid4())
        sb.table("video_artifacts").insert({
            "artifact_id": artifact_id,
            "property_id": property_id,
            "kind": "clip",
            "direction_id": direction_id,
            "concept_id": direction.get("concept_id"),
            "photo_id": photo_id,
            "input_hash": input_hash,
            "storage_url": r2_url,
            "duration_seconds": duration,
            "model": model,
            "vendor": "runway",
            "beat_ordinal": ordinal,
            "requested_motion": requested_motion,
            "technique": beat.get("technique", "generative"),
            "motion_params": {
                "prompt": actual_prompt,
                "actual_motion": actual_motion,
                "downgraded": downgraded,
                "original_motion": requested_motion if downgraded else None,
                "original_prompt": motion_prompt if downgraded else None,
            },
            "cost_estimate_usd": clip_cost,
            "status": "ready",
            "created_by_agent": "skills/generate_motion",
        }).execute()

        emit_cost(sb, run_id, property_id,
                  vendor="runway", service=model,
                  units=duration, unit_name="seconds",
                  unit_cost=RUNWAY_COST.get(model, 0.05),
                  total_cost=clip_cost,
                  workflow_name="generate_motion",
                  generation_reason="motion_clip",
                  discriminator=f"beat_{ordinal}")

        clips_rendered += 1
        motion_label = f"{requested_motion}→{actual_motion}" if downgraded else actual_motion
        logger.info("[motion] Beat %d: %s %ds model=%s cost=$%.4f",
                   ordinal, motion_label, duration, model, clip_cost)

    complete_step(sb, step_id, status="complete", metadata={
        "clips_rendered": clips_rendered,
        "clips_rejected": clips_rejected,
        "clips_cached": clips_cached,
        "cost_usd": round(total_cost, 4),
    })
    complete_run(sb, run_id, status="complete")

    return SkillResult.ok({
        "clips_rendered": clips_rendered,
        "clips_rejected": clips_rejected,
        "clips_cached": clips_cached,
        "cost_usd": round(total_cost, 4),
        "run_id": run_id,
    })
