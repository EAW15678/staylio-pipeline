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

# Valid camera treatment values. The director sets one per beat.
#   bounded    — Creatomate pan animation on the still image. $0. DEFAULT.
#   static     — Camera holds still. Content may animate via finishing (FINISH-A).
#   locked     — Legacy alias for static. Accepted for backward compat.
#   depth      — DepthFlow parallax on Modal GPU. Authored camera, overscan crop. ~$0.01.
#   generative — Runway gen, camera moves freely. Legacy path.
VALID_TECHNIQUES = {"bounded", "static", "locked", "generative", "depth"}

# Cost per second by model
RUNWAY_COST = {
    "gen4_turbo": 0.05,
    "gen4.5": 0.12,
}

# Locked-camera prompt template.
# PHASE0-1 (2026-08-21): Rewritten per Runway's own Gen-4 prompting guide.
# The previous template was ~two-thirds negative phrasing ("does not move",
# "no wobble", "no camera shake", "no micro-drift", "no oscillation") plus
# a closing sentence re-describing scene elements ("architecture, railings,
# structural elements"). Runway documents that negative phrasing "may produce
# unpredictable or even opposite results" and re-describing image content
# "reduces motion and produces unexpected results." G78 recorded handheld
# wobble as an unfixable characteristic of locked — this negative-heavy
# template is the most plausible cause. G78 reopened pending test.
#
# {content_motion} is filled from the beat's motion_prompt — the director
# describes what should move (water, foliage, fire, etc.).
_LOCKED_PROMPT_TEMPLATE = (
    "The locked-off camera remains perfectly still. "
    "{content_motion}"
)


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

    # ── Load observations for motion_risk and depth routing ──────────────
    obs_resp = sb.table("observations").select(
        "photo_id, motion_risk, depth_structure"
    ).eq("property_id", property_id).is_("superseded_at", "null").execute()
    motion_risk_by_photo = {
        o["photo_id"]: o.get("motion_risk") or []
        for o in (obs_resp.data or [])
    }
    depth_structure_by_photo = {
        o["photo_id"]: o.get("depth_structure") or ""
        for o in (obs_resp.data or [])
    }

    # ── Load photograph dimensions (needed for depth eligibility) ──────
    photo_ids = list({b.get("photo_id") for b in beats if b.get("photo_id")})
    photo_dims = {}
    for pid in photo_ids:
        p = sb.table("photographs").select("image_width, image_height").eq("photo_id", pid).limit(1).execute()
        if p.data:
            photo_dims[pid] = (p.data[0].get("image_width") or 0, p.data[0].get("image_height") or 0)

    # ── Load rendition URLs ─────────────────────────────────────────────
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
    technique_counts = {"bounded": 0, "static": 0, "generative": 0, "depth": 0}
    technique_costs = {"bounded": 0.0, "static": 0.0, "generative": 0.0, "depth": 0.0}

    for beat in beats:
        photo_id = beat.get("photo_id")
        if not photo_id:
            continue

        requested_motion = beat.get("requested_motion", "")
        motion_prompt = beat.get("motion_prompt", "")
        duration = beat.get("duration_seconds", 5)
        ordinal = beat.get("ordinal", 0)

        technique = beat.get("technique", "bounded")
        # Normalise legacy "locked" → "static" (camera-only semantics)
        if technique == "locked":
            technique = "static"

        if technique not in VALID_TECHNIQUES:
            logger.error("[motion] Beat %d: unknown technique '%s'", ordinal, technique)
            clips_rejected += 1
            continue

        # ── Source image URL (needed for all paths) ────────────────────
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
            technique_counts["bounded"] += 1
            logger.info("[motion] Beat %d: bounded %s %ds cost=$0",
                       ordinal, requested_motion, duration)
            continue

        # ── Depth: DepthFlow parallax on Modal GPU ─────────────────────
        if technique == "depth":
            from skills.depth_motion import check_depth_eligibility, render_depth_beat

            img_w, img_h = photo_dims.get(photo_id, (0, 0))
            depth_map_url = urls.get("depth_map", "")
            intensity = beat.get("intensity", "restrained")

            elig = check_depth_eligibility(
                photo_id=photo_id,
                image_width=img_w,
                image_height=img_h,
                has_depth_map=bool(depth_map_url),
                depth_structure=depth_structure_by_photo.get(photo_id, ""),
                motion_risks=motion_risk_by_photo.get(photo_id, []),
                requested_motion=requested_motion,
                intensity=intensity,
            )

            if not elig["eligible"]:
                # Fall back to bounded — never fail, never swap images
                logger.info("[motion] Beat %d: depth ineligible (%s), falling back to bounded",
                           ordinal, elig["reason"])
                technique = "bounded"
                # Re-enter bounded path (duplicate the bounded logic for fallback)
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
                        "downgraded": True,
                        "original_technique": "depth",
                        "fallback_reason": elig["reason"],
                    },
                    "cost_estimate_usd": 0,
                    "status": "ready",
                    "created_by_agent": "skills/generate_motion",
                }).execute()
                clips_rendered += 1
                technique_counts["bounded"] += 1
                continue

            # Eligible — render via Modal
            try:
                hash_input = json.dumps({
                    "source_url": source_url,
                    "depth_map_url": depth_map_url,
                    "motion": requested_motion,
                    "technique": "depth",
                    "intensity": intensity,
                    "render_w": elig["render_w"],
                    "render_h": elig["render_h"],
                    "amp": elig["amp"],
                    "duration": duration,
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

                result = render_depth_beat(
                    image_url=source_url,
                    depth_map_url=depth_map_url,
                    requested_motion=requested_motion,
                    render_w=elig["render_w"],
                    render_h=elig["render_h"],
                    amp=elig["amp"],
                    duration=duration,
                )

                if "error" in result:
                    raise RuntimeError(result["error"][:300])

                artifact_id = str(uuid.uuid4())
                sb.table("video_artifacts").insert({
                    "artifact_id": artifact_id,
                    "property_id": property_id,
                    "kind": "clip",
                    "direction_id": direction_id,
                    "concept_id": direction.get("concept_id"),
                    "photo_id": photo_id,
                    "input_hash": input_hash,
                    "storage_url": result["storage_url"],
                    "duration_seconds": duration,
                    "model": "depthflow-1.0.0",
                    "vendor": "modal",
                    "beat_ordinal": ordinal,
                    "requested_motion": requested_motion,
                    "technique": "depth",
                    "motion_params": {
                        "intensity": intensity,
                        "amp": elig["amp"],
                        "render_w": elig["render_w"],
                        "render_h": elig["render_h"],
                        "margin": elig["margin"],
                        "crop_offset": result.get("crop_offset"),
                        "downgraded": False,
                    },
                    "cost_estimate_usd": 0.01,
                    "status": "ready",
                    "created_by_agent": "skills/generate_motion",
                }).execute()

                clips_rendered += 1
                technique_counts["depth"] += 1
                technique_costs["depth"] += 0.01
                total_cost += 0.01
                logger.info("[motion] Beat %d: depth %s %ds render=%dx%d margin=%.0f%% cost=$0.01",
                           ordinal, requested_motion, duration,
                           elig["render_w"], elig["render_h"], elig["margin"] * 100)
                continue

            except Exception as exc:
                error_msg = str(exc)[:300]
                logger.warning("[motion] Beat %d: depth render failed (%s), falling back to bounded",
                              ordinal, error_msg)
                # Fall back to bounded on render failure
                artifact_id = str(uuid.uuid4())
                sb.table("video_artifacts").insert({
                    "artifact_id": artifact_id,
                    "property_id": property_id,
                    "kind": "clip",
                    "direction_id": direction_id,
                    "concept_id": direction.get("concept_id"),
                    "photo_id": photo_id,
                    "input_hash": hashlib.sha256(f"depth-fallback-{photo_id}-{ordinal}".encode()).hexdigest(),
                    "storage_url": source_url,
                    "duration_seconds": duration,
                    "model": None,
                    "vendor": "creatomate",
                    "beat_ordinal": ordinal,
                    "requested_motion": requested_motion,
                    "technique": "bounded",
                    "motion_params": {
                        "downgraded": True,
                        "original_technique": "depth",
                        "failure_reason": error_msg,
                    },
                    "cost_estimate_usd": 0,
                    "status": "ready",
                    "created_by_agent": "skills/generate_motion",
                }).execute()
                clips_rendered += 1
                technique_counts["bounded"] += 1
                continue

        # ── Static: camera holds still, Runway animates content ─────────
        # (formerly "locked" — renamed to separate camera from content)
        if technique == "static":
            content_motion = motion_prompt or "Subtle natural movement within the scene."
            locked_prompt = _LOCKED_PROMPT_TEMPLATE.format(content_motion=content_motion)

            risk = motion_risk_by_photo.get(photo_id, [])
            model = _select_model(beat, risk)

            hash_input = json.dumps({
                "source_url": source_url,
                "prompt": locked_prompt,
                "technique": "static",
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

            try:
                import runwayml
                runway_client = runwayml.RunwayML(api_key=runway_key)

                # Runway accepts only 5 or 10 second durations (confirmed
                # against Runway's API docs, 2026-08-21). Five of six locked
                # beats failed silently on this limit. Snap defensively — the
                # director is told the rule, this is the backstop.
                runway_duration = 10 if duration > 7.5 else 5
                if runway_duration != duration:
                    logger.info("[motion] Beat %d: quantizing duration %.1f → %ds for Runway",
                               ordinal, duration, runway_duration)

                task = runway_client.image_to_video.create(
                    model=model,
                    prompt_image=source_url,
                    prompt_text=locked_prompt,
                    duration=runway_duration,
                    ratio="1280:720" if aspect_ratio == "16:9" else "720:1280",
                )

                import time
                task_id = task.id
                for _ in range(180):
                    time.sleep(5)
                    task_status = runway_client.tasks.retrieve(task_id)
                    if task_status.status == "SUCCEEDED":
                        break
                    elif task_status.status in ("FAILED", "CANCELLED"):
                        raise RuntimeError(f"Runway task {task_id} {task_status.status}")
                else:
                    raise RuntimeError(f"Runway task {task_id} timed out")

                output_url = task_status.output[0] if task_status.output else None
                if not output_url:
                    raise RuntimeError("Runway returned no output URL")

                import httpx
                video_resp = httpx.get(output_url, timeout=60, follow_redirects=True)
                video_resp.raise_for_status()
                video_bytes = video_resp.content

            except Exception as exc:
                error_str = str(exc)
                # G67 / MOTION-1: downgrade to bounded, never drop. A dropped
                # beat is how a 29.4s direction became a 10.6s master (5 of 6
                # locked beats silently vanished, 2026-08-21).
                logger.warning("[motion] Runway locked failed for beat %d: %s — downgrading to bounded",
                              ordinal, error_str[:300])

                # Fall back to bounded at the beat's directed duration
                fallback_id = str(uuid.uuid4())
                sb.table("video_artifacts").insert({
                    "artifact_id": fallback_id,
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
                    "requested_motion": "push_in",
                    "technique": "bounded",
                    "motion_params": {
                        "prompt": None,
                        "actual_motion": "push_in",
                        "downgraded": True,
                        "original_technique": "static",
                        "original_motion": requested_motion,
                        "original_prompt": locked_prompt,
                        "failure_reason": error_str[:300],
                    },
                    "cost_estimate_usd": 0,
                    "status": "ready",
                    "created_by_agent": "skills/generate_motion",
                }).execute()

                clips_rendered += 1
                technique_counts["bounded"] = technique_counts.get("bounded", 0) + 1
                clips_rejected += 1
                continue

            key = f"{property_id}/clips/{input_hash[:12]}.mp4"
            r2_url = skills_r2_upload(key, video_bytes, "video/mp4")

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
                "technique": "static",
                "motion_params": {
                    "prompt": locked_prompt,
                    "actual_motion": "static",
                    "downgraded": False,
                    "original_motion": None,
                    "original_prompt": None,
                    "content_motion": content_motion,
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
                      generation_reason="locked_clip",
                      discriminator=f"beat_{ordinal}")

            clips_rendered += 1
            technique_counts["static"] += 1
            technique_costs["static"] += clip_cost
            logger.info("[motion] Beat %d: static %ds model=%s cost=$%.4f",
                       ordinal, duration, model, clip_cost)
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
        technique_counts["generative"] += 1
        technique_costs["generative"] += clip_cost
        motion_label = f"{requested_motion}→{actual_motion}" if downgraded else actual_motion
        logger.info("[motion] Beat %d: %s %ds model=%s cost=$%.4f",
                   ordinal, motion_label, duration, model, clip_cost)

    complete_step(sb, step_id, status="complete", metadata={
        "clips_rendered": clips_rendered,
        "clips_rejected": clips_rejected,
        "clips_cached": clips_cached,
        "cost_usd": round(total_cost, 4),
        "technique_counts": technique_counts,
        "technique_costs": {k: round(v, 4) for k, v in technique_costs.items()},
    })
    complete_run(sb, run_id, status="complete")

    return SkillResult.ok({
        "clips_rendered": clips_rendered,
        "clips_rejected": clips_rejected,
        "clips_cached": clips_cached,
        "cost_usd": round(total_cost, 4),
        "run_id": run_id,
    })
