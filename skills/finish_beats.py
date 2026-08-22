"""
FINISH-A: Aleph finishing for production beats.

Consumes Director-authored content_motion and atmosphere per beat.
Executes approved, observation-grounded environmental motion and
restrained atmosphere via Runway Aleph (video_to_video, model=aleph2).

The finisher owns execution feasibility ONLY. It does not invent
content motion, atmosphere, or camera treatment. If Director-authored
intent survives grounding, it is executed.

Skip Aleph only when BOTH: grounded content_motion is empty AND
atmosphere is "preserve".
"""

import hashlib
import json
import logging
import os
import time
import uuid as uuid_mod

from skills.contract import (
    SkillResult, get_substrate, require_env,
    record_run, record_step, complete_step, complete_run, emit_cost,
    skills_r2_upload,
)

logger = logging.getLogger(__name__)

# ── Content-motion grounding map ──────────────────────────────────────
# Maps director vocabulary → observation fields that defensibly
# establish the element is visible in the photograph.
GROUNDING_RULES = {
    "water": {
        "check": lambda obs: "water_surface" in (obs.get("motion_risk") or []),
        "prompt_phrase": "gentle movement to the existing water surface",
    },
    "water_surface": {
        "check": lambda obs: "water_surface" in (obs.get("motion_risk") or []),
        "prompt_phrase": "gentle movement to the existing water surface",
    },
    "foliage": {
        "check": lambda obs: _any_in_lists(obs, ["tree", "palm", "plant", "foliage",
                                                   "shrub", "garden", "hedge", "vegetation"]),
        "prompt_phrase": "subtle swaying to the existing foliage",
    },
    "fire": {
        "check": lambda obs: _any_in_lists(obs, ["fire", "fireplace", "firepit", "fire pit"]),
        "prompt_phrase": "gentle flickering of the existing fire",
    },
    "clouds": {
        "check": lambda obs: _any_in_fields(obs, ["sky", "cloud", "clouds"],
                                             ["beyond_frame_element", "visual_summary", "foreground_elements"]),
        "prompt_phrase": "subtle movement of the existing clouds",
    },
    "curtains": {
        "check": lambda obs: _any_in_lists(obs, ["curtain", "curtains", "drape", "drapes"]),
        "prompt_phrase": "gentle movement of the existing curtains",
    },
    "steam": {
        "check": lambda obs: _any_in_lists(obs, ["steam", "hot tub", "spa", "jacuzzi"]),
        "prompt_phrase": "subtle wisps of steam from the existing water",
    },
}

# ── Atmosphere vocabulary ─────────────────────────────────────────────
ATMOSPHERE_PROMPTS = {
    "preserve": None,  # no atmospheric clause
    "warm": "Warm late-afternoon sunlight with soft golden highlights on surfaces.",
}


def _any_in_lists(obs, keywords):
    """Check if any keyword appears in foreground_elements or located_amenities."""
    fe = obs.get("foreground_elements") or []
    la = obs.get("located_amenities") or []
    combined = " ".join(str(x).lower() for x in fe + la)
    return any(kw.lower() in combined for kw in keywords)


def _any_in_fields(obs, keywords, field_names):
    """Check if any keyword appears in the named observation fields."""
    for field in field_names:
        val = obs.get(field)
        if val is None:
            continue
        text = " ".join(str(x).lower() for x in val) if isinstance(val, list) else str(val).lower()
        if any(kw.lower() in text for kw in keywords):
            return True
    return False


def _build_aleph_prompt(grounded_phrases, atmosphere):
    """Build the Aleph prompt from grounded content-motion phrases and atmosphere."""
    parts = ["Preserve the existing scene and camera movement."]
    if grounded_phrases:
        parts.append("Add " + " and ".join(grounded_phrases) + ".")
    atmo_clause = ATMOSPHERE_PROMPTS.get(atmosphere)
    if atmo_clause:
        parts.append(atmo_clause)
    return " ".join(parts)


def _truth_gate(pre_aleph_path, post_aleph_path):
    """Compare pre-Aleph and post-Aleph renders for structural truth.

    Checks FIRST + MIDDLE + LAST frames at ~1280px.

    Returns dict with pass/fail, details per frame pair, and honest limits.

    Protected: architecture, geometry, permanent materials/fixtures,
    furniture identity/layout, views, spatial relationships.
    Free: light, shadow, reflections, water, foliage, fire, approved atmosphere.
    Uniform colour-temperature shift = relighting, NOT violation.

    Honest limits: detects meaningful regeneration (room shape, furniture
    replacement, major structural change). Cannot guarantee pixel-level
    fidelity or detect every subtle material change.
    """
    import subprocess as sp
    import numpy as np
    from PIL import Image

    # Get frame count
    probe = sp.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=nb_frames", "-of", "csv=p=0",
                    pre_aleph_path], capture_output=True, text=True, timeout=10)
    nb_frames = int(probe.stdout.strip() or "90")
    mid_frame = nb_frames // 2

    frame_indices = [0, mid_frame, nb_frames - 1]
    frame_names = ["first", "middle", "last"]
    results = []

    for idx, name in zip(frame_indices, frame_names):
        # Extract frames at ~1280px width
        pre_f = f"/tmp/truth_pre_{name}.jpg"
        post_f = f"/tmp/truth_post_{name}.jpg"
        sp.run(["ffmpeg", "-y", "-i", pre_aleph_path,
                "-vf", f"select=eq(n\\,{idx}),scale=1280:-1",
                "-vframes", "1", "-q:v", "2", pre_f],
               capture_output=True, timeout=15)
        sp.run(["ffmpeg", "-y", "-i", post_aleph_path,
                "-vf", f"select=eq(n\\,{idx}),scale=1280:-1",
                "-vframes", "1", "-q:v", "2", post_f],
               capture_output=True, timeout=15)

        try:
            pre_img = np.array(Image.open(pre_f).convert("RGB")).astype(float)
            post_img = np.array(Image.open(post_f).convert("RGB")).astype(float)

            if pre_img.shape != post_img.shape:
                results.append({
                    "frame": name, "pass": False,
                    "reason": f"dimension mismatch: {pre_img.shape} vs {post_img.shape}",
                })
                continue

            # Structural similarity: compare edge maps (Sobel) rather than pixels.
            # This catches geometry changes while being tolerant of colour shifts.
            from PIL import ImageFilter
            pre_edges = np.array(Image.open(pre_f).convert("L").filter(ImageFilter.FIND_EDGES)).astype(float)
            post_edges = np.array(Image.open(post_f).convert("L").filter(ImageFilter.FIND_EDGES)).astype(float)

            # Normalise edge maps
            pre_edges /= max(pre_edges.max(), 1)
            post_edges /= max(post_edges.max(), 1)

            # Structural diff: mean absolute difference of edge maps
            edge_diff = float(np.abs(pre_edges - post_edges).mean())

            # Also check overall composition: divide into a 4x4 grid and compare
            # mean brightness per cell. Large shifts = structural rearrangement.
            h, w = pre_img.shape[:2]
            grid_diffs = []
            for gy in range(4):
                for gx in range(4):
                    y0, y1 = gy * h // 4, (gy + 1) * h // 4
                    x0, x1 = gx * w // 4, (gx + 1) * w // 4
                    pre_mean = pre_img[y0:y1, x0:x1].mean()
                    post_mean = post_img[y0:y1, x0:x1].mean()
                    grid_diffs.append(abs(pre_mean - post_mean))
            max_grid_diff = max(grid_diffs)
            mean_grid_diff = sum(grid_diffs) / len(grid_diffs)

            # Thresholds (empirically reasonable)
            # edge_diff > 0.15 = major structural change
            # max_grid_diff > 60 = a large region changed dramatically
            frame_pass = edge_diff < 0.15 and max_grid_diff < 60

            results.append({
                "frame": name,
                "pass": frame_pass,
                "edge_diff": round(edge_diff, 4),
                "max_grid_diff": round(max_grid_diff, 2),
                "mean_grid_diff": round(mean_grid_diff, 2),
                "reason": "structural invariants intact" if frame_pass else
                          f"edge_diff={edge_diff:.4f} max_grid={max_grid_diff:.1f}",
            })
        except Exception as exc:
            results.append({"frame": name, "pass": False, "reason": str(exc)[:200]})

    overall_pass = all(r["pass"] for r in results)
    return {
        "pass": overall_pass,
        "frames_checked": frame_names,
        "review_width": 1280,
        "results": results,
        "limits": (
            "Detects meaningful regeneration (room shape, furniture replacement, "
            "major structural change). Cannot guarantee pixel-level fidelity or "
            "detect every subtle material change. Tolerant of uniform colour-"
            "temperature shifts (relighting). Edge-map comparison + grid-cell "
            "brightness analysis."
        ),
    }


def finish_beats(
    property_id: str,
    direction_id: str,
    *,
    force: bool = False,
) -> SkillResult:
    """Apply Aleph finishing to eligible beats in a direction.

    For each ready clip:
    1. Load Director's content_motion + atmosphere from the beat
    2. Ground content_motion against observation evidence
    3. Skip if grounded content_motion empty AND atmosphere=="preserve"
    4. Build Aleph prompt from grounded intent
    5. Call Aleph video_to_video, truth-gate the result
    6. On pass: create finished artifact, supersede raw
    7. On fail: keep raw, log reason
    """
    sb = get_substrate()

    try:
        runway_key = require_env("RUNWAYML_API_SECRET", "Runway Aleph finishing")
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    # Load direction
    dir_resp = sb.table("directions").select("beats").eq(
        "direction_id", direction_id
    ).is_("superseded_at", "null").limit(1).execute()
    if not dir_resp.data:
        return SkillResult.failed(reason=f"Direction {direction_id[:12]} not found")
    beats = dir_resp.data[0].get("beats") or []
    beat_by_ordinal = {b.get("ordinal"): b for b in beats}

    # Load observations for grounding
    obs_resp = sb.table("observations").select(
        "photo_id, motion_risk, foreground_elements, located_amenities, "
        "beyond_frame_element, visual_summary, contains_text"
    ).eq("property_id", property_id).is_("superseded_at", "null").execute()
    obs_map = {o["photo_id"]: o for o in (obs_resp.data or [])}

    # Load ready clips for this direction (non-superseded)
    clips_resp = sb.table("video_artifacts").select(
        "artifact_id, input_hash, photo_id, beat_ordinal, storage_url, "
        "technique, duration_seconds, motion_params"
    ).eq("direction_id", direction_id).eq("kind", "clip").eq(
        "status", "ready"
    ).is_("superseded_at", "null").execute()
    clips = clips_resp.data or []

    if not clips:
        return SkillResult.noop("No ready clips for this direction", {})

    run_id = record_run(sb, property_id, "monthly_cycle")
    step_id = record_step(sb, run_id, "finish_beats")

    finished = 0
    skipped = 0
    truth_failed = 0
    aleph_failed = 0
    total_cost = 0.0
    beat_reports = []

    import runwayml
    runway_client = runwayml.RunwayML(api_key=runway_key)

    for clip in sorted(clips, key=lambda c: c.get("beat_ordinal", 0)):
        ordinal = clip.get("beat_ordinal", 0)
        photo_id = clip.get("photo_id", "")
        beat = beat_by_ordinal.get(ordinal, {})
        content_motion = beat.get("content_motion") or []
        atmosphere = beat.get("atmosphere", "preserve") or "preserve"

        report = {
            "beat": ordinal,
            "photo_id": photo_id[:12],
            "technique": clip.get("technique"),
        }

        # Skip: text-bearing frame
        obs = obs_map.get(photo_id, {})
        if obs.get("contains_text"):
            report["action"] = "skipped"
            report["reason"] = "text-bearing frame"
            beat_reports.append(report)
            skipped += 1
            continue

        # Ground content_motion
        grounded_phrases = []
        removed_elements = []
        for element in content_motion:
            rule = GROUNDING_RULES.get(element)
            if rule and rule["check"](obs):
                grounded_phrases.append(rule["prompt_phrase"])
            else:
                removed_elements.append(element)

        if removed_elements:
            logger.info("[finish] Beat %d: removed unsupported content_motion: %s",
                       ordinal, removed_elements)

        report["grounded_content_motion"] = [p.split("existing ")[-1] for p in grounded_phrases]
        report["removed_elements"] = removed_elements
        report["atmosphere"] = atmosphere

        # Skip check: empty grounded content_motion AND atmosphere=="preserve"
        if not grounded_phrases and atmosphere == "preserve":
            report["action"] = "skipped"
            report["reason"] = "no grounded content motion and atmosphere=preserve"
            beat_reports.append(report)
            skipped += 1
            continue

        # Build prompt
        prompt = _build_aleph_prompt(grounded_phrases, atmosphere)
        report["aleph_prompt"] = prompt

        # Idempotency check
        hash_input = json.dumps({
            "pre_aleph_hash": clip.get("input_hash", ""),
            "prompt": prompt,
            "model": "aleph2",
        }, sort_keys=True)
        aleph_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        if not force:
            existing = sb.table("video_artifacts").select("artifact_id", count="exact").eq(
                "input_hash", aleph_hash
            ).eq("kind", "clip").eq("status", "ready").is_(
                "superseded_at", "null"
            ).limit(0).execute()
            if existing.count > 0:
                report["action"] = "cached"
                report["reason"] = "matching finished artifact exists"
                beat_reports.append(report)
                skipped += 1
                continue

        # Call Aleph
        video_url = clip["storage_url"]
        try:
            task = runway_client.video_to_video.create(
                model="aleph2",  # type: ignore
                prompt_text=prompt,
                video_uri=video_url,
            )
            task_id = task.id
            estimated_cost = getattr(task, 'estimatedCost', {})
            credits_used = estimated_cost.get('credits', 84) if isinstance(estimated_cost, dict) else 84

            # Poll
            for _ in range(180):
                time.sleep(5)
                task_status = runway_client.tasks.retrieve(task_id)
                if task_status.status == "SUCCEEDED":
                    break
                elif task_status.status in ("FAILED", "CANCELLED"):
                    raise RuntimeError(f"Aleph task {task_id} {task_status.status}: "
                                      f"{getattr(task_status, 'failure', '')}")
            else:
                raise RuntimeError(f"Aleph task {task_id} timed out")

            output_url = task_status.output[0] if task_status.output else None
            if not output_url:
                raise RuntimeError("Aleph returned no output URL")

            # Download Aleph output
            import httpx
            video_resp = httpx.get(output_url, timeout=60, follow_redirects=True)
            video_resp.raise_for_status()
            aleph_bytes = video_resp.content

        except Exception as exc:
            error_msg = str(exc)[:300]
            logger.warning("[finish] Beat %d: Aleph failed — %s", ordinal, error_msg)
            report["action"] = "aleph_failed"
            report["reason"] = error_msg
            beat_reports.append(report)
            aleph_failed += 1
            continue

        # Download pre-Aleph for truth comparison
        try:
            pre_resp = httpx.get(video_url, timeout=60, follow_redirects=True)
            pre_resp.raise_for_status()
            pre_path = f"/tmp/finish_pre_{ordinal}.mp4"
            post_path = f"/tmp/finish_post_{ordinal}.mp4"
            with open(pre_path, "wb") as f:
                f.write(pre_resp.content)
            with open(post_path, "wb") as f:
                f.write(aleph_bytes)

            # Truth gate
            truth = _truth_gate(pre_path, post_path)
            report["truth_gate"] = truth

            if not truth["pass"]:
                logger.warning("[finish] Beat %d: truth gate FAILED — using pre-Aleph render", ordinal)
                report["action"] = "truth_failed"
                report["reason"] = "truth gate failed, using pre-Aleph fallback"
                beat_reports.append(report)
                truth_failed += 1
                continue

        except Exception as exc:
            logger.warning("[finish] Beat %d: truth gate error — %s, using pre-Aleph", ordinal, str(exc)[:200])
            report["action"] = "truth_error"
            report["reason"] = str(exc)[:200]
            beat_reports.append(report)
            truth_failed += 1
            continue

        # Truth passed — upload finished clip to R2
        r2_key = f"{property_id}/clips/aleph_{aleph_hash[:12]}.mp4"
        finished_url = skills_r2_upload(r2_key, aleph_bytes, content_type="video/mp4")

        # Create finished artifact
        new_artifact_id = str(uuid_mod.uuid4())
        sb.table("video_artifacts").insert({
            "artifact_id": new_artifact_id,
            "property_id": property_id,
            "kind": "clip",
            "direction_id": direction_id,
            "concept_id": beat.get("concept_id") or clip.get("concept_id"),
            "photo_id": photo_id,
            "input_hash": aleph_hash,
            "storage_url": finished_url,
            "duration_seconds": clip.get("duration_seconds"),
            "model": "aleph2",
            "vendor": "runway",
            "beat_ordinal": ordinal,
            "requested_motion": beat.get("requested_motion"),
            "technique": clip.get("technique"),  # preserve camera treatment
            "motion_params": {
                "finishing": "aleph",
                "aleph_prompt": prompt,
                "pre_aleph_artifact_id": clip["artifact_id"],
                "pre_aleph_input_hash": clip.get("input_hash"),
                "grounded_content_motion": report.get("grounded_content_motion"),
                "atmosphere": atmosphere,
                "truth_gate": "pass",
            },
            "cost_estimate_usd": credits_used * 0.01,
            "status": "ready",
            "created_by_agent": "skills/finish_beats",
        }).execute()

        # Supersede the raw artifact (NOT delete — remains retrievable)
        sb.table("video_artifacts").update({
            "superseded_at": "now()",
        }).eq("artifact_id", clip["artifact_id"]).execute()

        clip_cost = credits_used * 0.01
        total_cost += clip_cost
        finished += 1
        report["action"] = "finished"
        report["finished_url"] = finished_url
        report["cost"] = clip_cost
        beat_reports.append(report)

        logger.info("[finish] Beat %d: Aleph finished, truth PASS, cost=$%.2f",
                   ordinal, clip_cost)

    complete_step(sb, step_id, status="complete",
                  metadata={"finished": finished, "skipped": skipped,
                            "truth_failed": truth_failed, "aleph_failed": aleph_failed})
    complete_run(sb, run_id, status="complete")

    if finished > 0:
        emit_cost(sb, run_id, property_id,
                  vendor="runway", service="aleph2",
                  units=finished, unit_name="clips",
                  unit_cost=total_cost / max(finished, 1),
                  total_cost=total_cost,
                  workflow_name="finish_beats",
                  generation_reason="aleph_finishing",
                  discriminator="finish_a")

    return SkillResult.ok({
        "finished": finished,
        "skipped": skipped,
        "truth_failed": truth_failed,
        "aleph_failed": aleph_failed,
        "total_cost": round(total_cost, 2),
        "beat_reports": beat_reports,
    })
