"""
Skill: score_music — generate background music from a direction's music_brief.

Ports Agent 8 Stage 5. ElevenLabs Music v2 (music_v2 model explicitly —
v1 must NOT be used). The director's music_brief is used verbatim.
Validator 10 (no artist references) already guards the brief.

Usage:
    from skills.score_music import score_music
    result = score_music("property_id", direction_id="uuid")
"""

import hashlib
import json
import logging
import os

import httpx

from skills.contract import (
    SkillResult, get_substrate, require_env,
    record_run, record_step, complete_step, complete_run, emit_cost,
    skills_r2_upload,
)

logger = logging.getLogger(__name__)

ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"
ELEVENLABS_MUSIC_MODEL = "music_v2"  # Explicitly v2 — v1 must NOT be fallback


def score_music(
    property_id: str,
    direction_id: str,
    *,
    force: bool = False,
) -> SkillResult:
    """Generate background music from a direction's music_brief.

    Returns SkillResult.ok({artifact_id, duration_seconds, cost_usd})
    """
    try:
        sb = get_substrate()
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    try:
        el_key = require_env("ELEVENLABS_API_KEY", "ElevenLabs Music")
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    # ── Load direction ──────────────────────────────────────────────────
    dir_resp = sb.table("directions").select(
        "direction_id, concept_id, music_brief, target_duration_sec"
    ).eq("direction_id", direction_id).is_("superseded_at", "null").limit(1).execute()

    if not dir_resp.data:
        return SkillResult.failed(reason=f"Direction {direction_id[:12]} not found")
    direction = dir_resp.data[0]
    music_brief = direction.get("music_brief") or {}

    if not music_brief:
        return SkillResult.noop("No music brief in this direction.", {})

    # Build prompt from brief
    prompt_text = " ".join(
        f"{k}: {v}" for k, v in music_brief.items() if v
    ) if isinstance(music_brief, dict) else str(music_brief)

    target_duration = direction.get("target_duration_sec") or 30
    duration_ms = int(target_duration * 1000)

    # ── Input hash ──────────────────────────────────────────────────────
    hash_input = json.dumps({
        "prompt_text": prompt_text,
        "duration_ms": duration_ms,
        "model": ELEVENLABS_MUSIC_MODEL,
    }, sort_keys=True)
    input_hash = hashlib.sha256(hash_input.encode()).hexdigest()

    if not force:
        existing = sb.table("video_artifacts").select("artifact_id", count="exact").eq(
            "input_hash", input_hash
        ).eq("kind", "music").eq("status", "ready").is_(
            "superseded_at", "null"
        ).limit(0).execute()
        if existing.count > 0:
            return SkillResult.noop("Music already exists (input_hash match).", {})

    # ── Record run ──────────────────────────────────────────────────────
    run_id = record_run(sb, property_id, "monthly_cycle")
    step_id = record_step(sb, run_id, "score_music")

    # ── Call ElevenLabs Music v2 ────────────────────────────────────────
    try:
        resp = httpx.post(
            f"{ELEVENLABS_API_BASE}/music",
            headers={
                "xi-api-key": el_key,
                "Content-Type": "application/json",
            },
            json={
                "prompt": prompt_text,
                "duration_seconds": target_duration,
                "model_id": ELEVENLABS_MUSIC_MODEL,
            },
            timeout=120,
        )
        resp.raise_for_status()
        audio_bytes = resp.content

    except Exception as exc:
        error_str = str(exc)
        complete_step(sb, step_id, status="failed", error_message=error_str[:200])
        complete_run(sb, run_id, status="failed")
        return SkillResult.failed(
            reason=f"ElevenLabs Music failed: {error_str[:200]}",
            attempted=1, succeeded=0, failed_count=1, error_class="vendor",
        )

    # ── Upload to R2 ────────────────────────────────────────────────────
    key = f"{property_id}/music/{input_hash[:12]}.mp3"
    r2_url = skills_r2_upload(key, audio_bytes, "audio/mpeg")

    # Estimate duration from bytes (MP3 ~128kbps)
    duration_seconds = round(len(audio_bytes) / (128 * 1024 / 8), 2)

    # ── Write video_artifacts ───────────────────────────────────────────
    import uuid
    from datetime import datetime, timezone
    artifact_id = str(uuid.uuid4())

    # Supersede prior music for this direction
    now_iso = datetime.now(timezone.utc).isoformat()
    sb.table("video_artifacts").update(
        {"superseded_at": now_iso}
    ).eq("direction_id", direction_id).eq("kind", "music").is_(
        "superseded_at", "null"
    ).execute()

    # Cost: ElevenLabs Music v2 pricing is per-generation, ~$0.08 for 30s
    music_cost = round(target_duration / 60 * 0.15, 4)

    sb.table("video_artifacts").insert({
        "artifact_id": artifact_id,
        "property_id": property_id,
        "kind": "music",
        "direction_id": direction_id,
        "concept_id": direction.get("concept_id"),
        "input_hash": input_hash,
        "storage_url": r2_url,
        "duration_seconds": duration_seconds,
        "model": ELEVENLABS_MUSIC_MODEL,
        "vendor": "elevenlabs",
        "prompt_text": prompt_text,
        "brief": music_brief,
        "status": "ready",
        "cost_estimate_usd": music_cost,
        "created_by_agent": "skills/score_music",
    }).execute()

    emit_cost(sb, run_id, property_id,
              vendor="elevenlabs", service="music_v2",
              units=1, unit_name="tracks",
              unit_cost=music_cost, total_cost=music_cost,
              workflow_name="score_music", generation_reason="background_music")

    complete_step(sb, step_id, status="complete", metadata={
        "artifact_id": artifact_id,
        "duration_seconds": duration_seconds,
        "cost_usd": music_cost,
    })
    complete_run(sb, run_id, status="complete")

    return SkillResult.ok({
        "artifact_id": artifact_id,
        "duration_seconds": duration_seconds,
        "cost_usd": music_cost,
        "run_id": run_id,
    })
