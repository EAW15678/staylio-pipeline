"""
Skill: narrate — generate narration audio from a direction's narration_brief.

Ports Agent 8 Stage 4. Edges only change — reads from directions table
(not shot_spec), writes to video_artifacts(kind='narration').

Voice selection by vibe bucket (ElevenLabs). Gender must match the guest
who wrote the entry for guest_book narration. Fail loudly if a vibe has
no configured voice bucket — do not substitute.

Usage:
    from skills.narrate import narrate
    result = narrate("property_id", direction_id="uuid")
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

# Voice IDs by vibe — from ElevenLabs buckets.
# Only multigenerational exists today (G45). Others must be configured
# before use — the skill fails loudly per Ruling 6 if missing.
VIBE_VOICE_IDS = {
    "multigenerational_retreat": os.environ.get("VOICE_MULTIGENERATIONAL", ""),
}

VIBE_VOICE_LABELS = {
    "multigenerational_retreat": "Multi-Generational voice",
}

ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"
ELEVENLABS_MODEL = "eleven_multilingual_v2"


def narrate(
    property_id: str,
    direction_id: str,
    *,
    force: bool = False,
) -> SkillResult:
    """Generate narration audio from a direction's narration_brief.

    Returns SkillResult.ok({artifact_id, duration_seconds, ...})
    """
    try:
        sb = get_substrate()
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    try:
        el_key = require_env("ELEVENLABS_API_KEY", "ElevenLabs TTS")
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    # ── Load direction ──────────────────────────────────────────────────
    dir_resp = sb.table("directions").select(
        "direction_id, concept_id, narration_brief, narration_provenance"
    ).eq("direction_id", direction_id).is_("superseded_at", "null").limit(1).execute()

    if not dir_resp.data:
        return SkillResult.failed(
            reason=f"Direction {direction_id[:12]} not found",
            attempted=0, succeeded=0, failed_count=0,
        )
    direction = dir_resp.data[0]
    script = direction.get("narration_brief") or ""

    if not script.strip():
        return SkillResult.noop("No narration brief in this direction.", {})

    # ── Resolve voice ───────────────────────────────────────────────────
    prop = sb.table("properties").select("vibe_profile").eq(
        "id", property_id
    ).limit(1).execute()
    vibe = (prop.data[0].get("vibe_profile") if prop.data else "") or ""

    voice_id = VIBE_VOICE_IDS.get(vibe, "")
    voice_label = VIBE_VOICE_LABELS.get(vibe, f"{vibe} voice")

    if not voice_id:
        # Fail loudly — do not substitute. Ruling 6.
        return SkillResult.failed(
            reason=f"No voice bucket configured for vibe '{vibe}'. "
                   f"Set VOICE_{vibe.upper()} env var with an ElevenLabs voice ID.",
            attempted=1, succeeded=0, failed_count=1,
            error_class="config", human_required=True,
        )

    # ── Input hash for idempotency ──────────────────────────────────────
    hash_input = json.dumps({
        "script": script,
        "voice_id": voice_id,
        "model": ELEVENLABS_MODEL,
    }, sort_keys=True)
    input_hash = hashlib.sha256(hash_input.encode()).hexdigest()

    if not force:
        existing = sb.table("video_artifacts").select("artifact_id", count="exact").eq(
            "input_hash", input_hash
        ).eq("kind", "narration").eq("status", "ready").is_(
            "superseded_at", "null"
        ).limit(0).execute()
        if existing.count > 0:
            return SkillResult.noop(
                f"Narration already exists (input_hash match).",
                {"input_hash": input_hash[:12]},
            )

    # ── Record run ──────────────────────────────────────────────────────
    run_id = record_run(sb, property_id, "monthly_cycle")
    step_id = record_step(sb, run_id, "narrate")

    # ── Call ElevenLabs TTS ─────────────────────────────────────────────
    try:
        resp = httpx.post(
            f"{ELEVENLABS_API_BASE}/text-to-speech/{voice_id}",
            headers={
                "xi-api-key": el_key,
                "Content-Type": "application/json",
            },
            json={
                "text": script,
                "model_id": ELEVENLABS_MODEL,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.3,
                },
            },
            timeout=60,
        )
        resp.raise_for_status()
        audio_bytes = resp.content

    except Exception as exc:
        error_str = str(exc)
        complete_step(sb, step_id, status="failed", error_message=error_str[:200])
        complete_run(sb, run_id, status="failed")
        return SkillResult.failed(
            reason=f"ElevenLabs TTS failed: {error_str[:200]}",
            attempted=1, succeeded=0, failed_count=1, error_class="vendor",
        )

    # ── Upload to R2 ────────────────────────────────────────────────────
    key = f"{property_id}/narration/{input_hash[:12]}.mp3"
    r2_url = skills_r2_upload(key, audio_bytes, "audio/mpeg")

    # Estimate duration (MP3 at ~128kbps)
    duration_seconds = len(audio_bytes) / (128 * 1024 / 8)

    # ── Write video_artifacts ───────────────────────────────────────────
    import uuid
    artifact_id = str(uuid.uuid4())

    # Supersede prior narrations for this direction
    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()
    sb.table("video_artifacts").update(
        {"superseded_at": now_iso}
    ).eq("direction_id", direction_id).eq("kind", "narration").is_(
        "superseded_at", "null"
    ).execute()

    sb.table("video_artifacts").insert({
        "artifact_id": artifact_id,
        "property_id": property_id,
        "kind": "narration",
        "direction_id": direction_id,
        "concept_id": direction.get("concept_id"),
        "input_hash": input_hash,
        "storage_url": r2_url,
        "duration_seconds": round(duration_seconds, 2),
        "model": ELEVENLABS_MODEL,
        "vendor": "elevenlabs",
        "script_text": script,
        "voice_id": voice_id,
        "voice_label": voice_label,
        "language": "en",
        "provenance": direction.get("narration_provenance") or "original",
        "contains_verbatim_source": direction.get("narration_provenance") == "guest_book",
        "status": "ready",
        "cost_estimate_usd": round(len(script) * 0.0002, 4),
        "created_by_agent": "skills/narrate",
    }).execute()

    # ── Cost ────────────────────────────────────────────────────────────
    char_cost = round(len(script) * 0.0002, 4)
    emit_cost(sb, run_id, property_id,
              vendor="elevenlabs", service="tts",
              units=len(script), unit_name="characters",
              unit_cost=0.0002, total_cost=char_cost,
              workflow_name="narrate", generation_reason="narration_tts")

    complete_step(sb, step_id, status="complete", metadata={
        "artifact_id": artifact_id,
        "duration_seconds": round(duration_seconds, 2),
        "character_count": len(script),
        "cost_usd": char_cost,
    })
    complete_run(sb, run_id, status="complete")

    return SkillResult.ok({
        "artifact_id": artifact_id,
        "duration_seconds": round(duration_seconds, 2),
        "character_count": len(script),
        "voice_label": voice_label,
        "cost_usd": char_cost,
        "run_id": run_id,
    })
