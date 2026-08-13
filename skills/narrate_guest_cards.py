"""
Skill: narrate_guest_cards — generate narration for guest book entries.

Each guest_evidence row gets its own narration → video_artifacts(kind='narration',
provenance='guest_book', source_reference={reviewer_name}).

Validator 9 applies: the narrated text must be the guest's own words, whole
sentences only, typo corrections allowed, no paraphrase.

Voice gender must match the writer. Gender-neutral attributions ("The Smith
Family", "Class of 2004 Reunion") are unconstrained.

For entries with both written_text and verbal_text (Eileen-style cards), the
SAME voice reads written ¶1 and verbal ¶2 as one flowing delivery.

Usage:
    from skills.narrate_guest_cards import narrate_guest_cards
    result = narrate_guest_cards("property_id")
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
from skills.direct import _split_sentences, _sentences_match

logger = logging.getLogger(__name__)

ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"
ELEVENLABS_MODEL = "eleven_multilingual_v2"

# Gender-neutral patterns — voice choice is unconstrained
_NEUTRAL_PATTERNS = ["family", "class of", "reunion", "group", "team", "club"]


def _is_gender_neutral(reviewer_name: str) -> bool:
    """Check if the reviewer name is gender-neutral (unconstrained voice)."""
    name_lower = (reviewer_name or "").lower()
    return any(p in name_lower for p in _NEUTRAL_PATTERNS)


def _select_voice(reviewer_name: str, vibe_voice_id: str) -> tuple:
    """Select voice for a guest entry.

    Gender-neutral attributions use the vibe's default voice.
    Gendered names would need male/female routing (not yet implemented
    — only gender-neutral entries exist for Vista Azule).

    Returns (voice_id, voice_label, reason).
    """
    if _is_gender_neutral(reviewer_name):
        return vibe_voice_id, "Multi-Generational voice (gender-neutral attribution)", "unconstrained"
    # TODO: gendered voice routing when male/female voice buckets exist
    return vibe_voice_id, "Multi-Generational voice (default)", "no gendered bucket configured"


def narrate_guest_cards(
    property_id: str,
    *,
    force: bool = False,
) -> SkillResult:
    """Generate narration audio for each guest book entry.

    Returns SkillResult.ok({entries_narrated, entries_skipped, cost_usd})
    """
    try:
        sb = get_substrate()
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    try:
        el_key = require_env("ELEVENLABS_API_KEY", "ElevenLabs TTS")
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    # Resolve vibe voice
    prop = sb.table("properties").select("vibe_profile").eq("id", property_id).limit(1).execute()
    vibe = (prop.data[0].get("vibe_profile") if prop.data else "") or ""
    vibe_voice_id = os.environ.get("VOICE_MULTIGENERATIONAL", "")
    if not vibe_voice_id:
        return SkillResult.failed(
            reason=f"No voice bucket configured for vibe '{vibe}'.",
            error_class="config", human_required=True,
        )

    # Load guest evidence
    ev_resp = sb.table("guest_evidence").select("*").eq(
        "property_id", property_id
    ).eq("is_guest_book", True).execute()
    entries = ev_resp.data or []

    if not entries:
        return SkillResult.noop("No guest book entries.", {})

    run_id = record_run(sb, property_id, "monthly_cycle")
    step_id = record_step(sb, run_id, "narrate_guest_cards")

    narrated = 0
    skipped = 0
    total_cost = 0.0
    voice_choices = []

    for entry in entries:
        reviewer = (entry.get("reviewer_name") or "Guest").strip()
        written = (entry.get("written_text") or "").strip()
        verbal = (entry.get("verbal_text") or "").strip()

        if not written:
            skipped += 1
            continue

        # Build the narration text: written + verbal as one flowing delivery
        # Only include verbal if it's substantive and not contaminated
        narration_text = written
        if verbal and len(verbal) > 20:
            narration_text = f"{written} {verbal}"

        # ── Voice selection ─────────────────────────────────────────────
        voice_id, voice_label, reason = _select_voice(reviewer, vibe_voice_id)
        voice_choices.append({
            "reviewer": reviewer,
            "voice_label": voice_label,
            "reason": reason,
        })

        # ── Input hash ──────────────────────────────────────────────────
        hash_input = json.dumps({
            "script": narration_text,
            "voice_id": voice_id,
            "model": ELEVENLABS_MODEL,
            "reviewer": reviewer,
        }, sort_keys=True)
        input_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        if not force:
            existing = sb.table("video_artifacts").select("artifact_id", count="exact").eq(
                "input_hash", input_hash
            ).eq("kind", "narration").eq("status", "ready").is_(
                "superseded_at", "null"
            ).limit(0).execute()
            if existing.count > 0:
                skipped += 1
                continue

        # ── Call ElevenLabs TTS ─────────────────────────────────────────
        try:
            resp = httpx.post(
                f"{ELEVENLABS_API_BASE}/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": el_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": narration_text,
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
            logger.warning("[narrate_guest_cards] TTS failed for %s: %s", reviewer, str(exc)[:80])
            skipped += 1
            continue

        # ── Upload to R2 ────────────────────────────────────────────────
        key = f"{property_id}/guest_narration/{input_hash[:12]}.mp3"
        r2_url = skills_r2_upload(key, audio_bytes, "audio/mpeg")
        duration = round(len(audio_bytes) / (128 * 1024 / 8), 2)

        # ── Write artifact ──────────────────────────────────────────────
        import uuid
        artifact_id = str(uuid.uuid4())
        char_cost = round(len(narration_text) * 0.0002, 4)
        total_cost += char_cost

        sb.table("video_artifacts").insert({
            "artifact_id": artifact_id,
            "property_id": property_id,
            "kind": "narration",
            "input_hash": input_hash,
            "storage_url": r2_url,
            "duration_seconds": duration,
            "model": ELEVENLABS_MODEL,
            "vendor": "elevenlabs",
            "script_text": narration_text,
            "voice_id": voice_id,
            "voice_label": voice_label,
            "language": "en",
            "provenance": "guest_book",
            "source_reference": {"reviewer_name": reviewer},
            "contains_verbatim_source": True,
            "status": "ready",
            "cost_estimate_usd": char_cost,
            "created_by_agent": "skills/narrate_guest_cards",
        }).execute()

        emit_cost(sb, run_id, property_id,
                  vendor="elevenlabs", service="tts_guest",
                  units=len(narration_text), unit_name="characters",
                  unit_cost=0.0002, total_cost=char_cost,
                  workflow_name="narrate_guest_cards",
                  generation_reason="guest_card_narration",
                  discriminator=reviewer[:20])

        narrated += 1
        logger.info("[narrate_guest_cards] %s: %d chars, %.2fs, $%.4f",
                   reviewer, len(narration_text), duration, char_cost)

    complete_step(sb, step_id, status="complete", metadata={
        "entries_narrated": narrated,
        "entries_skipped": skipped,
        "cost_usd": round(total_cost, 4),
        "voice_choices": voice_choices,
    })
    complete_run(sb, run_id, status="complete")

    return SkillResult.ok({
        "entries_narrated": narrated,
        "entries_skipped": skipped,
        "cost_usd": round(total_cost, 4),
        "voice_choices": voice_choices,
        "run_id": run_id,
    })
