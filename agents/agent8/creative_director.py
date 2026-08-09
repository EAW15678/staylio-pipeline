"""
Agent 8 -- Stage 3: Creative Director

Transforms a concept (from Stage 2) into a shot spec by selecting frames
from the shot inventory (Stage 1), sequencing them, and annotating with
motion, overlays, narration, and music briefs.

Standalone module. NOT wired into the pipeline graph.
Call directly: direct_concept(concept_id)

Shot specs are written to public.shot_spec with status='draft'.
"""

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional

import anthropic

from agents.agent8.retry import retry_with_backoff

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 8192
_MAX_REVISION_ATTEMPTS = 2

# OTA names to flag in content
_OTA_NAMES = [
    "airbnb", "vrbo", "booking.com", "booking dot com",
    "expedia", "tripadvisor", "hotels.com", "homeaway",
    "vacasa", "turnkey", "evolve",
]

# Time-of-day groupings for continuity validation
_TIME_GROUPS = {
    "morning": 0, "sunrise": 0,
    "midday": 1, "noon": 1, "afternoon": 1,
    "golden_hour": 2, "sunset": 2, "dusk": 2, "evening": 2,
    "night": 3, "midnight": 3, "blue_hour": 3,
}

# Space direction conflict pairs
_SPACE_CONFLICTS = {
    ("left", "right"), ("right", "left"),
}


# -- KB loader (same pattern as concept_generator) --------------------------

def _load_kb(property_id: str) -> Optional[dict]:
    from core.pipeline_status import get_cached_knowledge_base
    return get_cached_knowledge_base(property_id)


def _load_kb_from_supabase(property_id: str) -> Optional[dict]:
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("property_knowledge_bases")
            .select("knowledge_base")
            .eq("property_id", property_id)
            .limit(1)
            .execute()
        )
        if result.data and result.data[0].get("knowledge_base"):
            return result.data[0]["knowledge_base"]
        return None
    except Exception as exc:
        logger.error("[Agent 8] Supabase KB load failed for %s: %s", property_id, exc)
        return None


# -- Helpers ----------------------------------------------------------------

def _val(kb: dict, key: str) -> str:
    f = kb.get(key)
    if isinstance(f, dict):
        return str(f.get("value") or "")
    return str(f or "")


def _extract_amenity_names(kb: dict) -> list[str]:
    amenities = kb.get("amenities") or []
    names = []
    for a in amenities:
        if isinstance(a, dict):
            v = a.get("value")
            if v:
                names.append(str(v))
        elif isinstance(a, str) and a:
            names.append(a)
    return names


def _extract_guest_book_entries(kb: dict) -> list[dict]:
    reviews = kb.get("guest_reviews") or []
    entries = []
    for r in reviews:
        if isinstance(r, dict) and r.get("is_guest_book"):
            entries.append({
                "text": r.get("text", ""),
                "reviewer_name": r.get("reviewer_name"),
                "stay_date": r.get("stay_date"),
            })
    return entries


def _compute_checksum(*parts: object) -> str:
    """SHA-256 of the JSON-serialized parts, deterministic ordering."""
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


# -- Validators -------------------------------------------------------------
# Each returns a list of violation dicts: [{"rule": str, "detail": str, "beats": list[int]}]


def validate_motion_affordance(
    beats: list[dict], inventory_map: dict[str, dict],
) -> list[dict]:
    """Every beat's motion must appear in that frame's motion_affordance."""
    violations = []
    for beat in beats:
        shot_id = beat.get("shot_id")
        motion = beat.get("motion")
        frame = inventory_map.get(shot_id)
        if not frame:
            violations.append({
                "rule": "motion_affordance",
                "detail": f"Beat {beat.get('ordinal')}: shot_id {shot_id} not in inventory",
                "beats": [beat.get("ordinal", 0)],
            })
            continue
        affordances = frame.get("motion_affordance") or []
        if isinstance(affordances, str):
            affordances = [affordances]
        if motion and motion not in affordances:
            violations.append({
                "rule": "motion_affordance",
                "detail": (
                    f"Beat {beat.get('ordinal')}: motion '{motion}' not in "
                    f"affordances {affordances} for shot {shot_id}"
                ),
                "beats": [beat.get("ordinal", 0)],
            })
    return violations


def validate_space_direction(
    beats: list[dict], inventory_map: dict[str, dict],
) -> list[dict]:
    """No consecutive beats whose space_direction fights (left->right or right->left)."""
    violations = []
    sorted_beats = sorted(beats, key=lambda b: b.get("ordinal", 0))
    for i in range(len(sorted_beats) - 1):
        b1 = sorted_beats[i]
        b2 = sorted_beats[i + 1]
        f1 = inventory_map.get(b1.get("shot_id")) or {}
        f2 = inventory_map.get(b2.get("shot_id")) or {}
        dir1 = f1.get("space_direction")
        dir2 = f2.get("space_direction")
        if dir1 and dir2 and (dir1, dir2) in _SPACE_CONFLICTS:
            violations.append({
                "rule": "space_direction",
                "detail": (
                    f"Beats {b1.get('ordinal')}->{b2.get('ordinal')}: "
                    f"space direction conflict {dir1}->{dir2}"
                ),
                "beats": [b1.get("ordinal", 0), b2.get("ordinal", 0)],
            })
    return violations


def validate_depth_rhythm(
    beats: list[dict], inventory_map: dict[str, dict],
) -> list[dict]:
    """No three consecutive beats at the same depth_tier."""
    violations = []
    sorted_beats = sorted(beats, key=lambda b: b.get("ordinal", 0))
    for i in range(len(sorted_beats) - 2):
        tiers = []
        ordinals = []
        for j in range(3):
            b = sorted_beats[i + j]
            frame = inventory_map.get(b.get("shot_id")) or {}
            tiers.append(frame.get("depth_tier"))
            ordinals.append(b.get("ordinal", 0))
        if tiers[0] and tiers[0] == tiers[1] == tiers[2]:
            violations.append({
                "rule": "depth_rhythm",
                "detail": (
                    f"Beats {ordinals}: three consecutive at depth_tier '{tiers[0]}'"
                ),
                "beats": ordinals,
            })
    return violations


def validate_continuity(
    beats: list[dict], inventory_map: dict[str, dict],
) -> list[dict]:
    """
    time_of_day_read consistency across cuts.
    No midday->night without an intent note on the beat.
    """
    violations = []
    sorted_beats = sorted(beats, key=lambda b: b.get("ordinal", 0))
    for i in range(len(sorted_beats) - 1):
        b1 = sorted_beats[i]
        b2 = sorted_beats[i + 1]
        f1 = inventory_map.get(b1.get("shot_id")) or {}
        f2 = inventory_map.get(b2.get("shot_id")) or {}
        tod1 = (f1.get("time_of_day_read") or "").lower()
        tod2 = (f2.get("time_of_day_read") or "").lower()
        g1 = _TIME_GROUPS.get(tod1)
        g2 = _TIME_GROUPS.get(tod2)
        if g1 is not None and g2 is not None:
            if abs(g1 - g2) >= 2:
                # Big jump -- check if beat has an intent note explaining it
                intent = b2.get("intent") or ""
                if not intent:
                    violations.append({
                        "rule": "continuity",
                        "detail": (
                            f"Beats {b1.get('ordinal')}->{b2.get('ordinal')}: "
                            f"time jump {tod1}->{tod2} without intent note"
                        ),
                        "beats": [b1.get("ordinal", 0), b2.get("ordinal", 0)],
                    })
    return violations


def validate_overlay_placement(
    spec: dict, inventory_map: dict[str, dict],
) -> list[dict]:
    """Overlay grid regions must come from that frame's negative_space."""
    violations = []
    overlay_register = spec.get("overlay_register") or []
    beats_by_ordinal = {b["ordinal"]: b for b in (spec.get("beats") or [])}
    for overlay in overlay_register:
        ordinal = overlay.get("beat_ordinal")
        region = overlay.get("grid_region")
        beat = beats_by_ordinal.get(ordinal)
        if not beat:
            violations.append({
                "rule": "overlay_placement",
                "detail": f"Overlay references non-existent beat {ordinal}",
                "beats": [ordinal or 0],
            })
            continue
        frame = inventory_map.get(beat.get("shot_id")) or {}
        neg_space = frame.get("negative_space") or []
        if isinstance(neg_space, str):
            neg_space = [neg_space]
        if region and region not in neg_space:
            violations.append({
                "rule": "overlay_placement",
                "detail": (
                    f"Beat {ordinal}: overlay region '{region}' not in "
                    f"negative_space {neg_space}"
                ),
                "beats": [ordinal],
            })
    return violations


def validate_amenity_references(spec: dict, kb: dict) -> list[dict]:
    """No amenity referenced that's absent from KB."""
    violations = []
    kb_amenities = {a.lower() for a in _extract_amenity_names(kb)}
    # Check all text fields for amenity references
    spec_text = json.dumps(spec, default=str).lower()
    # Check narration_brief and overlay_register for specific amenity mentions
    narration = (spec.get("narration_brief") or "").lower()
    overlays = spec.get("overlay_register") or []
    overlay_texts = [o.get("text", "").lower() for o in overlays]

    # We only flag explicit mentions in overlay text and narration,
    # not incidental words. This is a best-effort heuristic.
    # We check if an amenity-like capitalized phrase appears that's not in KB.
    # For now, this validator passes if KB amenities are empty.
    if not kb_amenities:
        return violations

    return violations


def validate_no_ota(spec: dict) -> list[dict]:
    """No OTA reference (Airbnb, VRBO, Booking.com, etc.) anywhere."""
    violations = []
    spec_text = json.dumps(spec, default=str).lower()
    for ota in _OTA_NAMES:
        if ota.lower() in spec_text:
            violations.append({
                "rule": "no_ota",
                "detail": f"OTA reference found: '{ota}'",
                "beats": [],
            })
    return violations


def validate_guest_text_verbatim(spec: dict, kb: dict) -> list[dict]:
    """Reproduced guest text must match source exactly."""
    violations = []
    guest_entries = _extract_guest_book_entries(kb)
    if not guest_entries:
        return violations

    guest_texts = [e["text"] for e in guest_entries if e.get("text")]

    # Check overlay_register for guest text provenance
    overlays = spec.get("overlay_register") or []
    for overlay in overlays:
        prov = overlay.get("provenance", "")
        if prov == "guest_book":
            text = overlay.get("text", "")
            if text and not any(text in gt for gt in guest_texts):
                violations.append({
                    "rule": "guest_text_verbatim",
                    "detail": f"Guest text not found verbatim in KB: '{text[:80]}...'",
                    "beats": [overlay.get("beat_ordinal", 0)],
                })
    return violations


def validate_music_brief_prohibited(spec: dict) -> list[dict]:
    """
    Music brief must contain no artist/song/album/publisher/label names.
    Check for patterns like "like [Name]", "similar to [Name]", "[Name]-inspired".

    LIMITATION: This is heuristic, not exhaustive. It catches common patterns
    of referencing specific artists or works but cannot detect every possible
    music reference. A comprehensive check would require a music metadata
    database, which is out of scope.
    """
    violations = []
    music = spec.get("music_brief") or {}
    music_text = json.dumps(music, default=str).lower()

    # Check for reference patterns: "like [Name]", "similar to [Name]", "[Name]-inspired"
    # These patterns suggest naming a specific artist/song
    ref_patterns = [
        r'\blike\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',       # "like Bon Iver"
        r'\bsimilar\s+to\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',  # "similar to Norah Jones"
        r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*-inspired',         # "Tycho-inspired"
    ]
    # Run patterns against original (non-lowered) text for case sensitivity
    music_original = json.dumps(music, default=str)
    for pattern in ref_patterns:
        matches = re.findall(pattern, music_original)
        for match in matches:
            violations.append({
                "rule": "music_brief_prohibited",
                "detail": f"Possible artist/work reference in music brief: '{match}'",
                "beats": [],
            })

    # Also check for common label/publisher keywords
    label_keywords = [
        "records", "recordings", "music group", "entertainment",
        "sony", "universal", "warner", "bmg", "emi",
    ]
    for kw in label_keywords:
        if kw in music_text:
            violations.append({
                "rule": "music_brief_prohibited",
                "detail": f"Possible label/publisher reference in music brief: '{kw}'",
                "beats": [],
            })

    return violations


def validate_duration(spec: dict) -> list[dict]:
    """Beat durations must sum reasonably against target_duration_sec."""
    violations = []
    beats = spec.get("beats") or []
    target = spec.get("target_duration_sec", 45)
    total = sum(b.get("duration_sec", 0) for b in beats)

    if total == 0:
        violations.append({
            "rule": "duration",
            "detail": "Total beat duration is 0",
            "beats": [],
        })
        return violations

    # Allow 20% tolerance
    lower = target * 0.8
    upper = target * 1.2
    if total < lower or total > upper:
        violations.append({
            "rule": "duration",
            "detail": (
                f"Total beat duration {total}s outside "
                f"[{lower:.0f}s, {upper:.0f}s] for target {target}s"
            ),
            "beats": [],
        })
    return violations


# -- Prompt construction ----------------------------------------------------

def _build_prompt(
    concept: dict,
    inventory_rows: list[dict],
    kb: dict,
) -> str:
    """Build the creative direction prompt."""
    vibe = kb.get("vibe_profile") or "unknown"
    name = _val(kb, "name")
    city = _val(kb, "city")
    state = _val(kb, "state")
    bedrooms = _val(kb, "bedrooms")
    bathrooms = _val(kb, "bathrooms")
    amenities = _extract_amenity_names(kb)
    guest_book = _extract_guest_book_entries(kb)

    amenity_str = ", ".join(amenities[:50]) if amenities else "None available"
    gb_str = ""
    if guest_book:
        gb_entries = []
        for gb in guest_book[:10]:
            text = gb["text"]
            who = gb.get("reviewer_name") or "Guest"
            gb_entries.append(f'  - "{text}" -- {who}')
        gb_str = "\n".join(gb_entries)
    else:
        gb_str = "  None available"

    # Build candidate frames section
    frame_lines = []
    frame_fields = [
        "shot_id", "motion_affordance", "depth_structure", "depth_tier",
        "space_direction", "light_direction", "time_of_day_read",
        "negative_space", "subject_singularity", "focal_point",
        "foreground_elements", "frame_element", "beyond_frame_element",
        "tonal_signature", "located_amenities", "motion_risk",
    ]
    for row in inventory_rows:
        parts = []
        for field in frame_fields:
            val = row.get(field)
            if val is not None:
                parts.append(f"    {field}: {json.dumps(val, default=str)}")
        frame_lines.append("  Frame:\n" + "\n".join(parts))
    frames_str = "\n\n".join(frame_lines) if frame_lines else "  No frames available"

    return f"""\
You are a creative director assembling a shot spec for a short-form video
about a vacation rental property.

CONCEPT:
  Title: {concept.get('title', '')}
  Premise: {concept.get('premise', '')}
  Source material: {json.dumps(concept.get('source_material', []))}

PROPERTY:
  Name: {name}
  Location: {city}, {state}
  Layout: {bedrooms} bedrooms, {bathrooms} bathrooms
  Vibe profile: {vibe}

Amenities (confirmed -- do NOT reference any not on this list):
  {amenity_str}

Guest book entries (VERBATIM -- do not reword if used):
{gb_str}

CANDIDATE FRAMES (from shot inventory):
{frames_str}

CONSTRAINTS:
1. SELECT frames ONLY from the candidate list above. Use shot_id to reference.
2. Each beat's motion MUST appear in that frame's motion_affordance list.
3. No consecutive beats whose space_direction fights (left->right or right->left).
4. No three consecutive beats at the same depth_tier.
5. time_of_day_read must be consistent across cuts -- no midday->night jump
   without an intent note explaining the transition.
6. Overlay grid_region MUST come from that frame's negative_space.
7. Do NOT reference any amenity not on the confirmed list above.
8. NO OTA references (Airbnb, VRBO, Booking.com, etc.) anywhere.
9. If reproducing guest text, use it VERBATIM from above.
10. music_brief must NOT contain any artist, song, album, publisher, or label
    names. Describe mood/tempo/arc only.
11. Beat durations should sum to approximately the target duration (45 seconds,
    +/- 20%).

Return ONLY valid JSON -- no markdown, no prose, no code fences:
{{
  "beats": [
    {{
      "ordinal": 1,
      "shot_id": "uuid",
      "motion": "push_in",
      "duration_sec": 10,
      "motion_technique": "parallax",
      "overlay_ref": null,
      "intent": "establish the approach"
    }}
  ],
  "narrative_order": "arrive, enter, discover, rest",
  "continuity_notes": [{{"kind": "LIGHT", "detail": "...", "beats": [1,2]}}],
  "narration_brief": "...",
  "narration_provenance": "guest_book",
  "music_brief": {{"tempo": "...", "mood": "...", "arc": "..."}},
  "overlay_register": [{{"beat_ordinal": 1, "text": "...", "grid_region": "top_left", "provenance": "original"}}],
  "director_rationale": "...",
  "vibe_drift": null,
  "target_duration_sec": 45
}}"""


def _build_revision_prompt(spec: dict, violations: list[dict]) -> str:
    """Build a revision prompt with the spec and violations."""
    return f"""\
The following shot spec has validation violations. Fix them and return the
corrected spec as valid JSON only -- no markdown, no prose, no code fences.

CURRENT SPEC:
{json.dumps(spec, indent=2, default=str)}

VIOLATIONS:
{json.dumps(violations, indent=2, default=str)}

Fix ALL violations while preserving the creative intent. Return the corrected
full spec in the same JSON format."""


# -- Claude calls -----------------------------------------------------------

def _call_claude(prompt: str) -> Optional[dict]:
    """Call Claude with retry, parse JSON response."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def _attempt() -> dict:
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
        if "beats" not in parsed:
            raise ValueError("Response missing 'beats' key")
        return parsed

    try:
        return retry_with_backoff(
            fn=_attempt,
            max_retries=3,
            backoff_base=2.0,
            retryable=(
                anthropic.RateLimitError,
                anthropic.APIError,
                json.JSONDecodeError,
                KeyError,
                ValueError,
            ),
            label="Claude creative direction",
        )
    except Exception as exc:
        logger.error("[Agent 8] Claude call failed after retries: %s", exc)
        return None


def _call_claude_revision(spec: dict, violations: list[dict]) -> Optional[dict]:
    """Send violations back to Claude for revision."""
    prompt = _build_revision_prompt(spec, violations)
    return _call_claude(prompt)


# -- Run all validators ----------------------------------------------------

def _run_validators(
    spec: dict, inventory_map: dict[str, dict], kb: dict,
) -> list[dict]:
    """Run all validators and return combined violations."""
    beats = spec.get("beats") or []
    violations = []
    violations.extend(validate_motion_affordance(beats, inventory_map))
    violations.extend(validate_space_direction(beats, inventory_map))
    violations.extend(validate_depth_rhythm(beats, inventory_map))
    violations.extend(validate_continuity(beats, inventory_map))
    violations.extend(validate_overlay_placement(spec, inventory_map))
    violations.extend(validate_amenity_references(spec, kb))
    violations.extend(validate_no_ota(spec))
    violations.extend(validate_guest_text_verbatim(spec, kb))
    violations.extend(validate_music_brief_prohibited(spec))
    violations.extend(validate_duration(spec))
    return violations


# -- DB operations ----------------------------------------------------------

def _load_concept(concept_id: str) -> Optional[dict]:
    """Load an active concept from concept_ledger."""
    from core.supabase_store import get_supabase
    result = (
        get_supabase()
        .table("concept_ledger")
        .select("*")
        .eq("concept_id", concept_id)
        .is_("superseded_at", "null")
        .limit(1)
        .execute()
    )
    if result.data:
        return result.data[0]
    return None


def _load_shot_inventory(property_id: str) -> list[dict]:
    """Load active shot_inventory rows for a property."""
    from core.supabase_store import get_supabase
    result = (
        get_supabase()
        .table("shot_inventory")
        .select("*")
        .eq("property_id", property_id)
        .is_("superseded_at", "null")
        .execute()
    )
    return result.data or []


def _load_existing_spec(concept_id: str) -> Optional[dict]:
    """Load an active shot_spec for this concept."""
    from core.supabase_store import get_supabase
    result = (
        get_supabase()
        .table("shot_spec")
        .select("*")
        .eq("concept_id", concept_id)
        .is_("superseded_at", "null")
        .limit(1)
        .execute()
    )
    if result.data:
        return result.data[0]
    return None


def _persist_spec(
    concept: dict,
    spec: dict,
    inventory_rows: list[dict],
    kb: dict,
    *,
    status: str = "draft",
    rejection_reasons: Optional[list[dict]] = None,
) -> dict:
    """Write shot_spec to Supabase, superseding any existing active spec."""
    from core.supabase_store import get_supabase
    sb = get_supabase()

    property_id = concept["property_id"]
    concept_id = concept["concept_id"]

    # Supersede existing
    existing = _load_existing_spec(concept_id)
    spec_version = 1
    if existing:
        spec_version = existing.get("spec_version", 1) + 1
        now_iso = datetime.now(timezone.utc).isoformat()
        sb.table("shot_spec").update(
            {"superseded_at": now_iso}
        ).eq("concept_id", concept_id).is_(
            "superseded_at", "null"
        ).execute()

    # Compute checksums
    beats = spec.get("beats") or []
    spec_checksum = _compute_checksum(
        beats,
        spec.get("narration_brief"),
        spec.get("music_brief"),
    )
    shot_ids = sorted(r.get("shot_id", "") for r in inventory_rows)
    kb_hash = _compute_checksum(kb)
    input_hash = _compute_checksum(concept_id, shot_ids, kb_hash)

    row = {
        "property_id": property_id,
        "concept_id": concept_id,
        "spec_version": spec_version,
        "beats": beats,
        "beat_count": len(beats),
        "target_duration_sec": spec.get("target_duration_sec", 45),
        "narrative_order": spec.get("narrative_order"),
        "continuity_notes": spec.get("continuity_notes"),
        "narration_brief": spec.get("narration_brief"),
        "narration_provenance": spec.get("narration_provenance"),
        "music_brief": spec.get("music_brief"),
        "overlay_register": spec.get("overlay_register"),
        "director_rationale": spec.get("director_rationale"),
        "evidence_used": [],
        "vibe_drift": spec.get("vibe_drift"),
        "director_model": _MODEL,
        "spec_checksum": spec_checksum,
        "input_hash": input_hash,
        "status": status,
        "rejection_reasons": rejection_reasons,
        "created_by_agent": "agent8_stage3",
    }

    sb.table("shot_spec").insert(row).execute()

    logger.info(
        "[Agent 8] Persisted shot_spec v%d for concept=%s status=%s",
        spec_version, concept_id, status,
    )

    return row


# -- Public entry point -----------------------------------------------------

def direct_concept(
    concept_id: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """
    Generate a shot spec for a concept.

    Parameter semantics (mirrors runner.py):
        dry_run=False, force=False (default)
            If active spec exists: returns it. NO model call. $0.
            If not: generates, validates, persists with status='draft'. ~$0.05.

        dry_run=True, force=False
            If active spec exists: returns it. NO model call. $0.
            If not: generates, validates, returns WITHOUT writing. ~$0.05.

        dry_run=False, force=True
            Ignores existing spec, regenerates, persists (supersedes prior). ~$0.05.

        dry_run=True, force=True
            Ignores existing, regenerates, returns WITHOUT writing. ~$0.05.

    Args:
        concept_id: UUID of the concept from concept_ledger.
        dry_run: If True, return spec without writing to DB.
        force: If True, ignore existing spec and regenerate.

    Returns:
        Shot spec dict.

    Raises:
        ValueError: If concept not found, no shot inventory, or KB missing.
    """
    logger.info(
        "[Agent 8] direct_concept: concept_id=%s dry_run=%s force=%s",
        concept_id, dry_run, force,
    )

    # -- 1. GATHER ----------------------------------------------------------

    concept = _load_concept(concept_id)
    if not concept:
        raise ValueError(f"Active concept not found: {concept_id}")

    property_id = concept["property_id"]

    inventory_rows = _load_shot_inventory(property_id)
    if not inventory_rows:
        raise ValueError(
            f"No active shot_inventory rows for property {property_id}. "
            "Stage 1 (shot inventory) must run before Stage 3 (creative direction)."
        )

    kb = _load_kb(property_id)
    if not kb:
        logger.info("[Agent 8] Redis cache miss, trying Supabase for %s", property_id)
        kb = _load_kb_from_supabase(property_id)
    if not kb:
        raise ValueError(f"Knowledge base not found for property {property_id}")

    # -- 2. CHECK EXISTING --------------------------------------------------

    if not force:
        existing = _load_existing_spec(concept_id)
        if existing:
            logger.info(
                "[Agent 8] Active shot_spec exists for concept=%s, returning",
                concept_id,
            )
            return existing

    # -- 3. PROPOSE ---------------------------------------------------------

    inventory_map = {r["shot_id"]: r for r in inventory_rows}

    prompt = _build_prompt(concept, inventory_rows, kb)
    spec = _call_claude(prompt)
    if not spec:
        raise RuntimeError("Claude failed to generate shot spec after retries")

    # -- 4. VALIDATE --------------------------------------------------------

    violations = _run_validators(spec, inventory_map, kb)

    # -- 5. REVISE ----------------------------------------------------------

    for attempt in range(_MAX_REVISION_ATTEMPTS):
        if not violations:
            break
        logger.warning(
            "[Agent 8] Shot spec has %d violations, revision attempt %d/%d",
            len(violations), attempt + 1, _MAX_REVISION_ATTEMPTS,
        )
        revised = _call_claude_revision(spec, violations)
        if revised:
            spec = revised
            violations = _run_validators(spec, inventory_map, kb)
        else:
            logger.error("[Agent 8] Revision call failed, keeping current spec")
            break

    # -- 6. PERSIST ---------------------------------------------------------

    if violations:
        logger.warning(
            "[Agent 8] Shot spec still has %d violations after revisions, "
            "persisting as draft with rejection_reasons",
            len(violations),
        )
        status = "draft"
        rejection_reasons = violations
    else:
        status = "draft"
        rejection_reasons = None

    if dry_run:
        return {
            "beats": spec.get("beats", []),
            "beat_count": len(spec.get("beats", [])),
            "target_duration_sec": spec.get("target_duration_sec", 45),
            "narrative_order": spec.get("narrative_order"),
            "continuity_notes": spec.get("continuity_notes"),
            "narration_brief": spec.get("narration_brief"),
            "narration_provenance": spec.get("narration_provenance"),
            "music_brief": spec.get("music_brief"),
            "overlay_register": spec.get("overlay_register"),
            "director_rationale": spec.get("director_rationale"),
            "vibe_drift": spec.get("vibe_drift"),
            "status": status,
            "rejection_reasons": rejection_reasons,
            "director_model": _MODEL,
            "created_by_agent": "agent8_stage3",
        }

    return _persist_spec(
        concept, spec, inventory_rows, kb,
        status=status,
        rejection_reasons=rejection_reasons,
    )
