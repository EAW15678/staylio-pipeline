"""
Skill: direct — the director generates a direction from a concept.

Reads OBSERVATIONS directly (the shot_inventory relay is dead).
Vibe, wow_factor, guest_evidence, owner_context are inputs the
director reasons with — never a switch into a template library.

Includes all 11 validators ported from agents/agent8/creative_director.py
(mechanical rename: shot_inventory → observations, shot_spec → direction,
shot_id → photo_id, duration_sec → duration_seconds). Plus a revision
loop: failed validation triggers re-direction, not just rejection.

Validator 9 (guest_word_fidelity) is DETERMINISTIC — string comparison,
no model, no cost. Guest book narration must use the guest's own words:
whole sentences only, typo corrections allowed, no paraphrasing.

Usage:
    from skills.direct import direct
    result = direct("property_id", concept_id="uuid")
"""

import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone

from skills.contract import (
    SkillResult, get_substrate, require_env,
    record_run, record_step, complete_step, complete_run, emit_cost,
)

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-6"
_MAX_REVISION_ATTEMPTS = 2

# Quality self-assessment thresholds — first guesses, need tuning against
# real runs. Nobody has ever seen what scores this director actually produces.
_QUALITY_MIN_DIMENSION = 6   # any single dimension below this = shortfall
_QUALITY_MIN_AVERAGE = 7.0   # mean across all nine below this = shortfall

# The nine quality dimensions. "Template risk" is deliberately excluded —
# a director looking at one property has no idea what the previous twenty
# films looked like, so it cannot self-assess cross-property sameness.
# That needs comparison against prior directions, a different mechanism.
_QUALITY_DIMENSIONS = [
    "hook_strength", "property_differentiation", "experience",
    "motion_quality", "feature_prioritisation", "visual_variety",
    "emotional_arc", "authenticity", "booking_desire",
]

# Weak-frame threshold: frames below this width (in pixels) receive
# constrained motion when used as the opener (slow push-in only).
# Basis: Runway Gen-4 produces acceptable results at 768px+ but
# hallucinates geometry below ~600px. 768 is the lower boundary of
# "usable for motion" based on measured Brant Cottage results.
WEAK_OPENER_WIDTH_THRESHOLD = 768

# Sections that NEVER open a hero regardless of placement.
_OPENER_ALWAYS_DISQUALIFIED = {"Bedrooms", "Bathrooms"}

# Sections that can open ONLY if the frame's placement is 'outdoor'.
# Kitchen, Living Areas, and Extras may contain outdoor spaces (decks,
# outdoor kitchens) that qualify when the curator confirms placement.
_OPENER_OUTDOOR_ONLY_SECTIONS = {"Kitchen", "Living Areas", "Extras"}

# Common words filtered from guest name parts to avoid false positives.
# "The Hillis Family" → parts ["hillis"] (not ["the","hillis","family"]).
_GUEST_NAME_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "at", "in", "on", "to", "for",
    "by", "mr", "mrs", "ms", "dr", "family", "group", "team", "guest",
    "guests", "host", "hosts",
}

# OTA names — literal scan
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
_SPACE_CONFLICTS = {("left", "right"), ("right", "left")}


def _format_voice_candidates(voice_candidates) -> str:
    """Format voice candidates for the director prompt."""
    if not voice_candidates:
        return "  No voices available"
    lines = []
    for v in voice_candidates:
        gender = v.get("gender") or "unspecified"
        lines.append(f'  voice_id={v["voice_id"]}  name="{v["name"]}"  gender={gender}')
    return "\n".join(lines)


# ── Validators (ported from creative_director.py) ────────────────────────
# Each returns a list of violation dicts:
#   [{"rule": str, "detail": str, "beats": list[int]}]
# Rename: shot_id → photo_id, inventory_map → obs_map, duration_sec → duration_seconds


def validate_motion_affordance(beats: list, obs_map: dict) -> list:
    """Every beat's motion must appear in that frame's motion_affordance."""
    violations = []
    for beat in beats:
        photo_id = beat.get("photo_id")
        motion = beat.get("requested_motion")
        frame = obs_map.get(photo_id)
        if not frame:
            violations.append({
                "rule": "motion_affordance",
                "detail": f"Beat {beat.get('ordinal')}: photo_id {str(photo_id)[:8]} not in observations",
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
                    f"affordances {affordances}"
                ),
                "beats": [beat.get("ordinal", 0)],
            })
    return violations


def validate_space_direction(beats: list, obs_map: dict) -> list:
    """No consecutive beats whose space_direction fights (left↔right)."""
    violations = []
    sorted_beats = sorted(beats, key=lambda b: b.get("ordinal", 0))
    for i in range(len(sorted_beats) - 1):
        b1, b2 = sorted_beats[i], sorted_beats[i + 1]
        f1 = obs_map.get(b1.get("photo_id")) or {}
        f2 = obs_map.get(b2.get("photo_id")) or {}
        dir1 = f1.get("space_direction")
        dir2 = f2.get("space_direction")
        if dir1 and dir2 and (dir1, dir2) in _SPACE_CONFLICTS:
            violations.append({
                "rule": "space_direction",
                "detail": f"Beats {b1.get('ordinal')}->{b2.get('ordinal')}: conflict {dir1}->{dir2}",
                "beats": [b1.get("ordinal", 0), b2.get("ordinal", 0)],
            })
    return violations


def validate_depth_rhythm(beats: list, obs_map: dict) -> list:
    """No two consecutive beats at the same depth_tier.

    DIRECTOR-4 (2026-08-21): tightened from three to two consecutive,
    matching Erick's new constraint 3.
    """
    violations = []
    sorted_beats = sorted(beats, key=lambda b: b.get("ordinal", 0))
    for i in range(len(sorted_beats) - 1):
        b1 = sorted_beats[i]
        b2 = sorted_beats[i + 1]
        f1 = obs_map.get(b1.get("photo_id")) or {}
        f2 = obs_map.get(b2.get("photo_id")) or {}
        t1 = f1.get("depth_tier")
        t2 = f2.get("depth_tier")
        if t1 and t1 == t2:
            violations.append({
                "rule": "depth_rhythm",
                "detail": f"Beats {b1.get('ordinal', 0)}-{b2.get('ordinal', 0)}: two consecutive at depth_tier '{t1}'",
                "beats": [b1.get("ordinal", 0), b2.get("ordinal", 0)],
            })
    return violations


def validate_continuity(beats: list, obs_map: dict) -> list:
    """No time-of-day jumps ≥2 groups without an intent note."""
    violations = []
    sorted_beats = sorted(beats, key=lambda b: b.get("ordinal", 0))
    for i in range(len(sorted_beats) - 1):
        b1, b2 = sorted_beats[i], sorted_beats[i + 1]
        f1 = obs_map.get(b1.get("photo_id")) or {}
        f2 = obs_map.get(b2.get("photo_id")) or {}
        tod1 = (f1.get("time_of_day_read") or "").lower()
        tod2 = (f2.get("time_of_day_read") or "").lower()
        g1, g2 = _TIME_GROUPS.get(tod1), _TIME_GROUPS.get(tod2)
        if g1 is not None and g2 is not None and abs(g1 - g2) >= 2:
            intent = b2.get("intent") or ""
            if not intent:
                violations.append({
                    "rule": "continuity",
                    "detail": f"Beats {b1.get('ordinal')}->{b2.get('ordinal')}: time jump {tod1}->{tod2} without intent note",
                    "beats": [b1.get("ordinal", 0), b2.get("ordinal", 0)],
                })
    return violations


def _extract_regions(neg_space) -> set:
    """Extract region names from negative_space, handling both formats.

    Dict format (current): [{"region": "top_center", "size": ..., "contrast": ...}]
    Flat format (legacy):  ["top_center", "top_left"]
    """
    regions = set()
    if not neg_space or not isinstance(neg_space, list):
        return regions
    for item in neg_space:
        if isinstance(item, dict):
            r = item.get("region")
            if r:
                regions.add(r)
        elif isinstance(item, str):
            regions.add(item)
    return regions


def validate_overlay_placement(direction: dict, obs_map: dict) -> list:
    """Overlay grid regions must be in that frame's negative_space."""
    violations = []
    overlay_register = direction.get("overlay_register") or []
    beats_by_ordinal = {b["ordinal"]: b for b in (direction.get("beats") or [])}
    for overlay in overlay_register:
        ordinal = overlay.get("beat_ordinal")
        region = overlay.get("grid_region")
        beat = beats_by_ordinal.get(ordinal)
        if not beat:
            violations.append({"rule": "overlay_placement", "detail": f"Overlay references non-existent beat {ordinal}", "beats": [ordinal or 0]})
            continue
        frame = obs_map.get(beat.get("photo_id")) or {}
        regions = _extract_regions(frame.get("negative_space"))
        if region and region not in regions:
            violations.append({
                "rule": "overlay_placement",
                "detail": f"Beat {ordinal}: overlay region '{region}' not in negative_space regions {sorted(regions)}",
                "beats": [ordinal],
            })
    return violations


def validate_amenity_references(direction: dict, amenity_names: list) -> list:
    """Best-effort: passes if KB amenities are empty."""
    if not amenity_names:
        return []
    return []  # Same as original — best-effort heuristic, currently a pass-through


def validate_no_ota(direction: dict) -> list:
    """No OTA reference anywhere in the direction."""
    violations = []
    text = json.dumps(direction, default=str).lower()
    for ota in _OTA_NAMES:
        if ota.lower() in text:
            violations.append({"rule": "no_ota", "detail": f"OTA reference found: '{ota}'", "beats": []})
    return violations


def validate_no_guest_names(direction: dict, guest_names: list) -> list:
    """DETERMINISTIC, absolute. No guest name in narration or overlays."""
    violations = []
    if not guest_names:
        return violations

    surfaces = []
    # Check the SCRIPT (what's spoken), not the brief (instruction)
    narration = direction.get("narration_script") or direction.get("narration_brief") or ""
    if narration:
        surfaces.append((narration, "narration_script"))
    for overlay in direction.get("overlay_register") or []:
        text = overlay.get("text") or ""
        if text:
            surfaces.append((text, f"overlay beat {overlay.get('beat_ordinal', '?')}"))

    for name in guest_names:
        name_lower = name.lower()
        name_parts = [p for p in name_lower.split()
                      if len(p) > 2 and p not in _GUEST_NAME_STOPWORDS]
        for text, location in surfaces:
            text_lower = text.lower()
            if name_lower in text_lower:
                violations.append({"rule": "no_guest_names", "detail": f"Guest name '{name}' found in {location}", "beats": []})
                continue
            for part in name_parts:
                patterns = [
                    f"— {part}", f"- {part}", f"– {part}",
                    f"as {part} ", f"as {part},", f"as {part}.",
                    f"{part} said", f"{part} told", f"{part} wrote",
                    f"{part} m.", f"{part} w.",
                ]
                for pattern in patterns:
                    if pattern in text_lower:
                        violations.append({"rule": "no_guest_names", "detail": f"Guest name pattern '{pattern}' in {location} (guest: {name})", "beats": []})
                        break
    return violations


def _split_sentences(text: str) -> list:
    """Split text into sentences. Handles ., !, ? followed by space or end."""
    if not text:
        return []
    # Split on sentence-ending punctuation followed by whitespace or end
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if s.strip()]


def _normalize_for_comparison(text: str) -> list:
    """Normalize a sentence to a word list for typo-tolerant comparison.

    Strips punctuation, lowercases, expands symbols (&→and).
    Returns a list of words.
    """
    t = text.lower()
    t = t.replace("&", "and")
    # Remove punctuation except apostrophes in contractions
    t = re.sub(r"[^\w\s']", " ", t)
    # Collapse whitespace
    return t.split()


def _sentences_match(narrated: str, source: str) -> bool:
    """Check if a narrated sentence matches a source sentence.

    ALLOWED differences (typo/TTS normalisation only):
      - whitespace, punctuation
      - & → and
      - single-word spelling corrections (e.g. "grand daughter" → "granddaughter")

    FAILING differences: any word ADDED, REMOVED, or REORDERED.
    """
    n_words = _normalize_for_comparison(narrated)
    s_words = _normalize_for_comparison(source)

    if n_words == s_words:
        return True

    # Allow single-word spelling corrections: "grand daughter" (2 tokens)
    # matching "granddaughter" (1 token) or vice versa. We check if
    # concatenating adjacent word pairs produces a match.
    def _expand_compounds(words: list) -> list:
        """Try joining adjacent words to handle split/compound differences."""
        expanded = set()
        expanded.add(tuple(words))
        for i in range(len(words) - 1):
            merged = words[:i] + [words[i] + words[i + 1]] + words[i + 2:]
            expanded.add(tuple(merged))
        return expanded

    n_variants = _expand_compounds(n_words)
    s_variants = _expand_compounds(s_words)

    # Match if any variant of narrated matches any variant of source
    return bool(n_variants & s_variants)


def validate_quote_fidelity(direction: dict, guest_texts: list) -> list:
    """DETERMINISTIC guest word fidelity check. No model, no cost, always on.

    Guest book narration MUST be the guest's own words:
      - WHOLE SENTENCES ONLY (select which to include, not which words)
      - Typo corrections allowed (& → and, spelling fixes)
      - No paraphrasing, no mid-sentence cuts, no added/removed/reordered words

    Every narrated sentence must match a source sentence whole-to-whole.
    A narrated passage must never end anywhere but a source sentence boundary
    (the mid-flow guard).
    """
    violations = []
    if not guest_texts:
        return violations

    # Collect guest-attributed passages from the direction
    passages_to_check = []
    for overlay in direction.get("overlay_register") or []:
        if overlay.get("provenance") == "guest_book":
            text = overlay.get("text") or ""
            if text:
                passages_to_check.append((text, f"overlay beat {overlay.get('beat_ordinal', '?')}"))

    narration_prov = direction.get("narration_provenance") or ""
    # Validator 9 checks the SCRIPT (what's spoken), not the brief (instruction)
    narration = direction.get("narration_script") or ""
    if "guest_book" in narration_prov and narration:
        passages_to_check.append((narration, "narration_script"))

    if not passages_to_check:
        return violations

    # Build the pool of all source sentences
    source_sentences = []
    for text in guest_texts:
        source_sentences.extend(_split_sentences(text))

    for passage, location in passages_to_check:
        narrated_sentences = _split_sentences(passage)

        if not narrated_sentences:
            continue

        for ns in narrated_sentences:
            # Check if this narrated sentence matches ANY source sentence
            matched = False
            for ss in source_sentences:
                if _sentences_match(ns, ss):
                    matched = True
                    break

            if not matched:
                violations.append({
                    "rule": "guest_word_fidelity",
                    "detail": (
                        f"Narrated sentence in {location} has no whole-sentence match "
                        f"in source: '{ns[:80]}'"
                    ),
                    "beats": [],
                })

        # ── Mid-flow guard: passage must end at a source sentence boundary ──
        # The last narrated sentence must match a source sentence exactly
        # (already checked above). Additionally, if the source text continues
        # beyond the last matched sentence, the narration must not end mid-
        # sentence. This is structurally guaranteed by the sentence-level
        # matching: since every narrated sentence is a WHOLE source sentence,
        # the passage can only end at a sentence boundary. A mid-sentence
        # truncation would fail the word-level match above.

    return violations


def validate_music_brief_prohibited(direction: dict) -> list:
    """Music brief must not contain artist/song/label names."""
    violations = []
    music = direction.get("music_brief") or {}
    music_text = json.dumps(music, default=str).lower()
    music_original = json.dumps(music, default=str)

    ref_patterns = [
        r'\blike\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
        r'\bsimilar\s+to\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
        r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*-inspired',
    ]
    for pattern in ref_patterns:
        for match in re.findall(pattern, music_original):
            violations.append({"rule": "music_brief_prohibited", "detail": f"Possible artist reference in music brief: '{match}'", "beats": []})

    for kw in ["records", "recordings", "music group", "entertainment", "sony", "universal", "warner", "bmg", "emi"]:
        if kw in music_text:
            violations.append({"rule": "music_brief_prohibited", "detail": f"Possible label reference: '{kw}'", "beats": []})

    return violations


def validate_duration(direction: dict) -> list:
    """Beat durations must sum to 28-36 seconds (hard range).

    DIRECTOR-4 (2026-08-21): ceiling widened from 32 to 36, matching
    Erick's new constraint 8. Narration must never be cut off mid-sentence.
    """
    violations = []
    beats = direction.get("beats") or []
    total = sum(b.get("duration_seconds", 0) for b in beats)
    if total == 0:
        violations.append({"rule": "duration", "detail": "Total beat duration is 0", "beats": []})
        return violations
    if total < 28 or total > 36:
        violations.append({"rule": "duration", "detail": f"Total {total}s outside hard range [28s, 36s]", "beats": []})
    return violations


def validate_opening_establishes(direction: dict, obs_map: dict, photo_widths: dict = None) -> list:
    """Beat 1 must open on the property, its setting, or its standout feature.

    The director DECLARES which kind the opener is (opening_type field).
    This validator checks the declaration against the observation data.

    Deterministic — no model-assisted judgement.
    """
    violations = []
    beats = direction.get("beats") or []
    if not beats:
        return violations

    beat1 = sorted(beats, key=lambda b: b.get("ordinal", 0))[0]
    photo_id = beat1.get("photo_id")
    opening_type = direction.get("opening_type")  # "property" | "setting" | "feature"
    obs = obs_map.get(photo_id) or {}

    section = obs.get("curated_section") or ""

    placement = obs.get("placement") or "unknown"

    # FAIL: Bedrooms and Bathrooms NEVER open, regardless of placement
    if section in _OPENER_ALWAYS_DISQUALIFIED:
        violations.append({
            "rule": "opening_establishes",
            "detail": f"Beat 1 opens on '{section}' — Bedrooms and Bathrooms may never lead.",
            "beats": [beat1.get("ordinal", 1)],
        })
        return violations

    # FAIL: sections like Kitchen/Living Areas/Extras open ONLY if placement='outdoor'
    if section in _OPENER_OUTDOOR_ONLY_SECTIONS and placement != "outdoor":
        violations.append({
            "rule": "opening_establishes",
            "detail": f"Beat 1 opens on '{section}' with placement='{placement}' — only outdoor frames from this section may lead.",
            "beats": [beat1.get("ordinal", 1)],
        })
        return violations

    if not opening_type:
        violations.append({
            "rule": "opening_establishes",
            "detail": "Direction missing opening_type — the director must declare what beat 1 establishes (property, setting, or feature).",
            "beats": [1],
        })
        return violations

    # Validate the declaration against observation data
    if opening_type == "setting":
        if not obs.get("is_setting"):
            violations.append({
                "rule": "opening_establishes",
                "detail": f"Declared opening_type='setting' but frame {str(photo_id)[:8]} has is_setting=false.",
                "beats": [1],
            })

    elif opening_type == "property":
        # The frame must show the building as a whole (shows_structure=true)
        if not obs.get("shows_structure"):
            violations.append({
                "rule": "opening_establishes",
                "detail": f"Declared opening_type='property' but frame has shows_structure=false — the building must be legible as a whole.",
                "beats": [1],
            })

    elif opening_type == "feature":
        # The frame should show a specific distinguishing feature —
        # section or located_amenities should support the claim
        # This is a loose check: we just verify it's not in a detail section
        # (already checked above) and that SOMETHING is notable
        pass  # Feature is the loosest claim — any non-detail section qualifies

    # FAIL: weak opener with motion other than push_in
    photo_width = (photo_widths or {}).get(photo_id, 0)
    requested_motion = beat1.get("requested_motion") or ""
    if photo_width > 0 and photo_width < WEAK_OPENER_WIDTH_THRESHOLD:
        if requested_motion != "push_in":
            violations.append({
                "rule": "opening_establishes",
                "detail": (
                    f"Weak opener ({photo_width}px < {WEAK_OPENER_WIDTH_THRESHOLD}px threshold) "
                    f"assigned '{requested_motion}' — only 'push_in' is permitted on weak frames."
                ),
                "beats": [1],
            })

    # ── Text-bearing frames: bounded push_in ONLY ────────────────────────
    # Text in a frame (signage, house numbers, etc.) corrupts under any
    # generative model — Runway re-renders letterforms and produces
    # artefacts like BEACCH (GUARDRAIL-1). Constraint: every beat.
    # Motion must be push_in AND technique must be bounded.
    for beat in beats:
        beat_photo = beat.get("photo_id")
        beat_obs = obs_map.get(beat_photo) or {}
        beat_motion = beat.get("requested_motion") or ""
        beat_technique = beat.get("technique", "bounded")
        if beat_obs.get("contains_text"):
            if beat_motion != "push_in":
                violations.append({
                    "rule": "opening_establishes",
                    "detail": (
                        f"Beat {beat.get('ordinal', '?')}: frame contains text "
                        f"but assigned '{beat_motion}' — only 'push_in' is permitted "
                        f"on text-bearing frames."
                    ),
                    "beats": [beat.get("ordinal", 0)],
                })
            if beat_technique in ("locked", "generative"):
                violations.append({
                    "rule": "opening_establishes",
                    "detail": (
                        f"Beat {beat.get('ordinal', '?')}: frame contains text "
                        f"but technique='{beat_technique}' — generative models corrupt "
                        f"letterforms. Text-bearing frames must be technique='bounded'."
                    ),
                    "beats": [beat.get("ordinal", 0)],
                })

    return violations


def _assess_quality(direction_result: dict) -> list:
    """Assess the director's self-reported quality scores against thresholds.

    Returns a list of shortfall dicts, each naming the dimension, its score,
    and the director's own stated reason. Empty list = no shortfalls.

    Self-scoring is a soft signal — a model grading its own work grades
    generously. This may well return 8s across the board on a film Erick
    would call boring. The value is partly that the act of scoring changes
    what the director produces upstream — plausible, but unproven. Do not
    present this as a quality guarantee.
    """
    scores = direction_result.get("quality_self_score") or {}
    if not scores:
        return []

    shortfalls = []
    total = 0
    count = 0

    for dim in _QUALITY_DIMENSIONS:
        entry = scores.get(dim) or {}
        score = entry.get("score", 0) if isinstance(entry, dict) else 0
        why = entry.get("why", "") if isinstance(entry, dict) else ""
        total += score
        count += 1

        if score < _QUALITY_MIN_DIMENSION:
            shortfalls.append({
                "dimension": dim,
                "score": score,
                "threshold": _QUALITY_MIN_DIMENSION,
                "reason": "below_minimum",
                "director_says": why,
            })

    if count > 0:
        avg = total / count
        if avg < _QUALITY_MIN_AVERAGE:
            shortfalls.append({
                "dimension": "_average",
                "score": round(avg, 2),
                "threshold": _QUALITY_MIN_AVERAGE,
                "reason": "average_below_minimum",
                "director_says": f"Mean {avg:.1f} across {count} dimensions",
            })

    return shortfalls


def _determine_status(all_violations: list) -> str:
    """Determine direction status from validator violations ONLY.

    Quality shortfalls are deliberately excluded from this function's
    input — they cannot reach status by construction. This is the
    structural proof that Erick's ruling holds: shortfalls alert but
    never change status. Extracted so this separation is testable
    against the real code.
    """
    if all_violations:
        return "draft"
    return "approved"


def _alert_on_quality_shortfalls(sb, property_id, prop_data, shortfalls,
                                  run_id=None, attempts=1):
    """Surface quality shortfalls for review. NEVER raises, NEVER alters
    status — the direction ships either way.

    Erick's ruling, option C, 2026-08-21: alert AND ship. Not halt-only
    (property could get stuck with no video), not ship-silently (problems
    disappear).

    ⚠️ This is a deliberate, narrow exception to the standing rule that
    "quality annotations (needs_review) do NOT trigger alerts" (from
    escalate_halt's docstring). Erick ruled this one case an exception.
    Do not revert as a mistake — the tracker is updated.

    Extracted so this ruling is testable against the real code path
    rather than a simulation of it.
    """
    logger.info("[direct] Quality shortfalls at final attempt: %s",
                [(s["dimension"], s["score"]) for s in shortfalls])
    try:
        from skills.contract import escalate_halt
        shortfall_detail = "; ".join(
            f"{s['dimension']}: {s['score']}/{s['threshold']} — {s['director_says']}"
            for s in shortfalls
        )
        escalate_halt(
            sb, property_id,
            queue_type="content_review",
            reason_code="quality_below_threshold",
            title=f"Direction quality below threshold for {prop_data.get('name', property_id[:12])}",
            detail=f"Quality shortfalls after {attempts} attempts: {shortfall_detail}",
            property_name=prop_data.get("name"),
            run_id=run_id,
            step_name="direct",
        )
    except Exception as alert_exc:
        # A failed alert must never break a successful direction
        logger.warning("[direct] Quality alert failed: %s — continuing", str(alert_exc)[:100])


def _run_all_validators(direction: dict, obs_map: dict, guest_names: list, guest_texts: list, amenity_names: list, photo_widths: dict = None) -> list:
    """Run all 12 validators and return combined violations."""
    beats = direction.get("beats") or []
    violations = []
    # DIRECTOR-4 (2026-08-21, Erick's direction): three validators disabled
    # to test whether constraints — not the brief — produce identical videos.
    # Functions intact; can be restored by uncommenting.
    # violations += validate_motion_affordance(beats, obs_map)  # DIRECTOR-4: disabled
    violations += validate_space_direction(beats, obs_map)
    violations += validate_depth_rhythm(beats, obs_map)
    # violations += validate_continuity(beats, obs_map)  # DIRECTOR-4: disabled
    # violations += validate_overlay_placement(direction, obs_map)  # DIRECTOR-4: disabled
    violations += validate_amenity_references(direction, amenity_names)
    violations += validate_no_ota(direction)
    violations += validate_no_guest_names(direction, guest_names)
    violations += validate_quote_fidelity(direction, guest_texts)
    violations += validate_music_brief_prohibited(direction)
    violations += validate_duration(direction)
    violations += validate_opening_establishes(direction, obs_map, photo_widths)
    return violations


# ── Main skill ───────────────────────────────────────────────────────────


def direct(
    property_id: str,
    concept_id: str,
    *,
    force: bool = False,
) -> SkillResult:
    """Generate a direction (shot spec) for a concept.

    Reads observations, owner_context, guest_evidence, concept premise.
    Validates with 11 deterministic + 1 model-assisted check.
    Revision loop: up to 2 re-attempts on validation failure.

    Returns SkillResult.ok({direction_id, beat_count, violations, ...})
    """
    try:
        sb = get_substrate()
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    try:
        require_env("ANTHROPIC_API_KEY", "Claude for creative direction")
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    # ── Check existing direction ────────────────────────────────────────
    # Fetches the actual direction_id, not just a count. The noop response
    # must carry direction_id so onboard.py can pass it to narrate/music/
    # motion/assemble — same bug class as conceive's missing concept_id
    # that broke Vista Azule live on 2026-08-20.
    if not force:
        existing = sb.table("directions").select("direction_id").eq(
            "concept_id", concept_id
        ).is_("superseded_at", "null").limit(1).execute()
        if existing.data:
            return SkillResult.noop(
                f"Direction already exists for concept {concept_id[:12]}.",
                {"directions_existing": 1, "direction_id": existing.data[0]["direction_id"]},
            )

    # ── Load concept ────────────────────────────────────────────────────
    concept_resp = sb.table("concepts").select("*").eq(
        "concept_id", concept_id
    ).is_("superseded_at", "null").limit(1).execute()
    if not concept_resp.data:
        return SkillResult.failed(reason=f"Concept {concept_id[:12]} not found or superseded")
    concept = concept_resp.data[0]

    # ── Load observations → obs_map ─────────────────────────────────────
    obs_resp = sb.table("observations").select(
        "observation_id, photo_id, "
        "motion_affordance, motion_risk, depth_structure, depth_tier, "
        "space_direction, light_direction, light_quality, time_of_day_read, "
        "negative_space, foreground_elements, frame_element, "
        "beyond_frame_element, subject_singularity, focal_point, "
        "tonal_signature, located_amenities, "
        "role, curated_section, quality_score, alt_text, "
        "is_setting, setting_subject, placement, located_amenities, "
        "shows_structure, structure_view, contains_text, text_content"
    ).eq("property_id", property_id).is_("superseded_at", "null").execute()
    observations = obs_resp.data or []

    if not observations:
        return SkillResult.failed(reason="No observations — run observe first")

    obs_map = {o["photo_id"]: o for o in observations}

    # Get rendition URLs
    photo_ids = [o["photo_id"] for o in observations]
    renditions_by_photo = {}
    for pid in photo_ids:
        rend = sb.table("renditions").select("kind, storage_url").eq("photo_id", pid).execute()
        for r in (rend.data or []):
            renditions_by_photo.setdefault(pid, {})[r["kind"]] = r["storage_url"]

    # ── Load photo widths for weak-opener check ──────────────────────────
    photo_width_resp = sb.table("photographs").select("photo_id, image_width").eq(
        "property_id", property_id).eq("is_canonical", True).execute()
    photo_widths = {p["photo_id"]: p.get("image_width") or 0 for p in (photo_width_resp.data or [])}

    # ── Load context ────────────────────────────────────────────────────
    prop = sb.table("properties").select("name, vibe_profile, city, state_region, amenities").eq("id", property_id).limit(1).execute()
    prop_data = prop.data[0] if prop.data else {}
    from core.page_builder.amenity_taxonomy import extract_amenity_names
    amenity_names = extract_amenity_names(prop_data.get("amenities") or [])

    ctx_resp = sb.table("owner_context").select("owner_story, wow_factor, hidden_gems, guest_love").eq(
        "property_id", property_id).is_("superseded_at", "null").order("version", desc=True).limit(1).execute()
    owner_ctx = ctx_resp.data[0] if ctx_resp.data else {}

    ev_resp = sb.table("guest_evidence").select("written_text, verbal_text, reviewer_name, is_guest_book").eq(
        "property_id", property_id).eq("is_guest_book", True).limit(10).execute()
    guest_evidence = ev_resp.data or []
    guest_names = [e["reviewer_name"] for e in guest_evidence if e.get("reviewer_name")]
    guest_texts = [e.get("written_text") or e.get("verbal_text") or "" for e in guest_evidence if (e.get("written_text") or e.get("verbal_text"))]

    # ── Build frames for prompt ─────────────────────────────────────────
    frames = []
    for obs in observations:
        pid = obs["photo_id"]
        urls = renditions_by_photo.get(pid, {})
        display_url = urls.get("enhanced") or urls.get("original", "")
        if not display_url:
            continue
        frames.append({**obs, "url": display_url})

    if not frames:
        return SkillResult.failed(reason="No frames with rendition URLs")

    # ── Fetch voice candidates LIVE from ElevenLabs ─────────────────────
    vibe = prop_data.get("vibe_profile") or ""
    try:
        from skills.voice_buckets import fetch_vibe_voices
        voice_candidates = fetch_vibe_voices(sb, vibe)
    except ValueError as e:
        return SkillResult.failed(
            reason=str(e),
            attempted=1, succeeded=0, failed_count=1,
            error_class="config", human_required=True,
        )

    # ── Record run ──────────────────────────────────────────────────────
    run_id = record_run(sb, property_id, "monthly_cycle")
    step_id = record_step(sb, run_id, "direct")

    # ── Direction + Revision Loop ───────────────────────────────────────
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    direction_result = None
    all_violations = []
    quality_shortfalls = []
    attempt = 0

    for attempt in range(_MAX_REVISION_ATTEMPTS + 1):
        if attempt == 0:
            direction_result = _call_director(client, concept, frames, prop_data, owner_ctx, guest_evidence, voice_candidates)
        else:
            # Build revision issues from both violations AND quality shortfalls
            revision_issues = list(all_violations)
            for sf in quality_shortfalls:
                revision_issues.append({
                    "rule": f"quality_{sf['dimension']}",
                    "detail": f"Self-score {sf['score']}/{sf['threshold']} — {sf['director_says']}",
                    "beats": [],
                })
            direction_result = _call_revision(client, concept, frames, prop_data, owner_ctx, guest_evidence, voice_candidates, direction_result, revision_issues)

        if not direction_result:
            break

        # Validate (11 content validators + voice membership check)
        all_violations = _run_all_validators(direction_result, obs_map, guest_names, guest_texts, amenity_names, photo_widths)

        # Validate voice choice is a member of the vibe's collection
        chosen_voice = direction_result.get("narration_voice_id")
        if chosen_voice:
            valid_ids = {v["voice_id"] for v in voice_candidates}
            if chosen_voice not in valid_ids:
                all_violations.append({
                    "rule": "voice_membership",
                    "detail": f"Chosen voice '{chosen_voice}' is not in the vibe's collection",
                    "beats": [],
                })

        # Assess quality (separate from violations — shortfalls do not
        # change status or appear in rejection_reasons)
        quality_shortfalls = _assess_quality(direction_result)

        needs_revision = bool(all_violations) or bool(quality_shortfalls)
        if not needs_revision:
            logger.info("[direct] Attempt %d: all validators passed, quality OK", attempt + 1)
            break
        else:
            logger.info("[direct] Attempt %d: %d violations, %d quality shortfalls, rules=%s, quality=%s",
                       attempt + 1, len(all_violations), len(quality_shortfalls),
                       [v["rule"] for v in all_violations],
                       [s["dimension"] for s in quality_shortfalls])

    if not direction_result:
        complete_step(sb, step_id, status="failed", error_message="Director returned no result")
        complete_run(sb, run_id, status="failed")
        return SkillResult.failed(reason="Director returned no result", attempted=1, succeeded=0, failed_count=1, error_class="vendor")

    # ── Determine status ────────────────────────────────────────────────
    # Quality shortfalls are deliberately excluded — _determine_status
    # takes only validator violations. See its docstring for why.
    status = _determine_status(all_violations)
    if all_violations:
        logger.warning("[direct] Persisting with %d violations after %d attempts", len(all_violations), attempt + 1)

    final_shortfalls = quality_shortfalls if quality_shortfalls else None
    if final_shortfalls:
        _alert_on_quality_shortfalls(
            sb, property_id, prop_data, final_shortfalls,
            run_id=run_id, attempts=attempt + 1)

    # ── Compute input hash ──────────────────────────────────────────────
    hash_input = json.dumps({
        "concept_id": concept_id,
        "frame_ids": sorted([f["photo_id"] for f in frames]),
        "premise": concept.get("premise"),
    }, sort_keys=True)
    input_hash = hashlib.sha256(hash_input.encode()).hexdigest()

    # ── Supersede + write ───────────────────────────────────────────────
    now_iso = datetime.now(timezone.utc).isoformat()
    sb.table("directions").update({"superseded_at": now_iso}).eq(
        "concept_id", concept_id).is_("superseded_at", "null").execute()

    direction_id = str(uuid.uuid4())
    sb.table("directions").insert({
        "direction_id": direction_id,
        "property_id": property_id,
        "concept_id": concept_id,
        "beats": direction_result.get("beats", []),
        "beat_count": len(direction_result.get("beats", [])),
        "target_duration_sec": direction_result.get("target_duration_sec", 30),
        "opening_type": direction_result.get("opening_type"),
        "narrative_order": direction_result.get("narrative_order"),
        "continuity_notes": direction_result.get("continuity_notes", []),
        "narration_brief": direction_result.get("narration_brief"),
        "narration_script": direction_result.get("narration_script"),
        "narration_provenance": direction_result.get("narration_provenance"),
        "narration_voice_id": direction_result.get("narration_voice_id"),
        "narration_voice_rationale": direction_result.get("narration_voice_rationale"),
        "music_brief": direction_result.get("music_brief", {}),
        "overlay_register": direction_result.get("overlay_register", []),
        "director_rationale": direction_result.get("director_rationale"),
        "hero_proposition": direction_result.get("hero_proposition"),
        "booking_drivers": direction_result.get("booking_drivers"),
        "directing_mode": direction_result.get("directing_mode"),
        "mode_rationale": direction_result.get("mode_rationale"),
        "director_model": _MODEL,
        "evidence_used": direction_result.get("evidence_used", []),
        "vibe_drift": direction_result.get("vibe_drift"),
        "input_hash": input_hash,
        "quality_self_score": direction_result.get("quality_self_score"),
        "quality_shortfalls": final_shortfalls,
        "rejection_reasons": all_violations if all_violations else None,
        "status": status,
        "created_by_agent": "skills/direct",
    }).execute()

    # ── Cost ────────────────────────────────────────────────────────────
    cost = 0.05 * (attempt + 1)  # ~$0.05 per attempt
    emit_cost(sb, run_id, property_id,
              vendor="anthropic", service="claude_creative_direction",
              units=attempt + 1, unit_name="attempts",
              unit_cost=0.05, total_cost=round(cost, 4),
              workflow_name="direct", generation_reason="creative_direction")

    complete_step(sb, step_id, status="complete", metadata={
        "direction_id": direction_id,
        "beat_count": len(direction_result.get("beats", [])),
        "frames_considered": len(frames),
        "attempts": attempt + 1,
        "violations": len(all_violations),
        "status": status,
    })
    complete_run(sb, run_id, status="complete")

    return SkillResult.ok({
        "direction_id": direction_id,
        "beat_count": len(direction_result.get("beats", [])),
        "frames_considered": len(frames),
        "attempts": attempt + 1,
        "violations": len(all_violations),
        "violation_rules": [v["rule"] for v in all_violations],
        "status": status,
        "run_id": run_id,
    })


def _call_director(client, concept, frames, prop, owner_ctx, guest_evidence, voice_candidates) -> dict:
    """Initial direction call to Claude."""
    return _call_claude(client, _build_prompt(concept, frames, prop, owner_ctx, guest_evidence, voice_candidates))


def _call_revision(client, concept, frames, prop, owner_ctx, guest_evidence, voice_candidates, prev_direction, violations) -> dict:
    """Revision call: send previous direction + violations back to Claude."""
    base_prompt = _build_prompt(concept, frames, prop, owner_ctx, guest_evidence, voice_candidates)
    violation_text = "\n".join(f"- [{v['rule']}] {v['detail']}" for v in violations)
    revision_prompt = f"""{base_prompt}

YOUR PREVIOUS DIRECTION HAD THESE VIOLATIONS:
{violation_text}

Fix each violation while preserving the creative intent. Return the corrected direction as JSON."""

    return _call_claude(client, revision_prompt)


def _call_claude(client, prompt: str) -> dict:
    """Call Claude and extract JSON from response."""
    try:
        response = client.messages.create(
            model=_MODEL,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        # Find the first '{' and use raw_decode to parse exactly one
        # JSON object, ignoring any trailing text Claude appends.
        start = text.find('{')
        if start == -1:
            return None
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text, start)
        return obj
    except Exception as exc:
        logger.error("[direct] Claude call failed: %s", str(exc)[:120])
        return None


def _build_prompt(concept, frames, prop, owner_ctx, guest_evidence, voice_candidates=None) -> str:
    """Build the creative direction prompt for the LANDING PAGE HERO VIDEO.

    This is specifically the hero film for the property's own landing page —
    seen by someone who has already clicked through and is evaluating whether
    to book. It is NOT the brief for social media video generation.

    Social video will need its own brief when that path is built. The two
    are expected to share a majority of craft guidance but differ on:
    - Audience intent: landing page = evaluating; social = hasn't decided to care
    - Length: hero = 28-32s; social = shorter, platform-dependent
    - Aspect ratio: hero = 16:9; social = predominantly 9:16
    - Opening: hero establishes the property; social leads with the most
      arresting frame, whatever it is
    - Repetition: one hero per property; social is continuous, many pieces
    - Narration/music: hero = full narrated script; social = often without

    Do NOT add conditionals or mode flags for social in this function.
    Build the social director against its own requirements when the path
    exists, not guessed at here.
    """
    vibe = prop.get("vibe_profile") or "multigenerational_retreat"
    name = prop.get("name") or "the property"

    guest_snippets = []
    for ev in guest_evidence[:5]:
        text = ev.get("written_text") or ev.get("verbal_text") or ""
        if text.strip():
            reviewer = ev.get("reviewer_name") or "Guest"
            guest_snippets.append(f'  - "{text.strip()[:200]}" — {reviewer}')

    frame_lines = []
    fields = ["photo_id", "motion_affordance", "motion_risk", "depth_tier",
              "space_direction", "time_of_day_read", "negative_space",
              "focal_point", "subject_singularity", "located_amenities",
              "curated_section", "quality_score"]
    for f in frames[:30]:
        parts = [f"    {k}: {json.dumps(f.get(k), default=str)}" for k in fields if f.get(k) is not None]
        frame_lines.append("  Frame:\n" + "\n".join(parts))

    from core.page_builder.amenity_taxonomy import extract_amenity_names as _ean
    amenities = _ean(prop.get("amenities") or [])
    amenity_str = ", ".join(amenities[:50]) if amenities else "None available"

    return f"""You are the creative director of a 30-second film about a vacation property. This is the hero film for the property's own landing page — seen by someone who has already clicked through and is evaluating whether to book.

Your job is not to show eight attractive images. It is to make a viewer imagine themselves already on vacation at this specific property.

Do not make a video about the property's photographs. Make a video about the vacation those photographs make possible. The property must stay truthful; the experience can come alive.

Every film should answer three questions: Why should I stop scrolling? What makes this property special? What would it feel like to stay here?

═══════════════════════════════════════════════════════════════════
PART 1 — THE BRIEF: THINK BEFORE YOU SELECT BEATS
═══════════════════════════════════════════════════════════════════

Before selecting any beats, reason through these steps and record your conclusions in the output JSON.

HERO PROPOSITION: Identify the single strongest reason someone would book this property, and disproportionately emphasise it. Rank the top three booking drivers visible in the frames or verifiable from the property data; name #1. Examples: beachfront escape, spectacular pool house, luxury family gathering, romantic retreat, entertainer's house, chef's kitchen, extraordinary views, indoor-outdoor living, seclusion.

DIRECTING MODE: There is no universal storyboard. Infer from the assets and the property what kind of film this property deserves, then work with freedom inside that choice:
  - The Reveal (architecture/view)
  - The Weekend (experiential)
  - The Escape (relaxation)
  - The Gathering (group/family)
  - The Playground (amenities/activity)
  - The Retreat (privacy/romance)
  - The Design Film (interiors)
  - The Destination (property plus surroundings)
Inferred, never rotated randomly.

RANK THE MATERIAL: Not every frame deserves screen time. Score available frames on Visual Impact x Guest Appeal x Uniqueness x Motion Potential. Best-scoring material gets the strongest beats. Unique beats generic. Spend screen time according to booking value — a spectacular pool earns far more than a secondary bathroom. Avoid redundant coverage: two frames of essentially the same bedroom rarely deserve two beats.

SCORE YOUR OWN WORK: After selecting beats and building the film, honestly assess the result on nine dimensions (1-10 each, with a one-line justification). These scores are checked against thresholds — shortfalls trigger revision. Be honest, not generous.
  - hook_strength: Would the first two seconds stop someone scrolling?
  - property_differentiation: Could this film belong to another rental, or is it distinctly THIS property?
  - experience: Can the viewer imagine being there?
  - motion_quality: Does this feel filmed rather than animated?
  - feature_prioritisation: Are the strongest attributes getting the most attention?
  - visual_variety: Are scale, movement, composition and pacing varied?
  - emotional_arc: Does the film build toward something?
  - authenticity: Is everything implied genuinely possible here?
  - booking_desire: Does this make the property more desirable?

═══════════════════════════════════════════════════════════════════
PART 2 — CRAFT GUIDANCE: EXERCISE YOUR JUDGEMENT
═══════════════════════════════════════════════════════════════════

These are not numbered rules. They are guidance you exercise judgement over.

STORY: Tell a micro-story, not a room sequence. Avoid Exterior > Living > Kitchen > Bedroom > Bath > Pool. Prefer an experiential progression: Arrival > Reveal > Exploration > Experience > Indulgence > Escape. Rooms are ingredients; the experience is the story. Build an emotional arc — intrigue, discovery, excitement, desire, satisfaction. The film should not sit at one emotional level throughout.

THE OPENING: The first 1-3 seconds must carry one of the strongest visual moments available and make the viewer ask "where is this?" or "what does the rest of this place look like?" Establish the property — but that does not mean one literal shot type. From the street, from the beach, from the air, sweeping through open doors, a rapid push toward the most distinctive thing about the building. Showing the house is the intent; a photograph of the front door is only one way to do it.

THE CLOSE: Should create desire, not finish coverage. Sunset over the property, a pullback from an illuminated exterior, a pool glowing at dusk, an aerial retreat. Last impression: "I want to stay there." Do not try to show everything — leave the viewer wanting to open the listing.

PEOPLE: People are permitted (ruled 2026-08-20 by Erick). The constraint is not "no people" — it is that the property stays the hero. People convey scale, energy and implied experience: a hand reaching for coffee, feet entering the pool, someone walking toward a balcony, silhouettes, a couple seen from behind, a group around a dining table, hands preparing food. Not: posed faces at camera, influencer-style performance, stock-family theatrics. Show what guests actually DO there — dining room becomes friends sharing dinner; pool becomes someone entering the water. But do not force people into every film — a secluded cabin may be stronger as fire, rain and steam with nobody in frame. Match lifestyle to property identity. How many, doing what, or none at all, is your judgement. Do NOT depict children — that remains unruled and is out of scope.

MOTION: Motion should reveal something — push through a doorway to reveal the view, orbit an island to reveal the living space, rise above a railing to reveal landscape. Avoid movement merely because animation is possible. Simulate a real camera: dolly, crane, drone, slider, steadicam, orbit, push, pull, reveal. Vary movement across the film — do not use eight consecutive slow push-ins, and do not repeat the same move more than twice unless it is a deliberate motif. Mix shot scale: establishing, wide, medium, detail, hero wide. Detail shots create texture and luxury. Use environmental motion aggressively — water, fire, curtains, steam, foliage, reflections. Static architecture plus a moving environment is often the most believable footage available to you.

PACING: Beats are NOT equal length. Eight beats should not become eight identical clips. Timing follows visual importance, motion and story — e.g. 2.0 > 2.8 > 3.5 > 2.1 > 4.2 > 3.0 > 3.7 > 4.5. Accelerate and decelerate intentionally: high-energy sequences cut faster, reveals deserve breathing room. A spectacular view may earn four seconds while a bathroom detail gets one and a half. Give the biggest reveal enough time — do not waste the property's strongest asset in a rapid montage. Open with energy — motion should not be slow in the first beats.

CONTINUITY: Cut on action — movement beginning in one shot should continue into the next. Camera moves right, next continues right. Someone approaches the pool, next shot begins at water level. This is what makes independently generated clips feel like one film.

AVOIDING SAMENESS: For decisions that are equally valid, vary: opening shot, camera direction, shot duration, human presence, time-of-day treatment, detail selection, motion intensity, sequencing, closing shot. Not randomness for its own sake — controlled variation inside the directing mode you chose. A $4M oceanfront house and a $220/night cabin should not be the same film with different images in it.

KNOW YOUR CURRENT LIMITS: You have two ways to create motion. Bounded motion moves a viewport across a flat still — real camera movement, but nothing in the frame itself moves. Locked technique holds the camera and animates the content — this is the only way to put genuine life into a frame. True parallax and layered foreground/midground/background motion are not available. Music is generated after your beats are sized, so you cannot cut to musical structure. Work with what you have rather than describing what you can't.

═══════════════════════════════════════════════════════════════════
PROPERTY & SOURCE MATERIAL
═══════════════════════════════════════════════════════════════════

CONCEPT:
  Title: {concept.get('title', '')}
  Premise: {concept.get('premise', '')}

PROPERTY:
  Name: {name}
  Location: {prop.get('city', '')}, {prop.get('state_region', '')}
  Vibe profile: {vibe}

Owner story: {owner_ctx.get('owner_story', 'N/A')}
Wow factor: {owner_ctx.get('wow_factor', 'N/A')}
Hidden gems: {owner_ctx.get('hidden_gems', 'N/A')}

Amenities (confirmed — do NOT reference any not on this list):
  {amenity_str}

Guest book entries (VERBATIM — do not reword if used):
{chr(10).join(guest_snippets) if guest_snippets else '  None available'}

CANDIDATE FRAMES (from observations):
{chr(10).join(frame_lines) if frame_lines else '  No frames available'}

AVAILABLE NARRATOR VOICES (choose ONE for the hero narration):
{_format_voice_candidates(voice_candidates)}

═══════════════════════════════════════════════════════════════════
PART 3 — HARD CONSTRAINTS (non-negotiable)
═══════════════════════════════════════════════════════════════════

The following are non-negotiable. Each is checked deterministically and
a violation sends this direction back for revision. They are a floor,
not the brief — everything above is where the film actually gets made.

1. SELECT frames ONLY from the candidate list above. Use photo_id to reference.

2. No consecutive beats whose space_direction fights (left>right or right>left).

3. No two consecutive beats at the same depth_tier.

4. Do NOT reference any amenity not on the confirmed list.

5. NO OTA references (Airbnb, VRBO, Booking.com, etc.) anywhere.

6. NO guest names in narration_script or overlays — not in any form.

7. Music brief: NO artist, song, album, publisher, or label names.
   No lyrics in music to compete with narration.

8. Beat duration_seconds MUST sum to 28-36 seconds (hard range).
   Narration must never be cut off mid-sentence.

9. narration_voice_id MUST be one of the voice_ids from AVAILABLE NARRATOR
   VOICES above. Choose the voice that best serves the vibe and narration tone.

10. THE OPENING RULE: Beat 1 MUST open on one of three kinds of shot:
    - "property" — the property itself, read as a whole (Exterior section)
    - "setting" — what situates the property (a frame with is_setting=true)
    - "feature" — an EXTERIOR standout feature (Pool, deck, rooftop, view)
    Beat 1 MUST NOT open on ANY interior section (Bedrooms, Bathrooms,
    Kitchen, Living Areas, Extras). Interior frames are fully available
    for beats 2 onward — this rule governs beat 1 ONLY.
    PREFERENCE: property or setting over feature. Open on the property
    itself or on what situates it whenever a frame of either is available
    and strong enough to carry the opening. Use a feature opener when it
    is genuinely the strongest opening available to you — not as a
    default, and not merely because a feature frame is the prettiest
    photograph in the set. This is a preference you exercise judgment on,
    not a hard rule: a feature opener remains permitted and is not a
    violation.
    You MUST declare opening_type in the output JSON.
    If the opener is a frame below 768px wide (a weak opener), assign ONLY
    "push_in" as its requested_motion — no parallax, no pans.

NARRATION FIELDS — TWO SEPARATE OUTPUTS:
- narration_brief: your INSTRUCTION to yourself about tone, pacing, approach.
  NOT sent to TTS. Example: "Open warm, let spaces breathe, close emotional."
- narration_script: the EXACT WORDS the narrator speaks. This is what the
  audience hears. Write it as finished speech, not as direction.
  If narration_provenance is "guest_book", narration_script must use whole
  source sentences only (validated deterministically).

11. TECHNIQUE PER BEAT: each beat has a "technique" field. Choose per
    beat based on what the frame actually contains — there is no default.
    - "bounded": the camera is a viewport travelling across the still
      photograph. Nothing in the frame moves. $0 per clip.
    - "depth": the camera performs an authored parallax move through
      a depth-projected version of the still photograph. The scene gains
      dimensional depth — foreground and background separate and move at
      different rates, like a real camera. Best for wide-angle exteriors,
      pool/deck shots, and rooms with strong foreground-to-background
      layering. depth_structure must be "deep" for this to work.
      Depth beats can be any duration. Set intensity to "restrained"
      (4% displacement, subtle) or "moderate" (8%, visible but clean).
      ⚠️ depth does NOT animate content — water stays still, foliage
      does not sway. If the frame contains genuine life (water, fire,
      foliage), prefer "locked" to bring that life to the film.
      ⚠️ depth is INELIGIBLE for frames with thin foreground structures
      (chandeliers, pendant lights, balusters, thin chair legs) — these
      smear under reprojection. The system will automatically fall back
      to bounded if ineligible; you do not need to guard against this.
    - "locked": the camera holds still and the CONTENT animates — water
      rippling, a pool surface catching light, fire flickering, foliage
      and grass moving, steam rising, curtains breathing, clouds
      drifting. Costs Runway rates. State in motion_prompt exactly what
      you expect to move.
      A locked beat MUST be exactly 5.0 or 10.0 seconds — Runway
      accepts no other duration. Plan your pacing around this: locked
      beats are 5s or 10s, bounded beats can be any length. A 3.5-second
      locked beat is impossible and will fail.
    - "generative": full Runway generation with camera movement.
    REACH FOR "locked" ON ANY FRAME WITH GENUINE LIFE IN IT. A pool, the
    ocean, a fire pit, trees, sky, a hot tub, anything with water or
    flame or foliage — these are the frames that make a film feel filmed
    rather than assembled. A film composed entirely of bounded beats is
    a slideshow, and a slideshow is a FAILED direction regardless of how
    well it scores on anything else.
    Use "depth" for exterior or wide-angle frames that have strong
    depth layering but nothing that should naturally move — a driveway
    at dusk, a building facade, a rooftop view, an empty deck.
    Bounded is the right choice for frames with nothing moving AND
    shallow depth — a made bed, a vanity, a detail shot. Use it there
    deliberately, not as a fallback.
    Text-bearing frames (contains_text=true) MUST be technique="bounded"
    with requested_motion="push_in". Never locked, never generative,
    never depth.

Return ONLY valid JSON:
{{
  "hero_proposition": "the single strongest booking driver, one line",
  "booking_drivers": ["#1 driver", "#2 driver", "#3 driver"],
  "directing_mode": "The Reveal | The Weekend | The Escape | The Gathering | The Playground | The Retreat | The Design Film | The Destination",
  "mode_rationale": "why this mode for this property, one line",
  "beats": [
    {{"ordinal": 1, "photo_id": "uuid", "requested_motion": "tilt_up", "motion_prompt": "...", "duration_seconds": 4, "technique": "locked"}},
    {{"ordinal": 2, "photo_id": "uuid", "requested_motion": "lateral_right", "motion_prompt": "", "duration_seconds": 3, "technique": "depth", "intensity": "restrained"}},
    {{"ordinal": 3, "photo_id": "uuid", "requested_motion": "push_in", "motion_prompt": "...", "duration_seconds": 3, "technique": "bounded"}},
    ...
  ],
  "opening_type": "property" or "setting" or "feature",
  "narration_brief": "Instruction to the director about tone and approach",
  "narration_script": "The exact words the narrator speaks aloud",
  "narration_provenance": "original" or "guest_book",
  "narration_voice_id": "the voice_id you chose from the AVAILABLE NARRATOR VOICES list",
  "narration_voice_rationale": "one-line reason for the voice choice",
  "music_brief": {{"mood": "...", "tempo": "...", "instruments": "..."}},
  "overlay_register": [],
  "narrative_order": "...",
  "continuity_notes": [],
  "director_rationale": "...",
  "evidence_used": [],
  "vibe_drift": null,
  "quality_self_score": {{
    "hook_strength": {{"score": 1-10, "why": "one line"}},
    "property_differentiation": {{"score": 1-10, "why": "one line"}},
    "experience": {{"score": 1-10, "why": "one line"}},
    "motion_quality": {{"score": 1-10, "why": "one line"}},
    "feature_prioritisation": {{"score": 1-10, "why": "one line"}},
    "visual_variety": {{"score": 1-10, "why": "one line"}},
    "emotional_arc": {{"score": 1-10, "why": "one line"}},
    "authenticity": {{"score": 1-10, "why": "one line"}},
    "booking_desire": {{"score": 1-10, "why": "one line"}}
  }},
  "target_duration_sec": 30
}}"""
