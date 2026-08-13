"""
VIDEO-1B: Tests for all 11 direction validators.

Each validator is tested with a passing case and a failing case.
Validator 9 (quote_fidelity) is tested in stubbed mode (returns "uncertain").
"""

from skills.direct import (
    validate_motion_affordance,
    validate_space_direction,
    validate_depth_rhythm,
    validate_continuity,
    validate_overlay_placement,
    validate_amenity_references,
    validate_no_ota,
    validate_no_guest_names,
    validate_quote_fidelity,
    validate_music_brief_prohibited,
    validate_duration,
    _run_all_validators,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

OBS_MAP = {
    "photo-A": {
        "motion_affordance": ["push_in", "pan_left"],
        "space_direction": "left",
        "depth_tier": "foreground",
        "time_of_day_read": "morning",
        "negative_space": ["top_left", "top_center"],
    },
    "photo-B": {
        "motion_affordance": ["pan_right", "tilt_up"],
        "space_direction": "right",
        "depth_tier": "midground",
        "time_of_day_read": "midday",
        "negative_space": ["bottom_center"],
    },
    "photo-C": {
        "motion_affordance": ["push_in", "orbit"],
        "space_direction": "center",
        "depth_tier": "background",
        "time_of_day_read": "golden_hour",
        "negative_space": ["top_right"],
    },
    "photo-D": {
        "motion_affordance": ["pan_left"],
        "space_direction": "left",
        "depth_tier": "foreground",
        "time_of_day_read": "morning",
        "negative_space": [],
    },
}


# ── 1. Motion Affordance ─────────────────────────────────────────────────

def test_motion_affordance_pass():
    beats = [{"ordinal": 1, "photo_id": "photo-A", "requested_motion": "push_in"}]
    assert validate_motion_affordance(beats, OBS_MAP) == []


def test_motion_affordance_fail():
    beats = [{"ordinal": 1, "photo_id": "photo-A", "requested_motion": "orbit"}]
    violations = validate_motion_affordance(beats, OBS_MAP)
    assert len(violations) == 1
    assert violations[0]["rule"] == "motion_affordance"


def test_motion_affordance_missing_photo():
    beats = [{"ordinal": 1, "photo_id": "photo-MISSING", "requested_motion": "push_in"}]
    violations = validate_motion_affordance(beats, OBS_MAP)
    assert len(violations) == 1
    assert "not in observations" in violations[0]["detail"]


# ── 2. Space Direction ───────────────────────────────────────────────────

def test_space_direction_pass():
    beats = [
        {"ordinal": 1, "photo_id": "photo-A"},  # left
        {"ordinal": 2, "photo_id": "photo-C"},  # center
    ]
    assert validate_space_direction(beats, OBS_MAP) == []


def test_space_direction_fail():
    beats = [
        {"ordinal": 1, "photo_id": "photo-A"},  # left
        {"ordinal": 2, "photo_id": "photo-B"},  # right — conflict!
    ]
    violations = validate_space_direction(beats, OBS_MAP)
    assert len(violations) == 1
    assert violations[0]["rule"] == "space_direction"


# ── 3. Depth Rhythm ──────────────────────────────────────────────────────

def test_depth_rhythm_pass():
    beats = [
        {"ordinal": 1, "photo_id": "photo-A"},  # foreground
        {"ordinal": 2, "photo_id": "photo-B"},  # midground
        {"ordinal": 3, "photo_id": "photo-A"},  # foreground
    ]
    assert validate_depth_rhythm(beats, OBS_MAP) == []


def test_depth_rhythm_fail():
    beats = [
        {"ordinal": 1, "photo_id": "photo-A"},  # foreground
        {"ordinal": 2, "photo_id": "photo-D"},  # foreground
        {"ordinal": 3, "photo_id": "photo-A"},  # foreground — 3 in a row!
    ]
    violations = validate_depth_rhythm(beats, OBS_MAP)
    assert len(violations) == 1
    assert violations[0]["rule"] == "depth_rhythm"


# ── 4. Continuity ────────────────────────────────────────────────────────

def test_continuity_pass():
    beats = [
        {"ordinal": 1, "photo_id": "photo-A"},  # morning (group 0)
        {"ordinal": 2, "photo_id": "photo-B"},  # midday (group 1) — 1 step, OK
    ]
    assert validate_continuity(beats, OBS_MAP) == []


def test_continuity_fail():
    beats = [
        {"ordinal": 1, "photo_id": "photo-A"},  # morning (group 0)
        {"ordinal": 2, "photo_id": "photo-C"},  # golden_hour (group 2) — 2 step jump!
    ]
    violations = validate_continuity(beats, OBS_MAP)
    assert len(violations) == 1
    assert violations[0]["rule"] == "continuity"


def test_continuity_pass_with_intent():
    beats = [
        {"ordinal": 1, "photo_id": "photo-A"},  # morning
        {"ordinal": 2, "photo_id": "photo-C", "intent": "Time lapse transition"},  # golden_hour
    ]
    assert validate_continuity(beats, OBS_MAP) == []


# ── 5. Overlay Placement ────────────────────────────────────────────────

def test_overlay_placement_pass():
    direction = {
        "beats": [{"ordinal": 1, "photo_id": "photo-A"}],
        "overlay_register": [{"beat_ordinal": 1, "grid_region": "top_left"}],
    }
    assert validate_overlay_placement(direction, OBS_MAP) == []


def test_overlay_placement_fail():
    direction = {
        "beats": [{"ordinal": 1, "photo_id": "photo-A"}],
        "overlay_register": [{"beat_ordinal": 1, "grid_region": "bottom_right"}],
    }
    violations = validate_overlay_placement(direction, OBS_MAP)
    assert len(violations) == 1
    assert violations[0]["rule"] == "overlay_placement"


# ── 6. Amenity References ───────────────────────────────────────────────

def test_amenity_references_pass_empty():
    """Passes when no KB amenities (best-effort heuristic)."""
    assert validate_amenity_references({}, []) == []


# ── 7. No OTA ────────────────────────────────────────────────────────────

def test_no_ota_pass():
    direction = {"narration_brief": "A beautiful beachfront property"}
    assert validate_no_ota(direction) == []


def test_no_ota_fail():
    direction = {"narration_brief": "Book this Airbnb today!"}
    violations = validate_no_ota(direction)
    assert len(violations) >= 1
    assert violations[0]["rule"] == "no_ota"


# ── 8. No Guest Names ───────────────────────────────────────────────────

def test_no_guest_names_pass():
    direction = {"narration_brief": "Our guests love the sunset views"}
    assert validate_no_guest_names(direction, ["Sarah Johnson"]) == []


def test_no_guest_names_fail_exact():
    direction = {"narration_brief": "As Sarah Johnson once said, it was perfect"}
    violations = validate_no_guest_names(direction, ["Sarah Johnson"])
    assert len(violations) >= 1
    assert violations[0]["rule"] == "no_guest_names"


def test_no_guest_names_fail_attribution():
    direction = {"narration_brief": "The pool was incredible — Sarah"}
    violations = validate_no_guest_names(direction, ["Sarah"])
    assert len(violations) >= 1
    assert "no_guest_names" in violations[0]["rule"]


# ── 9. Guest Word Fidelity (DETERMINISTIC) ───────────────────────────────

def test_quote_fidelity_no_quotes():
    """No guest-provenance quotes → no violations."""
    direction = {"narration_brief": "Original narration", "narration_provenance": "original", "overlay_register": []}
    assert validate_quote_fidelity(direction, ["Some guest text."]) == []


def test_quote_fidelity_exact_match_passes():
    """Exact source sentence used → passes."""
    direction = {
        "narration_brief": "The pool was incredible.",
        "narration_provenance": "guest_book",
        "overlay_register": [],
    }
    assert validate_quote_fidelity(direction, ["The pool was incredible."]) == []


def test_quote_fidelity_paraphrase_fails():
    """Paraphrased sentence → fails."""
    direction = {
        "narration_brief": "The swimming pool was amazing.",
        "narration_provenance": "guest_book",
        "overlay_register": [],
    }
    violations = validate_quote_fidelity(direction, ["The pool was incredible."])
    assert len(violations) >= 1
    assert violations[0]["rule"] == "guest_word_fidelity"


# ── 10. Music Brief Prohibited ───────────────────────────────────────────

def test_music_brief_pass():
    direction = {"music_brief": {"mood": "warm and inviting", "tempo": "moderate"}}
    assert validate_music_brief_prohibited(direction) == []


def test_music_brief_fail_label():
    direction = {"music_brief": {"mood": "warm", "label": "Published by Sony entertainment"}}
    violations = validate_music_brief_prohibited(direction)
    assert len(violations) >= 1
    assert violations[0]["rule"] == "music_brief_prohibited"


# ── 11. Duration ─────────────────────────────────────────────────────────

def test_duration_pass():
    direction = {
        "beats": [{"duration_seconds": 5}, {"duration_seconds": 5}, {"duration_seconds": 5},
                  {"duration_seconds": 5}, {"duration_seconds": 5}, {"duration_seconds": 5}],
        "target_duration_sec": 30,
    }
    assert validate_duration(direction) == []  # 30s exactly


def test_duration_fail_too_short():
    direction = {
        "beats": [{"duration_seconds": 3}],
        "target_duration_sec": 30,
    }
    violations = validate_duration(direction)
    assert len(violations) == 1
    assert violations[0]["rule"] == "duration"


def test_duration_fail_zero():
    direction = {"beats": [{"duration_seconds": 0}], "target_duration_sec": 30}
    violations = validate_duration(direction)
    assert len(violations) == 1
    assert "duration is 0" in violations[0]["detail"]


# ── Combined: all validators ─────────────────────────────────────────────

def test_all_validators_clean_direction():
    """A well-formed direction passes all 11 validators.

    Fixture avoids: space conflicts (A-left, C-center, A-left, C-center...),
    depth monotony (foreground, background alternating), and time jumps
    (morning→golden_hour is 2-group jump, so use A→B→C with intent notes).
    """
    direction = {
        "beats": [
            {"ordinal": 1, "photo_id": "photo-A", "requested_motion": "push_in", "duration_seconds": 5},
            {"ordinal": 2, "photo_id": "photo-C", "requested_motion": "push_in", "duration_seconds": 5},
            {"ordinal": 3, "photo_id": "photo-A", "requested_motion": "pan_left", "duration_seconds": 5},
            {"ordinal": 4, "photo_id": "photo-C", "requested_motion": "orbit", "duration_seconds": 5},
            {"ordinal": 5, "photo_id": "photo-A", "requested_motion": "push_in", "duration_seconds": 5},
            {"ordinal": 6, "photo_id": "photo-C", "requested_motion": "push_in", "duration_seconds": 5},
        ],
        "narration_brief": "Welcome to a beachfront paradise",
        "narration_provenance": "original",
        "music_brief": {"mood": "warm and relaxing", "tempo": "gentle"},
        "overlay_register": [],
        "target_duration_sec": 30,
    }
    # A=left/foreground/morning, C=center/background/golden_hour
    # No left→right conflicts (left→center is OK)
    # No 3x same depth (foreground, background alternating)
    # Time: morning↔golden_hour is a 2-group jump — intent note needed
    # on the RECEIVING beat (the one that follows the jump)
    for beat in direction["beats"]:
        beat["intent"] = "Intentional time transition"

    violations = _run_all_validators(direction, OBS_MAP, [], [], [])
    assert violations == [], f"Unexpected violations: {violations}"
