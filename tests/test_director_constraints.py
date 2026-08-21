"""
DIRECTOR-4: Tests for the constraint edit experiment.

Tests verify: 11 renumbered constraints, removed constraints absent,
disabled validators, re-aligned depth_rhythm and duration, text-bearing
frame rule preserved.
"""

import sys
import re

sys.path.insert(0, ".")

from skills.direct import (
    _build_prompt, _run_all_validators,
    validate_depth_rhythm, validate_duration,
)


def _make_prompt():
    concept = {"title": "Test", "premise": "A retreat."}
    frames = [{"photo_id": "f1", "motion_affordance": ["push_in"],
               "curated_section": "Exterior", "quality_score": 0.9}]
    prop = {"name": "Test", "city": "X", "state_region": "NC",
            "vibe_profile": "romantic_escape", "amenities": []}
    return _build_prompt(concept, frames, prop, {}, [])


# ── Test 0: constraint 11 has all three techniques + text rule ──────────────

def test_constraint_11_techniques_and_text_rule():
    """Constraint 11 contains bounded, locked, AND generative (all three),
    plus the text-bearing frame rule (G49a / BEACCH guard)."""
    prompt = _make_prompt()

    # Find constraint 11
    c11_start = prompt.find("11.")
    assert c11_start > 0, "Constraint 11 must exist"

    # Get the text from 11 to the end of constraints (before "Return ONLY")
    c11_to_end = prompt[c11_start:prompt.find("Return ONLY")]

    assert '"bounded"' in c11_to_end, "Constraint 11 must list bounded"
    assert '"locked"' in c11_to_end, "Constraint 11 must list locked"
    assert '"generative"' in c11_to_end, "Constraint 11 must list generative"
    assert "Text-bearing" in c11_to_end or "text-bearing" in c11_to_end.lower(), \
        "Constraint 11 must include text-bearing frame rule"
    assert "push_in" in c11_to_end, "Text-bearing rule must specify push_in"


# ── Test 1: exactly 11 numbered constraints ─────────────────────────────────

def test_exactly_11_constraints():
    """The prompt contains constraints numbered 1-11, not 12-14."""
    prompt = _make_prompt()

    for i in range(1, 12):
        assert f"\n{i}." in prompt or prompt.startswith(f"{i}.") or f"\n{i}. " in prompt, \
            f"Constraint {i} must be present"

    # 12, 13, 14 should NOT exist as constraint numbers
    # (12 may appear in other contexts like "12 validators" — check specifically for numbered constraints)
    constraints_section = prompt[prompt.find("PART 3"):]
    assert "\n12." not in constraints_section, "Constraint 12 should not exist (renumbered to 11)"
    assert "\n13." not in constraints_section, "Constraint 13 should not exist (renumbered)"
    assert "\n14." not in constraints_section, "Constraint 14 should not exist (renumbered)"


# ── Test 2: removed constraints absent ──────────────────────────────────────

def test_removed_constraints_absent():
    """motion_affordance, time_of_day continuity, and overlay negative_space
    constraints are no longer in the prompt."""
    prompt = _make_prompt()
    constraints = prompt[prompt.find("PART 3"):]

    assert "motion_affordance" not in constraints, "motion_affordance constraint must be removed"
    assert "time_of_day_read must be consistent" not in constraints, \
        "time_of_day continuity constraint must be removed"
    assert "Overlay grid_region MUST come from" not in constraints, \
        "overlay negative_space constraint must be removed"


# ── Test 3: disabled validators not called ──────────────────────────────────

def test_disabled_validators_not_called():
    """_run_all_validators no longer calls the three disabled validators."""
    import inspect
    source = inspect.getsource(_run_all_validators)

    # The calls should be commented out
    assert "# violations += validate_motion_affordance" in source or \
           "validate_motion_affordance" not in source.replace("#", "").split("validate_motion_affordance")[0], \
        "validate_motion_affordance should be commented out"

    # Simpler check: run it and verify no motion_affordance violations
    direction = {
        "beats": [{"ordinal": 1, "photo_id": "p1", "requested_motion": "INVALID_MOVE"}],
        "opening_type": "property",
    }
    obs_map = {"p1": {"motion_affordance": ["push_in"], "curated_section": "Exterior",
                       "placement": "outdoor", "depth_tier": "deep"}}

    violations = _run_all_validators(direction, obs_map, [], [], [])
    rules = [v["rule"] for v in violations]

    assert "motion_affordance" not in rules, \
        "motion_affordance validator must not produce violations"
    assert "continuity" not in rules, \
        "continuity validator must not produce violations"
    assert "overlay_placement" not in rules, \
        "overlay_placement validator must not produce violations"


# ── Test 4: depth_rhythm flags TWO consecutive same-tier ────────────────────

def test_depth_rhythm_flags_two_consecutive():
    """validate_depth_rhythm flags TWO consecutive beats at the same
    depth_tier (tightened from three)."""
    beats = [
        {"ordinal": 1, "photo_id": "p1"},
        {"ordinal": 2, "photo_id": "p2"},
        {"ordinal": 3, "photo_id": "p3"},
    ]
    obs_map = {
        "p1": {"depth_tier": "medium"},
        "p2": {"depth_tier": "medium"},  # same as p1 — violation
        "p3": {"depth_tier": "deep"},
    }

    violations = validate_depth_rhythm(beats, obs_map)
    assert len(violations) >= 1, f"Should flag 2 consecutive same-tier, got {violations}"
    assert violations[0]["rule"] == "depth_rhythm"

    # Three different tiers — no violation
    obs_map_good = {
        "p1": {"depth_tier": "shallow"},
        "p2": {"depth_tier": "medium"},
        "p3": {"depth_tier": "deep"},
    }
    assert validate_depth_rhythm(beats, obs_map_good) == [], "Different tiers should pass"


# ── Test 5: duration accepts 34s, rejects 27s and 38s ──────────────────────

def test_duration_range_28_36():
    """validate_duration accepts 34s (inside 28-36) and rejects 27s and 38s."""
    # 34s — should pass
    direction_34 = {"beats": [{"duration_seconds": 34}]}
    assert validate_duration(direction_34) == [], "34s should pass (inside 28-36)"

    # 27s — should fail (below 28)
    direction_27 = {"beats": [{"duration_seconds": 27}]}
    violations_27 = validate_duration(direction_27)
    assert len(violations_27) == 1, f"27s should fail, got {violations_27}"
    assert "28s, 36s" in violations_27[0]["detail"], f"Should reference 28-36 range"

    # 38s — should fail (above 36)
    direction_38 = {"beats": [{"duration_seconds": 38}]}
    violations_38 = validate_duration(direction_38)
    assert len(violations_38) == 1, f"38s should fail, got {violations_38}"


# ── Test 6: prompt duration text matches validator ──────────────────────────

def test_prompt_duration_matches_validator():
    """The prompt says 28-36, matching the validator."""
    prompt = _make_prompt()
    assert "28-36" in prompt, "Prompt must say 28-36 seconds"
    assert "28-32" not in prompt, "Old range 28-32 must not appear"
