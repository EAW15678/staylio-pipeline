"""
DIRECTOR-1: Tests for the rebuilt creative direction prompt.

Verifies structure (brief before constraints), content (hero proposition,
directing mode, people guidance, constraint-11 fix), and the new output
fields in the JSON schema.
"""

import sys
sys.path.insert(0, ".")

from skills.direct import _build_prompt


def _make_prompt():
    """Build a prompt with minimal synthetic data."""
    concept = {"title": "Test Film", "premise": "A coastal retreat."}
    frames = [{"photo_id": "f1", "motion_affordance": ["push_in"],
               "curated_section": "Exterior", "quality_score": 0.9}]
    prop = {"name": "Test Property", "city": "Wilmington", "state_region": "NC",
            "vibe_profile": "romantic_escape", "amenities": []}
    owner_ctx = {"owner_story": "We love this place", "wow_factor": "The view",
                 "hidden_gems": "The garden"}
    guest_evidence = []
    return _build_prompt(concept, frames, prop, owner_ctx, guest_evidence)


# ── Test 1: brief appears BEFORE constraints ────────────────────────────────

def test_brief_before_constraints():
    """The brief section appears before the constraints section."""
    prompt = _make_prompt()

    brief_pos = prompt.find("PART 1")
    craft_pos = prompt.find("PART 2")
    constraints_pos = prompt.find("PART 3")

    assert brief_pos > 0, "PART 1 (brief) must be present"
    assert craft_pos > 0, "PART 2 (craft guidance) must be present"
    assert constraints_pos > 0, "PART 3 (constraints) must be present"

    assert brief_pos < craft_pos < constraints_pos, (
        f"Order must be PART 1 ({brief_pos}) < PART 2 ({craft_pos}) < PART 3 ({constraints_pos})"
    )

    # The key test: guidance comes BEFORE constraints
    assert prompt.find("Why should I stop scrolling") < constraints_pos, \
        "Brief content must appear before constraints"
    assert prompt.find("non-negotiable") > craft_pos, \
        "Constraints intro must appear after craft guidance"


# ── Test 2: hero proposition and directing mode instructions ────────────────

def test_hero_proposition_and_directing_mode():
    """The prompt instructs on hero proposition and directing mode."""
    prompt = _make_prompt()

    assert "HERO PROPOSITION" in prompt, "Must instruct on hero proposition"
    assert "booking driver" in prompt.lower(), "Must mention booking drivers"
    assert "DIRECTING MODE" in prompt, "Must instruct on directing mode"

    # All 8 directing modes listed
    modes = ["The Reveal", "The Weekend", "The Escape", "The Gathering",
             "The Playground", "The Retreat", "The Design Film", "The Destination"]
    for mode in modes:
        assert mode in prompt, f"Directing mode '{mode}' must be listed"


# ── Test 3: people guidance AND no-children line ────────────────────────────

def test_people_guidance_and_no_children():
    """The prompt contains people guidance and the explicit no-children line."""
    prompt = _make_prompt()

    assert "People are permitted" in prompt or "people are permitted" in prompt, \
        "Must state people are permitted"
    assert "property stays the hero" in prompt, \
        "Must state the property stays the hero"
    assert "Do NOT depict children" in prompt, \
        "Must explicitly prohibit depicting children"


# ── Test 4: constraint 11 is whole — bug fixed ─────────────────────────────

def test_constraint_11_is_whole():
    """Constraint 11 contains both the 28-32 range and the narration
    sentence-sizing rule together. The orphan no longer sits inside
    constraint 13."""
    prompt = _make_prompt()

    # Find constraint 11
    c11_start = prompt.find("11.")
    assert c11_start > 0, "Constraint 11 must exist"

    # Find constraint 12
    c12_start = prompt.find("12.")
    assert c12_start > c11_start, "Constraint 12 must follow 11"

    # The content between 11. and 12. should contain BOTH the duration
    # range AND the narration sentence rule
    c11_text = prompt[c11_start:c12_start]
    assert "28-32" in c11_text, f"Constraint 11 must contain '28-32', got: {c11_text[:100]}"
    assert "narration" in c11_text.lower() and "sentence" in c11_text.lower(), (
        f"Constraint 11 must contain narration sentence-sizing rule, got: {c11_text[:200]}"
    )

    # The orphan must NOT be inside constraint 13
    c13_start = prompt.find("13.")
    c14_start = prompt.find("14.")
    if c13_start > 0 and c14_start > 0:
        c13_text = prompt[c13_start:c14_start]
        assert "narration must never be cut off mid-sentence" not in c13_text, \
            "The narration sentence orphan must be removed from constraint 13"


# ── Test 5: all 14 constraints present and unmodified ───────────────────────

def test_all_14_constraints_present():
    """All 14 original constraints are present."""
    prompt = _make_prompt()

    # Check each constraint number exists
    for i in range(1, 15):
        assert f"{i}." in prompt, f"Constraint {i} must be present"

    # Key constraint content unchanged
    assert "SELECT frames ONLY" in prompt, "Constraint 1 content"
    assert "motion_affordance" in prompt, "Constraint 2 content"
    assert "space_direction" in prompt, "Constraint 3 content"
    assert "depth_tier" in prompt, "Constraint 4 content"
    assert "time_of_day_read" in prompt, "Constraint 5 content"
    assert "grid_region" in prompt, "Constraint 6 content"
    assert "amenity not on the confirmed" in prompt, "Constraint 7 content"
    assert "OTA references" in prompt, "Constraint 8 content"
    assert "guest names" in prompt.lower(), "Constraint 9 content"
    assert "artist, song, album" in prompt, "Constraint 10 content"
    assert "narration_voice_id MUST be one of" in prompt, "Constraint 12 content"
    assert "OPENING RULE" in prompt, "Constraint 13 content"
    assert "TECHNIQUE PER BEAT" in prompt, "Constraint 14 content"


# ── Test 6: output JSON schema includes new fields ──────────────────────────

def test_output_json_includes_new_fields():
    """The output JSON schema in the prompt includes the four new fields."""
    prompt = _make_prompt()

    new_fields = ["hero_proposition", "booking_drivers", "directing_mode", "mode_rationale"]
    for field in new_fields:
        assert f'"{field}"' in prompt, f"Output JSON must include '{field}'"


# ── Test 7: opening preference language present ─────────────────────────────

def test_opening_preference_present():
    """The prompt contains the property/setting preference and no longer
    contains the old 'ORDER is YOUR judgment' line."""
    prompt = _make_prompt()

    assert "property or setting over feature" in prompt, \
        "Must state preference for property/setting over feature"
    assert "The ORDER among the three kinds is YOUR judgment" not in prompt, \
        "Old unconstrained judgment line must be removed"
    assert "Supersedes G65" in prompt, \
        "Must note the G65 supersede"


# ── Test 8: feature opener still permitted ──────────────────────────────────

def test_feature_opener_still_permitted():
    """The preference is stated as a preference, not a prohibition.
    Feature opener must remain explicitly permitted."""
    prompt = _make_prompt()

    assert "feature opener remains permitted" in prompt, \
        "Must explicitly state feature opener is still permitted"
    assert "not a hard rule" in prompt, \
        "Must state the preference is not a hard rule"


# ── Test 9: validator still passes feature opener ───────────────────────────

def test_validator_still_passes_feature_opener():
    """validate_opening_establishes accepts opening_type='feature' —
    regression guard proving this stayed a prompt change and did not
    leak into enforcement."""
    from skills.direct import validate_opening_establishes

    direction = {
        "beats": [{"ordinal": 1, "photo_id": "p1"}],
        "opening_type": "feature",
    }
    obs_map = {
        "p1": {
            "curated_section": "Pool",
            "placement": "outdoor",
            "is_setting": False,
        },
    }

    violations = validate_opening_establishes(direction, obs_map)
    assert violations == [], (
        f"Feature opener must not produce violations, got: {violations}"
    )
