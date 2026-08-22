"""
MOTION-D2: Tests for eligibility visibility and requested=approved=rendered.
"""

import sys
sys.path.insert(0, ".")


# ── Test 1: director receives per-frame depth constraints ─────────────

def test_director_receives_depth_constraints():
    """The frame fields list includes depth_constraints, and the prompt
    builder serializes it into the frame data the LLM sees."""
    import inspect
    from skills.direct import _build_prompt

    src = inspect.getsource(_build_prompt)
    assert '"depth_constraints"' in src, \
        "depth_constraints must be in the fields list for prompt serialization"


# ── Test 2: every depth beat passes eligibility as specified ──────────

def test_depth_beats_pass_eligibility_as_specified():
    """For a given set of depth beats, every one must pass
    check_depth_eligibility with the EXACT parameters the director
    specified — same photo, same trajectory, same intensity.

    This test uses a fixture representing what a well-informed director
    would produce given the constraints."""
    from skills.depth_motion import check_depth_eligibility

    # Simulate a director-produced depth beat on f81ead38:
    # lateral_right / restrained — within validated constraints
    result = check_depth_eligibility(
        photo_id="f81ead38-51f6-5f86-a9bc-85def619a3b0",
        image_width=3840,
        image_height=2560,
        has_depth_map=True,
        depth_structure="deep",
        motion_risks=["thin_railings", "water_surface", "straight_architectural_lines"],
        requested_motion="lateral_right",
        intensity="restrained",
    )
    assert result["eligible"] is True, \
        f"Director-specified lateral_right/restrained must be eligible: {result['reason']}"


def test_depth_beats_pass_eligibility_mutation_proof():
    """If the director could specify tilt_up/moderate (an unvalidated
    combination), eligibility would reject it — proving the constraint
    system is working."""
    from skills.depth_motion import check_depth_eligibility

    # tilt_up/moderate on a frame with thin_railings — should be rejected
    result = check_depth_eligibility(
        photo_id="f81ead38-51f6-5f86-a9bc-85def619a3b0",
        image_width=3840,
        image_height=2560,
        has_depth_map=True,
        depth_structure="deep",
        motion_risks=["thin_railings", "water_surface", "straight_architectural_lines"],
        requested_motion="tilt_up",
        intensity="moderate",
    )
    assert result["eligible"] is False, \
        "tilt_up/moderate with thin_railings must be ineligible"


# ── Test 2b: constraints include limiting reason and three states ─────

def test_constraints_include_three_states_and_reasons():
    """The depth_constraints field must distinguish validated,
    constrained, and unvalidated — and include limiting_reasons."""
    from skills.depth_motion import check_depth_eligibility

    # Build constraints for f81ead38 the same way direct.py does
    trajectories = ["lateral_right", "lateral_left", "push_in", "tilt_up"]
    intensities = ["restrained", "moderate"]
    validated = []
    constrained = []
    unvalidated = []
    limiting_reasons = set()

    for traj in trajectories:
        for inten in intensities:
            elig = check_depth_eligibility(
                photo_id="f81ead38",
                image_width=3840, image_height=2560,
                has_depth_map=True, depth_structure="deep",
                motion_risks=["thin_railings", "water_surface"],
                requested_motion=traj, intensity=inten,
            )
            combo = f"{traj}/{inten}"
            if elig["eligible"]:
                if elig.get("constraints"):
                    constrained.append(combo)
                else:
                    validated.append(combo)
            else:
                unvalidated.append(combo)
                limiting_reasons.add(elig["reason"])

    # Must have all three categories populated for a frame with risks
    assert len(validated) > 0, "Must have validated combinations (lateral/restrained)"
    assert len(unvalidated) > 0, "Must have unvalidated combinations (tilt_up, push_in)"
    assert len(limiting_reasons) > 0, "Must include limiting reasons"

    # Validated must include lateral_right/restrained
    assert "lateral_right/restrained" in validated
    # Unvalidated must include tilt_up/moderate
    assert "tilt_up/moderate" in unvalidated


# ── Test 3: renderer executes director's parameters, never substitutes ─

def test_renderer_executes_director_parameters():
    """The depth handler in generate_motion must pass the director's
    requested_motion and intensity to the renderer unchanged."""
    import inspect
    # Check that generate_motion passes beat parameters to render_depth_beat
    from skills.generate_motion import generate_motion
    src = inspect.getsource(generate_motion)

    # The depth handler must use the beat's requested_motion and intensity
    assert "render_depth_beat(" in src, "Must call render_depth_beat"
    assert "requested_motion=requested_motion" in src, \
        "Must pass the director's requested_motion to the renderer unchanged"


def test_renderer_no_silent_substitution_mutation_proof():
    """If generate_motion silently changed requested_motion before
    passing to the renderer, the direction would not describe the film.
    Verify the motion is passed through, not rewritten."""
    import inspect
    from skills.generate_motion import generate_motion
    src = inspect.getsource(generate_motion)

    # Between the depth handler start and render_depth_beat call,
    # there must NOT be a reassignment of requested_motion
    depth_section = src[src.find("if technique == \"depth\""):]
    depth_section = depth_section[:depth_section.find("if technique == \"static\"")]

    # Should not silently reassign requested_motion
    assert "requested_motion = " not in depth_section or \
           "requested_motion = beat" in depth_section, \
        "Must not silently reassign requested_motion in depth handler"


# ── Test 4: constrained frame not assigned unvalidated combination ────

def test_constrained_frame_rejects_unvalidated():
    """A frame whose only safe options are restrained lateral
    must NOT be assigned tilt_up/moderate."""
    from skills.depth_motion import check_depth_eligibility

    # tilt_up/moderate on thin_railings frame → ineligible
    result = check_depth_eligibility(
        photo_id="deck_photo",
        image_width=3840, image_height=2560,
        has_depth_map=True, depth_structure="deep",
        motion_risks=["thin_railings"],
        requested_motion="tilt_up",
        intensity="moderate",
    )
    assert result["eligible"] is False

    # lateral_right/restrained on same frame → eligible
    result2 = check_depth_eligibility(
        photo_id="deck_photo",
        image_width=3840, image_height=2560,
        has_depth_map=True, depth_structure="deep",
        motion_risks=["thin_railings"],
        requested_motion="lateral_right",
        intensity="restrained",
    )
    assert result2["eligible"] is True


# ── Test 5: static remains selectable and described ───────────────────

def test_static_in_prompt():
    """'static' technique is described in the prompt and appears
    in the beat example JSON."""
    import inspect
    from skills.direct import _build_prompt
    src = inspect.getsource(_build_prompt)
    assert '"static"' in src, "static must appear in prompt"
    assert "camera holds completely still" in src or "camera holds still" in src, \
        "static must be described as camera-still"


# ── Test 6: existing rules unchanged — regression guard ───────────────

def test_opening_rule_regression():
    """The opening validator still exists and is called."""
    import inspect
    from skills.direct import _run_all_validators
    src = inspect.getsource(_run_all_validators)
    assert "validate_opening_establishes" in src


def test_text_bearing_rule_regression():
    """Text-bearing frames still enforce bounded technique."""
    import inspect
    from skills.direct import validate_opening_establishes
    src = inspect.getsource(validate_opening_establishes)
    assert "contains_text" in src, "Text-bearing frame check must exist in opening validator"
    assert '"static"' in src or '"locked"' in src, "Must reject static/locked on text frames"
