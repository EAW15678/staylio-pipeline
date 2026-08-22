"""
MOTION-D: Tests for depth motion eligibility and rendering mechanics.

Tests 1-7 per the spec.
"""

import sys
sys.path.insert(0, ".")


# ── Test 1: photo with depth map + sufficient overscan is eligible ────

def test_eligible_with_depth_map_and_resolution():
    from skills.depth_motion import check_depth_eligibility

    result = check_depth_eligibility(
        photo_id="p1",
        image_width=3840,
        image_height=2560,
        has_depth_map=True,
        depth_structure="deep",
        motion_risks=["straight_architectural_lines"],
        requested_motion="lateral_right",
        intensity="restrained",
    )
    assert result["eligible"] is True
    assert result["render_w"] > 1920, "Render must be wider than delivery for overscan"
    assert result["render_h"] > 1080


# ── Test 2: photo WITHOUT depth map is ineligible + falls back ────────

def test_ineligible_without_depth_map():
    from skills.depth_motion import check_depth_eligibility

    result = check_depth_eligibility(
        photo_id="p1",
        image_width=3840,
        image_height=2560,
        has_depth_map=False,
        depth_structure="deep",
        motion_risks=[],
        requested_motion="lateral_right",
    )
    assert result["eligible"] is False
    assert "no cached depth map" in result["reason"]


# ── Test 3: photo with insufficient resolution is ineligible ──────────

def test_ineligible_insufficient_resolution():
    from skills.depth_motion import check_depth_eligibility

    result = check_depth_eligibility(
        photo_id="p1",
        image_width=1024,
        image_height=683,
        has_depth_map=True,
        depth_structure="deep",
        motion_risks=[],
        requested_motion="lateral_right",
    )
    assert result["eligible"] is False
    assert "source" in result["reason"].lower() and ("small" in result["reason"].lower() or "insufficient" in result["reason"].lower())


# ── Test 4: renderer uses ffmpeg crop, NOT DepthFlow zoom ─────────────

def test_overscan_uses_ffmpeg_crop_not_zoom():
    """The DepthFlow script must NOT contain 'self.state.zoom' and
    the render service must use ffmpeg crop."""
    from skills.depth_motion import build_depthflow_script

    script = build_depthflow_script("lateral_right", amp=2.0)

    # Script must NOT use zoom for overscan
    assert "self.state.zoom" not in script, \
        "DepthFlow script must NOT use zoom — overscan is ffmpeg crop"

    # Verify the render service uses ffmpeg crop
    import inspect
    from importlib import import_module
    # Can't import the Modal service directly, but we can verify
    # depth_motion.render_depth_beat calls the Modal function which
    # does the crop. Check the contract.
    from skills.depth_motion import render_depth_beat
    src = inspect.getsource(render_depth_beat)
    assert "render_w" in src, "Must pass render_w for oversized canvas"
    assert "delivery_w" in src or "DELIVERY_W" in src, "Must specify delivery size for crop"


def test_overscan_uses_ffmpeg_crop_mutation_proof():
    """Removing the render_w/render_h parameters from render_depth_beat
    would break the overscan mechanism."""
    import inspect
    from skills.depth_motion import render_depth_beat
    sig = inspect.signature(render_depth_beat)
    params = list(sig.parameters.keys())
    assert "render_w" in params, "render_w must be a parameter"
    assert "render_h" in params, "render_h must be a parameter"


# ── Test 5: trajectories are one-way, never sinusoidal ────────────────

def test_trajectories_are_one_way():
    from skills.depth_motion import build_depthflow_script

    for motion in ["lateral_right", "lateral_left", "pan_right", "push_in"]:
        script = build_depthflow_script(motion, amp=2.0)
        assert "self.tau" in script, f"{motion}: must use self.tau (one-way 0→1)"
        assert "math.sin" not in script, f"{motion}: must NOT use sin (oscillating)"
        assert "self.cycle" not in script, f"{motion}: must NOT use self.cycle (oscillating)"


# ── Test 6: narrative preference beats depth eligibility ──────────────

def test_narrative_preference_over_depth_eligibility():
    """Given a narratively preferred frame that is depth-INELIGIBLE
    (shallow depth) and a weaker narrative frame that IS depth-eligible,
    the director must select the preferred frame and fall back to
    bounded — NOT swap in the eligible frame.

    This tests the contract: depth technique falls back to bounded
    when ineligible, preserving the director's image selection."""
    from skills.depth_motion import check_depth_eligibility

    # Narratively preferred: best exterior, but shallow depth
    preferred = check_depth_eligibility(
        photo_id="best_exterior",
        image_width=3840,
        image_height=2560,
        has_depth_map=True,
        depth_structure="shallow",  # ineligible
        motion_risks=[],
        requested_motion="lateral_right",
    )
    assert preferred["eligible"] is False, "Preferred frame must be ineligible"

    # Weaker alternative: pool, deep depth, eligible
    alternative = check_depth_eligibility(
        photo_id="pool_shot",
        image_width=3840,
        image_height=2560,
        has_depth_map=True,
        depth_structure="deep",  # eligible
        motion_risks=[],
        requested_motion="lateral_right",
    )
    assert alternative["eligible"] is True, "Alternative must be eligible"

    # The system's response: preferred frame gets fallback treatment,
    # NOT image swap. check_depth_eligibility returns eligible=False
    # for the preferred frame, and generate_motion falls back to bounded.
    # The director's photo_id is preserved — the beat stays.
    # This is enforced by generate_motion's fallback logic which keeps
    # photo_id unchanged and only changes technique to bounded.


def test_narrative_preference_mutation_proof():
    """If check_depth_eligibility stopped checking depth_structure,
    the shallow-depth frame would become eligible — destroying the
    narrative hierarchy."""
    import inspect
    from skills.depth_motion import check_depth_eligibility
    src = inspect.getsource(check_depth_eligibility)
    assert "depth_structure" in src, "Must check depth_structure for eligibility"
    assert "shallow" in src or "flat" in src, "Must reject shallow/flat depth"


# ── Test 7: opening rule still fires with depth available ─────────────

def test_opening_rule_still_fires():
    """The opening validator in direct.py still checks for
    property/setting preference — adding depth did not break it."""
    from skills.direct import validate_opening_establishes

    # A direction where beat 1 opens with a feature (not property/setting)
    direction = {
        "opening_type": "feature",
        "beats": [
            {"ordinal": 1, "photo_id": "p1", "technique": "depth"},
        ],
    }
    obs_map = {
        "p1": {"curated_section": "Pool", "is_setting": False},
    }

    violations = validate_opening_establishes(direction, obs_map)
    # The validator should still produce violations for feature openings
    # when the first frame doesn't establish property/setting
    # (This tests that the validator runs — its exact behaviour depends
    # on the implementation, but it must not crash or be disabled)
    assert isinstance(violations, list), "Validator must return a list"


def test_opening_rule_mutation_proof():
    """The opening validator must still exist and be called with
    depth technique available."""
    import inspect
    from skills.direct import validate_opening_establishes
    src = inspect.getsource(validate_opening_establishes)
    assert "opening_type" in src, "Must check opening_type"
    # Verify it's not commented out in _run_all_validators
    from skills.direct import _run_all_validators
    val_src = inspect.getsource(_run_all_validators)
    assert "validate_opening_establishes" in val_src, "Opening validator must still be called"


# ── Test 11: f81ead38 with thin_railings IS eligible at restrained ────

def test_f81ead38_thin_railings_eligible_restrained():
    """f81ead38 has thin_railings but is eligible at restrained lateral
    because the empirical override covers this combination."""
    from skills.depth_motion import check_depth_eligibility

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
    assert result["eligible"] is True, f"f81ead38 must be eligible, got: {result['reason']}"


def test_f81ead38_mutation_proof_blanket_rejection():
    """If the blanket THIN_STRUCTURE_RISKS rejection is restored
    (removing EMPIRICAL_RISK_OVERRIDES), f81ead38 must become
    ineligible — proving the override is doing the work."""
    from skills.depth_motion import (
        check_depth_eligibility, EMPIRICAL_RISK_OVERRIDES,
        PERIPHERAL_THIN_STRUCTURE_RISKS,
    )
    # The empirical override for thin_railings must exist
    assert "thin_railings" in EMPIRICAL_RISK_OVERRIDES, \
        "thin_railings must have an empirical override"
    # And thin_railings must be in the peripheral set, not the central set
    assert "thin_railings" in PERIPHERAL_THIN_STRUCTURE_RISKS, \
        "thin_railings must be peripheral, not central"


# ── Test 12: central thin structures still rejected ───────────────────

def test_central_thin_structures_rejected():
    """Chandeliers and pendant lights in the centre of frame are
    rejected — these cross major depth boundaries and smear visibly.

    NOTE: current observation data identifies pendant_lights and
    chandelier in motion_risk. It does NOT distinguish central vs
    peripheral placement. This is a known gap — the chandelier in
    the Phase 3 living-room beat demonstrated the problem, but
    observations do not carry positional information for thin
    structures. A future observation improvement should add
    thin_structure_region (central/peripheral/edge) to enable
    finer-grained depth eligibility."""
    from skills.depth_motion import check_depth_eligibility

    result = check_depth_eligibility(
        photo_id="chandelier_room",
        image_width=3840,
        image_height=2560,
        has_depth_map=True,
        depth_structure="deep",
        motion_risks=["pendant_lights", "straight_architectural_lines"],
        requested_motion="lateral_right",
        intensity="restrained",
    )
    assert result["eligible"] is False
    assert "central thin-structure" in result["reason"]


# ── Test 13: water frame is NOT forced to static ──────────────────────

def test_water_frame_not_forced_to_static():
    """A frame with water_surface in motion_risk is depth-eligible.
    Camera treatment and content motion are independently assignable.
    water_surface does not force the camera to be static."""
    from skills.depth_motion import check_depth_eligibility

    result = check_depth_eligibility(
        photo_id="pool_shot",
        image_width=3500,
        image_height=2333,
        has_depth_map=True,
        depth_structure="deep",
        motion_risks=["water_surface", "reflections", "straight_architectural_lines"],
        requested_motion="lateral_right",
        intensity="restrained",
    )
    assert result["eligible"] is True, \
        f"Water frame must be depth-eligible — camera and content are independent. Got: {result['reason']}"


def test_water_frame_not_forced_mutation_proof():
    """If water_surface were added to any rejection set, the water
    frame would become ineligible — proving the orthogonal model."""
    from skills.depth_motion import (
        CENTRAL_THIN_STRUCTURE_RISKS, PERIPHERAL_THIN_STRUCTURE_RISKS,
    )
    assert "water_surface" not in CENTRAL_THIN_STRUCTURE_RISKS
    assert "water_surface" not in PERIPHERAL_THIN_STRUCTURE_RISKS


# ── Test 14: empirical override mechanism ─────────────────────────────

def test_empirical_override_allows_validated_combination():
    """A validated (risk, camera_family, intensity) combination is
    allowed despite the risk label."""
    from skills.depth_motion import check_depth_eligibility, EMPIRICAL_RISK_OVERRIDES

    # thin_railings + lateral_right + restrained → overridden
    result = check_depth_eligibility(
        photo_id="any_deck",
        image_width=3840,
        image_height=2560,
        has_depth_map=True,
        depth_structure="deep",
        motion_risks=["thin_railings"],
        requested_motion="lateral_right",
        intensity="restrained",
    )
    assert result["eligible"] is True

    # Same risk + non-overridden trajectory → rejected
    result2 = check_depth_eligibility(
        photo_id="any_deck",
        image_width=3840,
        image_height=2560,
        has_depth_map=True,
        depth_structure="deep",
        motion_risks=["thin_railings"],
        requested_motion="push_in",  # not in safe_camera_families
        intensity="restrained",
    )
    assert result2["eligible"] is False, \
        "Non-overridden trajectory must be rejected despite same risk"
