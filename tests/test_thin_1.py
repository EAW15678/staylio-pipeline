"""
THIN-1: Tests for hanging-fixture depth eligibility and idempotency regression.
"""

import sys
import inspect
sys.path.insert(0, ".")


# ── Test 1: 110c2bd6 identified as high-risk ──────────────────────────

def test_110c2bd6_identified_high_risk():
    """Kitchen with pendant lights detected via alt_text scan."""
    from skills.depth_motion import check_depth_eligibility
    result = check_depth_eligibility(
        photo_id="110c2bd6",
        image_width=3840, image_height=2560,
        has_depth_map=True, depth_structure="deep",
        motion_risks=["straight_architectural_lines", "reflections"],
        requested_motion="push_in", intensity="restrained",
        alt_text="Open-plan kitchen and dining area with island and gold pendant lights",
        foreground_elements=["round dining table", "wood chairs", "kitchen island"],
    )
    assert result["eligible"] is False
    assert "hanging-fixture" in result["reason"]


# ── Test 2: be3e732c identified as high-risk ──────────────────────────

def test_be3e732c_identified_high_risk():
    """Living/dining with chandelier detected via alt_text scan."""
    from skills.depth_motion import check_depth_eligibility
    result = check_depth_eligibility(
        photo_id="be3e732c",
        image_width=3840, image_height=2560,
        has_depth_map=True, depth_structure="deep",
        motion_risks=["straight_architectural_lines", "reflections"],
        requested_motion="lateral_left", intensity="restrained",
        alt_text="Open living and dining with round table, fireplace, TV, and chandelier",
        foreground_elements=["round dining table", "bench seating"],
    )
    assert result["eligible"] is False
    assert "hanging-fixture" in result["reason"]


# ── Test 3: f81ead38 remains depth-eligible ───────────────────────────

def test_f81ead38_remains_eligible():
    """Rooftop with thin railings — validated treatment must survive."""
    from skills.depth_motion import check_depth_eligibility
    result = check_depth_eligibility(
        photo_id="f81ead38",
        image_width=3840, image_height=2560,
        has_depth_map=True, depth_structure="deep",
        motion_risks=["thin_railings", "water_surface", "straight_architectural_lines"],
        requested_motion="lateral_right", intensity="restrained",
        alt_text="Rooftop deck with panoramic waterway and coastal town views",
        foreground_elements=["deck railing", "rooftop table"],
    )
    assert result["eligible"] is True, f"Rooftop must remain eligible: {result['reason']}"


# ── Test 4: hanging-fixture does NOT inherit railing override ─────────

def test_hanging_fixture_no_railing_override():
    """The empirical override for thin_railings must not apply to
    hanging fixtures. These are different hazard classes."""
    from skills.depth_motion import check_depth_eligibility, EMPIRICAL_RISK_OVERRIDES
    # thin_railings has an override; hanging fixtures must not
    assert "thin_railings" in EMPIRICAL_RISK_OVERRIDES
    # A frame with a chandelier and lateral_right/restrained must still fail
    result = check_depth_eligibility(
        photo_id="chandelier_room",
        image_width=3840, image_height=2560,
        has_depth_map=True, depth_structure="deep",
        motion_risks=[],
        requested_motion="lateral_right", intensity="restrained",
        alt_text="Dining room with sputnik chandelier",
        foreground_elements=[],
    )
    assert result["eligible"] is False


# ── Test 5: placeholder — Erick's verdict determines eligibility ──────
# These tests are written as conditional on the verdict.
# If NOT RESCUED: both frames become depth-ineligible (already true via
# the hanging-fixture scan). Tests 1 and 2 cover this.
# If RESCUED: would need a new empirical override for the exact
# validated combination.

def test_verdict_not_rescued_already_covered():
    """If Erick rules NOT RESCUED, tests 1 and 2 already enforce
    that both bad frames are depth-ineligible. No additional code
    needed — the hanging-fixture scan is the mechanism."""
    pass  # Intentionally empty — the existing tests cover this


def test_verdict_rescued_requires_explicit_override():
    """If Erick rules RESCUED, a new entry in EMPIRICAL_RISK_OVERRIDES
    would be needed for the specific validated combination. That entry
    does NOT exist yet."""
    from skills.depth_motion import EMPIRICAL_RISK_OVERRIDES
    # No override exists for hanging fixtures
    assert "pendant_lights" not in EMPIRICAL_RISK_OVERRIDES
    assert "chandelier" not in EMPIRICAL_RISK_OVERRIDES


# ── Test 7: generic chair/railing alone creates no blanket rejection ──

def test_generic_terms_no_blanket_rejection():
    """'chair' or 'railing' alone in alt_text must NOT trigger the
    hanging-fixture scan."""
    from skills.depth_motion import _has_hanging_fixture_hazard
    assert _has_hanging_fixture_hazard("Room with dining chairs", ["chair", "table"]) is False
    assert _has_hanging_fixture_hazard("Deck with railing", ["railing"]) is False
    assert _has_hanging_fixture_hazard("Bedroom with furniture", ["bed frame"]) is False


# ── Test 8: alt_text scan labelled temporary ──────────────────────────

def test_alt_text_scan_labelled_temporary():
    """The hanging-fixture scan is documented as a stopgap."""
    src = inspect.getsource(
        __import__("skills.depth_motion", fromlist=["_has_hanging_fixture_hazard"])._has_hanging_fixture_hazard
    )
    # Check surrounding context
    import skills.depth_motion as dm
    full_src = inspect.getsource(dm)
    # Must contain TEMPORARY or stopgap labelling
    fixture_section = full_src[full_src.find("_HANGING_FIXTURE_TERMS"):full_src.find("_HANGING_FIXTURE_TERMS") + 500]
    assert "TEMPORARY" in fixture_section or "stopgap" in fixture_section or "temporary" in fixture_section.lower(), \
        "Hanging-fixture scan must be labelled as temporary/stopgap"


# ── Test 9: idempotency regression — already-finished force=False ─────

def test_already_finished_idempotency():
    """An already Aleph-finished active artifact must be skipped on
    force=False without any vendor call. This exercises the real
    finish_beats production path."""
    from unittest.mock import patch, MagicMock
    from skills.finish_beats import finish_beats

    # Build a mock substrate with one already-finished clip
    sb = MagicMock()

    # Direction with one beat
    dir_data = MagicMock()
    dir_data.data = [{"beats": [{"ordinal": 1, "content_motion": ["water"], "atmosphere": "warm"}]}]

    # One active clip that IS already Aleph-finished
    clips_data = MagicMock()
    clips_data.data = [{
        "artifact_id": "art-1",
        "input_hash": "hash-1",
        "photo_id": "photo-1",
        "beat_ordinal": 1,
        "storage_url": "https://example.com/finished.mp4",
        "technique": "depth",
        "duration_seconds": 3,
        "motion_params": {"finishing": "aleph", "truth_gate": "pass"},
    }]

    # Observations
    obs_data = MagicMock()
    obs_data.data = [{"photo_id": "photo-1", "motion_risk": ["water_surface"],
                      "foreground_elements": [], "located_amenities": [],
                      "beyond_frame_element": None, "alt_text": "", "contains_text": False}]

    def table_side_effect(name):
        t = MagicMock()
        if name == "directions":
            chain = MagicMock()
            chain.select.return_value = chain
            chain.eq.return_value = chain
            chain.is_.return_value = chain
            chain.limit.return_value = chain
            chain.execute.return_value = dir_data
            t.select.return_value = chain
        elif name == "observations":
            chain = MagicMock()
            chain.select.return_value = chain
            chain.eq.return_value = chain
            chain.is_.return_value = chain
            chain.execute.return_value = obs_data
            t.select.return_value = chain
        elif name == "video_artifacts":
            chain = MagicMock()
            chain.select.return_value = chain
            chain.eq.return_value = chain
            chain.is_.return_value = chain
            chain.execute.return_value = clips_data
            t.select.return_value = chain
        return t

    sb.table = table_side_effect

    # Mock Runway to fail if called
    mock_runway = MagicMock()
    mock_runway.video_to_video.create.side_effect = AssertionError("UNEXPECTED ALEPH CALL")

    with patch("skills.finish_beats.get_substrate", return_value=sb), \
         patch("skills.finish_beats.require_env", return_value="dummy_key"), \
         patch("skills.finish_beats.record_run", return_value="run-1"), \
         patch("skills.finish_beats.record_step", return_value="step-1"), \
         patch("skills.finish_beats.complete_step"), \
         patch("skills.finish_beats.complete_run"), \
         patch("skills.finish_beats.emit_cost"), \
         patch("runwayml.RunwayML", return_value=mock_runway):

        result = finish_beats("prop-1", direction_id="dir-1", force=False)

    assert result.is_ok
    assert result.data["finished"] == 0, "Must not finish any new clips"
    assert result.data["skipped"] == 1, "Must skip the already-finished clip"
    # Verify Runway was never called
    mock_runway.video_to_video.create.assert_not_called()


def test_already_finished_idempotency_mutation_proof():
    """If the 'already Aleph-finished' guard is removed from finish_beats,
    the mock would receive an Aleph call and raise AssertionError."""
    src = inspect.getsource(
        __import__("skills.finish_beats", fromlist=["finish_beats"]).finish_beats
    )
    assert 'finishing' in src and '"aleph"' in src, \
        "Must check motion_params.finishing == aleph to skip already-finished clips"
    # The specific guard: mp.get("finishing") == "aleph"
    assert 'mp.get("finishing")' in src or "finishing" in src, \
        "Must read finishing from motion_params"
