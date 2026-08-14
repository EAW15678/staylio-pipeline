"""
MOTION-2: Verify technique gates the FRAME_EXITING_MOVES downgrade.

Bounded beats skip the downgrade entirely — the bounds assertion is
the enforcement. Generative beats still downgrade. Text-bearing frame
constraint applies to BOTH techniques.

All mocked — no vendor calls, $0.
"""

import pytest

from skills.bounded_motion import compute_bounded_animation, assert_viewport_in_bounds
from skills.generate_motion import FRAME_EXITING_MOVES
from skills.direct import validate_opening_establishes


# ── Bounded beats: NOT downgraded ─────────────────────────────────────


def test_bounded_tilt_up_not_downgraded():
    """Bounded tilt_up renders as tilt_up, NOT downgraded to push_in."""
    anim = compute_bounded_animation("tilt_up", 768, 852)
    start_y = float(anim["start_y"].rstrip("%"))
    end_y = float(anim["end_y"].rstrip("%"))
    # tilt_up: camera tilts up = y decreases (start_y > end_y)
    assert start_y > end_y, "tilt_up should have start_y > end_y"
    assert not anim.get("reduced"), "tilt_up on 768x852 should not be reduced"


def test_bounded_pull_back_not_downgraded():
    """Bounded pull_back renders as pull_back, NOT downgraded to push_in."""
    anim = compute_bounded_animation("pull_back", 1024, 768)
    start_s = float(anim["start_scale"].rstrip("%"))
    end_s = float(anim["end_scale"].rstrip("%"))
    # pull_back: zooms out = start_scale < end_scale
    assert start_s < end_s, "pull_back should zoom out (start < end)"


def test_bounded_tilt_up_bounds_pass():
    """Bounded tilt_up on 768x852 passes the bounds assertion.

    compute_bounded_animation asserts internally with raw floats
    before formatting to 1 decimal place. If this call returns
    without raising, the viewport stayed in bounds.
    """
    anim = compute_bounded_animation("tilt_up", 768, 852)
    # The internal assertion already fired — verify the output is sane
    assert anim["type"] == "pan"
    start_y = float(anim["start_y"].rstrip("%"))
    end_y = float(anim["end_y"].rstrip("%"))
    assert start_y > end_y  # tilt_up = y decreases


def test_bounded_pull_back_bounds_pass():
    """Bounded pull_back on 1024x768 passes the bounds assertion."""
    anim = compute_bounded_animation("pull_back", 1024, 768)
    for prefix in ("start", "end"):
        x = float(anim[f"{prefix}_x"].rstrip("%"))
        y = float(anim[f"{prefix}_y"].rstrip("%"))
        s = float(anim[f"{prefix}_scale"].rstrip("%")) / 100
        assert_viewport_in_bounds(x, y, s, 1024, 768)


# ── Generative beats: STILL downgraded ───────────────────────────────


def test_generative_tilt_up_still_in_frame_exiting():
    """tilt_up is still in FRAME_EXITING_MOVES (generative downgrade)."""
    assert "tilt_up" in FRAME_EXITING_MOVES


def test_generative_pull_back_still_in_frame_exiting():
    """pull_back is still in FRAME_EXITING_MOVES (generative downgrade)."""
    assert "pull_back" in FRAME_EXITING_MOVES


def test_generative_push_in_not_in_frame_exiting():
    """push_in is NOT in FRAME_EXITING_MOVES — never downgraded."""
    assert "push_in" not in FRAME_EXITING_MOVES


# ── Text constraint: applies to BOTH techniques ─────────────────────


def _make_direction_with_beat(photo_id, motion, opening_type="property"):
    return {
        "beats": [{"ordinal": 1, "photo_id": photo_id, "requested_motion": motion}],
        "opening_type": opening_type,
    }


def test_text_frame_tilt_up_fails_regardless_of_technique():
    """A text-bearing frame requesting tilt_up fails validation.

    The text constraint in validate_opening_establishes checks
    contains_text and requested_motion — it does NOT check technique.
    """
    direction = _make_direction_with_beat("text-photo", "tilt_up", "feature")
    obs_map = {
        "text-photo": {
            "contains_text": True,
            "text_content": "BEACH",
            "curated_section": "Exterior",
            "placement": "outdoor",
            "shows_structure": False,
            "is_setting": False,
        }
    }
    violations = validate_opening_establishes(direction, obs_map)
    text_violations = [v for v in violations if "contains text" in v.get("detail", "")]
    assert len(text_violations) >= 1, "Text-bearing frame with tilt_up must fail"


def test_text_frame_push_in_passes():
    """A text-bearing frame with push_in passes the text constraint."""
    direction = _make_direction_with_beat("text-photo", "push_in", "feature")
    obs_map = {
        "text-photo": {
            "contains_text": True,
            "text_content": "BEACH",
            "curated_section": "Exterior",
            "placement": "outdoor",
            "shows_structure": False,
            "is_setting": False,
        }
    }
    violations = validate_opening_establishes(direction, obs_map)
    text_violations = [v for v in violations if "contains text" in v.get("detail", "")]
    assert len(text_violations) == 0, "Text-bearing frame with push_in should pass"


def test_no_text_frame_tilt_up_passes():
    """A non-text frame with tilt_up passes the text constraint."""
    direction = _make_direction_with_beat("normal-photo", "tilt_up", "feature")
    obs_map = {
        "normal-photo": {
            "contains_text": False,
            "curated_section": "Pool",
            "placement": "outdoor",
            "shows_structure": True,
            "is_setting": False,
        }
    }
    violations = validate_opening_establishes(direction, obs_map)
    text_violations = [v for v in violations if "contains text" in v.get("detail", "")]
    assert len(text_violations) == 0


# ── 384x288 worst case: all 8 motions bounded, all pass ──────────────


@pytest.mark.parametrize("motion", [
    "push_in", "pull_back", "pan_left", "pan_right",
    "tilt_up", "tilt_down", "parallax", "hold",
])
def test_384x288_bounded_all_motions_pass(motion):
    """384x288 (smallest Brant photo): every motion stays in bounds."""
    anim = compute_bounded_animation(motion, 384, 288)
    assert anim["type"] == "pan"
    for prefix in ("start", "end"):
        x = float(anim[f"{prefix}_x"].rstrip("%"))
        y = float(anim[f"{prefix}_y"].rstrip("%"))
        s = float(anim[f"{prefix}_scale"].rstrip("%")) / 100
        assert_viewport_in_bounds(x, y, s, 384, 288)
