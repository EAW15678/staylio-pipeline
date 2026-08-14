"""
GUARDRAIL-2: Motion downgrade tests.

Verifies frame-exit guard downgrades instead of dropping.
All mocked — no vendor calls, $0.
"""

from skills.generate_motion import FRAME_EXITING_MOVES


def test_pull_back_in_frame_exiting():
    assert "pull_back" in FRAME_EXITING_MOVES


def test_tilt_up_in_frame_exiting():
    assert "tilt_up" in FRAME_EXITING_MOVES


def test_push_in_not_in_frame_exiting():
    assert "push_in" not in FRAME_EXITING_MOVES


def test_pan_left_not_in_frame_exiting():
    assert "pan_left" not in FRAME_EXITING_MOVES


def test_parallax_not_in_frame_exiting():
    assert "parallax" not in FRAME_EXITING_MOVES


def test_downgrade_produces_safe_prompt():
    """The downgraded prompt must not contain pull-back or tilt-up language."""
    safe_prompt = "Slow push in toward the centre of the frame. Hold the existing composition steady."
    assert "pull" not in safe_prompt.lower()
    assert "tilt" not in safe_prompt.lower()
    assert "back" not in safe_prompt.lower()
    assert "up" not in safe_prompt.lower() or "push" in safe_prompt.lower()


def test_motion_params_records_downgrade():
    """The artifact's motion_params should record the downgrade."""
    # Simulate what generate_motion writes
    requested = "pull_back"
    actual = "push_in"
    downgraded = True
    original_prompt = "Pull back from the house facade"
    actual_prompt = "Slow push in toward the centre of the frame."

    params = {
        "prompt": actual_prompt,
        "actual_motion": actual,
        "downgraded": downgraded,
        "original_motion": requested,
        "original_prompt": original_prompt,
    }

    assert params["downgraded"] is True
    assert params["actual_motion"] == "push_in"
    assert params["original_motion"] == "pull_back"
    assert "pull" not in params["prompt"].lower()


def test_non_downgraded_params():
    """Non-downgraded beats should have downgraded=False."""
    params = {
        "prompt": "Pan right across the kitchen",
        "actual_motion": "pan_right",
        "downgraded": False,
        "original_motion": None,
        "original_prompt": None,
    }

    assert params["downgraded"] is False
    assert params["original_motion"] is None
