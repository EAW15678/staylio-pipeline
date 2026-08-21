"""
MOTION-3: Locked technique tests.

Verifies the three-technique routing: bounded ($0), locked (Runway,
camera-locked), generative (Runway, camera-free). Text-bearing frames
cannot use locked or generative.

All mocked — no vendor calls, $0.
"""

import pytest

from skills.generate_motion import (
    VALID_TECHNIQUES,
    _LOCKED_PROMPT_TEMPLATE,
    FRAME_EXITING_MOVES,
)
from skills.direct import validate_opening_establishes


# ── Technique validation ──────────────────────────────────────────────


def test_valid_techniques_has_three():
    """Three techniques: bounded, locked, generative."""
    assert VALID_TECHNIQUES == {"bounded", "locked", "generative"}


def test_locked_prompt_template_has_placeholder():
    """Template has {content_motion} placeholder."""
    assert "{content_motion}" in _LOCKED_PROMPT_TEMPLATE


def test_locked_prompt_template_positive_only():
    """PHASE0-1: Template uses positive phrasing only — no negatives.
    Runway documents negatives as producing opposite results."""
    t = _LOCKED_PROMPT_TEMPLATE.lower()
    assert "locked-off" in t or "still" in t, "Must state camera is still (positive)"
    assert "does not move" not in t, "No negative phrasing"
    assert "wobble" not in t, "No wobble negatives"


def test_locked_prompt_template_locks_camera():
    """Template states camera is still using positive phrasing."""
    t = _LOCKED_PROMPT_TEMPLATE.lower()
    assert "still" in t, "Must state camera is still"


def test_locked_prompt_fills_content_motion():
    """Template fills content_motion from director's beat."""
    filled = _LOCKED_PROMPT_TEMPLATE.format(
        content_motion="Pool water ripples with reflected sunlight."
    )
    assert "Pool water ripples" in filled
    assert "still" in filled.lower()


# ── Text constraint with technique ────────────────────────────────────


def _make_direction(photo_id, motion, technique, opening_type="feature"):
    return {
        "beats": [{
            "ordinal": 1,
            "photo_id": photo_id,
            "requested_motion": motion,
            "technique": technique,
        }],
        "opening_type": opening_type,
    }


def _text_obs_map(photo_id):
    return {
        photo_id: {
            "contains_text": True,
            "text_content": "BEACH",
            "curated_section": "Exterior",
            "placement": "outdoor",
            "shows_structure": False,
            "is_setting": False,
        }
    }


def _clean_obs_map(photo_id):
    return {
        photo_id: {
            "contains_text": False,
            "curated_section": "Pool",
            "placement": "outdoor",
            "shows_structure": True,
            "is_setting": False,
        }
    }


def test_text_frame_locked_fails():
    """contains_text + technique='locked' → FAILS."""
    direction = _make_direction("txt", "push_in", "locked")
    violations = validate_opening_establishes(direction, _text_obs_map("txt"))
    technique_violations = [v for v in violations if "technique" in v.get("detail", "")]
    assert len(technique_violations) >= 1


def test_text_frame_generative_fails():
    """contains_text + technique='generative' → FAILS."""
    direction = _make_direction("txt", "push_in", "generative")
    violations = validate_opening_establishes(direction, _text_obs_map("txt"))
    technique_violations = [v for v in violations if "technique" in v.get("detail", "")]
    assert len(technique_violations) >= 1


def test_text_frame_bounded_push_in_passes():
    """contains_text + technique='bounded' + push_in → PASSES."""
    direction = _make_direction("txt", "push_in", "bounded")
    violations = validate_opening_establishes(direction, _text_obs_map("txt"))
    # Should have no text-related violations
    text_violations = [v for v in violations
                       if "contains text" in v.get("detail", "")
                       or "technique" in v.get("detail", "")]
    assert len(text_violations) == 0


def test_clean_frame_locked_passes():
    """Non-text frame + technique='locked' → no text violation."""
    direction = _make_direction("pool", "push_in", "locked")
    violations = validate_opening_establishes(direction, _clean_obs_map("pool"))
    technique_violations = [v for v in violations if "technique" in v.get("detail", "")]
    assert len(technique_violations) == 0


def test_clean_frame_bounded_passes():
    """Non-text frame + technique='bounded' → no text violation."""
    direction = _make_direction("pool", "push_in", "bounded")
    violations = validate_opening_establishes(direction, _clean_obs_map("pool"))
    technique_violations = [v for v in violations if "technique" in v.get("detail", "")]
    assert len(technique_violations) == 0


# ── Unknown technique ────────────────────────────────────────────────


def test_unknown_technique_not_valid():
    """An unknown technique value is not in VALID_TECHNIQUES."""
    assert "cinematic" not in VALID_TECHNIQUES
    assert "ai_magic" not in VALID_TECHNIQUES


# ── Existing techniques unchanged ────────────────────────────────────


def test_frame_exiting_moves_unchanged():
    """FRAME_EXITING_MOVES still contains pull_back and tilt_up."""
    assert "pull_back" in FRAME_EXITING_MOVES
    assert "tilt_up" in FRAME_EXITING_MOVES
    assert "push_in" not in FRAME_EXITING_MOVES
