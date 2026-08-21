"""
PHASE0-1: Tests for the rewritten Runway locked prompt.

Runway's Gen-4 guide: negative phrasing produces opposite results;
re-describing image content reduces motion. The old template violated
both rules and is the suspected cause of the wobble recorded in G78.
"""

import sys
sys.path.insert(0, ".")

from skills.generate_motion import _LOCKED_PROMPT_TEMPLATE


# ── Test 1: no negative phrasing ────────────────────────────────────────────

def test_no_negative_phrasing():
    """The template contains no negative phrasing — Runway documents this
    as producing opposite results."""
    prompt = _LOCKED_PROMPT_TEMPLATE.format(content_motion="Water ripples gently.")

    negatives = [
        "does not",
        "no wobble",
        "no camera shake",
        "no micro-drift",
        "no oscillation",
        "no handheld",
        "do not move",
        "don't move",
    ]
    for neg in negatives:
        assert neg not in prompt.lower(), (
            f"Template must not contain negative phrasing '{neg}' — "
            f"Runway documents this as producing opposite results"
        )


# ── Test 2: no scene element re-description ─────────────────────────────────

def test_no_scene_redescription():
    """The template does not re-describe scene elements already in the image —
    Runway documents this as reducing motion."""
    prompt = _LOCKED_PROMPT_TEMPLATE.format(content_motion="Fire flickers.")

    redescriptions = ["architecture", "railings", "structural elements",
                      "fixed objects", "motionless"]
    for term in redescriptions:
        assert term not in prompt.lower(), (
            f"Template must not re-describe scene elements ('{term}') — "
            f"Runway says this reduces motion"
        )


# ── Test 3: content_motion interpolated correctly ───────────────────────────

def test_content_motion_interpolated():
    """{content_motion} is interpolated and reaches the final prompt."""
    test_motion = "Water surface catches afternoon light, ripples spread outward."
    prompt = _LOCKED_PROMPT_TEMPLATE.format(content_motion=test_motion)

    assert test_motion in prompt, (
        f"content_motion must appear in the final prompt, got: {prompt[:200]}"
    )
    assert "locked-off camera" in prompt.lower(), \
        "Must contain the positive camera-still phrasing"
