"""
DIRECTOR-5: Tests for the technique rebalancing — locked as a real choice.
"""

import sys
sys.path.insert(0, ".")

from skills.direct import _build_prompt


def _make_prompt():
    concept = {"title": "Test", "premise": "A retreat."}
    frames = [{"photo_id": "f1", "motion_affordance": ["push_in"],
               "curated_section": "Exterior", "quality_score": 0.9}]
    prop = {"name": "Test", "city": "X", "state_region": "NC",
            "vibe_profile": "romantic_escape", "amenities": []}
    return _build_prompt(concept, frames, prop, {}, [])


# ── Test 1: no "(DEFAULT)" or "Use for MOST beats" ─────────────────────────

def test_no_bounded_default():
    """The prompt no longer marks bounded as DEFAULT or says 'Use for MOST beats'."""
    prompt = _make_prompt()
    assert '"bounded" (DEFAULT)' not in prompt, "bounded must not be marked DEFAULT"
    assert "Use for MOST beats" not in prompt, "'Use for MOST beats' must be removed"


# ── Test 2: slideshow-is-a-failure language ─────────────────────────────────

def test_slideshow_failure_language():
    """The prompt contains language that a film of all bounded beats is a
    slideshow and a failed direction."""
    prompt = _make_prompt()
    assert "slideshow" in prompt.lower(), "Must mention slideshow"
    assert "FAILED direction" in prompt, "Must say all-bounded is a FAILED direction"


# ── Test 3: JSON example does NOT default to bounded ───────────────────────

def test_json_example_not_bounded():
    """The JSON example beat does not specify technique='bounded'."""
    prompt = _make_prompt()
    # Find the first example beat in the JSON schema
    example_start = prompt.find('"ordinal": 1')
    assert example_start > 0, "Must have an example beat"
    # Get the line containing the first example
    example_end = prompt.find("}", example_start)
    example = prompt[example_start:example_end]
    assert '"technique": "bounded"' not in example, (
        f"First example beat must not default to bounded, got: {example}"
    )


# ── Test 4: "bounded variation" phrase gone ─────────────────────────────────

def test_no_bounded_variation_phrase():
    """The phrase 'bounded variation' no longer appears."""
    prompt = _make_prompt()
    assert "bounded variation" not in prompt, \
        "'bounded variation' must be replaced with 'controlled variation'"
    assert "controlled variation" in prompt, "Must use 'controlled variation' instead"


# ── Test 5: KNOW YOUR CURRENT LIMITS mentions locked ───────────────────────

def test_limits_mentions_locked():
    """KNOW YOUR CURRENT LIMITS mentions locked as an available capability."""
    prompt = _make_prompt()
    limits_start = prompt.find("KNOW YOUR CURRENT LIMITS")
    assert limits_start > 0, "Must have KNOW YOUR CURRENT LIMITS section"
    limits_section = prompt[limits_start:limits_start + 500]
    assert "Locked" in limits_section or "locked" in limits_section.lower(), \
        "LIMITS section must mention locked technique"
    assert "animates the content" in limits_section or "animate" in limits_section.lower(), \
        "Must describe what locked does (animates content)"


# ── Test 6: text-bearing rule still present ─────────────────────────────────

def test_text_bearing_rule_preserved():
    """The text-bearing frame rule is still present — G49a / BEACCH guard."""
    prompt = _make_prompt()
    assert 'Text-bearing frames (contains_text=true) MUST be technique="bounded"' in prompt, \
        "Text-bearing rule must be present verbatim"
    assert "Never locked, never generative" in prompt, \
        "Text-bearing prohibition must be present"
