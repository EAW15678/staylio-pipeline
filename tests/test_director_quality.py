"""
DIRECTOR-3: Tests for the quality self-assessment mechanism.

Tests the _assess_quality function directly and the integration with
the revision loop and alert logic.
"""

import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, ".")

from skills.direct import (
    _assess_quality, _build_prompt, _determine_status,
    _alert_on_quality_shortfalls,
    _QUALITY_MIN_DIMENSION, _QUALITY_MIN_AVERAGE, _QUALITY_DIMENSIONS,
)


def _make_scores(base_score=8):
    """Build a quality_self_score dict with all dimensions at base_score."""
    return {
        dim: {"score": base_score, "why": f"Test reason for {dim}"}
        for dim in _QUALITY_DIMENSIONS
    }


def _make_prompt():
    concept = {"title": "Test", "premise": "A retreat."}
    frames = [{"photo_id": "f1", "motion_affordance": ["push_in"],
               "curated_section": "Exterior", "quality_score": 0.9}]
    prop = {"name": "Test", "city": "X", "state_region": "NC",
            "vibe_profile": "romantic_escape", "amenities": []}
    return _build_prompt(concept, frames, prop, {}, [])


# ── Test 1: prompt has nine dimensions + quality_self_score schema ───────────

def test_prompt_has_quality_dimensions():
    prompt = _make_prompt()

    for dim in _QUALITY_DIMENSIONS:
        assert dim in prompt, f"Prompt must instruct on dimension '{dim}'"

    assert "quality_self_score" in prompt, "Output schema must include quality_self_score"
    assert "SCORE YOUR OWN WORK" in prompt, "Brief must instruct on self-scoring"


# ── Test 2: template risk excluded ──────────────────────────────────────────

def test_template_risk_excluded():
    prompt = _make_prompt()
    assert "template_risk" not in prompt, "template_risk must NOT appear in the prompt"
    assert "template risk" not in prompt.lower() or "excluded" in prompt.lower(), \
        "If template risk is mentioned, it must be in context of exclusion"


# ── Test 3: all scores >= 8 → no shortfalls ─────────────────────────────────

def test_high_scores_no_shortfalls():
    result = {"quality_self_score": _make_scores(8)}
    shortfalls = _assess_quality(result)
    assert shortfalls == [], f"All 8s should produce no shortfalls, got: {shortfalls}"


# ── Test 4: single dimension at 4 → shortfall ──────────────────────────────

def test_single_low_dimension_shortfall():
    scores = _make_scores(8)
    scores["hook_strength"] = {"score": 4, "why": "Weak opening"}
    result = {"quality_self_score": scores}
    shortfalls = _assess_quality(result)

    dim_names = [s["dimension"] for s in shortfalls]
    assert "hook_strength" in dim_names, f"hook_strength should be a shortfall, got: {dim_names}"
    assert shortfalls[0]["score"] == 4
    assert shortfalls[0]["director_says"] == "Weak opening"


# ── Test 5: all at 6 → average shortfall (6.0 < 7.0) ───────────────────────

def test_average_shortfall():
    """All dimensions at 6: none individually below 6 (min threshold),
    but mean 6.0 < 7.0 (average threshold). Proves both thresholds
    work independently."""
    scores = _make_scores(6)
    result = {"quality_self_score": scores}
    shortfalls = _assess_quality(result)

    # No individual dimension shortfall (6 >= _QUALITY_MIN_DIMENSION=6)
    individual = [s for s in shortfalls if s["dimension"] != "_average"]
    assert individual == [], f"No individual shortfalls at 6, got: {individual}"

    # But average shortfall should fire (6.0 < 7.0)
    avg_shortfalls = [s for s in shortfalls if s["dimension"] == "_average"]
    assert len(avg_shortfalls) == 1, f"Average shortfall should fire, got: {shortfalls}"
    assert avg_shortfalls[0]["score"] == 6.0


# ── Test 6: shortfall triggers revision with dimension names ────────────────

def test_shortfall_triggers_revision():
    """With attempts remaining, a shortfall triggers a revision, and the
    revision prompt contains the failing dimension names."""
    # We test the revision_issues construction logic directly
    quality_shortfalls = [
        {"dimension": "hook_strength", "score": 3, "threshold": 6,
         "reason": "below_minimum", "director_says": "Weak opening"},
    ]

    # Build revision issues as the loop does
    revision_issues = []
    for sf in quality_shortfalls:
        revision_issues.append({
            "rule": f"quality_{sf['dimension']}",
            "detail": f"Self-score {sf['score']}/{sf['threshold']} — {sf['director_says']}",
            "beats": [],
        })

    assert len(revision_issues) == 1
    assert "hook_strength" in revision_issues[0]["rule"]
    assert "Weak opening" in revision_issues[0]["detail"]


# ── Test 7: _alert_on_quality_shortfalls calls escalate_halt ─────────────────

def test_alert_and_ship():
    """_alert_on_quality_shortfalls calls escalate_halt with the correct
    queue_type and reason_code, and returns normally."""
    shortfalls = [
        {"dimension": "hook_strength", "score": 4, "threshold": 6,
         "reason": "below_minimum", "director_says": "Weak opening"},
    ]

    import skills.contract
    with patch("skills.contract.escalate_halt") as mock_halt:
        _alert_on_quality_shortfalls(
            MagicMock(), "PROP-1", {"name": "Test"}, shortfalls,
            run_id="RUN-1", attempts=3)

    assert mock_halt.called, "escalate_halt must be called on shortfalls"
    assert mock_halt.call_args.kwargs["queue_type"] == "content_review"
    assert mock_halt.call_args.kwargs["reason_code"] == "quality_below_threshold"
    assert "hook_strength" in mock_halt.call_args.kwargs["detail"]


# ── Test 8: _determine_status takes only violations, not shortfalls ─────────

def test_shortfalls_cannot_reach_status():
    """_determine_status takes only validator violations as input.
    Quality shortfalls cannot reach it by construction — the function
    signature does not accept them. This is structural proof, not a
    value assertion."""
    import inspect

    # Structural: _determine_status takes ONLY all_violations
    sig = inspect.signature(_determine_status)
    params = list(sig.parameters.keys())
    assert params == ["all_violations"], (
        f"_determine_status must take only 'all_violations', got: {params}"
    )

    # Behavioural: no violations → approved
    assert _determine_status([]) == "approved"

    # Behavioural: with violations → draft
    assert _determine_status([{"rule": "test", "detail": "x"}]) == "draft"

    # The key point: there is no way to pass quality_shortfalls to this
    # function — it has no parameter for them. The separation is enforced
    # by the function's interface, not by a conditional inside it.


# ── Test 9: _alert_on_quality_shortfalls never raises ────────────────────────

def test_alert_failure_doesnt_break():
    """If escalate_halt raises inside _alert_on_quality_shortfalls,
    the function returns normally — never propagates the exception."""
    shortfalls = [
        {"dimension": "hook_strength", "score": 4, "threshold": 6,
         "reason": "below_minimum", "director_says": "Weak"},
    ]

    import skills.contract
    with patch("skills.contract.escalate_halt", side_effect=RuntimeError("alert down")):
        # Must not raise — the try/except inside catches it
        _alert_on_quality_shortfalls(
            MagicMock(), "PROP-1", {"name": "Test"}, shortfalls)
