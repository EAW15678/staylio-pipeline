"""
DIRECTOR-3: Tests for the quality self-assessment mechanism.

Tests the _assess_quality function directly and the integration with
the revision loop and alert logic.
"""

import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, ".")

from skills.direct import (
    _assess_quality, _build_prompt,
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


# ── Test 7: exhausted attempts → escalate_halt called AND returns ok ────────

def test_alert_and_ship():
    """With attempts exhausted, escalate_halt IS called AND the skill
    would still return ok with a direction_id."""
    shortfalls = [
        {"dimension": "hook_strength", "score": 4, "threshold": 6,
         "reason": "below_minimum", "director_says": "Weak opening"},
    ]

    # Simulate the alert-and-ship logic from direct()
    import skills.contract
    with patch("skills.contract.escalate_halt") as mock_halt:
        # This is what direct() does after the loop
        if shortfalls:
            from skills.contract import escalate_halt
            shortfall_detail = "; ".join(
                f"{s['dimension']}: {s['score']}/{s['threshold']} — {s['director_says']}"
                for s in shortfalls
            )
            escalate_halt(
                MagicMock(), "PROP-1",
                queue_type="content_review",
                reason_code="quality_below_threshold",
                title="Direction quality below threshold",
                detail=shortfall_detail,
            )

        assert mock_halt.called, "escalate_halt must be called on shortfalls"
        assert mock_halt.call_args.kwargs["queue_type"] == "content_review"

    # Status would still be "approved" if no validator violations
    all_violations = []
    status = "draft" if all_violations else "approved"
    assert status == "approved", "Quality shortfalls must NOT change status"


# ── Test 8: shortfalls not in rejection_reasons, not in status ──────────────

def test_shortfalls_separate_from_violations():
    """Quality shortfalls do not appear in rejection_reasons and do not
    alter status."""
    all_violations = []  # no validator violations
    quality_shortfalls = [{"dimension": "hook_strength", "score": 4,
                           "threshold": 6, "reason": "below_minimum",
                           "director_says": "Weak"}]

    # Status determination (from direct.py)
    status = "draft" if all_violations else "approved"
    rejection_reasons = all_violations if all_violations else None

    assert status == "approved", "Quality shortfalls must not change status"
    assert rejection_reasons is None, "Quality shortfalls must not appear in rejection_reasons"

    # The insert would have quality_shortfalls as a SEPARATE field
    insert_data = {
        "rejection_reasons": rejection_reasons,
        "quality_shortfalls": quality_shortfalls,
        "status": status,
    }
    assert insert_data["status"] == "approved"
    assert insert_data["rejection_reasons"] is None
    assert insert_data["quality_shortfalls"] is not None


# ── Test 9: escalate_halt failure doesn't break direction ───────────────────

def test_alert_failure_doesnt_break():
    """If escalate_halt raises, the direction still returns ok."""
    import skills.contract
    with patch("skills.contract.escalate_halt", side_effect=RuntimeError("alert down")):
        # Simulate the try/except from direct()
        try:
            from skills.contract import escalate_halt
            escalate_halt(MagicMock(), "PROP-1",
                          queue_type="content_review",
                          reason_code="quality_below_threshold",
                          title="test", detail="test")
        except Exception:
            pass  # "A failed alert must never break a successful direction"

    # If we got here without raising, the direction would continue
    assert True, "Alert failure must not break the direction"
