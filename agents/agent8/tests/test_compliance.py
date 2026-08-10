"""
Tests for Agent 8 Stage 8 compliance checker.
All mocked — $0.00 test cost.
"""

from unittest.mock import patch, MagicMock

import pytest

from agents.agent8.compliance import (
    check_subject,
    check_no_guest_names,
    check_no_ota,
    check_no_numeric_ratings,
    check_frame_exit,
    check_amenities,
    check_removal_rule,
    _compute_verdict,
    _collect_text_surfaces,
    DEFAULT_VERDICT,
    CHECKER_VERSION,
    GUIDELINES_VERSION,
    SKIPPED_RULES,
    DETERMINISTIC_RULES,
    MODEL_ASSISTED_RULES,
)


# ── Fixtures ────────────────────────────────────────────────────────────

MOCK_KB = {
    "amenities": [
        {"value": "pool", "source": "airbnb", "confidence": 0.9},
        {"value": "hot tub", "source": "intake_portal", "confidence": 1.0},
        {"value": "fireplace", "source": "airbnb", "confidence": 0.8},
    ],
    "guest_reviews": [
        {
            "text": "Wonderful stay!",
            "reviewer_name": "Sarah Johnson",
            "stay_date": "August 2025",
            "is_guest_book": True,
        },
        {
            "text": "Beautiful property",
            "reviewer_name": "Mike Chen",
            "stay_date": "July 2025",
            "is_guest_book": True,
        },
    ],
}

MOCK_MOTION_CLIP = {
    "id": "clip-001",
    "clip_id": "clip-001",
    "property_id": "prop-001",
    "spec_id": "spec-001",
    "technique": "generative",
    "requested_motion": "push_in",
    "source_image_url": "https://r2.example.com/photo_01.jpg",
    "r2_url": "https://r2.example.com/clip_01.mp4",
    "persistence_manifest": [
        {"element": "pool", "status": "present", "confidence": 0.95},
        {"element": "patio", "status": "present", "confidence": 0.90},
    ],
    "script_text": "",
    "overlay_register": [],
    "status": "ready",
}

MOCK_NARRATION = {
    "id": "narr-001",
    "narration_id": "narr-001",
    "property_id": "prop-001",
    "spec_id": "spec-001",
    "script_text": "Welcome to this beautiful mountain retreat.",
    "overlay_register": [],
    "status": "ready",
}


def _mock_supabase_select(data):
    """Build a mock Supabase select chain that returns data."""
    mock_result = MagicMock()
    mock_result.data = data

    mock_chain = MagicMock()
    mock_chain.select.return_value = mock_chain
    mock_chain.eq.return_value = mock_chain
    mock_chain.is_.return_value = mock_chain
    mock_chain.limit.return_value = mock_chain
    mock_chain.execute.return_value = mock_result
    return mock_chain


def _mock_supabase_insert():
    """Build a mock Supabase insert chain."""
    mock_chain = MagicMock()
    mock_chain.insert.return_value = mock_chain
    mock_chain.execute.return_value = MagicMock(data=[{"id": "new-id"}])
    return mock_chain


def _mock_supabase_update():
    """Build a mock Supabase update chain."""
    mock_chain = MagicMock()
    mock_chain.update.return_value = mock_chain
    mock_chain.eq.return_value = mock_chain
    mock_chain.is_.return_value = mock_chain
    mock_chain.execute.return_value = MagicMock(data=[])
    return mock_chain


# ── Tests ───────────────────────────────────────────────────────────────


class TestVerdictDefaults:
    """test_no_check_row_is_hold — a subject with no check = held."""

    def test_default_verdict_is_hold(self):
        assert DEFAULT_VERDICT == "hold"

    def test_no_check_row_is_hold(self):
        """A subject with no compliance check should be treated as held."""
        # The check_subject function initializes verdict to DEFAULT_VERDICT (hold)
        # and only changes it after all rules evaluate.
        # If no rules run, verdict stays hold.
        verdict = _compute_verdict(
            findings=[],
            rules_evaluated=[],
            rules_skipped=[],
        )
        assert verdict == "hold"


class TestMidRunCrash:
    """test_mid_run_crash_leaves_hold — exception during check -> verdict stays hold."""

    @patch("agents.agent8.compliance._load_subject")
    @patch("agents.agent8.compliance._load_kb")
    @patch("agents.agent8.compliance._persist_check")
    def test_mid_run_crash_leaves_hold(self, mock_persist, mock_kb, mock_load):
        """If a crash happens mid-check, verdict remains hold."""
        # _load_subject returns a subject, but _load_kb raises an exception
        mock_load.return_value = {"property_id": "prop-001"}
        mock_kb.side_effect = RuntimeError("Connection reset")

        result = check_subject("motion_clip", "clip-001", dry_run=True)
        assert result["verdict"] == "hold"


class TestRemovalRule:
    """test_missing_persistence_element_fails_removal — element in manifest not found -> fail."""

    def test_missing_persistence_element_fails_removal(self):
        subject = {
            "technique": "generative",
            "persistence_manifest": [
                {"element": "pool", "status": "present", "confidence": 0.95},
                {"element": "patio", "status": "missing", "confidence": 0.80},
            ],
        }
        result = check_removal_rule(subject)
        assert result["severity"] == "fail"
        assert "missing" in result["detail"].lower()

    def test_generative_never_not_applicable(self):
        """Generative clip -> removal_rule_result != not_applicable."""
        # Generative with empty manifest -> hold (not not_applicable)
        subject_no_manifest = {
            "technique": "generative",
            "persistence_manifest": [],
        }
        result = check_removal_rule(subject_no_manifest)
        assert result["severity"] != "pass" or "not applicable" not in result["detail"].lower()
        # The key assertion: generative technique never gets "not applicable"
        assert "not applicable" not in result["detail"].lower()

    def test_parallax_is_not_applicable(self):
        """Parallax (reprojection) -> removal rule not applicable."""
        subject = {"technique": "parallax"}
        result = check_removal_rule(subject)
        assert result["severity"] == "pass"
        assert "not applicable" in result["detail"].lower()


class TestGuestNameDeterministic:
    """test_guest_name_deterministic_no_model — fails without model call."""

    def test_guest_name_in_narration_fails(self):
        subject = {
            "script_text": "As Sarah Johnson described, the views are breathtaking.",
            "overlay_register": [],
        }
        result = check_no_guest_names(subject, MOCK_KB)
        assert result["severity"] == "fail"
        assert "Sarah Johnson" in result["detail"] or "sarah johnson" in result["detail"].lower()

    def test_no_guest_names_passes(self):
        subject = {
            "script_text": "The mountain views are breathtaking.",
            "overlay_register": [],
        }
        result = check_no_guest_names(subject, MOCK_KB)
        assert result["severity"] == "pass"


class TestOtaReferenceDeterministic:
    """test_ota_reference_deterministic — fails without model call."""

    def test_airbnb_reference_fails(self):
        subject = {
            "script_text": "Top rated on Airbnb!",
            "overlay_register": [],
        }
        result = check_no_ota(subject)
        assert result["severity"] == "fail"
        assert "airbnb" in result["detail"].lower()

    def test_no_ota_passes(self):
        subject = {
            "script_text": "Welcome to the mountain retreat.",
            "overlay_register": [],
        }
        result = check_no_ota(subject)
        assert result["severity"] == "pass"


class TestUndecidedPeopleRulesSkipped:
    """test_undecided_people_in_rules_skipped — not in findings, not passed."""

    def test_generated_people_in_skipped_rules(self):
        assert any(name == "generated_people" for name, _ in SKIPPED_RULES)

    @patch("agents.agent8.compliance._load_subject")
    @patch("agents.agent8.compliance._load_kb")
    @patch("agents.agent8.compliance._persist_check")
    @patch("agents.agent8.compliance._escalate")
    def test_skipped_rules_appear_in_check(self, mock_esc, mock_persist, mock_kb, mock_load):
        mock_load.return_value = MOCK_MOTION_CLIP
        mock_kb.return_value = MOCK_KB

        result = check_subject("motion_clip", "clip-001", dry_run=True)

        # generated_people must be in rules_skipped
        skipped_names = [s["rule"] for s in result["rules_skipped"]]
        assert "generated_people" in skipped_names

        # generated_people must NOT be in rules_evaluated
        assert "generated_people" not in result["rules_evaluated"]

        # generated_people must NOT appear in findings
        finding_rules = [f["rule"] for f in result["findings"]]
        assert "generated_people" not in finding_rules


class TestFailureEscalates:
    """test_failure_escalates — fail verdict -> hitl_queue_items insert."""

    @patch("agents.agent8.compliance._load_subject")
    @patch("agents.agent8.compliance._load_kb")
    @patch("agents.agent8.compliance._persist_check")
    @patch("agents.agent8.compliance._escalate")
    def test_failure_escalates(self, mock_escalate, mock_persist, mock_kb, mock_load):
        # Subject with guest name in narration -> fail
        subject_with_name = dict(MOCK_NARRATION)
        subject_with_name["script_text"] = "Sarah Johnson loved the views."
        mock_load.return_value = subject_with_name
        mock_kb.return_value = MOCK_KB

        result = check_subject("narration", "narr-001", dry_run=False)

        assert result["verdict"] in ("fail", "hold")
        # _escalate should have been called because verdict is fail or hold
        mock_escalate.assert_called_once()
        call_args = mock_escalate.call_args
        assert call_args[1]["property_id"] == "prop-001" or call_args[0][1] == "prop-001"


class TestRulesEvaluatedDistinguishes:
    """test_rules_evaluated_distinguishes — passed rules listed in rules_evaluated."""

    @patch("agents.agent8.compliance._load_subject")
    @patch("agents.agent8.compliance._load_kb")
    @patch("agents.agent8.compliance._persist_check")
    @patch("agents.agent8.compliance._escalate")
    def test_rules_evaluated_lists_run_rules(self, mock_esc, mock_persist, mock_kb, mock_load):
        mock_load.return_value = MOCK_MOTION_CLIP
        mock_kb.return_value = MOCK_KB

        result = check_subject("motion_clip", "clip-001", dry_run=True)

        # All deterministic and model-assisted rule names should be in rules_evaluated
        expected_rules = [name for name, _ in DETERMINISTIC_RULES] + [
            name for name, _ in MODEL_ASSISTED_RULES
        ]
        for rule_name in expected_rules:
            assert rule_name in result["rules_evaluated"], (
                f"Rule '{rule_name}' not in rules_evaluated: {result['rules_evaluated']}"
            )


class TestFindingsRetainedOnPass:
    """test_findings_retained_on_pass — clean check has findings with severity=pass."""

    @patch("agents.agent8.compliance._load_subject")
    @patch("agents.agent8.compliance._load_kb")
    @patch("agents.agent8.compliance._persist_check")
    @patch("agents.agent8.compliance._escalate")
    def test_findings_retained_on_pass(self, mock_esc, mock_persist, mock_kb, mock_load):
        # Build a clean subject that passes all deterministic checks
        clean_subject = {
            "id": "clip-clean",
            "clip_id": "clip-clean",
            "property_id": "prop-001",
            "technique": "parallax",  # parallax -> removal_rule not_applicable (pass)
            "requested_motion": "push_in",
            "source_image_url": "https://r2.example.com/photo_01.jpg",
            "r2_url": "https://r2.example.com/clip_clean.mp4",
            "persistence_manifest": [
                {"element": "pool", "status": "present", "confidence": 0.95},
            ],
            "script_text": "Beautiful mountain retreat with pool and hot tub.",
            "overlay_register": [],
            "status": "ready",
        }
        mock_load.return_value = clean_subject
        # KB with matching amenities so no fabrication is flagged
        mock_kb.return_value = MOCK_KB

        result = check_subject("motion_clip", "clip-clean", dry_run=True)

        # Findings should exist even on a passing check
        assert len(result["findings"]) > 0
        # Each finding should have severity=pass (for a clean subject)
        pass_findings = [f for f in result["findings"] if f["severity"] == "pass"]
        assert len(pass_findings) > 0

        # Note: verdict will be "hold" because SKIPPED_RULES is non-empty,
        # but each individual finding should be pass for a clean subject.
        for f in result["findings"]:
            assert f["severity"] == "pass", (
                f"Expected pass for clean subject, got {f['severity']} on rule {f['rule']}: {f['detail']}"
            )


class TestNumericRatings:
    """Additional deterministic check coverage."""

    def test_star_rating_fails(self):
        subject = {"script_text": "Rated 4.9/5 by guests!", "overlay_register": []}
        result = check_no_numeric_ratings(subject)
        assert result["severity"] == "fail"

    def test_out_of_ten_fails(self):
        subject = {"script_text": "A perfect 10/10 experience", "overlay_register": []}
        result = check_no_numeric_ratings(subject)
        assert result["severity"] == "fail"

    def test_stars_word_fails(self):
        subject = {"script_text": "5 stars from every guest", "overlay_register": []}
        result = check_no_numeric_ratings(subject)
        assert result["severity"] == "fail"

    def test_no_ratings_passes(self):
        subject = {"script_text": "Guests love this property", "overlay_register": []}
        result = check_no_numeric_ratings(subject)
        assert result["severity"] == "pass"


class TestFrameExit:
    """Frame-exit move check coverage."""

    def test_pull_back_fails(self):
        subject = {"requested_motion": "pull_back"}
        result = check_frame_exit(subject)
        assert result["severity"] == "fail"

    def test_push_in_passes(self):
        subject = {"requested_motion": "push_in"}
        result = check_frame_exit(subject)
        assert result["severity"] == "pass"
