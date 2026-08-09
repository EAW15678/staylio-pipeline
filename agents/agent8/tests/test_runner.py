"""
Tests for Agent 8 runner and shared retry.
All mocked — $0.00 test cost.
"""

import time
from datetime import date
from unittest.mock import patch, MagicMock

import pytest

from agents.agent8.runner import run_cycle, _check_existing_concepts
from agents.agent8.retry import retry_with_backoff

VISTA_AZULE_ID = "a1b2c3d4-0001-0001-0001-000000000001"
CYCLE = date(2026, 9, 1)


# ── Retry tests ──────────────────────────────────────────────────────────

class TestRetryWithBackoff:
    def test_succeeds_first_try(self):
        result = retry_with_backoff(
            fn=lambda: 42,
            max_retries=3,
            retryable=(ValueError,),
            label="test",
        )
        assert result == 42

    def test_retries_on_retryable_error(self):
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "ok"

        result = retry_with_backoff(
            fn=flaky,
            max_retries=3,
            backoff_base=0.01,  # fast for tests
            retryable=(ValueError,),
            label="test",
        )
        assert result == "ok"
        assert call_count == 3

    def test_raises_non_retryable_immediately(self):
        def bad():
            raise TypeError("wrong type")

        with pytest.raises(TypeError):
            retry_with_backoff(
                fn=bad,
                max_retries=3,
                retryable=(ValueError,),
                label="test",
            )

    def test_raises_after_max_retries(self):
        def always_fails():
            raise ValueError("always")

        with pytest.raises(ValueError, match="always"):
            retry_with_backoff(
                fn=always_fails,
                max_retries=2,
                backoff_base=0.01,
                retryable=(ValueError,),
                label="test",
            )

    def test_backoff_increases(self):
        """Verify backoff waits increase exponentially."""
        waits = []
        original_sleep = time.sleep

        def mock_sleep(secs):
            waits.append(secs)

        call_count = 0

        def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("fail")
            return "ok"

        with patch("core.retry.time.sleep", side_effect=mock_sleep):
            retry_with_backoff(
                fn=fails_twice,
                max_retries=3,
                backoff_base=2.0,
                retryable=(ValueError,),
                label="test",
            )

        assert len(waits) == 2
        assert waits[0] == 2.0   # 2^1
        assert waits[1] == 4.0   # 2^2


# ── Runner tests ─────────────────────────────────────────────────────────

MOCK_EXISTING_CONCEPTS = [
    {
        "concept_id": f"id-{i}",
        "concept_number": i,
        "title": f"Concept {i}",
        "premise": f"Premise {i}",
        "status": "draft",
        "concept_version": 1,
        "utm_content_slug": f"vista-azule_2026-09_c{i}",
    }
    for i in range(1, 9)
]


class TestRunCycle:
    def test_skips_generation_when_concepts_exist(self):
        """Default: existing active concepts are returned, no model call."""
        with patch(
            "agents.agent8.runner._check_existing_concepts",
            return_value=MOCK_EXISTING_CONCEPTS,
        ):
            result = run_cycle(VISTA_AZULE_ID, CYCLE)

        assert len(result) == 8
        assert result[0]["title"] == "Concept 1"

    def test_generates_when_no_concepts_exist(self):
        """Calls generate_concepts when concept_ledger is empty for this cycle."""
        mock_concepts = [{"title": f"New {i}", "concept_number": i} for i in range(1, 9)]

        with patch(
            "agents.agent8.runner._check_existing_concepts",
            return_value=None,
        ), patch(
            "agents.agent8.concept_generator.generate_concepts",
            return_value=mock_concepts,
        ) as mock_gen:
            result = run_cycle(VISTA_AZULE_ID, CYCLE)

        mock_gen.assert_called_once_with(
            property_id=VISTA_AZULE_ID,
            cycle_month=CYCLE,
            dry_run=False,
        )
        assert len(result) == 8

    def test_dry_run_with_existing_makes_no_model_call(self):
        """dry_run=True with existing concepts returns them — NO model call, $0."""
        with patch(
            "agents.agent8.runner._check_existing_concepts",
            return_value=MOCK_EXISTING_CONCEPTS,
        ) as mock_check:
            result = run_cycle(VISTA_AZULE_ID, CYCLE, dry_run=True)

        mock_check.assert_called_once()
        assert len(result) == 8
        assert result[0]["title"] == "Concept 1"

    def test_dry_run_without_existing_generates_without_writing(self):
        """dry_run=True without existing concepts generates but passes dry_run to generator."""
        mock_concepts = [{"title": f"Dry {i}", "concept_number": i} for i in range(1, 9)]

        with patch(
            "agents.agent8.runner._check_existing_concepts",
            return_value=None,
        ), patch(
            "agents.agent8.concept_generator.generate_concepts",
            return_value=mock_concepts,
        ) as mock_gen:
            result = run_cycle(VISTA_AZULE_ID, CYCLE, dry_run=True)

        mock_gen.assert_called_once_with(
            property_id=VISTA_AZULE_ID,
            cycle_month=CYCLE,
            dry_run=True,
        )
        assert len(result) == 8

    def test_force_bypasses_existing_check(self):
        """force=True ignores existing concepts and regenerates."""
        mock_concepts = [{"title": f"Forced {i}", "concept_number": i} for i in range(1, 9)]

        with patch(
            "agents.agent8.runner._check_existing_concepts",
        ) as mock_check, patch(
            "agents.agent8.concept_generator.generate_concepts",
            return_value=mock_concepts,
        ) as mock_gen:
            result = run_cycle(VISTA_AZULE_ID, CYCLE, force=True)

        mock_check.assert_not_called()
        mock_gen.assert_called_once_with(
            property_id=VISTA_AZULE_ID,
            cycle_month=CYCLE,
            dry_run=False,
        )
        assert result[0]["title"] == "Forced 1"

    def test_force_dry_run_generates_without_writing(self):
        """force=True + dry_run=True regenerates without writing — costs ~$0.03."""
        mock_concepts = [{"title": f"Preview {i}", "concept_number": i} for i in range(1, 9)]

        with patch(
            "agents.agent8.runner._check_existing_concepts",
        ) as mock_check, patch(
            "agents.agent8.concept_generator.generate_concepts",
            return_value=mock_concepts,
        ) as mock_gen:
            result = run_cycle(VISTA_AZULE_ID, CYCLE, force=True, dry_run=True)

        mock_check.assert_not_called()
        mock_gen.assert_called_once_with(
            property_id=VISTA_AZULE_ID,
            cycle_month=CYCLE,
            dry_run=True,
        )
        assert result[0]["title"] == "Preview 1"
