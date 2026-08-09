"""
Tests for Agent 8 Stage 4 narration rendering.
All mocked — $0.00 test cost.
"""

from unittest.mock import patch, MagicMock

import pytest

from agents.agent8.narration import (
    render_narration,
    _compute_input_hash,
    _detect_verbatim_source,
)

MOCK_SPEC = {
    "spec_id": "spec-001",
    "property_id": "prop-001",
    "concept_id": "concept-001",
    "narration_brief": "Three generations under one roof, and nobody got the short straw.",
    "narration_provenance": "original",
    "source_material": [],
    "vibe_drift": None,
}

MOCK_SPEC_GUEST_BOOK = {
    **MOCK_SPEC,
    "narration_brief": "Nobody felt like they got the bad room.",
    "narration_provenance": "guest_book",
    "source_material": ["guest_book: The Williamson Family"],
}

MOCK_SPEC_NO_NARRATION = {
    **MOCK_SPEC,
    "narration_brief": "",
}

MOCK_SPEC_WITH_NAME = {
    **MOCK_SPEC,
    "narration_brief": "As the Williamson Family put it, this was perfect.",
    "narration_provenance": "guest_book",
}

MOCK_KB = {
    "vibe_profile": "multigenerational_retreat",
    "guest_reviews": [
        {
            "text": "Nobody felt like they got the bad room. That never happens.",
            "reviewer_name": "The Williamson Family",
            "is_guest_book": True,
        },
    ],
    "amenities": [],
}

MOCK_CACHED_NARRATION = {
    "narration_id": "narr-cached",
    "status": "ready",
    "r2_url": "https://r2.example.com/narration/cached.mp3",
    "input_hash": "abc123",
}


def _patch_all(spec=MOCK_SPEC, kb=MOCK_KB, cached=None, render_result=None):
    """Return a context manager that patches all external calls."""
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        with patch("agents.agent8.narration._load_spec", return_value=spec), \
             patch("agents.agent8.narration._load_kb", return_value=kb), \
             patch("agents.agent8.narration._check_cached_narration", return_value=cached), \
             patch("agents.agent8.narration._render_elevenlabs", return_value=render_result), \
             patch("agents.agent8.narration._persist_narration"):
            yield

    return _ctx()


class TestRenderNarration:
    def test_no_narration_brief_returns_empty(self):
        """No narration_brief → clean return, no render."""
        with _patch_all(spec=MOCK_SPEC_NO_NARRATION):
            result = render_narration("spec-001", dry_run=True)
        assert result == []

    def test_cache_hit_makes_no_vendor_call(self):
        """Cached narration returned without rendering."""
        with _patch_all(cached=MOCK_CACHED_NARRATION) as _:
            result = render_narration("spec-001")
        assert len(result) == 1
        assert result[0]["status"] == "ready"
        assert result[0]["r2_url"] == "https://r2.example.com/narration/cached.mp3"

    def test_voice_selected_by_vibe(self):
        """Voice is determined by KB vibe_profile and recorded on the row."""
        render_result = {
            "r2_url": "https://r2.example.com/narration/new.mp3",
            "duration_seconds": 5.0,
            "alignment": None,
        }
        with _patch_all(render_result=render_result):
            result = render_narration("spec-001")
        assert len(result) == 1
        assert result[0]["vibe_prior"] == "multigenerational_retreat"
        assert result[0]["voice_label"] == "Multi-Generational voice"

    def test_guest_name_fails_before_vendor_call(self):
        """A guest name in narration_brief fails BEFORE any vendor call."""
        with _patch_all(spec=MOCK_SPEC_WITH_NAME) as _:
            with patch("agents.agent8.narration._render_elevenlabs") as mock_render:
                result = render_narration("spec-001")
                mock_render.assert_not_called()
        assert len(result) == 1
        assert result[0]["status"] == "failed"
        assert "Guest name" in result[0]["failure_reason"]

    def test_render_failure_lands_in_status(self):
        """Failed render → status='failed' + failure_reason, no raise."""
        with _patch_all(render_result=None):
            result = render_narration("spec-001")
        assert len(result) == 1
        assert result[0]["status"] == "failed"
        assert "failed after retries" in result[0]["failure_reason"]

    def test_dry_run_returns_pending_without_rendering(self):
        """dry_run returns asset spec with status='pending', no vendor call."""
        with _patch_all():
            with patch("agents.agent8.narration._render_elevenlabs") as mock_render:
                result = render_narration("spec-001", dry_run=True)
                mock_render.assert_not_called()
        assert len(result) == 1
        assert result[0]["status"] == "pending"
        assert result[0]["r2_url"] is None

    def test_force_bypasses_cache(self):
        """force=True ignores cached narration and re-renders."""
        render_result = {
            "r2_url": "https://r2.example.com/narration/forced.mp3",
            "duration_seconds": 5.0,
            "alignment": None,
        }
        with _patch_all(cached=MOCK_CACHED_NARRATION, render_result=render_result):
            result = render_narration("spec-001", force=True)
        assert len(result) == 1
        assert result[0]["r2_url"] == "https://r2.example.com/narration/forced.mp3"

    def test_contains_verbatim_source_set_for_guest_book(self):
        """Guest-book provenance → contains_verbatim_source=True + source_reference populated."""
        render_result = {
            "r2_url": "https://r2.example.com/narration/gb.mp3",
            "duration_seconds": 3.0,
            "alignment": None,
        }
        with _patch_all(spec=MOCK_SPEC_GUEST_BOOK, render_result=render_result):
            result = render_narration("spec-001")
        assert result[0]["contains_verbatim_source"] is True
        assert len(result[0]["source_reference"]) > 0

    def test_spec_not_found_raises(self):
        """Missing spec raises ValueError."""
        with patch("agents.agent8.narration._load_spec", return_value=None):
            with pytest.raises(ValueError, match="not found"):
                render_narration("nonexistent")


class TestInputHash:
    def test_same_inputs_same_hash(self):
        h1 = _compute_input_hash("Hello", "voice1", "model1", "en")
        h2 = _compute_input_hash("Hello", "voice1", "model1", "en")
        assert h1 == h2

    def test_different_script_different_hash(self):
        h1 = _compute_input_hash("Hello", "voice1", "model1", "en")
        h2 = _compute_input_hash("Goodbye", "voice1", "model1", "en")
        assert h1 != h2


class TestDetectVerbatimSource:
    def test_guest_book_detected(self):
        contains, refs = _detect_verbatim_source(MOCK_SPEC_GUEST_BOOK)
        assert contains is True
        assert len(refs) > 0

    def test_original_not_verbatim(self):
        contains, refs = _detect_verbatim_source(MOCK_SPEC)
        assert contains is False
        assert refs == []
