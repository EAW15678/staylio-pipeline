"""
Tests for Agent 8 Stage 5 music rendering.
All mocked — $0.00 test cost.
"""

from unittest.mock import patch, MagicMock

import pytest

from agents.agent8.music import (
    render_music,
    _compute_input_hash,
    _run_prohibited_input_check,
    _build_prompt_text,
    _build_composition_plan,
)


MOCK_SPEC = {
    "spec_id": "spec-music-001",
    "property_id": "prop-001",
    "concept_id": "concept-001",
    "music_brief": {
        "mood": "warm and inviting",
        "genre": "ambient",
        "tempo": "slow",
        "description": "Soft ambient pads with gentle piano",
    },
}

MOCK_SPEC_STRUCTURED = {
    "spec_id": "spec-music-002",
    "property_id": "prop-001",
    "concept_id": "concept-002",
    "music_brief": {
        "duration_ms": 30000,
        "sections": [
            {
                "description": "Gentle intro with soft piano",
                "duration_ms": 10000,
                "positive_styles": ["ambient", "peaceful"],
            },
            {
                "description": "Building warmth with strings",
                "duration_ms": 10000,
            },
            {
                "description": "Gentle resolution",
                "duration_ms": 10000,
                "negative_styles": ["harsh", "loud"],
            },
        ],
    },
}

MOCK_SPEC_NO_MUSIC = {
    "spec_id": "spec-music-003",
    "property_id": "prop-001",
    "concept_id": "concept-003",
    "music_brief": None,
}

MOCK_SPEC_PROHIBITED = {
    "spec_id": "spec-music-004",
    "property_id": "prop-001",
    "concept_id": "concept-004",
    "music_brief": {
        "description": "warm, like Bon Iver",
        "mood": "intimate",
    },
}

MOCK_CACHED_MUSIC = {
    "music_id": "music-cached",
    "status": "ready",
    "r2_url": "https://r2.example.com/music/cached.mp3",
    "input_hash": "abc123",
}

MOCK_AUDIO_BYTES = b"\xff\xfb\x90\x00" * 1000  # fake MP3 data


def _patch_all(spec=MOCK_SPEC, cached=None, audio_result=(MOCK_AUDIO_BYTES, "music_v2")):
    """Return a context manager that patches all external calls."""
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        with patch("agents.agent8.music._load_spec", return_value=spec), \
             patch("agents.agent8.music._check_cached_music", return_value=cached), \
             patch("core.elevenlabs.generate_music", return_value=audio_result) as mock_gen, \
             patch("core.r2_storage.upload_video", return_value="https://r2.example.com/music/new.mp3"), \
             patch("agents.agent8.music._persist_music"):
            yield mock_gen

    return _ctx()


class TestRenderMusic:
    def test_no_music_brief_returns_none(self):
        """No music_brief -> clean return (None)."""
        with _patch_all(spec=MOCK_SPEC_NO_MUSIC) as mock_gen:
            result = render_music("spec-music-003")
        assert result is None
        mock_gen.assert_not_called()

    def test_cache_hit_makes_no_vendor_call(self):
        """Cached music returned without rendering."""
        with _patch_all(cached=MOCK_CACHED_MUSIC) as mock_gen:
            result = render_music("spec-music-001")
        assert result["status"] == "ready"
        assert result["r2_url"] == "https://r2.example.com/music/cached.mp3"
        mock_gen.assert_not_called()

    def test_prohibited_input_fails_before_vendor_call(self):
        """Brief containing 'like Bon Iver' is rejected before any vendor call."""
        with _patch_all(spec=MOCK_SPEC_PROHIBITED) as mock_gen:
            result = render_music("spec-music-004")
        assert result is not None
        assert result["status"] == "failed"
        assert result["prohibited_input_check"] == "failed"
        assert "Bon Iver" in result["failure_reason"] or "bon iver" in result["failure_reason"].lower()
        mock_gen.assert_not_called()

    def test_prohibited_input_passes_clean_brief(self):
        """Clean brief passes prohibited check and proceeds to render."""
        with _patch_all() as mock_gen:
            result = render_music("spec-music-001")
        assert result is not None
        assert result["status"] == "ready"
        assert result["prohibited_input_check"] == "passed"
        mock_gen.assert_called_once()

    def test_composition_plan_path(self):
        """Structured sections produce a composition_plan call."""
        with _patch_all(spec=MOCK_SPEC_STRUCTURED) as mock_gen:
            result = render_music("spec-music-002")
        assert result is not None
        assert result["status"] == "ready"
        # Verify generate_music was called with composition_plan, not prompt_text
        call_kwargs = mock_gen.call_args
        assert call_kwargs[1]["composition_plan"] is not None or call_kwargs.kwargs.get("composition_plan") is not None
        assert call_kwargs[1].get("prompt_text") is None or call_kwargs.kwargs.get("prompt_text") is None

    def test_prompt_text_path(self):
        """Unstructured brief produces a prompt_text call."""
        with _patch_all() as mock_gen:
            result = render_music("spec-music-001")
        assert result is not None
        assert result["status"] == "ready"
        call_kwargs = mock_gen.call_args
        assert call_kwargs[1]["prompt_text"] is not None or call_kwargs.kwargs.get("prompt_text") is not None

    def test_failure_lands_in_status(self):
        """When generate_music returns None, asset has status='failed'."""
        with _patch_all(audio_result=(None, "music_v2")) as mock_gen:
            result = render_music("spec-music-001")
        assert result is not None
        assert result["status"] == "failed"
        assert "failed after retries" in result["failure_reason"]

    def test_dry_run_returns_pending(self):
        """dry_run=True returns asset with status='pending', no vendor call."""
        with _patch_all() as mock_gen:
            result = render_music("spec-music-001", dry_run=True)
        assert result is not None
        assert result["status"] == "pending"
        assert result["r2_url"] is None
        mock_gen.assert_not_called()

    def test_force_bypasses_cache(self):
        """force=True ignores cache and re-renders."""
        with _patch_all(cached=MOCK_CACHED_MUSIC) as mock_gen:
            result = render_music("spec-music-001", force=True)
        assert result is not None
        assert result["status"] == "ready"
        assert result["r2_url"] == "https://r2.example.com/music/new.mp3"
        mock_gen.assert_called_once()


class TestProhibitedInputCheck:
    def test_clean_brief_passes(self):
        status, reason = _run_prohibited_input_check({
            "mood": "warm and inviting",
            "genre": "ambient",
        })
        assert status == "passed"
        assert reason is None

    def test_artist_name_fails(self):
        status, reason = _run_prohibited_input_check({
            "description": "Something warm, like Bon Iver",
        })
        assert status == "failed"
        assert reason is not None

    def test_label_name_fails(self):
        status, reason = _run_prohibited_input_check({
            "description": "sony music style production",
        })
        assert status == "failed"
        assert reason is not None


class TestInputHash:
    def test_same_input_same_hash(self):
        h1 = _compute_input_hash("soft piano", None, 30000, "music_v2")
        h2 = _compute_input_hash("soft piano", None, 30000, "music_v2")
        assert h1 == h2

    def test_different_input_different_hash(self):
        h1 = _compute_input_hash("soft piano", None, 30000, "music_v2")
        h2 = _compute_input_hash("loud drums", None, 30000, "music_v2")
        assert h1 != h2
