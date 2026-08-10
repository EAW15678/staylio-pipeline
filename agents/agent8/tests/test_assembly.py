"""
Tests for Agent 8 Stage 7 assembly.
All mocked — $0.00 test cost.
"""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from agents.agent8.assembly import (
    assemble_master,
    _check_overlay_safe_zone,
    _build_render_payload,
    _apply_audio_ducking,
    _compute_input_hash,
    GRID_REGIONS_IN_MASTER,
    SAFE_LEFT,
    SAFE_TOP,
    SAFE_RIGHT,
    SAFE_BOTTOM,
    CREATOMATE_COST_PER_RENDER_MINUTE,
)


# ── Fixtures ────────────────────────────────────────────────────────────

MOCK_SPEC = {
    "spec_id": "spec-asm-001",
    "property_id": "prop-001",
    "concept_id": "concept-001",
    "creatomate_template_id": "tmpl-abc123",
    "overlay_register": [
        {"grid_region": "top_center", "text": "Welcome"},
    ],
}

MOCK_CLIPS = [
    {
        "clip_id": "clip-b",
        "spec_id": "spec-asm-001",
        "beat_index": 1,
        "r2_url": "https://r2.example.com/clip_b.mp4",
        "duration": 10,
        "status": "ready",
    },
    {
        "clip_id": "clip-a",
        "spec_id": "spec-asm-001",
        "beat_index": 0,
        "r2_url": "https://r2.example.com/clip_a.mp4",
        "duration": 8,
        "status": "ready",
    },
]

MOCK_NARRATION = {
    "narration_id": "narr-001",
    "spec_id": "spec-asm-001",
    "r2_url": "https://r2.example.com/narration.mp3",
    "duration_seconds": 15.0,
    "alignment": {"words": [{"word": "hello", "start": 0.0, "end": 0.5}]},
    "status": "ready",
}

MOCK_NARRATION_NO_ALIGNMENT = {
    "narration_id": "narr-002",
    "spec_id": "spec-asm-001",
    "r2_url": "https://r2.example.com/narration_flat.mp3",
    "duration_seconds": 12.0,
    "alignment": None,
    "status": "ready",
}

MOCK_MUSIC = {
    "music_id": "music-001",
    "spec_id": "spec-asm-001",
    "r2_url": "https://r2.example.com/music.mp3",
    "duration_seconds": 30.0,
    "status": "ready",
}


def _mock_load_spec(spec_id):
    if spec_id == MOCK_SPEC["spec_id"]:
        return dict(MOCK_SPEC)
    return None


def _mock_load_clips(spec_id):
    if spec_id == MOCK_SPEC["spec_id"]:
        return [dict(c) for c in MOCK_CLIPS]
    return []


def _mock_load_narration(spec_id):
    if spec_id == MOCK_SPEC["spec_id"]:
        return dict(MOCK_NARRATION)
    return None


def _mock_load_narration_no_alignment(spec_id):
    if spec_id == MOCK_SPEC["spec_id"]:
        return dict(MOCK_NARRATION_NO_ALIGNMENT)
    return None


def _mock_load_music(spec_id):
    if spec_id == MOCK_SPEC["spec_id"]:
        return dict(MOCK_MUSIC)
    return None


def _mock_load_no_music(spec_id):
    return None


def _mock_check_cached_assembly(prop_id, input_hash):
    return None


def _mock_persist_assembly(assembly, prop_id, input_hash):
    pass


# ── Tests ───────────────────────────────────────────────────────────────

class TestClipsInBeatOrder:
    """test_clips_in_beat_order — assembly uses spec.beats ordering."""

    @patch("agents.agent8.assembly._persist_assembly", _mock_persist_assembly)
    @patch("agents.agent8.assembly._check_cached_assembly", _mock_check_cached_assembly)
    @patch("agents.agent8.assembly._load_music", _mock_load_music)
    @patch("agents.agent8.assembly._load_narration", _mock_load_narration)
    @patch("agents.agent8.assembly._load_motion_clips", _mock_load_clips)
    @patch("agents.agent8.assembly._load_spec", _mock_load_spec)
    def test_clips_in_beat_order(self):
        result = assemble_master("spec-asm-001", dry_run=True)
        assert result is not None
        # clip-a (beat_index=0) should come before clip-b (beat_index=1)
        assert result["clip_ids"] == ["clip-a", "clip-b"]


class TestOverlaySafeZone:
    """test_overlay_outside_safe_zone_repositioned — moved to compliant region."""

    def test_overlay_outside_safe_zone_repositioned(self):
        # Use a region name that is NOT in GRID_REGIONS_IN_MASTER
        overlay = {"grid_region": "outside_canvas", "text": "Test"}
        checked, was_moved, orig = _check_overlay_safe_zone(overlay)
        assert was_moved is True
        assert orig == "outside_canvas"
        assert checked["grid_region"] in GRID_REGIONS_IN_MASTER

    def test_valid_region_not_moved(self):
        overlay = {"grid_region": "top_center", "text": "Test"}
        checked, was_moved, orig = _check_overlay_safe_zone(overlay)
        assert was_moved is False
        assert orig is None


class TestOverlayNoCompliantRegion:
    """test_overlay_no_compliant_region_held — held, not rendered."""

    @patch.dict("agents.agent8.assembly.GRID_REGIONS_IN_MASTER", {}, clear=True)
    def test_overlay_no_compliant_region_held(self):
        overlay = {"grid_region": "nonexistent", "text": "Test"}
        checked, was_moved, orig = _check_overlay_safe_zone(overlay)
        assert checked.get("_held") is True
        assert "nonexistent" in checked.get("_held_reason", "")


class TestNoLoopFlagsInPayload:
    """test_no_loop_flags_in_payload — render payload must not contain loop."""

    def test_no_loop_flags_in_payload(self):
        clips = [{"r2_url": "https://r2.example.com/c.mp4", "duration": 10}]
        mods = _build_render_payload(clips, None, None, [], "tmpl-x")
        payload_str = str(mods).lower()
        assert "loop" not in payload_str


class TestR2UrlNotVendorUrl:
    """test_r2_url_not_vendor_url — final URL must be R2, not vendor."""

    @patch("agents.agent8.assembly._persist_assembly", _mock_persist_assembly)
    @patch("agents.agent8.assembly._check_cached_assembly", _mock_check_cached_assembly)
    @patch("agents.agent8.assembly._load_music", _mock_load_no_music)
    @patch("agents.agent8.assembly._load_narration", lambda sid: None)
    @patch("agents.agent8.assembly._load_motion_clips", _mock_load_clips)
    @patch("agents.agent8.assembly._load_spec", _mock_load_spec)
    @patch("agents.agent8.assembly._apply_audio_ducking", return_value=(b"video", False))
    @patch("core.r2_storage.upload_video", return_value="https://r2.staylio.ai/master.mp4")
    @patch("agents.agent8.assembly.render_video", new_callable=AsyncMock)
    @patch("agents.agent8.assembly.emit_media_cost")
    def test_r2_url_not_vendor_url(self, mock_cost, mock_render, mock_upload, mock_duck):
        mock_render.return_value = (b"rendered_video", "render-id-123")
        result = assemble_master("spec-asm-001")
        assert result is not None
        assert result["r2_url"].startswith("https://r2.")
        assert "creatomate.com" not in (result["r2_url"] or "")


class TestHasAudioDuckingReflectsReality:
    """test_has_audio_ducking_reflects_reality — true when applied, false when not."""

    @patch("agents.agent8.assembly._persist_assembly", _mock_persist_assembly)
    @patch("agents.agent8.assembly._check_cached_assembly", _mock_check_cached_assembly)
    @patch("agents.agent8.assembly._load_music", _mock_load_music)
    @patch("agents.agent8.assembly._load_narration", _mock_load_narration)
    @patch("agents.agent8.assembly._load_motion_clips", _mock_load_clips)
    @patch("agents.agent8.assembly._load_spec", _mock_load_spec)
    @patch("core.r2_storage.upload_video", return_value="https://r2.staylio.ai/master.mp4")
    @patch("agents.agent8.assembly.render_video", new_callable=AsyncMock)
    @patch("agents.agent8.assembly.emit_media_cost")
    def test_ducking_true_when_applied(self, mock_cost, mock_render, mock_upload):
        mock_render.return_value = (b"rendered_video", "render-id-123")
        with patch(
            "agents.agent8.assembly._apply_audio_ducking",
            return_value=(b"ducked_video", True),
        ):
            result = assemble_master("spec-asm-001")
            assert result is not None
            assert result["has_audio_ducking"] is True
            assert len(result["post_process"]) == 1
            assert result["post_process"][0]["step"] == "audio_ducking"

    @patch("agents.agent8.assembly._persist_assembly", _mock_persist_assembly)
    @patch("agents.agent8.assembly._check_cached_assembly", _mock_check_cached_assembly)
    @patch("agents.agent8.assembly._load_music", _mock_load_no_music)
    @patch("agents.agent8.assembly._load_narration", lambda sid: None)
    @patch("agents.agent8.assembly._load_motion_clips", _mock_load_clips)
    @patch("agents.agent8.assembly._load_spec", _mock_load_spec)
    @patch("core.r2_storage.upload_video", return_value="https://r2.staylio.ai/master.mp4")
    @patch("agents.agent8.assembly.render_video", new_callable=AsyncMock)
    @patch("agents.agent8.assembly.emit_media_cost")
    def test_ducking_false_when_no_narration(self, mock_cost, mock_render, mock_upload):
        mock_render.return_value = (b"rendered_video", "render-id-123")
        with patch(
            "agents.agent8.assembly._apply_audio_ducking",
            return_value=(b"rendered_video", False),
        ):
            result = assemble_master("spec-asm-001")
            assert result is not None
            assert result["has_audio_ducking"] is False
            assert result["post_process"] == []


class TestNullAlignmentFallback:
    """test_null_alignment_fallback — flat narration, no crash."""

    @patch("agents.agent8.assembly._persist_assembly", _mock_persist_assembly)
    @patch("agents.agent8.assembly._check_cached_assembly", _mock_check_cached_assembly)
    @patch("agents.agent8.assembly._load_music", _mock_load_no_music)
    @patch("agents.agent8.assembly._load_narration", _mock_load_narration_no_alignment)
    @patch("agents.agent8.assembly._load_motion_clips", _mock_load_clips)
    @patch("agents.agent8.assembly._load_spec", _mock_load_spec)
    def test_null_alignment_fallback(self):
        result = assemble_master("spec-asm-001", dry_run=True)
        assert result is not None
        # Should not crash even though alignment is None
        assert result["status"] == "pending"
        assert result["narration_id"] is not None


class TestDryRunPending:
    """test_dry_run_pending — dry_run returns status=pending without rendering."""

    @patch("agents.agent8.assembly._persist_assembly", _mock_persist_assembly)
    @patch("agents.agent8.assembly._check_cached_assembly", _mock_check_cached_assembly)
    @patch("agents.agent8.assembly._load_music", _mock_load_music)
    @patch("agents.agent8.assembly._load_narration", _mock_load_narration)
    @patch("agents.agent8.assembly._load_motion_clips", _mock_load_clips)
    @patch("agents.agent8.assembly._load_spec", _mock_load_spec)
    def test_dry_run_pending(self):
        result = assemble_master("spec-asm-001", dry_run=True)
        assert result is not None
        assert result["status"] == "pending"
        assert result["r2_url"] is None
        assert result["renderer"] == "creatomate"


class TestForceBypassesCache:
    """test_force_bypasses_cache — force=True skips cache check."""

    @patch("agents.agent8.assembly._persist_assembly", _mock_persist_assembly)
    @patch("agents.agent8.assembly._load_music", _mock_load_music)
    @patch("agents.agent8.assembly._load_narration", _mock_load_narration)
    @patch("agents.agent8.assembly._load_motion_clips", _mock_load_clips)
    @patch("agents.agent8.assembly._load_spec", _mock_load_spec)
    @patch("agents.agent8.assembly._apply_audio_ducking", return_value=(b"video", False))
    @patch("core.r2_storage.upload_video", return_value="https://r2.staylio.ai/master.mp4")
    @patch("agents.agent8.assembly.render_video", new_callable=AsyncMock)
    @patch("agents.agent8.assembly.emit_media_cost")
    def test_force_bypasses_cache(self, mock_cost, mock_render, mock_upload, mock_duck):
        mock_render.return_value = (b"rendered_video", "render-id-123")

        # Even with a cached result existing, force=True should re-render
        with patch(
            "agents.agent8.assembly._check_cached_assembly",
            return_value={"status": "ready", "r2_url": "https://r2.staylio.ai/old.mp4"},
        ):
            result = assemble_master("spec-asm-001", force=True)
            assert result is not None
            # Should have rendered fresh, not returned cached
            mock_render.assert_called_once()
            assert result["status"] == "ready"


class TestCostEstimate:
    """test_cost_estimate — $0.84/min calculation."""

    @patch("agents.agent8.assembly._persist_assembly", _mock_persist_assembly)
    @patch("agents.agent8.assembly._check_cached_assembly", _mock_check_cached_assembly)
    @patch("agents.agent8.assembly._load_music", _mock_load_no_music)
    @patch("agents.agent8.assembly._load_narration", lambda sid: None)
    @patch("agents.agent8.assembly._load_motion_clips", _mock_load_clips)
    @patch("agents.agent8.assembly._load_spec", _mock_load_spec)
    def test_cost_estimate(self):
        result = assemble_master("spec-asm-001", dry_run=True)
        assert result is not None
        # Clips: 8s + 10s = 18s = 0.3 min
        # Cost: 0.3 * 0.84 = 0.252
        total_seconds = sum(c["duration"] for c in MOCK_CLIPS)
        expected_cost = round(
            (total_seconds / 60.0) * CREATOMATE_COST_PER_RENDER_MINUTE, 4
        )
        assert result["cost_estimate_usd"] == expected_cost
        assert result["cost_estimate_usd"] == round(18 / 60.0 * 0.84, 4)
