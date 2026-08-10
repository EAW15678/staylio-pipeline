"""
Tests for Agent 8 Stage 6 motion rendering.
All mocked — $0.00 test cost.
"""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from agents.agent8.motion import (
    render_beats,
    _select_model,
    _compute_input_hash,
    _build_prompt_text,
    FRAME_EXITING_MOVES,
)
from core.runway import RUNWAY_COST_PER_SEC


# ── Fixtures ────────────────────────────────────────────────────────────

MOCK_SPEC_WITH_BEATS = {
    "spec_id": "spec-motion-001",
    "property_id": "prop-001",
    "concept_id": "concept-001",
    "beats": [
        {
            "source_image_url": "https://r2.example.com/photo_01.jpg",
            "motion": "push_in",
            "intent": "revealing the pool area",
            "params": {},
            "duration": 10,
        },
    ],
}

MOCK_SPEC_NO_BEATS = {
    "spec_id": "spec-motion-002",
    "property_id": "prop-001",
    "concept_id": "concept-002",
    "beats": [],
}

MOCK_SPEC_PULL_BACK = {
    "spec_id": "spec-motion-003",
    "property_id": "prop-001",
    "concept_id": "concept-003",
    "beats": [
        {
            "source_image_url": "https://r2.example.com/photo_02.jpg",
            "motion": "pull_back",
            "intent": "wide reveal",
            "params": {},
            "duration": 10,
        },
    ],
}

MOCK_SPEC_MULTI_BEAT = {
    "spec_id": "spec-motion-004",
    "property_id": "prop-001",
    "concept_id": "concept-004",
    "beats": [
        {
            "source_image_url": "https://r2.example.com/photo_01.jpg",
            "motion": "push_in",
            "intent": "pool area",
            "params": {},
            "duration": 10,
        },
        {
            "source_image_url": "https://r2.example.com/photo_02.jpg",
            "motion": "pan_left",
            "intent": "living room",
            "params": {},
            "duration": 10,
        },
    ],
}

MOCK_INVENTORY_REFLECTIVE = [
    {
        "image_url": "https://r2.example.com/photo_01.jpg",
        "source_image_url": "https://r2.example.com/photo_01.jpg",
        "motion_risk": ["reflections"],
    },
]

MOCK_INVENTORY_WATER = [
    {
        "image_url": "https://r2.example.com/photo_01.jpg",
        "source_image_url": "https://r2.example.com/photo_01.jpg",
        "motion_risk": ["water_surface"],
    },
]

MOCK_INVENTORY_NORMAL = [
    {
        "image_url": "https://r2.example.com/photo_01.jpg",
        "source_image_url": "https://r2.example.com/photo_01.jpg",
        "motion_risk": [],
    },
]


def _mock_supabase_for_spec(spec_data):
    """Build a mock Supabase client that returns the given spec."""
    mock_sb = MagicMock()

    # shot_spec query
    spec_result = MagicMock()
    spec_result.data = [spec_data] if spec_data else []
    mock_sb.table.return_value.select.return_value.eq.return_value.is_.return_value.limit.return_value.execute.return_value = spec_result

    # shot_inventory query — needs separate table mock
    inv_result = MagicMock()
    inv_result.data = []
    # motion_clips cache check — no cache
    clip_result = MagicMock()
    clip_result.data = []

    return mock_sb


def _patch_stack(spec_data, inventory_data=None, cached_clip=None, video_bytes=b"fake_video"):
    """Return a dict of patch targets for common test setup."""
    return {
        "_load_spec": patch(
            "agents.agent8.motion._load_spec",
            return_value=spec_data,
        ),
        "_load_shot_inventory": patch(
            "agents.agent8.motion._load_shot_inventory",
            return_value=inventory_data or [],
        ),
        "_check_cached_clip": patch(
            "agents.agent8.motion._check_cached_clip",
            return_value=cached_clip,
        ),
        "_persist_clip": patch(
            "agents.agent8.motion._persist_clip",
        ),
        "generate_clip": patch(
            "agents.agent8.motion.generate_clip",
            new_callable=AsyncMock,
            return_value=(video_bytes, "gen4_turbo"),
        ),
        "upload_video": patch(
            "core.r2_storage.upload_video",
            return_value="https://r2.example.com/rendered.mp4",
        ),
        "emit_media_cost": patch(
            "agents.agent8.motion.emit_media_cost",
        ),
    }


# ── Tests ────────────────────────────────────────────────────────────────

class TestModelRouting:
    def test_reflective_frame_routes_gen4_5(self):
        """motion_risk with 'reflections' routes to gen4.5."""
        inv_row = {"motion_risk": ["reflections"]}
        result = _select_model({}, inv_row)
        assert result == "gen4.5"

    def test_water_surface_routes_gen4_5(self):
        """motion_risk with 'water_surface' routes to gen4.5."""
        inv_row = {"motion_risk": ["water_surface"]}
        result = _select_model({}, inv_row)
        assert result == "gen4.5"

    def test_non_reflective_routes_gen4_turbo(self):
        """No reflective signals routes to gen4_turbo."""
        inv_row = {"motion_risk": []}
        result = _select_model({}, inv_row)
        assert result == "gen4_turbo"

    def test_hero_does_not_route_gen4_5(self):
        """hero flag alone does NOT get gen4.5."""
        beat = {"is_hero": True}
        inv_row = {"motion_risk": []}
        result = _select_model(beat, inv_row)
        assert result == "gen4_turbo"

    def test_none_inventory_routes_gen4_turbo(self):
        """No inventory row at all defaults to gen4_turbo."""
        result = _select_model({}, None)
        assert result == "gen4_turbo"


class TestFrameExitConstraint:
    def test_frame_exiting_move_refused(self):
        """pull_back is rejected BEFORE vendor call."""
        patches = _patch_stack(MOCK_SPEC_PULL_BACK)
        with patches["_load_spec"], patches["_load_shot_inventory"], \
             patches["_check_cached_clip"], patches["_persist_clip"], \
             patches["generate_clip"] as mock_gen, \
             patches["upload_video"], patches["emit_media_cost"]:
            clips = render_beats("spec-motion-003")

        assert len(clips) == 1
        assert clips[0]["status"] == "rejected"
        assert "pull_back" in clips[0]["failure_reason"]
        assert "Guidelines v1.2" in clips[0]["failure_reason"]
        assert clips[0]["persistence_check"] == "not_applicable"
        # Vendor was NEVER called
        mock_gen.assert_not_awaited()


class TestCaching:
    def test_cache_hit_no_vendor_call(self):
        """Cached clip is returned without calling vendor."""
        cached = {
            "property_id": "prop-001",
            "spec_id": "spec-motion-001",
            "status": "ready",
            "r2_url": "https://r2.example.com/cached.mp4",
            "input_hash": "abc123",
        }
        patches = _patch_stack(MOCK_SPEC_WITH_BEATS, cached_clip=cached)
        with patches["_load_spec"], patches["_load_shot_inventory"], \
             patches["_check_cached_clip"], patches["_persist_clip"], \
             patches["generate_clip"] as mock_gen, \
             patches["upload_video"], patches["emit_media_cost"]:
            clips = render_beats("spec-motion-001")

        assert len(clips) == 1
        assert clips[0]["status"] == "ready"
        assert clips[0]["r2_url"] == "https://r2.example.com/cached.mp4"
        mock_gen.assert_not_awaited()

    def test_force_bypasses_cache(self):
        """force=True ignores cached clip and re-renders."""
        cached = {
            "property_id": "prop-001",
            "spec_id": "spec-motion-001",
            "status": "ready",
            "r2_url": "https://r2.example.com/cached.mp4",
            "input_hash": "abc123",
        }
        patches = _patch_stack(
            MOCK_SPEC_WITH_BEATS,
            cached_clip=cached,
            video_bytes=b"new_video",
        )
        # Override generate_clip to return gen4_turbo
        with patches["_load_spec"], patches["_load_shot_inventory"], \
             patches["_check_cached_clip"] as mock_cache, \
             patches["_persist_clip"], \
             patches["generate_clip"] as mock_gen, \
             patches["upload_video"], patches["emit_media_cost"]:
            mock_gen.return_value = (b"new_video", "gen4_turbo")
            clips = render_beats("spec-motion-001", force=True)

        # Cache check should NOT have been called
        mock_cache.assert_not_called()
        # Vendor WAS called
        mock_gen.assert_awaited_once()
        assert len(clips) == 1
        assert clips[0]["status"] == "ready"


class TestCostEstimate:
    def test_cost_estimate_matches_model_gen4_turbo(self):
        """gen4_turbo: 10s * $0.05 = $0.50."""
        patches = _patch_stack(MOCK_SPEC_WITH_BEATS)
        with patches["_load_spec"], patches["_load_shot_inventory"], \
             patches["_check_cached_clip"], patches["_persist_clip"], \
             patches["generate_clip"] as mock_gen, \
             patches["upload_video"], patches["emit_media_cost"]:
            mock_gen.return_value = (b"video", "gen4_turbo")
            clips = render_beats("spec-motion-001")

        assert len(clips) == 1
        assert clips[0]["cost_estimate_usd"] == 0.50

    def test_cost_estimate_matches_model_gen4_5(self):
        """gen4.5: 10s * $0.12 = $1.20."""
        spec = {
            **MOCK_SPEC_WITH_BEATS,
            "spec_id": "spec-motion-005",
        }
        patches = _patch_stack(spec, inventory_data=MOCK_INVENTORY_REFLECTIVE)
        with patches["_load_spec"], patches["_load_shot_inventory"], \
             patches["_check_cached_clip"], patches["_persist_clip"], \
             patches["generate_clip"] as mock_gen, \
             patches["upload_video"], patches["emit_media_cost"]:
            mock_gen.return_value = (b"video", "gen4.5")
            clips = render_beats("spec-motion-005")

        assert len(clips) == 1
        assert clips[0]["cost_estimate_usd"] == 1.20


class TestBatchResilience:
    def test_one_failed_beat_continues(self):
        """One failed beat does not abort the batch."""
        patches = _patch_stack(MOCK_SPEC_MULTI_BEAT)
        call_count = 0

        async def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Runway transient failure")
            return (b"video", "gen4_turbo")

        with patches["_load_spec"], patches["_load_shot_inventory"], \
             patches["_check_cached_clip"], patches["_persist_clip"], \
             patches["generate_clip"] as mock_gen, \
             patches["upload_video"], patches["emit_media_cost"]:
            mock_gen.side_effect = _side_effect
            clips = render_beats("spec-motion-004")

        assert len(clips) == 2
        assert clips[0]["status"] == "failed"
        assert clips[1]["status"] == "ready"


class TestPersistenceCheck:
    def test_persistence_check_never_not_applicable(self):
        """Generative technique always gets 'unchecked', never 'not_applicable'."""
        patches = _patch_stack(MOCK_SPEC_WITH_BEATS)
        with patches["_load_spec"], patches["_load_shot_inventory"], \
             patches["_check_cached_clip"], patches["_persist_clip"], \
             patches["generate_clip"], patches["upload_video"], \
             patches["emit_media_cost"]:
            clips = render_beats("spec-motion-001")

        assert len(clips) == 1
        assert clips[0]["persistence_check"] == "unchecked"
        assert clips[0]["technique"] == "generative"


class TestDryRun:
    def test_dry_run_pending(self):
        """dry_run returns pending status without vendor call."""
        patches = _patch_stack(MOCK_SPEC_WITH_BEATS)
        with patches["_load_spec"], patches["_load_shot_inventory"], \
             patches["_check_cached_clip"], patches["_persist_clip"], \
             patches["generate_clip"] as mock_gen, \
             patches["upload_video"], patches["emit_media_cost"]:
            clips = render_beats("spec-motion-001", dry_run=True)

        assert len(clips) == 1
        assert clips[0]["status"] == "pending"
        assert clips[0]["r2_url"] is None
        mock_gen.assert_not_awaited()
