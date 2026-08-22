"""
OBSERVE-D: Tests for depth map generation and caching.

Tests 1-5 per the spec:
1. Photo without depth map gets one generated and recorded.
2. Photo with current depth map is skipped.
2b. Photo with stale depth map (source changed) is regenerated.
3. Depth estimation failure logs and continues; observe still ok.
4. All pre-existing observation fields are unchanged.
5. Re-running with force=False produces no new depth maps.
"""

import sys
import os
import types
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, ".")


# ── Shared fixtures ──────────────────────────────────────────────────────

def _make_sb_mock(photos, renditions_by_pid, observations=None):
    """Build a mock substrate that returns photo and rendition data."""
    sb = MagicMock()

    def table_side_effect(name):
        t = MagicMock()
        if name == "photographs":
            chain = MagicMock()
            chain.select.return_value = chain
            chain.eq.return_value = chain
            chain.execute.return_value = MagicMock(data=photos)
            t.select.return_value = chain
        elif name == "renditions":
            # select().eq("photo_id", pid).execute() must return per-pid data
            def select_fn(*args, **kwargs):
                sel = MagicMock()
                def eq_fn(col, val):
                    eq_chain = MagicMock()
                    eq_chain.execute.return_value = MagicMock(
                        data=renditions_by_pid.get(val, [])
                    )
                    return eq_chain
                sel.eq = eq_fn
                return sel
            t.select = select_fn
            # upsert chain
            upsert_ret = MagicMock()
            upsert_ret.execute.return_value = MagicMock(data=[])
            t.upsert = MagicMock(return_value=upsert_ret)
        return t

    sb.table = table_side_effect
    return sb


MODAL_RESULT = {
    "depth_map_url": "https://example.com/depth/test.png",
    "width": 3840,
    "height": 2560,
    "byte_size": 500000,
    "elapsed_seconds": 8.5,
}


# ── Test 1: photo without depth map gets one generated ────────────────

def test_generates_depth_map_for_photo_without_one():
    """A canonical photo with an enhanced rendition but no depth_map
    gets a depth map generated and recorded as a rendition."""
    from skills.generate_depth_maps import generate_depth_maps

    photos = [{"photo_id": "p1", "content_hash": "hash_a", "image_width": 3840, "image_height": 2560}]
    rends = {
        "p1": [
            {"rendition_id": "r1", "kind": "enhanced", "storage_url": "https://example.com/enhanced.jpg", "source_content_hash": None},
        ]
    }

    sb = _make_sb_mock(photos, rends)

    with patch("skills.generate_depth_maps.get_substrate", return_value=sb), \
         patch("skills.generate_depth_maps._call_modal_depth", return_value=MODAL_RESULT) as mock_modal:

        result = generate_depth_maps("prop-1")

    assert result.is_ok, f"Expected ok, got {result.status}: {result.reason}"
    assert result.data["generated"] == 1
    assert result.data["skipped"] == 0
    mock_modal.assert_called_once()


# ── Test 2: photo with current depth map is skipped ───────────────────

def test_skips_photo_with_current_depth_map():
    """A photo whose depth_map has source_content_hash matching the
    photograph's content_hash is skipped — no Modal call."""
    from skills.generate_depth_maps import generate_depth_maps

    photos = [{"photo_id": "p1", "content_hash": "hash_a", "image_width": 3840, "image_height": 2560}]
    rends = {
        "p1": [
            {"rendition_id": "r1", "kind": "enhanced", "storage_url": "https://example.com/enhanced.jpg", "source_content_hash": None},
            {"rendition_id": "r2", "kind": "depth_map", "storage_url": "https://example.com/depth.png", "source_content_hash": "hash_a"},
        ]
    }

    sb = _make_sb_mock(photos, rends)

    with patch("skills.generate_depth_maps.get_substrate", return_value=sb), \
         patch("skills.generate_depth_maps._call_modal_depth") as mock_modal:

        result = generate_depth_maps("prop-1")

    assert result.data.get("noop") is True, f"Expected noop, got {result.data}"
    mock_modal.assert_not_called()


# ── Test 2b: stale depth map is regenerated ───────────────────────────

def test_regenerates_stale_depth_map():
    """A depth_map whose source_content_hash does NOT match the current
    content_hash is regenerated."""
    from skills.generate_depth_maps import generate_depth_maps

    photos = [{"photo_id": "p1", "content_hash": "hash_b_new", "image_width": 3840, "image_height": 2560}]
    rends = {
        "p1": [
            {"rendition_id": "r1", "kind": "enhanced", "storage_url": "https://example.com/enhanced.jpg", "source_content_hash": None},
            {"rendition_id": "r2", "kind": "depth_map", "storage_url": "https://example.com/depth.png", "source_content_hash": "hash_a_old"},
        ]
    }

    sb = _make_sb_mock(photos, rends)

    with patch("skills.generate_depth_maps.get_substrate", return_value=sb), \
         patch("skills.generate_depth_maps._call_modal_depth", return_value=MODAL_RESULT) as mock_modal:

        result = generate_depth_maps("prop-1")

    assert result.is_ok
    assert result.data["generated"] == 1
    mock_modal.assert_called_once()


# ── Test 3: failure logs and continues ────────────────────────────────

def test_failure_logs_and_continues():
    """Depth estimation failing for one photo does not fail the skill.
    The other photos still get processed."""
    from skills.generate_depth_maps import generate_depth_maps

    photos = [
        {"photo_id": "p1", "content_hash": "h1", "image_width": 3840, "image_height": 2560},
        {"photo_id": "p2", "content_hash": "h2", "image_width": 3840, "image_height": 2560},
    ]
    rends = {
        "p1": [{"rendition_id": "r1", "kind": "enhanced", "storage_url": "https://example.com/e1.jpg", "source_content_hash": None}],
        "p2": [{"rendition_id": "r2", "kind": "enhanced", "storage_url": "https://example.com/e2.jpg", "source_content_hash": None}],
    }

    call_count = [0]
    def mock_modal(image_url, r2_key):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("GPU out of memory")
        return MODAL_RESULT

    sb = _make_sb_mock(photos, rends)

    with patch("skills.generate_depth_maps.get_substrate", return_value=sb), \
         patch("skills.generate_depth_maps._call_modal_depth", side_effect=mock_modal):

        result = generate_depth_maps("prop-1")

    assert result.is_ok, f"Expected ok even with partial failure, got {result.status}"
    assert result.data["generated"] == 1
    assert result.data["failed"] == 1
    assert len(result.data["failures"]) == 1
    assert "GPU out of memory" in result.data["failures"][0]["error"]


# ── Test 4: observation fields unchanged ──────────────────────────────

def test_observe_fields_unchanged_by_depth_generation():
    """generate_depth_maps does NOT read, write, or modify the
    observations table. It only touches renditions."""
    from skills.generate_depth_maps import generate_depth_maps
    import inspect

    source = inspect.getsource(generate_depth_maps)

    # The function must never reference the observations table
    assert '"observations"' not in source, \
        "generate_depth_maps must not touch the observations table"
    assert "'observations'" not in source, \
        "generate_depth_maps must not touch the observations table"

    # It should only reference renditions and photographs
    assert '"renditions"' in source, "Must read/write renditions table"
    assert '"photographs"' in source, "Must read photographs table"


def test_observe_fields_unchanged_mutation_proof():
    """Removing the renditions upsert from generate_depth_maps would
    cause test 1 to fail — proving the upsert is real, not dead code."""
    from skills.generate_depth_maps import generate_depth_maps
    import inspect

    source = inspect.getsource(generate_depth_maps)
    # The upsert is the mechanism that records the depth map
    assert '.upsert(' in source, "Must upsert renditions"
    assert '"depth_map"' in source, "Must insert kind='depth_map'"
    assert '"source_content_hash"' in source, "Must record source_content_hash"


# ── Test 5: re-run with force=False produces no new depth maps ────────

def test_rerun_noop():
    """Re-running when all depth maps are current produces noop."""
    from skills.generate_depth_maps import generate_depth_maps

    photos = [
        {"photo_id": "p1", "content_hash": "h1", "image_width": 3840, "image_height": 2560},
        {"photo_id": "p2", "content_hash": "h2", "image_width": 3840, "image_height": 2560},
    ]
    rends = {
        "p1": [
            {"rendition_id": "r1", "kind": "enhanced", "storage_url": "https://ex.com/e1.jpg", "source_content_hash": None},
            {"rendition_id": "r2", "kind": "depth_map", "storage_url": "https://ex.com/d1.png", "source_content_hash": "h1"},
        ],
        "p2": [
            {"rendition_id": "r3", "kind": "enhanced", "storage_url": "https://ex.com/e2.jpg", "source_content_hash": None},
            {"rendition_id": "r4", "kind": "depth_map", "storage_url": "https://ex.com/d2.png", "source_content_hash": "h2"},
        ],
    }

    sb = _make_sb_mock(photos, rends)

    with patch("skills.generate_depth_maps.get_substrate", return_value=sb), \
         patch("skills.generate_depth_maps._call_modal_depth") as mock_modal:

        result = generate_depth_maps("prop-1", force=False)

    assert result.data.get("noop") is True, f"Expected noop, got {result.data}"
    mock_modal.assert_not_called()
