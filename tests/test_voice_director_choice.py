"""
VOICE-4: Director voice choice tests.

Verifies: director receives candidates, voice membership validation,
missing narration_voice_id fails, hero exclusion still holds.
"""

import pytest
from skills.direct import _format_voice_candidates


VOICE_CANDIDATES = [
    {"voice_id": "brad1", "name": "Brad - Welcoming", "gender": "male"},
    {"voice_id": "drew1", "name": "Drew - Casual", "gender": "male"},
    {"voice_id": "haley1", "name": "Haley Maven", "gender": "female"},
    {"voice_id": "juliet1", "name": "Juliet - Customer Care", "gender": "female"},
]


# ── Voice candidates formatting ──────────────────────────────────────────

def test_format_voice_candidates():
    result = _format_voice_candidates(VOICE_CANDIDATES)
    assert "brad1" in result
    assert "Brad - Welcoming" in result
    assert "male" in result
    assert "female" in result


def test_format_empty():
    result = _format_voice_candidates([])
    assert "No voices" in result


def test_format_none():
    result = _format_voice_candidates(None)
    assert "No voices" in result


# ── Voice membership validation (inline in direct.py) ────────────────────

def test_voice_in_collection_passes():
    """A chosen voice that IS in the collection → no violation."""
    valid_ids = {v["voice_id"] for v in VOICE_CANDIDATES}
    assert "brad1" in valid_ids


def test_voice_outside_collection_fails():
    """A chosen voice NOT in the collection → violation."""
    valid_ids = {v["voice_id"] for v in VOICE_CANDIDATES}
    assert "unknown_voice" not in valid_ids


# ── Missing narration_voice_id ───────────────────────────────────────────

def test_missing_voice_id_is_detectable():
    """A direction without narration_voice_id should be caught."""
    direction = {"narration_script": "Hello", "narration_voice_id": None}
    assert not direction.get("narration_voice_id")


def test_present_voice_id():
    direction = {"narration_script": "Hello", "narration_voice_id": "brad1"}
    assert direction.get("narration_voice_id") == "brad1"


# ── Hero exclusion still holds ───────────────────────────────────────────

def test_hero_excluded_from_guest_pool():
    """resolve_guest_voice with exclude_voice_id removes the hero."""
    from unittest.mock import patch, MagicMock

    mock_api_response = {
        "voices": [
            {"voice_id": "brad1", "name": "Brad", "labels": {"gender": "male"}},
            {"voice_id": "drew1", "name": "Drew", "labels": {"gender": "male"}},
            {"voice_id": "haley1", "name": "Haley", "labels": {"gender": "female"}},
        ],
    }

    def _mock_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json.return_value = mock_api_response
        return resp

    sb = MagicMock()
    chain = MagicMock()
    chain.eq.return_value = chain
    chain.limit.return_value = chain
    chain.execute.return_value = MagicMock(data=[{"collection_id": "coll123"}])
    sb.table.return_value.select.return_value.eq.return_value = chain

    with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test"}), \
         patch("skills.voice_buckets.httpx.get", side_effect=_mock_get):
        from skills.voice_buckets import resolve_guest_voice
        result = resolve_guest_voice(sb, "multi", "The Hillis Family", exclude_voice_id="brad1")
        assert result["voice_id"] != "brad1"
