"""
VOICE-3: Live voice resolution tests.

All ElevenLabs API calls are mocked — $0.
"""

import pytest
from unittest.mock import MagicMock, patch
from skills.voice_buckets import (
    fetch_vibe_voices,
    resolve_guest_voice,
    _is_gender_neutral,
    _get_collection_id,
)


# ── Mock helpers ─────────────────────────────────────────────────────────

def _mock_sb(collection_id="coll123"):
    """Mock Supabase that returns a collection_id from vibe_collections."""
    sb = MagicMock()
    chain = MagicMock()
    chain.eq.return_value = chain
    chain.limit.return_value = chain
    chain.execute.return_value = MagicMock(data=[{"collection_id": collection_id}])
    sb.table.return_value.select.return_value.eq.return_value = chain
    return sb


def _mock_sb_empty():
    """Mock Supabase that returns no collection mapping."""
    sb = MagicMock()
    chain = MagicMock()
    chain.eq.return_value = chain
    chain.limit.return_value = chain
    chain.execute.return_value = MagicMock(data=[])
    sb.table.return_value.select.return_value.eq.return_value = chain
    return sb


MOCK_API_RESPONSE = {
    "voices": [
        {"voice_id": "brad1", "name": "Brad - Welcoming", "labels": {"gender": "male", "language": "en"}},
        {"voice_id": "drew1", "name": "Drew - Casual", "labels": {"gender": "male", "language": "en"}},
        {"voice_id": "haley1", "name": "Haley Maven", "labels": {"gender": "female", "language": "en"}},
        {"voice_id": "juliet1", "name": "Juliet - Customer Care", "labels": {"gender": "female", "language": "en"}},
        {"voice_id": "nogender1", "name": "Mystery Voice", "labels": {"language": "en"}},
    ],
}


def _mock_httpx_get(url, **kwargs):
    """Mock httpx.get that returns the voice list."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = MOCK_API_RESPONSE
    return resp


def _mock_httpx_get_empty(url, **kwargs):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"voices": []}
    return resp


# ── Gender detection ─────────────────────────────────────────────────────

def test_gender_neutral_family():
    assert _is_gender_neutral("The Hillis Family")

def test_gender_neutral_class():
    assert _is_gender_neutral("The USC Class of 2004 Reunion")

def test_gender_not_neutral():
    assert not _is_gender_neutral("Eileen Breslin")


# ── Live fetch ───────────────────────────────────────────────────────────

@patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test_key"})
@patch("skills.voice_buckets.httpx.get", side_effect=_mock_httpx_get)
def test_fetch_vibe_voices(mock_get):
    sb = _mock_sb()
    voices = fetch_vibe_voices(sb, "multigenerational_retreat")
    assert len(voices) == 5
    assert voices[0]["voice_id"] == "brad1"
    assert voices[0]["gender"] == "male"
    assert voices[4]["gender"] is None  # no gender label


@patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test_key"})
@patch("skills.voice_buckets.httpx.get", side_effect=_mock_httpx_get_empty)
def test_fetch_empty_collection_fails(mock_get):
    sb = _mock_sb()
    with pytest.raises(ValueError, match="empty"):
        fetch_vibe_voices(sb, "multigenerational_retreat")


def test_missing_mapping_fails():
    sb = _mock_sb_empty()
    with pytest.raises(ValueError, match="No collection mapping"):
        _get_collection_id(sb, "nonexistent_vibe")


# ── Guest voice resolution ──────────────────────────────────────────────

@patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test_key"})
@patch("skills.voice_buckets.httpx.get", side_effect=_mock_httpx_get)
def test_guest_neutral_unconstrained(mock_get):
    """Gender-neutral name → all voices eligible including no-gender."""
    sb = _mock_sb()
    result = resolve_guest_voice(sb, "multi", "The Hillis Family")
    assert result["voice_id"] in ["brad1", "drew1", "haley1", "juliet1", "nogender1"]
    assert "unconstrained" in result["reason"]


@patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test_key"})
@patch("skills.voice_buckets.httpx.get", side_effect=_mock_httpx_get)
def test_guest_gender_match_female(mock_get):
    sb = _mock_sb()
    result = resolve_guest_voice(sb, "multi", "Eileen", reviewer_gender="female")
    assert result["voice_id"] in ["haley1", "juliet1"]
    assert result["gender"] == "female"


@patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test_key"})
@patch("skills.voice_buckets.httpx.get", side_effect=_mock_httpx_get)
def test_guest_gender_match_male(mock_get):
    sb = _mock_sb()
    result = resolve_guest_voice(sb, "multi", "Matthew", reviewer_gender="male")
    assert result["voice_id"] in ["brad1", "drew1"]
    assert result["gender"] == "male"


@patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test_key"})
@patch("skills.voice_buckets.httpx.get", side_effect=_mock_httpx_get)
def test_no_gender_voice_excluded_from_gendered_slot(mock_get):
    """A voice with no gender label is NOT eligible for gendered slots."""
    sb = _mock_sb()
    result = resolve_guest_voice(sb, "multi", "Eileen", reviewer_gender="female")
    assert result["voice_id"] != "nogender1"


@patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test_key"})
@patch("skills.voice_buckets.httpx.get", side_effect=_mock_httpx_get)
def test_hero_exclusion(mock_get):
    """Hero narrator excluded from guest pool."""
    sb = _mock_sb()
    result = resolve_guest_voice(sb, "multi", "The Hillis Family", exclude_voice_id="brad1")
    assert result["voice_id"] != "brad1"


# ── Variation + determinism ──────────────────────────────────────────────

@patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test_key"})
@patch("skills.voice_buckets.httpx.get", side_effect=_mock_httpx_get)
def test_variation_across_three(mock_get):
    sb = _mock_sb()
    names = ["The Hillis Family", "The Resni & Mitchell Family", "The USC Class of 2004 Reunion"]
    chosen = [resolve_guest_voice(sb, "multi", n)["voice_id"] for n in names]
    assert len(set(chosen)) >= 2, f"All same: {chosen}"


@patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test_key"})
@patch("skills.voice_buckets.httpx.get", side_effect=_mock_httpx_get)
def test_deterministic(mock_get):
    sb = _mock_sb()
    r1 = resolve_guest_voice(sb, "multi", "The Hillis Family")
    r2 = resolve_guest_voice(sb, "multi", "The Hillis Family")
    assert r1["voice_id"] == r2["voice_id"]
