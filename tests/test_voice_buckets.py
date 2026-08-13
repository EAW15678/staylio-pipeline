"""
VOICE-1: Voice bucket resolution tests (one pool, no role column).
"""

import pytest
from unittest.mock import MagicMock
from skills.voice_buckets import get_vibe_pool, resolve_guest_voice, _is_gender_neutral


def _mock_sb(rows):
    """Mock Supabase client returning `rows` for any chained query."""
    sb = MagicMock()
    chain = MagicMock()
    chain.eq.return_value = chain
    chain.execute.return_value = MagicMock(data=rows)
    sb.table.return_value.select.return_value.eq.return_value = chain
    return sb


POOL = [
    {"voice_id": "staylio1", "voice_name": "staylio_multigen", "gender": None},
    {"voice_id": "male1", "voice_name": "George", "gender": "male"},
    {"voice_id": "male2", "voice_name": "Chris", "gender": "male"},
    {"voice_id": "female1", "voice_name": "Sarah", "gender": "female"},
    {"voice_id": "female2", "voice_name": "Elise", "gender": "female"},
    {"voice_id": "neutral1", "voice_name": "River", "gender": "neutral"},
]


# ── Gender detection ─────────────────────────────────────────────────────

def test_gender_neutral_family():
    assert _is_gender_neutral("The Hillis Family")

def test_gender_neutral_class():
    assert _is_gender_neutral("The USC Class of 2004 Reunion")

def test_gender_not_neutral_individual():
    assert not _is_gender_neutral("Eileen Breslin")


# ── Pool ─────────────────────────────────────────────────────────────────

def test_pool_found():
    sb = _mock_sb(POOL)
    result = get_vibe_pool(sb, "multigenerational_retreat")
    assert len(result) == 6

def test_pool_missing_fails():
    sb = _mock_sb([])
    with pytest.raises(ValueError, match="No voices"):
        get_vibe_pool(sb, "unknown_vibe")


# ── Guest voice ──────────────────────────────────────────────────────────

def test_guest_neutral_unconstrained():
    sb = _mock_sb(POOL)
    result = resolve_guest_voice(sb, "multi", "The Hillis Family")
    assert result["voice_id"] in [v["voice_id"] for v in POOL]
    assert "unconstrained" in result["reason"]

def test_guest_gender_match():
    female_only = [v for v in POOL if v["gender"] == "female"]
    sb = _mock_sb(female_only)
    result = resolve_guest_voice(sb, "multi", "Eileen", reviewer_gender="female")
    assert result["voice_id"] in ["female1", "female2"]

def test_guest_exclude_hero_voice():
    """Hero narrator excluded from guest pool."""
    sb = _mock_sb(POOL)
    result = resolve_guest_voice(sb, "multi", "The Hillis Family", exclude_voice_id="staylio1")
    assert result["voice_id"] != "staylio1"

def test_guest_missing_fails():
    sb = _mock_sb([])
    with pytest.raises(ValueError, match="No voice"):
        resolve_guest_voice(sb, "missing", "Anyone")


# ── Variation + determinism ──────────────────────────────────────────────

def test_variation_across_three():
    sb = _mock_sb(POOL)
    names = ["The Hillis Family", "The Resni & Mitchell Family", "The USC Class of 2004 Reunion"]
    chosen = [resolve_guest_voice(sb, "multi", n)["voice_id"] for n in names]
    assert len(set(chosen)) >= 2, f"All same: {chosen}"

def test_deterministic():
    sb = _mock_sb(POOL)
    r1 = resolve_guest_voice(sb, "multi", "The Hillis Family")
    r2 = resolve_guest_voice(sb, "multi", "The Hillis Family")
    assert r1["voice_id"] == r2["voice_id"]
