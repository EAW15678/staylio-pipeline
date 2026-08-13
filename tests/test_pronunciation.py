"""
PHONETIC-1: Pronunciation dictionary tests.

All API calls mocked — $0.
"""

import pytest
from unittest.mock import MagicMock, patch
from skills.pronunciation import (
    get_pronunciation_locators,
    _compute_entries_hash,
    _create_dictionary,
)


def _mock_sb(global_entries=None, prop_entries=None, cache=None):
    """Mock Supabase with separate responses for global/prop/cache queries."""
    sb = MagicMock()

    call_count = [0]
    responses = [
        MagicMock(data=global_entries or []),  # global entries
        MagicMock(data=prop_entries or []),     # property entries
        MagicMock(data=cache or []),            # cache lookup
    ]

    def mock_execute():
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        return responses[idx]

    chain = MagicMock()
    chain.eq.return_value = chain
    chain.is_.return_value = chain
    chain.limit.return_value = chain
    chain.execute = mock_execute
    chain.insert.return_value = chain
    chain.update.return_value = chain
    sb.table.return_value.select.return_value = chain

    return sb


GLOBAL_ENTRIES = [
    {"term": "LBI", "pronunciation": "L B I", "rule_type": "alias", "alphabet": None},
    {"term": "VRBO", "pronunciation": "V R B O", "rule_type": "alias", "alphabet": None},
]

PROP_ENTRIES = [
    {"term": "Azule", "pronunciation": "ah-ZOO-lay", "rule_type": "alias", "alphabet": None},
]

CACHE_HIT = [
    {"id": "cache1", "dictionary_id": "dict123", "version_id": "ver456",
     "entries_hash": _compute_entries_hash(GLOBAL_ENTRIES + PROP_ENTRIES)},
]


# ── Hash computation ────────────────────────────────────────────────────

def test_hash_deterministic():
    h1 = _compute_entries_hash(GLOBAL_ENTRIES)
    h2 = _compute_entries_hash(GLOBAL_ENTRIES)
    assert h1 == h2


def test_hash_changes_on_different_entries():
    h1 = _compute_entries_hash(GLOBAL_ENTRIES)
    h2 = _compute_entries_hash(PROP_ENTRIES)
    assert h1 != h2


# ── Cache reuse ──────────────────────────────────────────────────────────

def test_cache_reuse():
    """Cached dictionary reused when entries haven't changed."""
    sb = _mock_sb(
        global_entries=GLOBAL_ENTRIES,
        prop_entries=PROP_ENTRIES,
        cache=CACHE_HIT,
    )
    locators = get_pronunciation_locators(sb, "prop123")
    assert len(locators) == 1
    assert locators[0]["pronunciation_dictionary_id"] == "dict123"
    assert locators[0]["version_id"] == "ver456"


# ── Empty entries ────────────────────────────────────────────────────────

def test_no_entries_returns_empty():
    """No pronunciation entries → empty locators (TTS runs without dictionary)."""
    sb = _mock_sb(global_entries=[], prop_entries=[], cache=[])
    locators = get_pronunciation_locators(sb, "prop123")
    assert locators == []


# ── Global only ──────────────────────────────────────────────────────────

@patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test_key"})
@patch("skills.pronunciation.httpx.post")
def test_global_only_creates_dict(mock_post):
    """Global entries only → creates dictionary."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"id": "new_dict", "version_id": "new_ver"}
    mock_post.return_value = mock_resp

    sb = _mock_sb(global_entries=GLOBAL_ENTRIES, prop_entries=[], cache=[])
    locators = get_pronunciation_locators(sb, "prop123")
    assert len(locators) == 1
    assert locators[0]["pronunciation_dictionary_id"] == "new_dict"

    # Verify the create call included the rules
    call_args = mock_post.call_args
    rules = call_args.kwargs.get("json", {}).get("rules", [])
    assert len(rules) == 2
    assert rules[0]["string_to_replace"] == "LBI"
    assert rules[0]["type"] == "alias"


# ── Per-property merge ───────────────────────────────────────────────────

@patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test_key"})
@patch("skills.pronunciation.httpx.post")
def test_property_merge(mock_post):
    """Per-property entries merge with global."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"id": "merged_dict", "version_id": "merged_ver"}
    mock_post.return_value = mock_resp

    sb = _mock_sb(global_entries=GLOBAL_ENTRIES, prop_entries=PROP_ENTRIES, cache=[])
    locators = get_pronunciation_locators(sb, "prop123")

    # Should have 3 rules (2 global + 1 property)
    call_args = mock_post.call_args
    rules = call_args.kwargs.get("json", {}).get("rules", [])
    assert len(rules) == 3


# ── API failure → Ruling 6 ──────────────────────────────────────────────

@patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test_key"})
@patch("skills.pronunciation.httpx.post")
def test_api_failure_raises(mock_post):
    """API error → ValueError (Ruling 6: fail loudly)."""
    mock_post.side_effect = Exception("API down")

    sb = _mock_sb(global_entries=GLOBAL_ENTRIES, prop_entries=[], cache=[])
    with pytest.raises(ValueError, match="Failed to create"):
        get_pronunciation_locators(sb, "prop123")


# ── 3-locator limit ─────────────────────────────────────────────────────

def test_locators_within_limit():
    """We return max 1 locator (merged dictionary), well within the 3 limit."""
    sb = _mock_sb(
        global_entries=GLOBAL_ENTRIES,
        prop_entries=PROP_ENTRIES,
        cache=CACHE_HIT,
    )
    locators = get_pronunciation_locators(sb, "prop123")
    assert len(locators) <= 3
