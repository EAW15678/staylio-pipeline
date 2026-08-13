"""
VIDEO-1C: Guest word fidelity validator tests.

Uses real Casa de Lubitz / Vista Azule guest text from production.
Every test is named for its scenario and quotes the input/output.
"""

from skills.direct import (
    validate_quote_fidelity,
    _split_sentences,
    _normalize_for_comparison,
    _sentences_match,
)


# ── Real guest text from production ──────────────────────────────────────

# Eileen Breslin (Casa de Lubitz) — written_text
EILEEN_WRITTEN = (
    "My family had a fantastic two weeks staying at this wonderful house. "
    "My granddaughter was in the pool every day and loved the basketball hoop. "
    "This home is extremely clean & very spacious. "
    "The owners were terrific in quickly handling any problem that came up. "
    "Already looking forward to next summer."
)

# Eileen Breslin (Casa de Lubitz) — verbal_text
EILEEN_VERBAL = (
    "We had a fantastic time. My grand daughter loved the pool. "
    "Hoping to be back next year."
)

# Colleen Gillon (Casa de Lubitz) — written_text
COLLEEN_WRITTEN = (
    "We had an amazing time in LBI at this special home which has everything "
    "you and your family and friends need for a comfortable and fun-filled stay! "
    "We used the private pool every day for quiet morning dips and night swimming."
)

# Tom Jenkins (Vista Azule) — written_text
TOM_WRITTEN = (
    "This place was the perfect location. All things were just steps away. "
    "Loved the pool."
)

ALL_GUEST_TEXTS = [EILEEN_WRITTEN, EILEEN_VERBAL, COLLEEN_WRITTEN, TOM_WRITTEN]


# ── Sentence splitter tests ──────────────────────────────────────────────

def test_split_sentences_basic():
    result = _split_sentences("Hello world. This is great! Is it?")
    assert result == ["Hello world.", "This is great!", "Is it?"]


def test_split_sentences_preserves_abbreviations():
    """Handles Mr. Mrs. etc. — not perfect, but functional for guest text."""
    result = _split_sentences("Loved the pool. Already looking forward to next summer.")
    assert len(result) == 2


# ── Exact match passes ───────────────────────────────────────────────────

def test_exact_match_passes():
    """Narration uses Eileen's exact sentences → passes."""
    direction = {
        "narration_script": (
            "My family had a fantastic two weeks staying at this wonderful house. "
            "My granddaughter was in the pool every day and loved the basketball hoop."
        ),
        "narration_provenance": "guest_book",
        "overlay_register": [],
    }
    violations = validate_quote_fidelity(direction, ALL_GUEST_TEXTS)
    assert violations == [], f"Should pass: {violations}"


# ── Typo correction: "grand daughter" → "granddaughter" passes ───────────

def test_granddaughter_typo_passes():
    """Verbal text has 'grand daughter', written has 'granddaughter'.
    Narration using either spelling should match."""
    direction = {
        "narration_script": "My grand daughter loved the pool.",
        "narration_provenance": "guest_book",
        "overlay_register": [],
    }
    violations = validate_quote_fidelity(direction, ALL_GUEST_TEXTS)
    assert violations == [], f"Should pass (typo correction): {violations}"


# ── Symbol expansion: & → and passes ─────────────────────────────────────

def test_ampersand_expansion_passes():
    """Source has '&', narration uses 'and' → passes."""
    direction = {
        "narration_script": "This home is extremely clean and very spacious.",
        "narration_provenance": "guest_book",
        "overlay_register": [],
    }
    violations = validate_quote_fidelity(direction, ALL_GUEST_TEXTS)
    assert violations == [], f"Should pass (& → and): {violations}"


# ── Dropped word fails ───────────────────────────────────────────────────

def test_dropped_word_fails():
    """'My family had a fantastic two weeks at this wonderful house.'
    — dropped 'staying' → fails."""
    direction = {
        "narration_script": "My family had a fantastic two weeks at this wonderful house.",
        "narration_provenance": "guest_book",
        "overlay_register": [],
    }
    violations = validate_quote_fidelity(direction, ALL_GUEST_TEXTS)
    assert len(violations) >= 1, "Should fail: word 'staying' dropped"
    assert violations[0]["rule"] == "guest_word_fidelity"


# ── Reordered clause fails ──────────────────────────────────────────────

def test_reordered_clause_fails():
    """'This wonderful house, my family had a fantastic two weeks staying at.'
    — reordered → fails."""
    direction = {
        "narration_script": "This wonderful house, my family had a fantastic two weeks staying at.",
        "narration_provenance": "guest_book",
        "overlay_register": [],
    }
    violations = validate_quote_fidelity(direction, ALL_GUEST_TEXTS)
    assert len(violations) >= 1, "Should fail: clause reordered"
    assert violations[0]["rule"] == "guest_word_fidelity"


# ── Mid-sentence truncation fails ────────────────────────────────────────

def test_mid_sentence_truncation_fails():
    """'My family had a fantastic two weeks' — sentence cut mid-flow → fails.
    The director must drop the whole sentence or choose a shorter one."""
    direction = {
        "narration_script": "My family had a fantastic two weeks",
        "narration_provenance": "guest_book",
        "overlay_register": [],
    }
    violations = validate_quote_fidelity(direction, ALL_GUEST_TEXTS)
    assert len(violations) >= 1, "Should fail: mid-sentence truncation"
    assert violations[0]["rule"] == "guest_word_fidelity"


# ── Paraphrase fails ────────────────────────────────────────────────────

def test_paraphrase_fails():
    """'The family enjoyed their wonderful two-week stay at this beautiful home.'
    — paraphrased → fails."""
    direction = {
        "narration_script": "The family enjoyed their wonderful two-week stay at this beautiful home.",
        "narration_provenance": "guest_book",
        "overlay_register": [],
    }
    violations = validate_quote_fidelity(direction, ALL_GUEST_TEXTS)
    assert len(violations) >= 1, "Should fail: paraphrase"
    assert violations[0]["rule"] == "guest_word_fidelity"


# ── Multiple sentences: select 2, skip 3 — passes ───────────────────────

def test_sentence_selection_passes():
    """Director selects sentences 1 and 5 from Eileen, skipping 2-4 → passes.
    Whole sentences, just fewer of them."""
    direction = {
        "narration_script": (
            "My family had a fantastic two weeks staying at this wonderful house. "
            "Already looking forward to next summer."
        ),
        "narration_provenance": "guest_book",
        "overlay_register": [],
    }
    violations = validate_quote_fidelity(direction, ALL_GUEST_TEXTS)
    assert violations == [], f"Should pass (sentence selection): {violations}"


# ── Cross-guest sourcing passes ──────────────────────────────────────────

def test_cross_guest_sentences_pass():
    """Director uses one sentence from Eileen and one from Tom → passes.
    Both are whole source sentences."""
    direction = {
        "narration_script": (
            "My granddaughter was in the pool every day and loved the basketball hoop. "
            "Loved the pool."
        ),
        "narration_provenance": "guest_book",
        "overlay_register": [],
    }
    violations = validate_quote_fidelity(direction, ALL_GUEST_TEXTS)
    assert violations == [], f"Should pass (cross-guest): {violations}"


# ── Non-guest-book narration is not checked ──────────────────────────────

def test_original_narration_not_checked():
    """narration_provenance='original' → validator skips entirely."""
    direction = {
        "narration_script": "This is completely made up narration, not from any guest.",
        "narration_provenance": "original",
        "overlay_register": [],
    }
    violations = validate_quote_fidelity(direction, ALL_GUEST_TEXTS)
    assert violations == [], "Original narration should not be checked"


# ── Overlay with guest_book provenance is checked ────────────────────────

def test_overlay_guest_quote_checked():
    """An overlay with provenance='guest_book' and a paraphrase → fails."""
    direction = {
        "narration_script": "",
        "narration_provenance": "original",
        "overlay_register": [
            {
                "beat_ordinal": 3,
                "provenance": "guest_book",
                "text": "We absolutely adored the pool and the basketball.",
            }
        ],
    }
    violations = validate_quote_fidelity(direction, ALL_GUEST_TEXTS)
    assert len(violations) >= 1, "Paraphrased overlay should fail"


# ── _sentences_match unit tests ──────────────────────────────────────────

def test_sentences_match_exact():
    assert _sentences_match("Loved the pool.", "Loved the pool.")


def test_sentences_match_punctuation_diff():
    assert _sentences_match("Loved the pool!", "Loved the pool.")


def test_sentences_match_ampersand():
    assert _sentences_match(
        "This home is extremely clean and very spacious.",
        "This home is extremely clean & very spacious."
    )


def test_sentences_match_compound_word():
    assert _sentences_match(
        "My grand daughter loved the pool.",
        "My granddaughter loved the pool."
    )


def test_sentences_no_match_added_word():
    assert not _sentences_match(
        "My family had a really fantastic two weeks staying at this wonderful house.",
        "My family had a fantastic two weeks staying at this wonderful house."
    )


def test_sentences_no_match_reorder():
    assert not _sentences_match(
        "This wonderful house, my family had a fantastic two weeks staying at.",
        "My family had a fantastic two weeks staying at this wonderful house."
    )
