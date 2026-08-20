"""
C1 FIX: acquire_listing must not re-insert identical OTA reviews.

Tests import and call the real _insert_review_if_new function from
skills/acquire_listing.py — not a local reimplementation.
"""

import sys
from unittest.mock import MagicMock

sys.path.insert(0, ".")

from skills.acquire_listing import _insert_review_if_new


class FakeSB:
    """Tracks inserts and supports dedupe queries."""

    def __init__(self):
        self.rows = []

    def table(self, name):
        return _Tbl(name, self)


class _Tbl:
    def __init__(self, name, fake_sb):
        self._name = name
        self._sb = fake_sb
        self._eq_filters = []

    def select(self, *a, **k):
        return self

    def eq(self, col, val):
        self._eq_filters.append((col, val))
        return self

    def limit(self, *a, **k):
        return self

    def execute(self):
        if self._name == "guest_evidence" and self._eq_filters:
            matches = [
                row for row in self._sb.rows
                if all(row.get(col) == val for col, val in self._eq_filters)
            ]
            self._eq_filters = []
            return MagicMock(data=matches, count=len(matches))
        self._eq_filters = []
        return MagicMock(data=[], count=0)

    def insert(self, payload):
        if self._name == "guest_evidence":
            self._sb.rows.append(payload)
        return self


class _Review:
    def __init__(self, text, reviewer_name, stay_date=None, star_rating=None):
        self.text = text
        self.reviewer_name = reviewer_name
        self.stay_date = stay_date
        self.star_rating = star_rating


# ── Test 1: Airbnb duplicate blocked ────────────────────────────────────────

def test_airbnb_duplicate_blocked():
    """Inserting the same Airbnb review twice results in exactly one row."""
    sb = FakeSB()
    r = _Review("Great place!", "John D.", star_rating=5)

    assert _insert_review_if_new(sb, "PROP-1", "airbnb", r) is True
    assert len(sb.rows) == 1

    assert _insert_review_if_new(sb, "PROP-1", "airbnb", r) is False
    assert len(sb.rows) == 1, f"Duplicate should be blocked, got {len(sb.rows)}"


# ── Test 2: VRBO duplicate blocked ─────────────────────────────────────────

def test_vrbo_duplicate_blocked():
    """Inserting the same VRBO review twice results in exactly one row."""
    sb = FakeSB()
    r = _Review("Amazing stay!", "Stephanie B.")

    assert _insert_review_if_new(sb, "PROP-1", "vrbo", r) is True
    assert len(sb.rows) == 1

    assert _insert_review_if_new(sb, "PROP-1", "vrbo", r) is False
    assert len(sb.rows) == 1


# ── Test 3: Different text from same reviewer is NOT skipped ────────────────

def test_different_text_not_skipped():
    """Same reviewer, different text → different review, should insert."""
    sb = FakeSB()
    r1 = _Review("Great place!", "John D.", star_rating=5)
    r2 = _Review("Came back again, still great!", "John D.", star_rating=5)

    _insert_review_if_new(sb, "PROP-1", "airbnb", r1)
    assert len(sb.rows) == 1

    _insert_review_if_new(sb, "PROP-1", "airbnb", r2)
    assert len(sb.rows) == 2, f"Different text should insert, got {len(sb.rows)}"
