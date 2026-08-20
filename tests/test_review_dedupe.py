"""
C1 FIX: acquire_listing must not re-insert identical OTA reviews.

Broke live on Vista Azule 2026-08-20 — Stephanie B. VRBO review appeared
twice after a crash-and-retry.
"""

import sys
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, ".")


class FakeSB:
    """Tracks inserts and supports dedupe queries."""

    def __init__(self):
        self.rows = []  # all inserted guest_evidence rows

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

    def order(self, *a, **k):
        return self

    def is_(self, *a, **k):
        return self

    def execute(self):
        if self._name == "guest_evidence" and self._eq_filters:
            # Check if any existing row matches ALL eq filters
            matches = []
            for row in self._sb.rows:
                match = all(row.get(col) == val for col, val in self._eq_filters)
                if match:
                    matches.append(row)
            self._eq_filters = []
            return MagicMock(data=matches, count=len(matches))
        self._eq_filters = []
        return MagicMock(data=[], count=0)

    def insert(self, payload):
        if self._name == "guest_evidence":
            self._sb.rows.append(payload)
        return self

    def update(self, *a, **k):
        return self

    def upsert(self, *a, **k):
        return self


# Minimal review object matching temp_kb.guest_reviews shape
class _Review:
    def __init__(self, text, reviewer_name, stay_date=None, star_rating=None):
        self.text = text
        self.reviewer_name = reviewer_name
        self.stay_date = stay_date
        self.star_rating = star_rating


def _run_airbnb_inserts(sb, property_id, reviews):
    """Simulate the Airbnb review insert loop from acquire_listing."""
    for r in reviews:
        if r.text:
            dup = sb.table("guest_evidence").select("evidence_id").eq(
                "property_id", property_id
            ).eq("source", "airbnb").eq(
                "reviewer_name", r.reviewer_name or ""
            ).eq("written_text", r.text).limit(1).execute()
            if dup.data:
                continue
            sb.table("guest_evidence").insert({
                "property_id": property_id,
                "written_text": r.text or "",
                "verbal_text": "",
                "reviewer_name": r.reviewer_name,
                "stay_date": r.stay_date,
                "source": "airbnb",
                "is_guest_book": False,
                "star_rating": r.star_rating,
            }).execute()


def _run_vrbo_inserts(sb, property_id, reviews):
    """Simulate the VRBO review insert loop from acquire_listing."""
    for r in reviews:
        if r.text:
            dup = sb.table("guest_evidence").select("evidence_id").eq(
                "property_id", property_id
            ).eq("source", "vrbo").eq(
                "reviewer_name", r.reviewer_name or ""
            ).eq("written_text", r.text).limit(1).execute()
            if dup.data:
                continue
            sb.table("guest_evidence").insert({
                "property_id": property_id,
                "written_text": r.text or "",
                "verbal_text": "",
                "reviewer_name": r.reviewer_name,
                "stay_date": r.stay_date,
                "source": "vrbo",
                "is_guest_book": False,
            }).execute()


# ── Test 1: Airbnb duplicate blocked ────────────────────────────────────────

def test_airbnb_duplicate_blocked():
    """Inserting the same Airbnb review twice results in exactly one row."""
    sb = FakeSB()
    reviews = [_Review("Great place!", "John D.", star_rating=5)]

    _run_airbnb_inserts(sb, "PROP-1", reviews)
    assert len(sb.rows) == 1, f"First insert: expected 1 row, got {len(sb.rows)}"

    _run_airbnb_inserts(sb, "PROP-1", reviews)
    assert len(sb.rows) == 1, f"Second insert: expected still 1 row, got {len(sb.rows)}"


# ── Test 2: VRBO duplicate blocked ─────────────────────────────────────────

def test_vrbo_duplicate_blocked():
    """Inserting the same VRBO review twice results in exactly one row."""
    sb = FakeSB()
    reviews = [_Review("Amazing stay!", "Stephanie B.")]

    _run_vrbo_inserts(sb, "PROP-1", reviews)
    assert len(sb.rows) == 1

    _run_vrbo_inserts(sb, "PROP-1", reviews)
    assert len(sb.rows) == 1, f"Duplicate should be blocked, got {len(sb.rows)} rows"


# ── Test 3: Different text from same reviewer is NOT skipped ────────────────

def test_different_text_not_skipped():
    """A review with the same reviewer name but different text is a different
    review and should insert."""
    sb = FakeSB()
    review1 = [_Review("Great place!", "John D.", star_rating=5)]
    review2 = [_Review("Came back again, still great!", "John D.", star_rating=5)]

    _run_airbnb_inserts(sb, "PROP-1", review1)
    assert len(sb.rows) == 1

    _run_airbnb_inserts(sb, "PROP-1", review2)
    assert len(sb.rows) == 2, f"Different text should insert, got {len(sb.rows)} rows"
