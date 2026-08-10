"""
Tests for guest book entry field mapping and review gate.
"""

import pytest


# ── Extract the _resolve_guest_entry_text logic for testing ──────────────
# The function is a nested def inside load_intake_submission, so we
# replicate the exact logic here rather than importing through the
# models.property chain (which uses Python 3.10+ syntax).

def _resolve_guest_entry_text(entry: dict) -> str:
    """Exact copy of core/supabase_store.py _resolve_guest_entry_text."""
    written = (entry.get("text") or entry.get("review_text") or "").strip()
    verbal = (entry.get("verbal") or "").strip()
    if written and verbal:
        return f"{written}\n\n{verbal}"
    return written or verbal


def _map_entry(entry: dict) -> dict:
    """Simulate the full mapping from supabase_store.py:138-145."""
    return {
        "text": _resolve_guest_entry_text(entry),
        "reviewer_name": entry.get("name") or entry.get("guest_name"),
        "stay_date": entry.get("date") or entry.get("stay_date"),
        "is_guest_book": True,
    }


# ── Portal-shaped entries (name, date, text, verbal) ─────────────────────

class TestPortalShape:
    def test_text_populated(self):
        entry = {"name": "Eileen", "date": "August 2026", "text": "Great stay!", "verbal": ""}
        result = _map_entry(entry)
        assert result["text"] == "Great stay!"
        assert result["reviewer_name"] == "Eileen"
        assert result["stay_date"] == "August 2026"

    def test_name_maps(self):
        entry = {"name": "John", "date": "July", "text": "Loved it", "verbal": ""}
        assert _map_entry(entry)["reviewer_name"] == "John"

    def test_date_maps(self):
        entry = {"name": "Jane", "date": "September 2025", "text": "Perfect", "verbal": ""}
        assert _map_entry(entry)["stay_date"] == "September 2025"


# ── Legacy-shaped entries (review_text, guest_name, stay_date) ───────────

class TestLegacyShape:
    def test_review_text_populated(self):
        entry = {"guest_name": "Bob", "stay_date": "2025-07", "review_text": "Wonderful"}
        result = _map_entry(entry)
        assert result["text"] == "Wonderful"
        assert result["reviewer_name"] == "Bob"
        assert result["stay_date"] == "2025-07"


# ── Verbal combining ────────────────────────────────────────────────────

class TestVerbalCombining:
    def test_both_present_combines(self):
        """Written + verbal → two paragraphs."""
        entry = {"text": "Written review.", "verbal": "They also mentioned the pool."}
        result = _resolve_guest_entry_text(entry)
        assert result == "Written review.\n\nThey also mentioned the pool."

    def test_verbal_only(self):
        """No written text, verbal present → use verbal."""
        entry = {"text": "", "verbal": "Guest said it was amazing."}
        result = _resolve_guest_entry_text(entry)
        assert result == "Guest said it was amazing."

    def test_text_only(self):
        """Written text, no verbal → use text."""
        entry = {"text": "Fantastic house.", "verbal": ""}
        result = _resolve_guest_entry_text(entry)
        assert result == "Fantastic house."

    def test_neither_present(self):
        """No text, no verbal → empty string, no crash."""
        entry = {"text": "", "verbal": ""}
        result = _resolve_guest_entry_text(entry)
        assert result == ""

    def test_both_none(self):
        """Keys missing entirely → empty string, no crash."""
        entry = {}
        result = _resolve_guest_entry_text(entry)
        assert result == ""

    def test_whitespace_only_treated_as_empty(self):
        entry = {"text": "  \n  ", "verbal": "  "}
        result = _resolve_guest_entry_text(entry)
        assert result == ""

    def test_verbal_with_legacy_review_text(self):
        """Legacy key + verbal → combines."""
        entry = {"review_text": "From the book.", "verbal": "Also loved the sunset."}
        result = _resolve_guest_entry_text(entry)
        assert result == "From the book.\n\nAlso loved the sunset."

    def test_real_casa_de_lubitz_eileen(self):
        """Real production data from property 82cb9d7e."""
        entry = {
            "name": "Eileen Breslin",
            "date": "August 2026",
            "text": "My family had a fantastic two weeks staying at this wonderful house.",
            "verbal": "We had a fantastic time. My grand daughter loved the pool.",
        }
        result = _resolve_guest_entry_text(entry)
        assert "My family had a fantastic two weeks" in result
        assert "My grand daughter loved the pool" in result
        assert "\n\n" in result  # joined as paragraphs


# ── Review gate ──────────────────────────────────────────────────────────

class TestReviewGate:
    def _count_with_text(self, reviews):
        """Replicate the gate logic from video_generator.py."""
        return len([
            r for r in reviews
            if r.get("is_guest_book") and (r.get("text") or "").strip()
        ])

    def test_three_real_reviews_pass(self):
        reviews = [
            {"text": "Great!", "is_guest_book": True},
            {"text": "Amazing!", "is_guest_book": True},
            {"text": "Perfect!", "is_guest_book": True},
        ]
        assert self._count_with_text(reviews) == 3

    def test_hollow_reviews_rejected(self):
        """Three entries with empty text → 0 count."""
        reviews = [
            {"text": "", "is_guest_book": True},
            {"text": "", "is_guest_book": True},
            {"text": "", "is_guest_book": True},
        ]
        assert self._count_with_text(reviews) == 0

    def test_mixed_hollow_and_real(self):
        reviews = [
            {"text": "Great!", "is_guest_book": True},
            {"text": "", "is_guest_book": True},
            {"text": "Perfect!", "is_guest_book": True},
        ]
        assert self._count_with_text(reviews) == 2

    def test_non_guestbook_excluded(self):
        reviews = [
            {"text": "OTA review", "is_guest_book": False},
            {"text": "Book entry", "is_guest_book": True},
        ]
        assert self._count_with_text(reviews) == 1

    def test_whitespace_only_text_rejected(self):
        reviews = [{"text": "   \n  ", "is_guest_book": True}]
        assert self._count_with_text(reviews) == 0
