"""
PHASH-2: Order-independence tests for the LIVE deduplicate module
(skills/deduplicate.py), NOT the dead skills/dedupe.py.

Test 5 is a regression guard — no failing baseline exists because
distinct resolutions already produce deterministic results with the
old single-key sort.

Test 6 covers exact-resolution ties — the case the tiebreak exists
for. It must fail on c7b3d15 (no tiebreak, result depends on row order).
"""

import itertools
from unittest.mock import MagicMock, patch

from skills.deduplicate import deduplicate, _SOURCE_PRIORITY, _best_source_rank


def _make_photo(photo_id, phash, width, height, source_system="vrbo", source_image_id=None):
    return {
        "photo_id": photo_id,
        "phash": phash,
        "image_width": width,
        "image_height": height,
        "is_canonical": True,
        "source_image_id": source_image_id,
        "source_systems": [source_system],
    }


def _run_deduplicate_with_photos(photos):
    """Run deduplicate with a mocked Supabase returning the given photos.

    Returns the list of (photo_id, is_canonical) updates that were applied.
    """
    updates = []

    def mock_table(name):
        t = MagicMock()
        if name == "photographs":
            # .select(...).eq(...).execute() → return photos
            select_chain = MagicMock()
            select_chain.execute = MagicMock(return_value=MagicMock(data=photos))
            t.select = MagicMock(return_value=MagicMock(
                eq=MagicMock(return_value=select_chain)
            ))
            # .update({...}).eq("photo_id", ...).execute()
            def make_update(data):
                update_chain = MagicMock()
                def eq_handler(col, val):
                    updates.append((val, data.get("is_canonical"), data))
                    exec_mock = MagicMock()
                    exec_mock.execute = MagicMock(return_value=MagicMock(data=[]))
                    return exec_mock
                update_chain.eq = eq_handler
                return update_chain
            t.update = make_update
        return t

    sb = MagicMock()
    sb.table = mock_table

    with patch("skills.deduplicate.get_substrate", return_value=sb), \
         patch("skills.deduplicate.record_run", return_value="fake-run"), \
         patch("skills.deduplicate.record_step", return_value="fake-step"), \
         patch("skills.deduplicate.complete_run"), \
         patch("skills.deduplicate.complete_step"):
        result = deduplicate("test-prop", force=True)

    return updates, result


def _get_canonical_id(updates):
    """Extract the photo_id marked as canonical from the update list."""
    for pid, is_canon, data in updates:
        if is_canon is True:
            return pid
    return None


# ── Test 5: distinct resolutions, same canonical regardless of order ──

def test_distinct_resolutions_order_independent():
    """deduplicate returns the same canonical across several permutations
    of the same photograph set with distinct resolutions.

    Regression guard — no failing baseline exists. With distinct
    resolutions the single-key sort already returns the same canonical,
    so this test passes on every commit. Its value is as a regression
    guard against future changes.
    """
    # Three photos that cluster together (identical pHash), distinct resolutions
    base_photos = [
        _make_photo("photo-large", "abcdef0123456789", 3840, 2567),   # 9.86MP
        _make_photo("photo-medium", "abcdef0123456789", 1200, 800),   # 0.96MP
        _make_photo("photo-small", "abcdef0123456789", 640, 480),     # 0.31MP
    ]

    canonical_ids = set()
    for perm in itertools.permutations(base_photos):
        updates, result = _run_deduplicate_with_photos(list(perm))
        canon = _get_canonical_id(updates)
        canonical_ids.add(canon)

    assert len(canonical_ids) == 1, (
        f"Expected one canonical across all permutations, got {canonical_ids}"
    )
    assert "photo-large" in canonical_ids, (
        f"Highest resolution photo-large should be canonical, got {canonical_ids}"
    )


# ── Test 6: exact resolution tie, tiebreak by source priority ─────────

def test_exact_tie_tiebreak_by_source():
    """Two photographs with identical resolution but different source_systems.
    The tiebreak (source priority) must produce the same canonical in
    every permutation of the tied pair.

    Baseline: fails on c7b3d15 — no tiebreak, result depends on row order.
    """
    # Two photos: same pHash, same resolution, different sources
    vrbo_photo = _make_photo("photo-vrbo", "abcdef0123456789", 1200, 800, source_system="vrbo")
    airbnb_photo = _make_photo("photo-airbnb", "abcdef0123456789", 1200, 800, source_system="airbnb")

    # Permute the tied pair explicitly
    permutations = [
        [vrbo_photo, airbnb_photo],
        [airbnb_photo, vrbo_photo],
    ]

    canonical_ids = set()
    for perm in permutations:
        updates, result = _run_deduplicate_with_photos(perm)
        canon = _get_canonical_id(updates)
        canonical_ids.add(canon)

    assert len(canonical_ids) == 1, (
        f"Expected one canonical across permutations of tied pair, got {canonical_ids}"
    )
    # VRBO (priority 1) beats Airbnb (priority 2) per _SOURCE_PRIORITY
    assert "photo-vrbo" in canonical_ids, (
        f"VRBO photo should win the tiebreak over Airbnb, got {canonical_ids}"
    )
