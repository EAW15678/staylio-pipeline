"""
G27: Order-independence invariant for canonical selection.

Proves that the canonical elected from a pHash cluster is the SAME
regardless of the order photographs were ingested.

The canonical_key sorts by:
  1. -megapixels  (highest resolution wins)
  2. source_rank  (intake_portal=0 > vrbo=1 > airbnb=2 > pmc=3)
  3. photo_id     (deterministic UUID tiebreak)

All three axes are order-independent by construction.
"""

from skills.dedupe import _megapixels, _best_source_rank, _hex_hamming, _UnionFind


def _canonical_key(photo):
    """Mirror the key from dedupe.py."""
    mp = _megapixels(photo.get("image_width"), photo.get("image_height"))
    src_rank = _best_source_rank(photo.get("source_systems", []))
    return (-mp, src_rank, photo["photo_id"])


def _select_canonical(photos):
    """Given a cluster of photos, return the canonical photo_id."""
    return sorted(photos, key=_canonical_key)[0]["photo_id"]


# ── Test fixtures ────────────────────────────────────────────────────────

VRBO_3840 = {
    "photo_id": "aaaa-vrbo",
    "image_width": 3840, "image_height": 2562,
    "source_systems": ["vrbo"],
    "phash": "f8e0c0c0c0e0f8f8",
}

AIRBNB_1200 = {
    "photo_id": "bbbb-airbnb",
    "image_width": 1200, "image_height": 800,
    "source_systems": ["airbnb"],
    "phash": "f8e0c0c0c0e0f8f8",  # same scene
}

OWNER_5712 = {
    "photo_id": "cccc-owner",
    "image_width": 5712, "image_height": 4284,
    "source_systems": ["intake_portal"],
    "phash": "f8e0c0c0c0e0f8f8",  # same scene
}

PMC_1280 = {
    "photo_id": "dddd-pmc",
    "image_width": 1280, "image_height": 853,
    "source_systems": ["pmc_website"],
    "phash": "f8e0c0c0c0e0f8f8",  # same scene
}


# ── Tests ────────────────────────────────────────────────────────────────

def test_vrbo_beats_airbnb_on_resolution():
    """VRBO 3840px wins over Airbnb 1200px."""
    assert _select_canonical([VRBO_3840, AIRBNB_1200]) == "aaaa-vrbo"
    assert _select_canonical([AIRBNB_1200, VRBO_3840]) == "aaaa-vrbo"


def test_owner_beats_vrbo_on_resolution():
    """Owner 5712px wins over VRBO 3840px (resolution, not source priority)."""
    assert _select_canonical([OWNER_5712, VRBO_3840]) == "cccc-owner"
    assert _select_canonical([VRBO_3840, OWNER_5712]) == "cccc-owner"


def test_source_priority_breaks_resolution_tie():
    """When resolution is equal, source priority wins."""
    vrbo_same = {**PMC_1280, "photo_id": "eeee-vrbo", "source_systems": ["vrbo"]}
    # Both 1280x853 — PMC (rank 3) vs VRBO (rank 1) → VRBO wins
    assert _select_canonical([PMC_1280, vrbo_same]) == "eeee-vrbo"
    assert _select_canonical([vrbo_same, PMC_1280]) == "eeee-vrbo"


def test_four_way_order_independence():
    """All 24 permutations of 4 photos produce the same canonical."""
    from itertools import permutations
    photos = [VRBO_3840, AIRBNB_1200, OWNER_5712, PMC_1280]
    results = set()
    for perm in permutations(photos):
        results.add(_select_canonical(list(perm)))
    assert len(results) == 1, f"Non-deterministic! Got: {results}"
    assert results.pop() == "cccc-owner"  # Owner 5712px is highest


def test_photo_id_breaks_all_ties():
    """When MP and source are identical, photo_id is the tiebreak."""
    a = {"photo_id": "aaaa", "image_width": 1920, "image_height": 1080, "source_systems": ["airbnb"]}
    b = {"photo_id": "zzzz", "image_width": 1920, "image_height": 1080, "source_systems": ["airbnb"]}
    assert _select_canonical([a, b]) == "aaaa"
    assert _select_canonical([b, a]) == "aaaa"


def test_hamming_distance():
    """pHash hamming distance computation."""
    assert _hex_hamming("f8e0c0c0c0e0f8f8", "f8e0c0c0c0e0f8f8") == 0
    assert _hex_hamming("f8e0c0c0c0e0f8f8", "f8e0c0c0c0e0f8f9") == 1
    # Distance > threshold should NOT cluster
    assert _hex_hamming("0000000000000000", "ffffffffffffffff") == 64


def test_union_find_clustering():
    """Union-Find clusters correctly."""
    uf = _UnionFind(4)
    uf.union(0, 1)
    uf.union(2, 3)
    assert uf.find(0) == uf.find(1)
    assert uf.find(2) == uf.find(3)
    assert uf.find(0) != uf.find(2)
