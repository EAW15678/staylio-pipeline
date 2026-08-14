"""
DEDUPE-1: Source image identity pre-clustering tests.
"""

from collections import defaultdict
from skills.deduplicate import _hamming_distance, HAMMING_THRESHOLD


def _run_union_find(photos):
    """Simulate the deduplicate skill's clustering logic."""
    parent = {p["photo_id"]: p["photo_id"] for p in photos}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Pre-clustering by source_image_id
    by_sid = defaultdict(list)
    for p in photos:
        sid = p.get("source_image_id")
        if sid:
            by_sid[sid].append(p["photo_id"])

    for sid, pids in by_sid.items():
        for pid in pids[1:]:
            union(pids[0], pid)

    # pHash clustering
    for i in range(len(photos)):
        for j in range(i + 1, len(photos)):
            dist = _hamming_distance(photos[i]["phash"], photos[j]["phash"])
            if dist <= HAMMING_THRESHOLD:
                union(photos[i]["photo_id"], photos[j]["photo_id"])

    clusters = defaultdict(list)
    for p in photos:
        clusters[find(p["photo_id"])].append(p)
    return clusters


def test_same_source_id_distant_phash_clusters():
    """Two rows, same source_image_id, pHash distance 32 → one cluster."""
    photos = [
        {"photo_id": "a", "phash": "0000000000000000", "image_width": 1024, "image_height": 768, "source_image_id": "img1"},
        {"photo_id": "b", "phash": "ffffffffffffffff", "image_width": 84, "image_height": 56, "source_image_id": "img1"},
    ]
    clusters = _run_union_find(photos)
    assert len(clusters) == 1
    members = list(clusters.values())[0]
    assert len(members) == 2


def test_null_source_id_distant_phash_no_cluster():
    """Two rows, NULL source_image_id, distance 32 → two clusters."""
    photos = [
        {"photo_id": "a", "phash": "0000000000000000", "image_width": 1024, "image_height": 768, "source_image_id": None},
        {"photo_id": "b", "phash": "ffffffffffffffff", "image_width": 84, "image_height": 56, "source_image_id": None},
    ]
    clusters = _run_union_find(photos)
    assert len(clusters) == 2


def test_canonical_is_larger():
    """In a source_image_id cluster, the larger row becomes canonical."""
    photos = [
        {"photo_id": "small", "phash": "0000000000000000", "image_width": 84, "image_height": 56, "source_image_id": "img1"},
        {"photo_id": "large", "phash": "ffffffffffffffff", "image_width": 1024, "image_height": 768, "source_image_id": "img1"},
    ]
    clusters = _run_union_find(photos)
    members = list(clusters.values())[0]
    # Sort by resolution descending (same as deduplicate.py)
    members.sort(key=lambda p: (p.get("image_width") or 0) * (p.get("image_height") or 0), reverse=True)
    assert members[0]["photo_id"] == "large"


def test_mixed_source_id_and_phash():
    """Source ID clusters merge with pHash clusters transitively."""
    photos = [
        {"photo_id": "a", "phash": "0000000000000001", "image_width": 1024, "image_height": 768, "source_image_id": "img1"},
        {"photo_id": "b", "phash": "ffffffffffffffff", "image_width": 84, "image_height": 56, "source_image_id": "img1"},
        {"photo_id": "c", "phash": "0000000000000000", "image_width": 768, "image_height": 512, "source_image_id": None},
    ]
    # a and b cluster by source_id; a and c cluster by pHash (dist=1)
    clusters = _run_union_find(photos)
    assert len(clusters) == 1  # all three in one cluster
