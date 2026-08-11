"""
Skill: deduplicate — find near-duplicate photographs via pHash.

Content-hash exact dupes are already impossible (Ruling 7 UNIQUE on
property_id + content_hash). This skill handles NEAR-duplicates by
computing pHash Hamming distance.

Marks duplicates on the photographs table itself:
  is_canonical = False, canonical_photo_id = <best version's photo_id>

Usage:
    from skills.deduplicate import deduplicate
    result = deduplicate("82cb9d7e-...")
"""

import logging
from collections import defaultdict

from skills.contract import (
    SkillResult, get_substrate,
    record_run, record_step, complete_step, complete_run,
)

logger = logging.getLogger(__name__)

# pHash Hamming distance threshold for near-duplicate grouping
HAMMING_THRESHOLD = 8


def _hamming_distance(h1, h2):
    """Compute Hamming distance between two hex-encoded pHash strings."""
    if not h1 or not h2 or len(h1) != len(h2):
        return 64  # Max distance
    try:
        i1 = int(h1, 16)
        i2 = int(h2, 16)
        return bin(i1 ^ i2).count("1")
    except ValueError:
        return 64


def deduplicate(
    property_id: str,
    *,
    force: bool = False,
) -> SkillResult:
    """Find near-duplicate photographs and mark canonical relationships.

    Uses Union-Find on pHash Hamming distance <= HAMMING_THRESHOLD.
    The photograph with the highest resolution in each cluster becomes
    canonical; others get is_canonical=False + canonical_photo_id set.
    """
    try:
        sb = get_substrate()
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    # ── Load photographs with pHash ──────────────────────────────────────
    photos_resp = sb.table("photographs").select(
        "photo_id, phash, image_width, image_height, is_canonical"
    ).eq("property_id", property_id).execute()
    photos = photos_resp.data or []

    if not photos:
        return SkillResult.failed(
            reason=f"No photographs for property {property_id}",
            attempted=0, succeeded=0, failed_count=0,
        )

    # Check if already deduped (any non-canonical photos exist)
    if not force:
        non_canonical = sum(1 for p in photos if not p.get("is_canonical", True))
        if non_canonical > 0:
            return SkillResult.noop(
                f"Already deduped ({non_canonical} non-canonical). Use force=True to re-dedupe.",
                {"non_canonical_existing": non_canonical},
            )

    # ── Record run ───────────────────────────────────────────────────────
    run_id = record_run(sb, property_id, "onboard")
    step_id = record_step(sb, run_id, "deduplicate")

    # ── Union-Find clustering ────────────────────────────────────────────
    photos_with_phash = [p for p in photos if p.get("phash")]

    if len(photos_with_phash) < 2:
        complete_step(sb, step_id, status="complete", metadata={
            "clusters": 0, "duplicates_marked": 0,
            "reason": "fewer than 2 photos with pHash",
        })
        complete_run(sb, run_id, status="complete")
        return SkillResult.ok({
            "clusters": 0,
            "duplicates_marked": 0,
            "total_photos": len(photos),
            "photos_with_phash": len(photos_with_phash),
            "run_id": run_id,
        })

    # Union-Find
    parent = {p["photo_id"]: p["photo_id"] for p in photos_with_phash}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Compare all pairs
    for i in range(len(photos_with_phash)):
        for j in range(i + 1, len(photos_with_phash)):
            p1 = photos_with_phash[i]
            p2 = photos_with_phash[j]
            dist = _hamming_distance(p1["phash"], p2["phash"])
            if dist <= HAMMING_THRESHOLD:
                union(p1["photo_id"], p2["photo_id"])

    # Group by cluster
    clusters = defaultdict(list)
    for p in photos_with_phash:
        clusters[find(p["photo_id"])].append(p)

    # ── Mark canonical/duplicate ─────────────────────────────────────────
    duplicates_marked = 0
    multi_clusters = 0

    for cluster_root, members in clusters.items():
        if len(members) <= 1:
            continue
        multi_clusters += 1

        # Best = highest resolution
        def _resolution(p):
            w = p.get("image_width") or 0
            h = p.get("image_height") or 0
            return w * h

        members.sort(key=_resolution, reverse=True)
        canonical = members[0]

        for member in members:
            if member["photo_id"] == canonical["photo_id"]:
                sb.table("photographs").update({
                    "is_canonical": True,
                    "canonical_photo_id": None,
                }).eq("photo_id", member["photo_id"]).execute()
            else:
                sb.table("photographs").update({
                    "is_canonical": False,
                    "canonical_photo_id": canonical["photo_id"],
                }).eq("photo_id", member["photo_id"]).execute()
                duplicates_marked += 1

    # ── Complete ─────────────────────────────────────────────────────────
    complete_step(sb, step_id, status="complete", metadata={
        "total_photos": len(photos),
        "photos_with_phash": len(photos_with_phash),
        "clusters": multi_clusters,
        "duplicates_marked": duplicates_marked,
    })
    complete_run(sb, run_id, status="complete")

    logger.info(
        "[deduplicate] Property %s: %d photos, %d with phash, %d clusters, %d duplicates marked",
        property_id[:12], len(photos), len(photos_with_phash), multi_clusters, duplicates_marked,
    )

    return SkillResult.ok({
        "total_photos": len(photos),
        "photos_with_phash": len(photos_with_phash),
        "clusters": multi_clusters,
        "duplicates_marked": duplicates_marked,
        "run_id": run_id,
    })
