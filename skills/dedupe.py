"""
Skill: dedupe — pHash-based visual deduplication on photographs.

Clusters photographs by perceptual hash (Hamming ≤ 8), then selects
the canonical within each cluster by:
  1. Megapixels (highest wins — not quality_tier buckets)
  2. Source priority (intake_portal > vrbo > airbnb > pmc)
  3. photo_id (deterministic tiebreak)

Order-independent: the result is the SAME regardless of the order
photographs were ingested. This is enforced by the invariant test.

Operates on the photographs table only. Backfills pHash for any row
that lacks one (downloads original rendition, computes pHash+dimensions).

Usage:
    from skills.dedupe import dedupe
    result = dedupe("82cb9d7e-...")
"""

import io
import logging
from collections import defaultdict
from datetime import datetime, timezone

import httpx

from skills.contract import (
    SkillResult, get_substrate,
    record_run, record_step, complete_step, complete_run,
)

logger = logging.getLogger(__name__)

PHASH_HAMMING_THRESHOLD = 8

# Source priority for tiebreak within a cluster (lower = preferred).
# Only consulted when megapixels are equal.
_SOURCE_PRIORITY = {
    "intake_portal": 0,
    "vrbo":          1,
    "airbnb":        2,
    "pmc_website":   3,
    "pmc":           3,
    "booking_com":   4,
    "claude_parsed": 5,
}


def _hex_hamming(h1: str, h2: str) -> int:
    """Hamming distance between two 16-char hex pHash strings (64-bit)."""
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


def _megapixels(w, h) -> float:
    """Compute megapixels from width/height, defaulting to 0."""
    if w and h:
        return (w * h) / 1_000_000
    return 0.0


def _best_source_rank(source_systems: list) -> int:
    """Return the best (lowest) source priority from a list of source_systems."""
    if not source_systems:
        return 99
    return min(_SOURCE_PRIORITY.get(s, 99) for s in source_systems)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def dedupe(
    property_id: str,
    *,
    force: bool = False,
) -> SkillResult:
    """Cluster photographs by pHash and elect canonicals by resolution.

    Returns SkillResult.ok({clusters, canonicals, duplicates, ...})
    """
    try:
        sb = get_substrate()
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    # ── Load all photographs ────────────────────────────────────────────
    resp = sb.table("photographs").select(
        "photo_id, content_hash, phash, source_systems, is_canonical, "
        "canonical_photo_id, image_width, image_height, quality_tier"
    ).eq("property_id", property_id).execute()
    photos = resp.data or []
    total = len(photos)

    if total == 0:
        return SkillResult.noop("No photographs to deduplicate.", {})

    run_id = record_run(sb, property_id, "onboard")
    step_id = record_step(sb, run_id, "dedupe")

    # ── Backfill pHash for any photos missing it ────────────────────────
    photos_by_id = {p["photo_id"]: p for p in photos}
    needs_backfill = [p for p in photos if not p.get("phash")]
    backfill_ok = 0
    backfill_fail = 0

    if needs_backfill:
        # Get original rendition URLs for backfill
        for photo in needs_backfill:
            pid = photo["photo_id"]
            rend = sb.table("renditions").select("storage_url").eq(
                "photo_id", pid
            ).eq("kind", "original").limit(1).execute()
            if rend.data:
                photo["_backfill_url"] = rend.data[0]["storage_url"]

        with httpx.Client(timeout=20, follow_redirects=True) as client:
            for photo in needs_backfill:
                url = photo.get("_backfill_url")
                if not url:
                    backfill_fail += 1
                    continue
                try:
                    resp = client.get(url)
                    resp.raise_for_status()
                    from agents.agent1.phash_dedupe import compute_phash_and_dimensions
                    phash_hex, w, h = compute_phash_and_dimensions(resp.content)
                    if not phash_hex:
                        backfill_fail += 1
                        continue

                    qt = "high" if _megapixels(w, h) >= 2.0 else ("medium" if _megapixels(w, h) >= 0.5 else "low")
                    sb.table("photographs").update({
                        "phash": phash_hex,
                        "image_width": w,
                        "image_height": h,
                        "quality_tier": qt,
                    }).eq("photo_id", pid).execute()

                    # Patch in-memory
                    photo["phash"] = phash_hex
                    photo["image_width"] = w
                    photo["image_height"] = h
                    photo["quality_tier"] = qt
                    photos_by_id[pid] = photo
                    backfill_ok += 1
                except Exception as exc:
                    logger.warning("[dedupe] Backfill failed %s: %s", pid[:8], str(exc)[:60])
                    backfill_fail += 1

    # ── Separate hashed vs unhashed ─────────────────────────────────────
    all_photos = list(photos_by_id.values())
    hashed = [p for p in all_photos if p.get("phash")]
    unhashed = [p for p in all_photos if not p.get("phash")]

    # ── Union-Find clustering ───────────────────────────────────────────
    n = len(hashed)
    uf = _UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            d = _hex_hamming(hashed[i]["phash"], hashed[j]["phash"])
            if d <= PHASH_HAMMING_THRESHOLD:
                uf.union(i, j)

    cluster_map = defaultdict(list)
    for i in range(n):
        cluster_map[uf.find(i)].append(i)

    # ── Canonical selection: megapixels → source priority → photo_id ────
    def _canonical_key(idx):
        p = hashed[idx]
        mp = _megapixels(p.get("image_width"), p.get("image_height"))
        src_rank = _best_source_rank(p.get("source_systems", []))
        return (-mp, src_rank, p["photo_id"])  # negative mp = highest first

    canonical_count = 0
    duplicate_count = 0
    updates_applied = 0

    for members in cluster_map.values():
        members_sorted = sorted(members, key=_canonical_key)
        canonical = hashed[members_sorted[0]]
        canonical_id = canonical["photo_id"]

        # Merge source_systems across cluster
        all_sources = set()
        for idx in members:
            all_sources.update(hashed[idx].get("source_systems", []))
        merged_sources = sorted(all_sources)

        # Update canonical
        if not canonical.get("is_canonical") or canonical.get("canonical_photo_id") is not None:
            sb.table("photographs").update({
                "is_canonical": True,
                "canonical_photo_id": None,
                "source_systems": merged_sources,
            }).eq("photo_id", canonical_id).execute()
            updates_applied += 1
        elif sorted(canonical.get("source_systems", [])) != merged_sources:
            sb.table("photographs").update({
                "source_systems": merged_sources,
            }).eq("photo_id", canonical_id).execute()
            updates_applied += 1
        canonical_count += 1

        # Update duplicates
        for idx in members_sorted[1:]:
            dup = hashed[idx]
            dup_id = dup["photo_id"]
            if dup.get("is_canonical") or dup.get("canonical_photo_id") != canonical_id:
                sb.table("photographs").update({
                    "is_canonical": False,
                    "canonical_photo_id": canonical_id,
                }).eq("photo_id", dup_id).execute()
                updates_applied += 1
            duplicate_count += 1

    # Unhashed photos stay as singleton canonicals
    for p in unhashed:
        canonical_count += 1

    cluster_count = len(cluster_map) + len(unhashed)

    # ── Source distribution of canonicals ────────────────────────────────
    source_dist = defaultdict(int)
    for members in cluster_map.values():
        members_sorted = sorted(members, key=_canonical_key)
        canonical = hashed[members_sorted[0]]
        for s in canonical.get("source_systems", []):
            source_dist[s] += 1

    complete_step(sb, step_id, status="complete", metadata={
        "total": total,
        "hashed": len(hashed),
        "clusters": cluster_count,
        "canonicals": canonical_count,
        "duplicates": duplicate_count,
        "updates_applied": updates_applied,
        "backfill_ok": backfill_ok,
        "backfill_fail": backfill_fail,
        "source_distribution": dict(source_dist),
    })
    complete_run(sb, run_id, status="complete")

    return SkillResult.ok({
        "total": total,
        "hashed": len(hashed),
        "clusters": cluster_count,
        "canonicals": canonical_count,
        "duplicates": duplicate_count,
        "updates_applied": updates_applied,
        "backfill_ok": backfill_ok,
        "backfill_fail": backfill_fail,
        "source_distribution": dict(source_dist),
        "run_id": run_id,
    })
