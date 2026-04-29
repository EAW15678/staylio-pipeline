"""
agents/agent1/phash_dedupe.py

pHash-based visual deduplication for source_assets.

Phase 1 (write-only): Agent 1 computes pHash + image dimensions during
ingestion and stores them on source_assets rows, then re-clusters all rows
for the property to update is_canonical / canonical_asset_id / source_origins
based on visual similarity and image quality.

Agent 3 is NOT modified in Phase 1 — it continues to receive the full
kb.photos list exactly as today.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Hamming distance ≤ 8 → same visual scene (validated by pHash sweep: T*=2..16 flat)
PHASH_HAMMING_THRESHOLD = 8

# Source tiebreak within a pHash cluster — used only when quality_tier and
# host_priority are equal. Lower = higher priority.
_SOURCE_TIEBREAK: dict[str, int] = {
    "pmc_website":   0,
    "vrbo":          1,
    "airbnb":        2,
    "intake_portal": 3,
    "booking_com":   4,
    "claude_parsed": 5,
}


@dataclass
class PhashDedupeResult:
    total_assets: int
    hashed_count: int
    hash_failures: int
    cluster_count: int
    canonical_count: int
    duplicate_count: int
    updates_applied: int
    updates_skipped: int
    backfill_needed: int
    backfill_succeeded: int
    backfill_failed: int


# ── Union-Find ────────────────────────────────────────────────────────────────

class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hex_hamming(h1: str, h2: str) -> int:
    """Hamming distance between two 16-char lowercase hex pHash strings (64-bit)."""
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


def _sources_match(desired: Optional[list], current: Optional[list]) -> bool:
    """
    Compare source_origins for idempotency check.
    Both must be None to match as null — None != [] to avoid false-positive skips.
    """
    if desired is None and current is None:
        return True
    if desired is None or current is None:
        return False
    return sorted(desired) == sorted(current)


def _is_noop(payload: dict, row: dict) -> bool:
    """Return True if every field in payload already matches the current row value."""
    for field, desired in payload.items():
        current = row.get(field)
        if field == "source_origins":
            if not _sources_match(desired, current):
                return False
        else:
            if desired != current:
                return False
    return True


# ── Public API ────────────────────────────────────────────────────────────────

def compute_phash_and_dimensions(
    image_bytes: bytes,
) -> tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Open the image once with PIL, compute pHash and capture dimensions.
    Returns (phash_hex, width, height) — any element None on any error.
    Never raises.
    """
    try:
        import io
        import imagehash
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = img.size
        h = imagehash.phash(img, hash_size=8)
        return str(h).lower().zfill(16), width, height
    except Exception as exc:
        logger.warning("[Agent 1] compute_phash_and_dimensions failed: %s", exc)
        return None, None, None


def compute_phash(image_bytes: bytes) -> Optional[str]:
    """Thin wrapper — returns just the hash string (or None)."""
    return compute_phash_and_dimensions(image_bytes)[0]


def compute_quality_tier(width: int, height: int) -> str:
    """
    Tier the image by megapixel count:
      high   — >= 2.0 MP  (e.g. 1920×1080 = 2.07 MP and up)
      medium — 0.5–2.0 MP (640×480 to ~1920×1040)
      low    — < 0.5 MP   (anything below ~720×480)
    """
    mp = (width * height) / 1_000_000
    if mp >= 2.0:
        return "high"
    if mp >= 0.5:
        return "medium"
    return "low"


def cluster_and_assign_canonical(property_id: str) -> PhashDedupeResult:
    """
    Re-cluster all source_assets rows for this property based on pHash.
    Pure read-modify-write within source_assets — no other tables touched.

    Steps:
      1. SELECT all rows for the property (includes pHash, dimensions, quality_tier,
         host_priority, source_origins — everything needed for clustering + idempotency).
      2. Backfill: for rows with phash IS NULL, fetch bytes via HTTP and compute
         pHash + dimensions. UPDATE the row and patch the in-memory dict.
      3. Union-Find single-linkage clustering at Hamming ≤ PHASH_HAMMING_THRESHOLD.
      4. Within each cluster, select canonical by:
           quality_tier rank (high=0 > medium=1 > low=2 > None=3)
           → host_priority (True=0 > False=1)
           → source_system tiebreak (_SOURCE_TIEBREAK)
           → lowest source_asset_id (UUID string, deterministic)
      5. Apply updates with idempotency guard — skip UPDATEs where all desired
         fields already match current values.

    Idempotent: calling twice with the same data produces identical DB state
    (second call skips all UPDATEs).

    Canonical row shape after cluster_and_assign:
        is_canonical:       True
        canonical_asset_id: None
        source_origins:     ["airbnb", "intake_portal", "vrbo"]  ← sorted unique source_systems
        host_priority:      True   ← inherited if ANY cluster member had host_priority=True
        phash:              "f8e0c0c0c0e0f8f8"
        quality_tier:       "high"

    Non-canonical row shape:
        is_canonical:       False
        canonical_asset_id: "f47ac10b-..."        ← points to canonical
        source_origins:     None
        phash:              "f8e0c0c0c0e0f8f8"    ← same visual scene, different URL
        quality_tier:       "medium"
        host_priority:      False
    """
    import httpx
    from core.supabase_store import get_supabase

    supabase = get_supabase()

    # 1. Fetch all source_assets for this property
    result = (
        supabase.table("source_assets")
        .select(
            "source_asset_id,phash,source_system,source_url,"
            "is_canonical,canonical_asset_id,source_origins,"
            "quality_tier,host_priority"
        )
        .eq("property_id", str(property_id))
        .execute()
    )
    rows = result.data or []
    total = len(rows)

    if total == 0:
        logger.info(
            "[Agent 1] pHash dedupe: property=%s total=0 — nothing to cluster.",
            property_id,
        )
        return PhashDedupeResult(
            total_assets=0, hashed_count=0, hash_failures=0,
            cluster_count=0, canonical_count=0, duplicate_count=0,
            updates_applied=0, updates_skipped=0,
            backfill_needed=0, backfill_succeeded=0, backfill_failed=0,
        )

    # Mutable index so backfill can patch rows in-place
    rows_by_id = {r["source_asset_id"]: r for r in rows}

    # 2. Backfill: compute pHash for rows that missed it during ingestion
    needs_backfill = [r for r in rows if not r.get("phash")]
    backfill_needed    = len(needs_backfill)
    backfill_succeeded = 0
    backfill_failed    = 0

    if needs_backfill:
        now_iso = datetime.now(timezone.utc).isoformat()
        with httpx.Client(timeout=15, follow_redirects=True,
                          headers={"User-Agent": "StaylioPhashBackfill/1.0"}) as client:
            for row in needs_backfill:
                url = row.get("source_url")
                if not url:
                    backfill_failed += 1
                    continue
                try:
                    resp = client.get(url)
                    if resp.status_code != 200:
                        logger.warning(
                            "[Agent 1] pHash backfill: HTTP %d for %s — skipping.",
                            resp.status_code, url[-60:],
                        )
                        backfill_failed += 1
                        continue

                    phash_hex, img_w, img_h = compute_phash_and_dimensions(resp.content)
                    if phash_hex is None:
                        backfill_failed += 1
                        continue

                    quality_tier = compute_quality_tier(img_w, img_h) if img_w and img_h else None
                    host_priority = (row["source_system"] == "intake_portal")

                    backfill_payload = {
                        "phash":             phash_hex,
                        "phash_computed_at": now_iso,
                        "image_width":       img_w,
                        "image_height":      img_h,
                        "quality_tier":      quality_tier,
                        "host_priority":     host_priority,
                    }
                    supabase.table("source_assets").update(backfill_payload).eq(
                        "source_asset_id", row["source_asset_id"]
                    ).execute()

                    # Patch in-memory row so Union-Find sees the new pHash
                    rows_by_id[row["source_asset_id"]].update(backfill_payload)
                    backfill_succeeded += 1

                except Exception as exc:
                    logger.warning(
                        "[Agent 1] pHash backfill failed for %s: %s",
                        url[-60:] if url else "<no url>", exc,
                    )
                    backfill_failed += 1

    logger.info(
        "[Agent 1] pHash backfill: property=%s needed=%d succeeded=%d failed=%d",
        property_id, backfill_needed, backfill_succeeded, backfill_failed,
    )

    # 3. Split using potentially-updated in-memory state
    all_rows    = list(rows_by_id.values())
    hashed_rows  = [r for r in all_rows if r.get("phash")]
    unhashed_rows = [r for r in all_rows if not r.get("phash")]
    hashed_count  = len(hashed_rows)
    hash_failures = len(unhashed_rows)

    # 4. Union-Find single-linkage clustering on hashed rows
    n  = len(hashed_rows)
    uf = _UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            d = _hex_hamming(hashed_rows[i]["phash"], hashed_rows[j]["phash"])
            if d <= PHASH_HAMMING_THRESHOLD:
                uf.union(i, j)

    # Group by cluster root
    cluster_map: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        cluster_map[uf.find(i)].append(i)

    # Total cluster count: pHash clusters + one singleton per unhashed row
    cluster_count = len(cluster_map) + hash_failures

    # 5. Determine canonical for each cluster and build update lists
    canonical_updates: list[tuple[str, dict]] = []  # (asset_id, payload)
    duplicate_updates: list[tuple[str, dict]] = []

    def _priority_key(row_idx: int) -> tuple:
        r = hashed_rows[row_idx]
        tier_rank   = {"high": 0, "medium": 1, "low": 2, None: 3}.get(r.get("quality_tier"), 3)
        host_rank   = 0 if r.get("host_priority") else 1
        source_rank = _SOURCE_TIEBREAK.get(r["source_system"], 99)
        return (tier_rank, host_rank, source_rank, r["source_asset_id"])

    for members in cluster_map.values():
        members_sorted = sorted(members, key=_priority_key)
        canonical_row  = hashed_rows[members_sorted[0]]
        canonical_id   = canonical_row["source_asset_id"]

        all_sources           = sorted({hashed_rows[i]["source_system"] for i in members})
        cluster_host_priority = any(bool(hashed_rows[i].get("host_priority")) for i in members)

        canonical_updates.append((
            canonical_id,
            {
                "is_canonical":       True,
                "canonical_asset_id": None,
                "source_origins":     all_sources,
                "host_priority":      cluster_host_priority,
            },
        ))

        for idx in members_sorted[1:]:
            row = hashed_rows[idx]
            duplicate_updates.append((
                row["source_asset_id"],
                {
                    "is_canonical":       False,
                    "canonical_asset_id": canonical_id,
                    "source_origins":     None,
                },
            ))

    # Unhashed rows — singleton canonicals (no pHash grouping possible)
    for row in unhashed_rows:
        canonical_updates.append((
            row["source_asset_id"],
            {
                "is_canonical":       True,
                "canonical_asset_id": None,
                "source_origins":     [row["source_system"]],
            },
        ))

    # 6. Apply updates with idempotency guard
    updates_applied = 0
    updates_skipped = 0

    for asset_id, payload in canonical_updates + duplicate_updates:
        current_row = rows_by_id[asset_id]
        if _is_noop(payload, current_row):
            updates_skipped += 1
            continue
        supabase.table("source_assets").update(payload).eq(
            "source_asset_id", asset_id
        ).execute()
        updates_applied += 1

    canonical_count = len(canonical_updates)
    duplicate_count = len(duplicate_updates)

    logger.info(
        "[Agent 1] pHash dedupe: property=%s total=%d hashed=%d "
        "clusters=%d canonical=%d duplicates=%d hash_failures=%d "
        "updates_applied=%d updates_skipped=%d "
        "backfill_needed=%d backfill_succeeded=%d backfill_failed=%d",
        property_id, total, hashed_count, cluster_count,
        canonical_count, duplicate_count, hash_failures,
        updates_applied, updates_skipped,
        backfill_needed, backfill_succeeded, backfill_failed,
    )

    return PhashDedupeResult(
        total_assets=total,
        hashed_count=hashed_count,
        hash_failures=hash_failures,
        cluster_count=cluster_count,
        canonical_count=canonical_count,
        duplicate_count=duplicate_count,
        updates_applied=updates_applied,
        updates_skipped=updates_skipped,
        backfill_needed=backfill_needed,
        backfill_succeeded=backfill_succeeded,
        backfill_failed=backfill_failed,
    )
