"""
agents/agent3/phash_resolver.py

Resolves canonical pHash values for CDN photo URLs by querying source_assets,
and backfills source_phash on existing media_assets rows.

Called by agent3_node() at the start of every pipeline run (idempotent).
No side effects in resolve_canonical_phashes — backfill writes only to
media_assets.source_phash.
"""

import hashlib
import logging
import os
import re

logger = logging.getLogger(__name__)


def resolve_canonical_phashes(property_id: str, urls: list[str]) -> dict[str, str]:
    """
    Given a list of CDN URLs for a property, return a dict mapping each URL
    to the CANONICAL pHash of its pHash cluster in source_assets.

    Why canonical pHash and not the row's own pHash: pHash dedup uses Hamming
    distance, so two cluster members can have slightly different pHash values.
    The canonical's pHash is the cluster's stable identity for cache purposes.

    Uses at most 2 SELECT queries:
      1. Fetch source_assets rows matching the given URLs
      2. Fetch canonical rows' phashes for any non-canonical rows found

    URLs not in source_assets, or whose cluster has phash IS NULL on both the
    row and its canonical (rare — backfill failure), are omitted. Caller treats
    omission as a cache miss — never a pipeline failure.

    No side effects. Idempotent.
    """
    if not urls:
        return {}

    from core.supabase_store import get_supabase
    supabase = get_supabase()

    # 1. Fetch source_assets rows matching the given CDN URLs
    rows = (
        supabase.table("source_assets")
        .select("source_url,phash,canonical_asset_id,source_asset_id")
        .eq("property_id", property_id)
        .in_("source_url", urls)
        .execute()
        .data or []
    )

    if not rows:
        return {}

    # 2. Collect canonical_asset_ids that need pHash resolution
    #    (non-canonical rows point to a different canonical row)
    canonical_ids_needed = {
        r["canonical_asset_id"]
        for r in rows
        if r.get("canonical_asset_id")
    }

    canonical_phash_by_id: dict[str, str] = {}
    if canonical_ids_needed:
        canonical_rows = (
            supabase.table("source_assets")
            .select("source_asset_id,phash")
            .in_("source_asset_id", list(canonical_ids_needed))
            .execute()
            .data or []
        )
        canonical_phash_by_id = {
            r["source_asset_id"]: r["phash"]
            for r in canonical_rows
            if r.get("phash")
        }

    # 3. Build URL → canonical pHash
    #    COALESCE logic: canonical rows (canonical_asset_id IS NULL) use their own phash;
    #    non-canonical rows follow the FK to get the canonical's phash.
    result: dict[str, str] = {}
    for r in rows:
        source_url = r.get("source_url")
        if not source_url:
            continue
        if r.get("canonical_asset_id") is None:
            # This row IS the canonical — use its own pHash
            ph = r.get("phash")
        else:
            # Non-canonical — follow FK to canonical row's pHash
            ph = canonical_phash_by_id.get(r["canonical_asset_id"])
        if ph:
            result[source_url] = ph

    return result


def backfill_media_assets_phash(property_id: str) -> dict:
    """
    One-time backfill: populate source_phash on existing media_assets rows that
    pre-date the pHash feature (where source_phash IS NULL).

    Bridge strategy: the R2 filename `photo_{NNN}_{md5_8}.jpg` was produced by
    _stable_filename(cdn_url, index) = f"photo_{index:03d}_{md5(url)[:8]}.jpg".
    We reverse the mapping by computing md5(source_url)[:8] for every
    source_assets row and matching it against the 8-char hex in the filename.

    Once matched, we resolve the canonical pHash via the same COALESCE logic
    as resolve_canonical_phashes (no additional DB query — data already fetched).

    Idempotent: the caller guards with a null-count check before calling this,
    so rows that already have source_phash are not re-processed.

    Returns a dict: {checked, matched, updated, unmatched}.
    """
    from core.supabase_store import get_supabase
    supabase = get_supabase()

    # 1. Fetch all source_assets for the property
    sa_rows = (
        supabase.table("source_assets")
        .select("source_asset_id,source_url,phash,canonical_asset_id")
        .eq("property_id", property_id)
        .execute()
        .data or []
    )

    if not sa_rows:
        logger.info(
            "[Agent 3] media_assets backfill: property=%s no source_assets — skipping",
            property_id,
        )
        return {"checked": 0, "matched": 0, "updated": 0, "unmatched": 0}

    # Build canonical pHash lookup (canonical rows only: canonical_asset_id IS NULL)
    canonical_phash_by_id: dict[str, str] = {
        r["source_asset_id"]: r["phash"]
        for r in sa_rows
        if r.get("phash") and r.get("canonical_asset_id") is None
    }

    # Build md5_8 → canonical_phash for every source_assets row
    md5_to_canonical_phash: dict[str, str] = {}
    for r in sa_rows:
        source_url = r.get("source_url")
        if not source_url:
            continue
        md5_8 = hashlib.md5(source_url.encode()).hexdigest()[:8]
        if r.get("canonical_asset_id") is None:
            ph = r.get("phash")                              # IS the canonical
        else:
            ph = canonical_phash_by_id.get(r["canonical_asset_id"])  # follow FK
        if ph:
            md5_to_canonical_phash[md5_8] = ph

    # 2. Fetch media_assets rows with source_phash IS NULL
    ma_rows = (
        supabase.table("media_assets")
        .select("asset_id,asset_url_original")
        .eq("property_id", property_id)
        .is_("source_phash", "null")
        .execute()
        .data or []
    )

    checked  = len(ma_rows)
    matched  = 0
    updated  = 0
    unmatched = 0

    # 3. For each unresolved media_assets row, extract md5_8 from filename and match
    for ma in ma_rows:
        url = ma.get("asset_url_original", "")
        basename = os.path.basename(url.split("?")[0])
        stem = os.path.splitext(basename)[0]
        # Filename pattern: photo_NNN_<md5_8>   (md5_8 is exactly 8 lowercase hex chars)
        m = re.match(r"^photo_\d+_([0-9a-f]{8})$", stem)
        if not m:
            unmatched += 1
            continue
        md5_8 = m.group(1)
        canonical_phash = md5_to_canonical_phash.get(md5_8)
        if not canonical_phash:
            unmatched += 1
            continue
        matched += 1
        supabase.table("media_assets").update(
            {"source_phash": canonical_phash}
        ).eq("asset_id", ma["asset_id"]).execute()
        updated += 1

    logger.info(
        "[Agent 3] media_assets backfill: property=%s checked=%d matched=%d "
        "updated=%d unmatched=%d",
        property_id, checked, matched, updated, unmatched,
    )

    return {"checked": checked, "matched": matched, "updated": updated, "unmatched": unmatched}
