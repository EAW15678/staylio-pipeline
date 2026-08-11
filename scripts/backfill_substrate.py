"""
Backfill bridge — resolves production data into the substrate schema.

This script reads from PRODUCTION (read-only) and resolves the identity
chain that was performed by hand for shot_inventory:
  curation.asset_id (R2 URL) → media_assets.asset_url_original
    → media_assets.source_phash → source_assets.phash
    → source_assets.content_hash → photograph.photo_id (uuid5)

It does NOT write to either database. Dry-run only.

Usage:
  # Dry-run: show resolution counts
  python scripts/backfill_substrate.py --dry-run

  # Dry-run for a specific property
  python scripts/backfill_substrate.py --dry-run --property-id a1b2c3d4-...

Env vars required (production READ-ONLY):
  SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import uuid

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("backfill")

# Namespace for deterministic photo_id: uuid5(NAMESPACE, content_hash)
PHOTO_ID_NAMESPACE = uuid.UUID("a8f3c2b1-4d5e-6f7a-8b9c-0d1e2f3a4b5c")


def resolve_property(sb, property_id: str) -> dict:
    """Resolve all photo identity for one property. READ-ONLY."""

    result = {
        "property_id": property_id,
        "source_assets_total": 0,
        "media_assets_total": 0,
        "resolved_to_content_hash": 0,
        "unresolved_no_phash": 0,
        "unresolved_no_source_match": 0,
        "unique_content_hashes": 0,
        "canonical_photos": 0,
        "renditions_original": 0,
        "renditions_enhanced": 0,
        "photographs": [],  # list of {photo_id, content_hash, phash, source_urls, renditions}
    }

    # Load source_assets: content_hash, phash, source_url
    sa_resp = sb.table("source_assets").select(
        "source_asset_id, source_url, content_hash, phash, is_canonical, canonical_asset_id, source_system"
    ).eq("property_id", property_id).execute()
    source_assets = sa_resp.data or []
    result["source_assets_total"] = len(source_assets)

    # Build phash → source_asset lookup (prefer canonical)
    phash_to_source = {}
    for sa in source_assets:
        p = sa.get("phash")
        if not p:
            continue
        if sa.get("is_canonical") or p not in phash_to_source:
            phash_to_source[p] = sa

    # Load media_assets: asset_url_original, source_phash, asset_url_enhanced
    ma_resp = sb.table("media_assets").select(
        "asset_url_original, asset_url_enhanced, source_phash, subject_category, "
        "category_rank, composition_score, brightness, dominant_colors, "
        "labels_enhanced, labels_original, hero_rank, is_hero, sharpness, "
        "safe_search_pass, provenance_flag, source, source_caption"
    ).eq("property_id", property_id).execute()
    media_assets = ma_resp.data or []
    result["media_assets_total"] = len(media_assets)

    # Resolve each media_asset to a content_hash via phash bridge
    photos_by_hash = {}  # content_hash → photo dict

    for ma in media_assets:
        source_phash = ma.get("source_phash")
        original_url = ma.get("asset_url_original", "")
        enhanced_url = ma.get("asset_url_enhanced")

        if not source_phash:
            result["unresolved_no_phash"] += 1
            continue

        source_row = phash_to_source.get(source_phash)
        if not source_row:
            result["unresolved_no_source_match"] += 1
            continue

        content_hash = source_row.get("content_hash")
        if not content_hash:
            result["unresolved_no_source_match"] += 1
            continue

        result["resolved_to_content_hash"] += 1

        # Deterministic photo_id
        photo_id = str(uuid.uuid5(PHOTO_ID_NAMESPACE, content_hash))

        if content_hash not in photos_by_hash:
            photos_by_hash[content_hash] = {
                "photo_id": photo_id,
                "content_hash": content_hash,
                "phash": source_phash,
                "source_urls": [],
                "source_systems": [],
                "is_canonical": source_row.get("is_canonical", True),
                "renditions": [],
            }

        photo = photos_by_hash[content_hash]

        # Collect source URLs
        source_url_val = source_row.get("source_url", "")
        if source_url_val and source_url_val not in photo["source_urls"]:
            photo["source_urls"].append(source_url_val)
        src_sys = source_row.get("source_system", "")
        if src_sys and src_sys not in photo["source_systems"]:
            photo["source_systems"].append(src_sys)

        # Renditions
        if original_url:
            photo["renditions"].append({
                "kind": "original",
                "storage_url": original_url,
            })
            result["renditions_original"] += 1
        if enhanced_url:
            photo["renditions"].append({
                "kind": "enhanced",
                "storage_url": enhanced_url,
            })
            result["renditions_enhanced"] += 1

    result["unique_content_hashes"] = len(photos_by_hash)
    result["canonical_photos"] = sum(
        1 for p in photos_by_hash.values() if p["is_canonical"]
    )
    result["photographs"] = list(photos_by_hash.values())

    return result


def main():
    parser = argparse.ArgumentParser(description="Backfill substrate schema (dry-run)")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Read-only mode (default, always on)")
    parser.add_argument("--property-id", type=str, default=None,
                        help="Resolve a specific property (default: all)")
    args = parser.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        logger.error("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        sys.exit(1)

    from supabase import create_client
    sb = create_client(url, key)

    # Get properties to resolve
    if args.property_id:
        property_ids = [args.property_id]
    else:
        props = sb.table("properties").select("id").execute()
        property_ids = [r["id"] for r in (props.data or [])]

    logger.info("Resolving %d properties (DRY RUN — no writes)", len(property_ids))

    total_resolved = 0
    total_unresolved = 0
    total_photos = 0

    for pid in property_ids:
        result = resolve_property(sb, pid)
        total_resolved += result["resolved_to_content_hash"]
        total_unresolved += result["unresolved_no_phash"] + result["unresolved_no_source_match"]
        total_photos += result["unique_content_hashes"]

        logger.info(
            "Property %s: source_assets=%d, media_assets=%d, "
            "resolved=%d, unresolved_no_phash=%d, unresolved_no_source_match=%d, "
            "unique_photos=%d, canonical=%d, "
            "renditions_original=%d, renditions_enhanced=%d",
            pid[:12],
            result["source_assets_total"],
            result["media_assets_total"],
            result["resolved_to_content_hash"],
            result["unresolved_no_phash"],
            result["unresolved_no_source_match"],
            result["unique_content_hashes"],
            result["canonical_photos"],
            result["renditions_original"],
            result["renditions_enhanced"],
        )

    logger.info(
        "TOTAL: %d properties, %d resolved, %d unresolved, %d unique photographs",
        len(property_ids), total_resolved, total_unresolved, total_photos,
    )


if __name__ == "__main__":
    main()
