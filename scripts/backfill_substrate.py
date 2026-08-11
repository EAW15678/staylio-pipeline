"""
Backfill bridge — resolves production data into the substrate schema.

Reads from PRODUCTION (read-only), writes to STAGING only.

Resolution chain for photographs:
  media_assets.source_phash → source_assets.phash → source_assets.content_hash
  → uuid5(PHOTO_NAMESPACE, content_hash) = photo_id

For the 2 unresolved photos (no source_phash): downloads from R2,
computes SHA-256 content_hash from the bytes.

Usage:
  python scripts/backfill_substrate.py
  python scripts/backfill_substrate.py --property-id a1b2c3d4-...

Env vars required:
  SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY  (production, READ-ONLY)
  STAGING_SUPABASE_URL, STAGING_SUPABASE_KEY  (staging, WRITES)
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone

import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("backfill")

PHOTO_NAMESPACE = uuid.UUID("a8f3c2b1-4d5e-6f7a-8b9c-0d1e2f3a4b5c")
RUN_ID = str(uuid.uuid4())


def _photo_id(content_hash: str) -> str:
    return str(uuid.uuid5(PHOTO_NAMESPACE, content_hash))


def _compute_quality_tier(w, h):
    if w and h:
        mp = (w * h) / 1_000_000
        if mp >= 2.0:
            return "high"
        elif mp >= 0.5:
            return "medium"
        else:
            return "low"
    return None


def _resolve_unresolved_photo(url: str) -> dict:
    """Download an image from R2 and compute its content_hash."""
    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        content_hash = hashlib.sha256(resp.content).hexdigest()
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(resp.content))
        return {
            "content_hash": content_hash,
            "width": img.width,
            "height": img.height,
            "resolved": True,
        }
    except Exception as exc:
        logger.warning("Could not resolve %s: %s", url[:60], exc)
        return {"resolved": False}


def backfill_property(prod_sb, staging_sb, property_id: str, steps: list):
    """Backfill one property from production to staging."""
    step_start = datetime.now(timezone.utc)

    # ── Load production data ────────────────────────────────────────────

    # source_assets
    sa_resp = prod_sb.table("source_assets").select(
        "source_asset_id, source_url, content_hash, phash, is_canonical, "
        "canonical_asset_id, source_system, source_origins, "
        "image_width, image_height, quality_tier"
    ).eq("property_id", property_id).execute()
    source_assets = sa_resp.data or []

    # media_assets
    ma_resp = prod_sb.table("media_assets").select("*").eq("property_id", property_id).execute()
    media_assets = ma_resp.data or []

    # Build phash → source_asset lookup
    phash_to_source = {}
    for sa in source_assets:
        p = sa.get("phash")
        if not p:
            continue
        if sa.get("is_canonical") or p not in phash_to_source:
            phash_to_source[p] = sa

    # ── Resolve photographs ─────────────────────────────────────────────

    photos_by_hash = {}  # content_hash → photo dict
    renditions = []  # list of rendition dicts

    for ma in media_assets:
        source_phash = ma.get("source_phash")
        original_url = ma.get("asset_url_original", "")
        enhanced_url = ma.get("asset_url_enhanced")

        content_hash = None
        phash_val = source_phash
        source_row = None
        width = None
        height = None

        if source_phash:
            source_row = phash_to_source.get(source_phash)
            if source_row:
                content_hash = source_row.get("content_hash")
                width = source_row.get("image_width")
                height = source_row.get("image_height")

        # Unresolved: download and compute
        if not content_hash and original_url:
            resolved = _resolve_unresolved_photo(original_url)
            if resolved.get("resolved"):
                content_hash = resolved["content_hash"]
                width = resolved.get("width")
                height = resolved.get("height")
                logger.info("Resolved unresolved photo: %s → %s", original_url[-40:], content_hash[:16])

        if not content_hash:
            logger.warning("Cannot resolve: %s (no phash, download failed)", original_url[-50:])
            continue

        photo_id = _photo_id(content_hash)

        if content_hash not in photos_by_hash:
            source_urls = []
            source_systems = []
            if source_row:
                su = source_row.get("source_url", "")
                if su:
                    source_urls.append(su)
                ss = source_row.get("source_system", "")
                if ss:
                    source_systems.append(ss)

            photos_by_hash[content_hash] = {
                "photo_id": photo_id,
                "property_id": property_id,
                "content_hash": content_hash,
                "phash": phash_val,
                "source_urls": source_urls,
                "source_systems": source_systems,
                "is_canonical": source_row.get("is_canonical", True) if source_row else True,
                "image_width": width,
                "image_height": height,
                "quality_tier": _compute_quality_tier(width, height),
            }

        # Renditions
        if original_url:
            renditions.append({
                "photo_id": photo_id,
                "kind": "original",
                "storage_url": original_url,
                "format": "jpg",
            })
        if enhanced_url:
            renditions.append({
                "photo_id": photo_id,
                "kind": "enhanced",
                "storage_url": enhanced_url,
                "format": "jpg",
            })

    # Deduplicate renditions by (photo_id, kind, format)
    seen_rend = set()
    unique_renditions = []
    for r in renditions:
        key = (r["photo_id"], r["kind"], r.get("format"))
        if key not in seen_rend:
            seen_rend.add(key)
            unique_renditions.append(r)

    # ── Load observations from curations ────────────────────────────────

    cur_resp = prod_sb.table("property_image_curations").select(
        "per_image_results, superseded_at, created_at"
    ).eq("property_id", property_id).eq("status", "complete").execute()
    curations = cur_resp.data or []

    observations = []
    for cur in curations:
        per_image = cur.get("per_image_results") or []
        if isinstance(per_image, str):
            per_image = json.loads(per_image)
        superseded_at = cur.get("superseded_at")

        for img in per_image:
            asset_id = img.get("asset_id", "")
            # Resolve asset_id (R2 URL) → photo_id via the same chain
            ma_match = next((m for m in media_assets if m.get("asset_url_original") == asset_id), None)
            if not ma_match:
                continue
            sp = ma_match.get("source_phash")
            if not sp:
                continue
            sr = phash_to_source.get(sp)
            if not sr or not sr.get("content_hash"):
                continue
            photo_id = _photo_id(sr["content_hash"])

            obs = {
                "property_id": property_id,
                "photo_id": photo_id,
                "observation_version": 1,
                "model": "claude-sonnet-4-6",
                "read_basis": "enhanced" if ma_match.get("asset_url_enhanced") else "original",
                "depth_structure": img.get("depth_structure"),
                "depth_tier": img.get("depth_tier"),
                "space_direction": img.get("space_direction"),
                "light_direction": img.get("light_direction"),
                "light_quality": img.get("light_quality"),
                "time_of_day_read": img.get("time_of_day_read"),
                "negative_space": img.get("negative_space") or [],
                "foreground_elements": img.get("foreground_elements") or [],
                "frame_element": img.get("frame_element"),
                "beyond_frame_element": img.get("beyond_frame_element"),
                "subject_singularity": img.get("subject_singularity"),
                "focal_point": img.get("visual_summary"),
                "located_amenities": img.get("located_amenities") or [],
                "motion_affordance": img.get("motion_affordance") or [],
                "motion_risk": img.get("motion_risk") or [],
                "persistence_manifest": [],
                "role": img.get("role"),
                "curated_section": img.get("curated_section"),
                "quality_score": img.get("quality_score"),
                "alt_text": img.get("alt"),
                "duplicate_group": img.get("duplicate_group"),
                "is_best_in_group": img.get("is_best_in_duplicate_group", True),
                "gallery_visible": img.get("gallery_visible", True),
                "tour_eligible": img.get("tour_eligible", True),
                "superseded_at": superseded_at,
            }
            observations.append(obs)

    # ── Load guest evidence ─────────────────────────────────────────────

    intake_resp = prod_sb.table("intake_answers").select(
        "answer_json"
    ).eq("property_id", property_id).eq("question_key", "guest_book_entries").execute()

    evidence_rows = []
    for row in (intake_resp.data or []):
        entries = row.get("answer_json") or []
        if isinstance(entries, str):
            entries = json.loads(entries)
        for entry in entries:
            evidence_rows.append({
                "property_id": property_id,
                "written_text": (entry.get("text") or "").strip(),
                "verbal_text": (entry.get("verbal") or "").strip(),
                "reviewer_name": entry.get("name"),
                "stay_date": entry.get("date"),
                "source": "intake_portal",
                "is_guest_book": True,
            })

    # Also load OTA reviews from property_knowledge_bases.data
    kb_resp = prod_sb.table("property_knowledge_bases").select("data").eq("property_id", property_id).execute()
    for kb_row in (kb_resp.data or []):
        data = kb_row.get("data") or {}
        if isinstance(data, str):
            data = json.loads(data)
        for rev in (data.get("guest_reviews") or []):
            if rev.get("is_guest_book"):
                continue  # Already from intake_answers
            evidence_rows.append({
                "property_id": property_id,
                "written_text": (rev.get("text") or "").strip(),
                "verbal_text": "",
                "reviewer_name": rev.get("reviewer_name"),
                "stay_date": rev.get("stay_date"),
                "source": rev.get("source", "unknown"),
                "is_guest_book": False,
            })

    # ── Write to staging ────────────────────────────────────────────────

    photo_list = list(photos_by_hash.values())

    # Check property exists in staging
    existing_prop = staging_sb.table("properties").select("id").eq("id", property_id).execute()

    if not existing_prop.data:
        # Copy property from production
        prop_resp = prod_sb.table("properties").select("*").eq("id", property_id).execute()
        if prop_resp.data:
            prop = prop_resp.data[0]
            # Map column names (production has client_id, staging has account_id)
            account_id = prop.get("account_id") or prop.get("client_id")

            # Ensure account exists
            existing_acct = staging_sb.table("accounts").select("id").eq("id", account_id).execute()
            if not existing_acct.data:
                acct_resp = prod_sb.table("accounts").select(
                    "id, name, legal_name, billing_email, primary_contact_name, "
                    "primary_contact_email, status, subscription_status, timezone, "
                    "stripe_customer_id, notes, created_at, updated_at"
                ).eq("id", account_id).execute()
                if acct_resp.data:
                    staging_sb.table("accounts").upsert(acct_resp.data[0], on_conflict="id").execute()
                    logger.info("  Account %s created in staging", account_id[:12])

            staging_sb.table("properties").upsert({
                "id": property_id,
                "account_id": account_id,
                "name": prop.get("name", "Unknown"),
                "slug": prop.get("slug"),
                "vibe_profile": prop.get("vibe_profile"),
                "property_type": prop.get("property_type"),
                "city": prop.get("city"),
                "state_region": prop.get("state_region") or prop.get("state"),
                "latitude": prop.get("latitude"),
                "longitude": prop.get("longitude"),
                "booking_url": prop.get("booking_url"),
                "airbnb_url": prop.get("airbnb_url"),
                "vrbo_url": prop.get("vrbo_url"),
                "status": prop.get("status", "active"),
            }, on_conflict="id").execute()
            logger.info("  Property %s created in staging", property_id[:12])

    # Photographs — upsert on content_hash (idempotent)
    photo_inserted = 0
    for photo in photo_list:
        try:
            staging_sb.table("photographs").upsert(
                photo, on_conflict="content_hash"
            ).execute()
            photo_inserted += 1
        except Exception as e:
            logger.warning("  Photo insert failed: %s", str(e)[:80])

    # Renditions — upsert on (photo_id, kind, format)
    rend_inserted = 0
    for rend in unique_renditions:
        try:
            staging_sb.table("renditions").upsert(
                rend, on_conflict="photo_id,kind,format"
            ).execute()
            rend_inserted += 1
        except Exception as e:
            logger.warning("  Rendition insert failed: %s", str(e)[:80])

    # Observations — insert (no upsert key that works well with superseded_at)
    obs_inserted = 0
    # Check if observations already exist for this property
    existing_obs = staging_sb.table("observations").select("observation_id", count="exact").eq("property_id", property_id).limit(0).execute()
    if existing_obs.count == 0:
        for obs in observations:
            try:
                staging_sb.table("observations").insert(obs).execute()
                obs_inserted += 1
            except Exception as e:
                logger.warning("  Observation insert failed: %s", str(e)[:80])
    else:
        logger.info("  Observations already exist (%d rows), skipping", existing_obs.count)

    # Guest evidence — check existing before insert
    ev_inserted = 0
    existing_ev = staging_sb.table("guest_evidence").select("evidence_id", count="exact").eq("property_id", property_id).limit(0).execute()
    if existing_ev.count == 0:
        for ev in evidence_rows:
            try:
                staging_sb.table("guest_evidence").insert(ev).execute()
                ev_inserted += 1
            except Exception as e:
                logger.warning("  Evidence insert failed: %s", str(e)[:80])
    else:
        logger.info("  Guest evidence already exists (%d rows), skipping", existing_ev.count)

    step_end = datetime.now(timezone.utc)
    steps.append({
        "run_id": RUN_ID,
        "step_name": f"backfill_property_{property_id[:8]}",
        "status": "complete",
        "started_at": step_start.isoformat(),
        "completed_at": step_end.isoformat(),
        "metadata": json.dumps({
            "photos": photo_inserted,
            "renditions": rend_inserted,
            "observations": obs_inserted,
            "evidence": ev_inserted,
        }),
    })

    logger.info(
        "  Property %s: %d photos, %d renditions, %d observations, %d evidence",
        property_id[:12], photo_inserted, rend_inserted, obs_inserted, ev_inserted,
    )
    return photo_inserted, rend_inserted, obs_inserted, ev_inserted


def backfill_visitor_data(prod_sb, staging_sb, steps: list):
    """Backfill visitor_sessions, page_events, cta_clicks, bookings, etc."""
    step_start = datetime.now(timezone.utc)
    counts = {}

    for table in ["visitor_sessions", "page_events", "cta_clicks", "bookings",
                   "booking_reports", "attribution_links"]:
        existing = staging_sb.table(table).select("id", count="exact").limit(0).execute()
        if existing.count > 0:
            logger.info("  %s: already has %d rows, skipping", table, existing.count)
            counts[table] = 0
            continue

        # Select only columns that exist in both production and staging
        prod_data = prod_sb.table(table).select("*").execute()
        rows = prod_data.data or []
        # Get staging column names to filter
        try:
            staging_cols_resp = staging_sb.table(table).select("*").limit(0).execute()
        except Exception:
            staging_cols_resp = None

        inserted = 0
        for row in rows:
            # Filter to only columns staging accepts
            filtered = {k: v for k, v in row.items()
                       if k not in ("client_id", "client_type", "external_property_ref")}
            try:
                staging_sb.table(table).insert(filtered).execute()
                inserted += 1
            except Exception as e:
                if "PGRST204" in str(e):
                    # Column mismatch — try stripping the offending column
                    col_name = str(e).split("'")[1] if "'" in str(e) else "unknown"
                    filtered.pop(col_name, None)
                    try:
                        staging_sb.table(table).insert(filtered).execute()
                        inserted += 1
                    except Exception as e2:
                        logger.warning("  %s insert retry failed: %s", table, str(e2)[:80])
                else:
                    logger.warning("  %s insert failed: %s", table, str(e)[:80])
        counts[table] = inserted
        logger.info("  %s: %d rows copied", table, inserted)

    step_end = datetime.now(timezone.utc)
    steps.append({
        "run_id": RUN_ID,
        "step_name": "backfill_visitor_data",
        "status": "complete",
        "started_at": step_start.isoformat(),
        "completed_at": step_end.isoformat(),
        "metadata": json.dumps(counts),
    })
    return counts


def main():
    parser = argparse.ArgumentParser(description="Backfill substrate schema")
    parser.add_argument("--property-id", type=str, default=None)
    args = parser.parse_args()

    prod_url = os.environ.get("SUPABASE_URL")
    prod_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    staging_url = os.environ.get("STAGING_SUPABASE_URL")
    staging_key = os.environ.get("STAGING_SUPABASE_KEY")

    if not all([prod_url, prod_key, staging_url, staging_key]):
        logger.error("Need SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, STAGING_SUPABASE_URL, STAGING_SUPABASE_KEY")
        sys.exit(1)

    from supabase import create_client
    prod_sb = create_client(prod_url, prod_key)
    staging_sb = create_client(staging_url, staging_key)

    run_start = datetime.now(timezone.utc)
    steps = []

    # Get properties
    if args.property_id:
        property_ids = [args.property_id]
    else:
        props = prod_sb.table("properties").select("id").execute()
        property_ids = [r["id"] for r in (props.data or [])]

    logger.info("Backfilling %d properties, run_id=%s", len(property_ids), RUN_ID[:12])

    totals = {"photos": 0, "renditions": 0, "observations": 0, "evidence": 0}
    for pid in property_ids:
        p, r, o, e = backfill_property(prod_sb, staging_sb, pid, steps)
        totals["photos"] += p
        totals["renditions"] += r
        totals["observations"] += o
        totals["evidence"] += e

    # Visitor/booking data
    visitor_counts = backfill_visitor_data(prod_sb, staging_sb, steps)

    # Record the run itself
    run_end = datetime.now(timezone.utc)
    try:
        staging_sb.table("runs").insert({
            "run_id": RUN_ID,
            "property_id": property_ids[0] if len(property_ids) == 1 else property_ids[0],
            "workflow": "onboard",
            "status": "complete",
            "started_at": run_start.isoformat(),
            "completed_at": run_end.isoformat(),
            "metadata": json.dumps({
                "type": "backfill",
                "properties": len(property_ids),
                "totals": totals,
                "visitor_data": visitor_counts,
            }),
        }).execute()

        for step in steps:
            staging_sb.table("run_steps").insert(step).execute()

        logger.info("Run recorded: %s (%d steps)", RUN_ID[:12], len(steps))
    except Exception as exc:
        logger.warning("Could not record run: %s", exc)

    logger.info(
        "BACKFILL COMPLETE: %d properties, %d photos, %d renditions, %d observations, %d evidence",
        len(property_ids), totals["photos"], totals["renditions"],
        totals["observations"], totals["evidence"],
    )


if __name__ == "__main__":
    main()
