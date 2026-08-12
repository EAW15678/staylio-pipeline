"""
Skill: ingest_intake — read portal intake answers into substrate entities.

Reads intake_answers + intake_sessions from the substrate (must be
backfilled first). Writes: properties fields, guest_evidence rows.

Guest book normalisation from 8c775c7 ports intact:
  Portal sends {name, date, text, verbal}
  Legacy sends {guest_name, stay_date, review_text}
  Both accepted. written_text and verbal_text stored SEPARATELY.

Usage:
    from skills.ingest_intake import ingest_intake
    result = ingest_intake("82cb9d7e-...")
"""

import json
import logging
import os
from datetime import datetime, timezone

from skills.contract import (
    SkillResult, get_substrate,
    record_run, record_step, complete_step, complete_run,
)

logger = logging.getLogger(__name__)

# Vibe mapping: portal short code → pipeline enum (from staylio-web actions.ts)
VIBE_MAP = {
    "romantic": "romantic_escape",
    "family": "family_adventure",
    "multigen": "multigenerational_retreat",
    "wellness": "wellness_retreat",
    "adventure": "adventure_base_camp",
    "social": "social_celebrations",
    "creative": "creative_remote_work",
}


def ingest_intake(
    property_id: str,
    *,
    force: bool = False,
) -> SkillResult:
    """Read intake answers and populate properties + guest_evidence.

    Idempotent: re-run updates properties deterministically and does
    not create duplicate guest_evidence rows (checks existing count).
    """
    try:
        sb = get_substrate()
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    # ── Load intake answers ──────────────────────────────────────────────
    # Try intake_answers first (portal path)
    answers_resp = sb.table("intake_answers").select(
        "question_key, answer_json"
    ).eq("property_id", property_id).execute()

    answers = {}
    for a in (answers_resp.data or []):
        answers[a["question_key"]] = a["answer_json"]

    if not answers:
        return SkillResult.failed(
            reason=f"No intake answers found for property {property_id}",
            attempted=0, succeeded=0, failed_count=0,
        )

    # ── Record run ───────────────────────────────────────────────────────
    run_id = record_run(sb, property_id, "onboard")
    step_id = record_step(sb, run_id, "ingest_intake")

    # ── Extract fields from answers ──────────────────────────────────────
    def _str(key):
        v = answers.get(key)
        if isinstance(v, str):
            return v.strip() or None
        return None

    def _list(key):
        v = answers.get(key)
        if isinstance(v, list):
            return v
        return []

    # Resolve vibe profile
    vibe_raw = _str("vibe_profile") or ""
    vibe_profile = VIBE_MAP.get(vibe_raw, vibe_raw)

    # ── Update properties ────────────────────────────────────────────────
    prop_update = {}
    if _str("property_name"):
        prop_update["name"] = _str("property_name")
    if _str("city"):
        prop_update["city"] = _str("city")
    if _str("state"):
        prop_update["state_region"] = _str("state")
    if _str("zip_code"):
        prop_update["postal_code"] = _str("zip_code")
    if vibe_profile:
        prop_update["vibe_profile"] = vibe_profile
    if _str("booking_url"):
        prop_update["booking_url"] = _str("booking_url")
    if _str("ical_url"):
        prop_update["ical_url"] = _str("ical_url")
    if _str("pmc_website_url"):
        prop_update["primary_listing_url"] = _str("pmc_website_url")

    # Listing URLs
    listing_urls = _list("listing_urls")
    if isinstance(listing_urls, list):
        for url_entry in listing_urls:
            url = url_entry if isinstance(url_entry, str) else (url_entry.get("url") if isinstance(url_entry, dict) else None)
            if url:
                url_lower = url.lower()
                if "airbnb" in url_lower:
                    prop_update["airbnb_url"] = url
                elif "vrbo" in url_lower or "vacationrentals" in url_lower:
                    prop_update["vrbo_url"] = url

    # Geocode city/state → lat/lng if coordinates are missing
    city = prop_update.get("city") or _str("city")
    state = prop_update.get("state_region") or _str("state")
    if city and state:
        # Check if lat/lng already set
        current_prop = sb.table("properties").select("latitude, longitude").eq("id", property_id).limit(1).execute()
        has_coords = current_prop.data and current_prop.data[0].get("latitude")
        if not has_coords:
            try:
                import httpx
                geocode_resp = httpx.get(
                    "https://maps.googleapis.com/maps/api/geocode/json",
                    params={"address": f"{city}, {state}", "key": os.environ.get("GOOGLE_PLACES_API_KEY", "")},
                    timeout=10,
                )
                geocode_data = geocode_resp.json()
                if geocode_data.get("results"):
                    loc = geocode_data["results"][0]["geometry"]["location"]
                    prop_update["latitude"] = loc["lat"]
                    prop_update["longitude"] = loc["lng"]
                    logger.info("[ingest_intake] Geocoded %s, %s → %.4f, %.4f", city, state, loc["lat"], loc["lng"])
            except Exception as exc:
                logger.warning("[ingest_intake] Geocoding failed: %s", str(exc)[:80])

    if prop_update:
        sb.table("properties").update(prop_update).eq("id", property_id).execute()
        logger.info("[ingest_intake] Updated %d property fields", len(prop_update))

    # ── Guest evidence ───────────────────────────────────────────────────
    # Check existing to avoid duplicates
    existing_ev = sb.table("guest_evidence").select("evidence_id", count="exact").eq(
        "property_id", property_id
    ).eq("is_guest_book", True).limit(0).execute()

    evidence_written = 0
    if existing_ev.count == 0 or force:
        # Delete existing guest_book entries on force
        if force and existing_ev.count > 0:
            sb.table("guest_evidence").delete().eq(
                "property_id", property_id
            ).eq("is_guest_book", True).execute()

        entries = _list("guest_book_entries")
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            # Portal keys: {name, date, text, verbal}
            # Legacy keys: {guest_name, stay_date, review_text}
            written = (entry.get("text") or entry.get("review_text") or "").strip()
            verbal = (entry.get("verbal") or "").strip()
            name = entry.get("name") or entry.get("guest_name")
            date = entry.get("date") or entry.get("stay_date")

            if not written and not verbal:
                continue

            sb.table("guest_evidence").insert({
                "property_id": property_id,
                "written_text": written,
                "verbal_text": verbal,
                "reviewer_name": name,
                "stay_date": date,
                "source": "intake_portal",
                "is_guest_book": True,
            }).execute()
            evidence_written += 1
    else:
        logger.info("[ingest_intake] Guest evidence already exists (%d rows), skipping", existing_ev.count)

    # ── Property facts (amenities, bedrooms from intake) ─────────────────
    amenities = _list("amenities")
    if amenities:
        prop_facts = {"amenities": amenities}
        sb.table("properties").update(prop_facts).eq("id", property_id).execute()
        logger.info("[ingest_intake] Set %d amenities on properties", len(amenities))

    # ── Owner context (all 26 narrative/seasonal/guest keys) ─────────────
    existing_ctx = sb.table("owner_context").select("context_id", count="exact").eq(
        "property_id", property_id
    ).is_("superseded_at", "null").limit(0).execute()

    context_written = False
    if existing_ctx.count == 0 or force:
        if force and existing_ctx.count > 0:
            from datetime import datetime, timezone as tz
            sb.table("owner_context").update(
                {"superseded_at": datetime.now(tz.utc).isoformat()}
            ).eq("property_id", property_id).is_("superseded_at", "null").execute()

        ctx = {
            "property_id": property_id,
            "owner_story": _str("owner_story"),
            "wow_factor": _str("wow_factor"),
            "hidden_gems": _str("hidden_gems"),
            "guest_love": _str("guest_love"),
            "arrival_info": _str("arrival_info"),
            "extra_notes": _str("extra_notes"),
            "seasonal_notes": _str("seasonal_notes"),
            "best_seasons": _list("best_seasons"),
            "local_events": _str("local_events"),
            "off_season_notes": _str("off_season_notes"),
            "area_vibe": _str("area_vibe"),
            "dont_miss": _str("dont_miss"),
            "surround_areas": _str("surround_areas"),
            "guest_types": _list("guest_types"),
            "group_size": _str("group_size"),
            "stay_length": _str("stay_length"),
            "host_style": _str("host_style"),
            "owner_years": _str("owner_years"),
            "vibe_q1_answer": _str("vibe_q1_answer"),
            "vibe_q2_answer": _str("vibe_q2_answer"),
            "vibe_notes": _str("vibe_notes"),
        }
        # Only write if there's actual content
        has_content = any(v for k, v in ctx.items() if k != "property_id" and v)
        if has_content:
            sb.table("owner_context").insert(ctx).execute()
            context_written = True
            logger.info("[ingest_intake] Owner context written (%d non-null fields)",
                       sum(1 for k, v in ctx.items() if k != "property_id" and v))
    else:
        logger.info("[ingest_intake] Owner context already exists, skipping")

    # ── Complete ─────────────────────────────────────────────────────────
    complete_step(sb, step_id, status="complete", metadata={
        "fields_updated": len(prop_update),
        "evidence_written": evidence_written,
        "answers_count": len(answers),
    })
    complete_run(sb, run_id, status="complete")

    return SkillResult.ok({
        "fields_updated": len(prop_update),
        "evidence_written": evidence_written,
        "answers_count": len(answers),
        "vibe_profile": vibe_profile,
        "run_id": run_id,
    })
