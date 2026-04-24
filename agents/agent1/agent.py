"""
Agent 1 — Content Ingestion Agent
LangGraph Node

Orchestrates all scraping for a single property:
  1. Loads intake submission from Supabase (seeded by the intake portal)
  2. Runs scrapers in parallel based on available URLs:
     - Firecrawl (TS-01): PMC website if pmc_website_url present
     - Apify Airbnb Actor (TS-02/TS-04b): Airbnb URL if present
     - Apify VRBO Actor (TS-03): VRBO URL if present
     - Claude fallback (TS-04c): any unrecognised OTA URL
  3. Merges all data using intake-priority merge policy
  4. Runs normalisation pass (Claude Haiku — fills gaps, dedupes amenities)
  5. Saves completed knowledge base to Supabase
  6. Caches in Redis for parallel downstream agent access
  7. Updates pipeline status throughout

LangGraph integration:
  - Receives a state dict with property_id
  - Returns updated state with knowledge_base_ready=True
  - Downstream agents (2, 3, 4) start in parallel after this node completes
"""

import asyncio
import concurrent.futures
import hashlib
import logging
import uuid as _uuid
from datetime import datetime, timezone
from typing import Any, TypedDict

from agents.agent1.apify_scraper import (
    scrape_ota_listing,
    scrape_airbnb_reviews_only,
    detect_ota_platform,
)
from agents.agent1.claude_parser import (
    parse_unknown_ota,
    normalise_and_fill_gaps,
)
from agents.agent1.firecrawl_scraper import scrape_pmc_website
from core.pipeline_status import (
    PipelineStepStatus,
    cache_knowledge_base,
    update_pipeline_status,
)
from core.supabase_store import load_intake_submission, save_knowledge_base
from models.property import ClientChannel, PropertyKnowledgeBase

logger = logging.getLogger(__name__)

AGENT_NUMBER = 1


# ── LangGraph State ───────────────────────────────────────────────────────

class PipelineState(TypedDict):
    property_id: str
    client_id: str
    knowledge_base_ready: bool
    knowledge_base: dict        # Serialised KB dict, shared with downstream agents via Redis
    errors: list[str]
    # Flags for downstream agents
    agent1_complete: bool
    agent2_ready: bool
    agent3_ready: bool
    agent4_ready: bool


# ── Main Agent Node ───────────────────────────────────────────────────────

def agent1_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node for Agent 1 — Content Ingestion.
    This function is synchronous — LangGraph calls it in the main thread.
    Internal scraping calls run in a ThreadPoolExecutor for parallelism.
    """
    property_id = state["property_id"]
    logger.info(f"[Agent 1] Starting content ingestion for property {property_id}")

    # ── Step 1: Update status to RUNNING ─────────────────────────────────
    update_pipeline_status(property_id, AGENT_NUMBER, PipelineStepStatus.RUNNING)

    # ── Step 2: Load intake submission ────────────────────────────────────
    kb = load_intake_submission(property_id)
    if kb is None:
        error = f"Agent 1: No intake submission found for property {property_id}"
        logger.error(error)
        update_pipeline_status(
            property_id, AGENT_NUMBER,
            PipelineStepStatus.FAILED,
            error_message=error,
        )
        return {
            **state,
            "errors": state.get("errors", []) + [error],
            "knowledge_base_ready": False,
            "agent1_complete": False,
        }

    # ── Step 2.5: Seed asset slots ────────────────────────────────────────
    _seed_asset_slots(property_id)

    # ── Step 3: Run scrapers based on available URLs ───────────────────────
    kb = _run_scrapers(kb)

    # ── Step 3.5: SHA-256 photo source deduplication ──────────────────────
    kb = _dedupe_photos(kb)

    # ── Step 4: Normalisation pass ────────────────────────────────────────
    try:
        kb = normalise_and_fill_gaps(kb)
    except Exception as exc:
        # Normalisation failure is non-fatal
        logger.warning(f"[Agent 1] Normalisation pass failed (non-fatal): {exc}")
        kb.ingestion_errors.append(f"Normalisation pass failed: {exc}")

    # ── Step 5: Generate slug if missing ─────────────────────────────────
    if not kb.slug and kb.name and kb.name.value:
        kb.slug = _generate_slug(kb.name.value)

    # ── Step 6: Mark ingestion complete ──────────────────────────────────
    kb.ingested_at = datetime.now(timezone.utc)
    kb.ingestion_complete = True

    # ── Step 7: Read enrichment before save (save overwrites the row) ────────
    enrichment = _read_enrichment_from_supabase(property_id)

    # ── Step 8: Save to Supabase ──────────────────────────────────────────
    saved = save_knowledge_base(kb)
    if not saved:
        # Save failure — still proceed but flag it
        kb.ingestion_errors.append("Supabase save failed — downstream agents will use Redis cache only")

    # ── Step 9: Cache in Redis for parallel downstream agents ─────────────
    kb_dict = kb.to_dict()
    for field, value in enrichment.items():
        if value:
            kb_dict[field] = value
    if enrichment:
        logger.info(f"[Agent 1] Merged enrichment fields: {list(enrichment.keys())}")

    cache_knowledge_base(property_id, kb_dict)

    # ── Step 10: Update pipeline status ──────────────────────────────────
    update_pipeline_status(
        property_id,
        AGENT_NUMBER,
        PipelineStepStatus.COMPLETE,
        metadata={
            "photos_collected": len(kb.photos),
            "reviews_collected": len(kb.guest_reviews),
            "amenities_collected": len(kb.amenities),
            "sources": kb.ingestion_sources,
            "errors": kb.ingestion_errors,
        },
    )

    logger.info(
        f"[Agent 1] Complete for property {property_id}. "
        f"Photos: {len(kb.photos)}, Reviews: {len(kb.guest_reviews)}, "
        f"Amenities: {len(kb.amenities)}, Sources: {kb.ingestion_sources}. "
        f"Errors: {len(kb.ingestion_errors)}"
    )

    return {
        **state,
        "knowledge_base": kb_dict,
        "knowledge_base_ready": True,
        "errors": state.get("errors", []) + kb.ingestion_errors,
        "agent1_complete": True,
        # Signal to LangGraph that agents 2, 3, 4 can now start in parallel
        "agent2_ready": True,
        "agent3_ready": True,
        "agent4_ready": True,
    }


# ── Scraper Orchestration ─────────────────────────────────────────────────

def _run_scrapers(kb: PropertyKnowledgeBase) -> PropertyKnowledgeBase:
    """
    Determines which scrapers to run based on available URLs,
    then runs them in parallel using a ThreadPoolExecutor.
    """
    tasks: list[tuple[str, callable]] = []

    # Determine scraping jobs
    if kb.pmc_website_url and kb.client_channel == ClientChannel.PMC:
        tasks.append(("firecrawl_pmc", lambda: scrape_pmc_website(kb.pmc_website_url, kb)))

    if kb.airbnb_url:
        if kb.client_channel == ClientChannel.IO:
            # Full property scrape + reviews for IO clients (TS-02)
            tasks.append(("apify_airbnb_io", lambda: scrape_ota_listing(kb.airbnb_url, kb, scrape_reviews=True)))
        else:
            # PMC: reviews only — property data comes from the PMC website (TS-04b)
            tasks.append(("apify_airbnb_reviews", lambda: scrape_airbnb_reviews_only(kb.airbnb_url, kb)))

    if kb.vrbo_url:
        tasks.append(("apify_vrbo", lambda: scrape_ota_listing(kb.vrbo_url, kb, scrape_reviews=True)))

    if kb.booking_com_url:
        platform = detect_ota_platform(kb.booking_com_url)
        if platform == "booking_com":
            tasks.append(("claude_fallback_booking", lambda: parse_unknown_ota(kb.booking_com_url, kb)))

    if not tasks:
        logger.warning(
            f"[Agent 1] No URLs available for property {kb.property_id}. "
            "Will rely on intake data only."
        )
        return kb

    logger.info(f"[Agent 1] Running {len(tasks)} scraper(s): {[t[0] for t in tasks]}")

    # NOTE: Scrapers mutate the same KB object.
    # ThreadPoolExecutor is used for I/O concurrency (network calls)
    # but all mutations must be serialised — scrapers use merge_field()
    # which is safe for concurrent reads but needs a lock for writes.
    # For simplicity at this scale, run sequentially.
    # TODO: At high-volume intake, add threading.Lock around KB mutations
    # and truly parallelise the scraping calls.
    for task_name, task_fn in tasks:
        try:
            logger.debug(f"[Agent 1] Running scraper: {task_name}")
            kb = task_fn()
        except Exception as exc:
            error_msg = f"Scraper '{task_name}' failed: {exc}"
            logger.error(f"[Agent 1] {error_msg}")
            kb.ingestion_errors.append(error_msg)

    return kb


# ── Utilities ─────────────────────────────────────────────────────────────

def _generate_slug(name: str) -> str:
    """Generate a URL-safe slug from a property name."""
    import re
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)       # remove special chars
    slug = re.sub(r"[\s_]+", "-", slug)         # spaces to hyphens
    slug = re.sub(r"-+", "-", slug)             # collapse multiple hyphens
    slug = slug.strip("-")                       # trim leading/trailing hyphens
    return slug[:60]                             # max 60 chars for subdomain safety


def _read_enrichment_from_supabase(property_id: str) -> dict:
    """
    Read owner-supplied enrichment fields from property_knowledge_bases BEFORE
    save_knowledge_base() overwrites the row. Returns a dict of non-empty fields.
    """
    _ENRICHMENT_FIELDS = [
        "owner_story",
        "wow_factor",
        "hidden_gems",
        "guest_reviews",
        "dont_miss_picks",
        "seasonal_notes",
        "area_vibe",
        "surround_areas",
        "vibe_profile",
    ]
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("property_knowledge_bases")
            .select(", ".join(_ENRICHMENT_FIELDS))
            .eq("property_id", property_id)
            .single()
            .execute()
        )
        row = result.data or {}
        return {field: row[field] for field in _ENRICHMENT_FIELDS if row.get(field)}
    except Exception as exc:
        logger.warning(
            f"[Agent 1] Could not read enrichment from property_knowledge_bases for {property_id}: {exc}"
        )
        return {}


# ── Asset Slot Seeding ────────────────────────────────────────────────────

_ASSET_SLOTS = [
    ("hero_video",              "video", "16_9"),
    ("hero_video",              "video", "9_16"),
    ("hero_narration",          "audio", "mp3"),
    ("guest_review_audio_1",    "audio", "mp3"),
    ("guest_review_audio_2",    "audio", "mp3"),
    ("guest_review_audio_3",    "audio", "mp3"),
    ("social_clip_vibe",        "video", "9_16"),
    ("social_clip_vibe",        "video", "16_9"),
    ("social_clip_walkthrough", "video", "9_16"),
    ("social_clip_local",       "video", "9_16"),
    ("social_clip_seasonal",    "video", "9_16"),
    ("page_html",               "html",  "html"),
    ("photo_hero",              "image", "jpg"),
    ("social_crop_square",      "image", "jpg"),
    ("social_crop_vertical",    "image", "jpg"),
    ("social_crop_landscape",   "image", "jpg"),
]


def _seed_asset_slots(property_id: str) -> None:
    """
    Idempotent insert of all 16 asset slots for a property.
    Called once at the start of every onboarding run before scrapers fire.
    Upsert on (property_id, slot_name, format) — existing slots untouched.
    Non-fatal: failure here does not block the pipeline.
    """
    try:
        from core.supabase_store import get_supabase
        rows = [
            {
                "slot_id":       str(_uuid.uuid4()),
                "property_id":   property_id,
                "slot_name":     slot_name,
                "artifact_kind": kind,
                "format":        fmt,
                "status":        "pending",
            }
            for slot_name, kind, fmt in _ASSET_SLOTS
        ]
        get_supabase().table("asset_slots").upsert(
            rows,
            on_conflict="property_id,slot_name,format",
        ).execute()
        logger.info(f"[Agent 1] Seeded {len(rows)} asset slots for property {property_id}")
    except Exception as exc:
        logger.warning(f"[Agent 1] Slot seeding failed (non-fatal): {exc}")


# ── Photo Source Deduplication ────────────────────────────────────────────

def _dedupe_photos(kb) -> object:
    """
    SHA-256 deduplication of photos before expensive Claid processing.
    On first run: fetches raw bytes, hashes, writes to source_assets.
    On repeat runs: pre-queries known URLs to skip unnecessary byte fetches.
    Duplicate photos (same bytes, different URLs) are removed from kb.photos.
    Non-fatal: if anything fails, kb.photos is returned unchanged.
    """
    import httpx
    from core.supabase_store import get_supabase

    if not kb.photos:
        return kb

    try:
        supabase = get_supabase()

        # Pre-query all known source URLs for this property — avoids re-fetching
        # bytes on repeat pipeline runs. One query up front, zero wasted fetches.
        existing_result = (
            supabase.table("source_assets")
            .select("source_url,is_canonical")
            .eq("property_id", kb.property_id)
            .execute()
        )
        existing_urls: dict[str, bool] = {
            row["source_url"]: row["is_canonical"]
            for row in (existing_result.data or [])
        }

        canonical_hashes: dict[str, str] = {}  # content_hash → source_asset_id
        canonical_photos = []
        skipped_known = 0

        with httpx.Client(timeout=30) as client:
            for photo in kb.photos:
                # Fast path: URL already recorded from a previous run
                if photo.url in existing_urls:
                    if existing_urls[photo.url]:
                        canonical_photos.append(photo)  # known canonical — keep
                    else:
                        skipped_known += 1              # known duplicate — drop
                    continue

                # Slow path: new URL — fetch bytes, hash, insert
                try:
                    resp = client.get(photo.url, follow_redirects=True)
                    if resp.status_code != 200:
                        # Can't fetch — keep photo, skip deduplication for it
                        canonical_photos.append(photo)
                        continue

                    content_hash = hashlib.sha256(resp.content).hexdigest()

                    if content_hash in canonical_hashes:
                        # Duplicate of a photo seen this run — record and drop
                        canonical_id = canonical_hashes[content_hash]
                        supabase.table("source_assets").insert({
                            "source_asset_id":    str(_uuid.uuid4()),
                            "property_id":        kb.property_id,
                            "content_hash":       content_hash,
                            "is_canonical":       False,
                            "canonical_asset_id": canonical_id,
                            "source_system":      photo.source if isinstance(photo.source, str) else photo.source.value,
                            "source_url":         photo.url,
                        }).execute()
                        logger.debug(f"[Agent 1] Duplicate photo skipped: {photo.url}")
                    else:
                        # New canonical photo — record and keep
                        asset_id = str(_uuid.uuid4())
                        supabase.table("source_assets").insert({
                            "source_asset_id":    asset_id,
                            "property_id":        kb.property_id,
                            "content_hash":       content_hash,
                            "is_canonical":       True,
                            "canonical_asset_id": None,
                            "source_system":      photo.source if isinstance(photo.source, str) else photo.source.value,
                            "source_url":         photo.url,
                        }).execute()
                        canonical_hashes[content_hash] = asset_id
                        canonical_photos.append(photo)

                except Exception as photo_exc:
                    # Per-photo failure — keep photo, log warning, continue
                    logger.warning(f"[Agent 1] Could not dedupe photo {photo.url}: {photo_exc}")
                    canonical_photos.append(photo)

        removed = len(kb.photos) - len(canonical_photos) - skipped_known
        logger.info(
            f"[Agent 1] Photo dedupe: {len(canonical_photos)} canonical, "
            f"{removed} new duplicates, {skipped_known} known duplicates skipped "
            f"(pre-query cache hit)"
        )
        kb.photos = canonical_photos
        return kb

    except Exception as exc:
        logger.warning(f"[Agent 1] Photo deduplication failed (non-fatal): {exc}")
        return kb


# ── LangGraph Graph Definition ────────────────────────────────────────────
# This section wires Agent 1 into the full pipeline graph.
# The complete graph is defined in booked/pipeline/graph.py
# This preview shows Agent 1's place in the execution order.

"""
Agent execution graph (defined in pipeline/graph.py):

    START
      │
      ▼
  [Agent 1: Content Ingestion]          ← This file
      │
      ├─────────────────┬──────────────────┐
      ▼                 ▼                  ▼
  [Agent 2:         [Agent 3:          [Agent 4:
  Content           Visual Media]      Local Discovery]
  Enhancement]          │                  │
      │                 │                  │
      └────────┬─────────┘                 │
               ▼                           │
           [Agent 5:                       │
           Website Builder] ◄──────────────┘
               │
               ▼
           [Agent 6:
           Social Media]
               │
               ▼
           [Agent 7:
           Analytics]
               │
               ▼
             END
"""
