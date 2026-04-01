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
import logging
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

    # ── Step 3: Run scrapers based on available URLs ───────────────────────
    kb = _run_scrapers(kb)

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

    # ── Step 7: Save to Supabase ──────────────────────────────────────────
    saved = save_knowledge_base(kb)
    if not saved:
        # Save failure — still proceed but flag it
        kb.ingestion_errors.append("Supabase save failed — downstream agents will use Redis cache only")

    # ── Step 8: Cache in Redis for parallel downstream agents ─────────────
    kb_dict = kb.to_dict()
    cache_knowledge_base(property_id, kb_dict)

    # ── Step 9: Update pipeline status ───────────────────────────────────
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

    if kb.vrbo_url and kb.client_channel == ClientChannel.IO:
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
