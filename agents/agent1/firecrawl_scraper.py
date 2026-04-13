"""
TS-01 — PMC Website Scraper
Tool: Firecrawl

FIX NOTES (v1 → v2):
  Switched from /extract (paid-only, 402 error) to /scrape (free tier).
  Returns Markdown passed to Claude parser for structured extraction.
"""

import os
import logging
from typing import Optional

import httpx

from pipeline_emitter import emit_media_cost

from models.property import (
    DataSource,
    PropertyKnowledgeBase,
)

logger = logging.getLogger(__name__)

FIRECRAWL_API_BASE = "https://api.firecrawl.dev/v1"
FIRECRAWL_API_KEY  = os.environ["FIRECRAWL_API_KEY"]
REQUEST_TIMEOUT    = 60


def scrape_pmc_website(
    pmc_website_url: str,
    knowledge_base: PropertyKnowledgeBase,
) -> PropertyKnowledgeBase:
    logger.info(f"[TS-01] Firecrawl scraping PMC website: {pmc_website_url}")

    try:
        markdown = _firecrawl_scrape(pmc_website_url)
    except Exception as exc:
        error_msg = f"Firecrawl scrape failed for {pmc_website_url}: {exc}"
        logger.error(error_msg)
        knowledge_base.ingestion_errors.append(error_msg)
        return knowledge_base

    if not markdown:
        knowledge_base.ingestion_errors.append(
            f"Firecrawl returned empty markdown for {pmc_website_url}"
        )
        return knowledge_base

    emit_media_cost(
        vendor="firecrawl",
        service="scrape",
        units=1,
        unit_name="pages",
        property_id=str(knowledge_base.property_id),
        workflow_name="listing_generation",
        slot_name="pmc_website_scrape",
        generation_reason="pmc_website_scrape",
    )

    from agents.agent1.claude_parser import _claude_extract, _apply_extraction

    extraction = _claude_extract(markdown, pmc_website_url)
    if not extraction:
        knowledge_base.ingestion_errors.append(
            f"Claude parser returned no data for {pmc_website_url}"
        )
        return knowledge_base

    knowledge_base = _apply_extraction(extraction, knowledge_base, DataSource.PMC_WEBSITE)

    if DataSource.PMC_WEBSITE not in knowledge_base.ingestion_sources:
        knowledge_base.ingestion_sources.append(DataSource.PMC_WEBSITE)

    logger.info(
        f"[TS-01] PMC website scrape complete. "
        f"Photos: {len(knowledge_base.photos)}, "
        f"Amenities: {len(knowledge_base.amenities)}"
    )
    return knowledge_base


def _firecrawl_scrape(url: str) -> Optional[str]:
    headers = {
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": True,
        "waitFor": 3000,
    }

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.post(
            f"{FIRECRAWL_API_BASE}/scrape",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    if not data.get("success"):
        logger.warning(f"[TS-01] Firecrawl scrape returned success=false for {url}: {data}")
        return None

    markdown = data.get("data", {}).get("markdown") or data.get("markdown")
    if not markdown:
        logger.warning(f"[TS-01] Firecrawl scrape returned no markdown for {url}")
        return None

    logger.info(f"[TS-01] Firecrawl scrape returned {len(markdown)} chars of markdown")
    return markdown
