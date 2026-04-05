"""
Agent 5 — Website Builder & Optimizer
LangGraph Node

Runs AFTER Agents 2, 3, and 4 complete (all parallel agents must finish).
Agent 6 (Social Media) waits for Agent 5 before publishing.

Pipeline:
  1. Load knowledge base from Redis (Agent 1)
  2. Load content package from Supabase (Agent 2)
  3. Load visual media package from Redis (Agent 3)
  4. Load local guide from Redis (Agent 4)
  5. Provision iCal calendar sync (TS-14)
  6. Build complete landing page HTML (all sections assembled)
  7. Deploy to Cloudflare Pages (TS-12)
  8. Save LandingPage record to Supabase
  9. Update pipeline status — signals Agent 6 to begin
"""

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from agents.agent5.calendar_sync import provision_calendar_sync
from agents.agent5.cloudflare_deployer import deploy_property_page
from agents.agent5.models import (
    DeployMode,
    LandingPage,
    PageStatus,
)
from agents.agent5.page_builder import build_landing_page_html
from core.pipeline_status import (
    PipelineStepStatus,
    cache_knowledge_base,
    get_cached_knowledge_base,
    update_pipeline_status,
)

logger = logging.getLogger(__name__)

AGENT_NUMBER = 5


def agent5_node(state: dict) -> dict:
    """
    LangGraph node for Agent 5 — Website Builder & Optimizer.
    Runs after Agents 2, 3, and 4 all complete.
    """
    property_id = state["property_id"]
    logger.info(f"[Agent 5] Building landing page for property {property_id}")

    update_pipeline_status(property_id, AGENT_NUMBER, PipelineStepStatus.RUNNING)

    # ── Step 1: Load all agent outputs ────────────────────────────────────
    kb = get_cached_knowledge_base(property_id) or state.get("knowledge_base") or {}
    content_package  = _load_from_cache_or_state(state, property_id, "content_package", "agent2")
    visual_media     = _load_from_cache_or_state(state, property_id, "visual_media_package", "visual_media")
    local_guide      = _load_from_cache_or_state(state, property_id, "local_guide", "local_guide")

    if not kb:
        error = f"Agent 5: Knowledge base not found for property {property_id}"
        update_pipeline_status(property_id, AGENT_NUMBER, PipelineStepStatus.FAILED, error)
        return {**state, "errors": state.get("errors", []) + [error], "agent5_complete": False}

    # ── Step 2: Determine slug and deploy mode ────────────────────────────
    slug = kb.get("slug") or _fallback_slug(kb)
    pmc_tier = _get_client_tier(kb.get("client_id"))
    custom_domain = kb.get("custom_domain")   # Set at intake for Portfolio tier
    deploy_mode = (
        DeployMode.CNAME_CUSTOM
        if pmc_tier == "portfolio" and custom_domain
        else DeployMode.STAYLIO_SUBDOMAIN
    )

    # ── Step 3: Calendar sync (TS-14) ─────────────────────────────────────
    ical_url = kb.get("ical_url")
    pms_type = kb.get("pms_type")
    pms_connected = kb.get("pms_api_connected", False)

    calendar_config = provision_calendar_sync(
        property_id=property_id,
        ical_url=ical_url,
        pms_type=pms_type,
        pms_api_connected=pms_connected,
        slug=slug,
    )

    # ── Step 4: Build page URL ────────────────────────────────────────────
    if deploy_mode == DeployMode.CNAME_CUSTOM and custom_domain:
        page_url = f"https://{custom_domain}"
    else:
        page_url = f"https://{slug}.staylio.ai"

    # ── Step 5: Build HTML ────────────────────────────────────────────────
    logger.info(f"[Agent 5] Assembling HTML for property {property_id}")
    html = build_landing_page_html(
        kb=kb,
        content_package=content_package,
        visual_media=visual_media,
        local_guide=local_guide,
        page_url=page_url,
        slug=slug,
        calendar_cache_endpoint=calendar_config.cache_endpoint,
    )

    if not html or len(html) < 1000:
        error = f"Agent 5: Page builder produced insufficient HTML for property {property_id}"
        update_pipeline_status(property_id, AGENT_NUMBER, PipelineStepStatus.FAILED, error)
        return {**state, "errors": state.get("errors", []) + [error], "agent5_complete": False}

    # ── Step 6: Deploy to Cloudflare Pages (TS-12) ────────────────────────
    logger.info(f"[Agent 5] Deploying to Cloudflare Pages: {page_url}")
    success, deployed_url, deployment_id = deploy_property_page(
        property_id=property_id,
        slug=slug,
        html_content=html,
        deploy_mode=deploy_mode,
        custom_domain=custom_domain,
    )

    if not success:
        error = f"Agent 5: Cloudflare Pages deployment failed for property {property_id}"
        update_pipeline_status(property_id, AGENT_NUMBER, PipelineStepStatus.FAILED, error)
        return {**state, "errors": state.get("errors", []) + [error], "agent5_complete": False}

    # ── Step 7: Save LandingPage record ───────────────────────────────────
    landing_page = LandingPage(
        property_id=property_id,
        slug=slug,
        page_url=deployed_url,
        deploy_mode=deploy_mode,
        content_version=1,
        last_built_at=datetime.now(timezone.utc).isoformat(),
        status=PageStatus.DEPLOYED,
        cloudflare_deployment_id=deployment_id,
        schema_generated=True,
        calendar_config=calendar_config,
    )
    _save_landing_page(landing_page)

    # Cache page URL for Agent 6 (needed for UTM links in social posts)
    cache_knowledge_base(
        f"{property_id}:landing_page",
        {"page_url": deployed_url, "slug": slug},
        ttl_seconds=86400 * 30,  # 30-day TTL — URL is stable
    )

    # ── Step 8: Update pipeline status ───────────────────────────────────
    update_pipeline_status(
        property_id, AGENT_NUMBER,
        PipelineStepStatus.COMPLETE,
        metadata={
            "page_url": deployed_url,
            "deploy_mode": deploy_mode,
            "slug": slug,
            "html_size_kb": len(html) // 1024,
            "deployment_id": deployment_id,
            "ical_synced": bool(calendar_config.cache_endpoint),
        },
    )

    logger.info(
        f"[Agent 5] Landing page live for property {property_id}: {deployed_url} "
        f"(deploy_mode={deploy_mode}, html={len(html)//1024}KB)"
    )

    return {
        **state,
        "page_url": deployed_url,
        "page_slug": slug,
        "agent5_complete": True,
        # Signal Agent 6 to start social publishing
        "agent6_ready": True,
    }


# ── Helpers ───────────────────────────────────────────────────────────────

def _load_from_cache_or_state(
    state: dict,
    property_id: str,
    state_key: str,
    agent_label: str,
) -> dict:
    """
    Load an agent's output from Redis cache first, fall back to state dict.
    Returns empty dict if neither is available (non-fatal — page degrades gracefully).
    """
    # Try Redis cache first (written by the producing agent)
    cached = get_cached_knowledge_base(f"{property_id}:{agent_label}")
    if cached:
        return cached

    # Fall back to LangGraph state dict
    result = state.get(state_key)
    if result:
        return result

    logger.warning(
        f"[Agent 5] {agent_label} output not found for property {property_id} "
        f"— page section will be omitted"
    )
    return {}


def _fallback_slug(kb: dict) -> str:
    """Generate a slug from property name if none was set by Agent 1."""
    import re
    name = ""
    name_field = kb.get("name")
    if isinstance(name_field, dict):
        name = name_field.get("value") or ""
    elif isinstance(name_field, str):
        name = name_field

    if not name:
        prop_id = kb.get("property_id", "property")
        return f"property-{prop_id[:8]}"

    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:50]


def _get_client_tier(client_id: Optional[str]) -> str:
    """
    Look up the client's pricing tier from Supabase.
    Returns "base", "portfolio", or "unknown".
    """
    if not client_id:
        return "base"
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("pmc_clients")
            .select("tier")
            .eq("client_id", client_id)
            .single()
            .execute()
        )
        return (result.data or {}).get("tier", "base")
    except Exception:
        return "base"


def _save_landing_page(landing_page: LandingPage) -> None:
    """Save the LandingPage record to Supabase."""
    try:
        from core.supabase_store import get_supabase
        get_supabase().table("landing_pages").upsert(
            {
                "property_id": landing_page.property_id,
                "slug": landing_page.slug,
                "page_url": landing_page.page_url,
                "deploy_mode": landing_page.deploy_mode,
                "status": landing_page.status,
                "content_version": landing_page.content_version,
                "last_built_at": landing_page.last_built_at,
                "cloudflare_deployment_id": landing_page.cloudflare_deployment_id,
                "schema_generated": landing_page.schema_generated,
            },
            on_conflict="property_id",
        ).execute()
    except Exception as exc:
        logger.error(f"[Agent 5] Failed to save landing page record: {exc}")
