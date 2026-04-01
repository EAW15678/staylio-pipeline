"""
Agent 7 — Analytics & Reporting Agent
LangGraph Node + Monthly Scheduled Job

TWO OPERATING MODES:

MODE 1 — PIPELINE NODE (runs once, immediately after Agent 5 deploys page):
  - Registers the property in GA4 + Segment for tracking
  - Generates and stores the conversion pixel snippet
  - Records the property's initial attribution tier
  - Sets up Ayrshare post performance polling
  - Does NOT generate a report (no data yet)

MODE 2 — MONTHLY JOB (runs last day of each month for all active properties):
  - Collects GA4 + Segment analytics for the month
  - Fetches pixel conversions (Tier 2) if pixel is active
  - Fetches PMS reservations (Tier 3) if PMS API connected
  - Computes month-over-month changes
  - Generates Claude Haiku report narrative
  - Renders PDF via Puppeteer
  - Stores PDF in R2, saves report record to Supabase
  - Queues email delivery

Both modes write to the same analytics_snapshots and monthly_reports tables
that power the real-time PMC dashboard.
"""

import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import anthropic

from agents.agent7.attribution_engine import (
    apply_pixel_data_to_snapshot,
    apply_pms_data_to_snapshot,
    fetch_pixel_conversions,
    fetch_pms_reservations,
    generate_pixel_snippet,
)
from agents.agent7.ga4_collector import fetch_property_analytics
from agents.agent7.models import (
    AnalyticsSnapshot,
    AttributionTier,
    MonthlyReport,
)
from agents.agent7.report_generator import generate_monthly_report
from core.pipeline_status import (
    PipelineStepStatus,
    get_cached_knowledge_base,
    update_pipeline_status,
)

logger = logging.getLogger(__name__)

AGENT_NUMBER = 7

_anthropic_client: Optional[anthropic.Anthropic] = None


def _get_anthropic() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _anthropic_client


# ── MODE 1: Pipeline node ─────────────────────────────────────────────────

def agent7_node(state: dict) -> dict:
    """
    LangGraph node for Agent 7 — runs immediately after Agent 5 deploys.
    Sets up analytics infrastructure for the property.
    Does not generate a report — no data yet.
    """
    property_id = state["property_id"]
    logger.info(f"[Agent 7] Setting up analytics for property {property_id}")

    update_pipeline_status(property_id, AGENT_NUMBER, PipelineStepStatus.RUNNING)

    # ── Load inputs ───────────────────────────────────────────────────────
    kb       = get_cached_knowledge_base(property_id) or state.get("knowledge_base", {})
    page_url = state.get("page_url", "")
    slug     = state.get("page_slug", "") or kb.get("slug", "")
    client_id = kb.get("client_id", "")

    # ── Determine attribution tier ────────────────────────────────────────
    tier = _determine_attribution_tier(kb)

    # ── Generate conversion pixel snippet ────────────────────────────────
    pixel_snippet = generate_pixel_snippet(property_id)

    # ── Register property in analytics system ─────────────────────────────
    _register_property_analytics(
        property_id=property_id,
        client_id=client_id,
        slug=slug,
        page_url=page_url,
        tier=tier,
        pixel_snippet=pixel_snippet,
    )

    # ── Update pipeline status ────────────────────────────────────────────
    update_pipeline_status(
        property_id, AGENT_NUMBER,
        PipelineStepStatus.COMPLETE,
        metadata={
            "attribution_tier": tier,
            "pixel_snippet_generated": True,
            "page_url": page_url,
            "tracking_active": bool(os.environ.get("GA4_PROPERTY_ID")),
        },
    )

    logger.info(
        f"[Agent 7] Analytics setup complete for property {property_id}. "
        f"Tier: {tier}. Page: {page_url}"
    )

    return {
        **state,
        "attribution_tier": tier,
        "pixel_snippet": pixel_snippet,
        "agent7_complete": True,
        "pipeline_complete": True,   # All agents done — property is fully live
    }


# ── MODE 2: Monthly report job ────────────────────────────────────────────

def run_monthly_report_job(
    property_id: str,
    client_id: str,
    report_month: str,   # "2026-03"
) -> Optional[MonthlyReport]:
    """
    Generate the monthly report for a single property.
    Called by a scheduled cron job on the last day of each month.
    Not part of the intake pipeline.

    Args:
        property_id:  Property UUID
        client_id:    Client UUID
        report_month: Year-month string "2026-03"
    """
    logger.info(f"[Agent 7] Monthly report job: property={property_id}, month={report_month}")

    # ── Load property metadata from Supabase ──────────────────────────────
    kb = _load_property_kb(property_id)
    if not kb:
        logger.error(f"[Agent 7] Could not load KB for property {property_id}")
        return None

    property_name  = (kb.get("name") or {}).get("value", "Property")
    slug           = kb.get("slug", property_id[:8])
    avg_rate       = _extract_float(kb, "avg_nightly_rate")
    pms_type       = kb.get("pms_type")
    pms_connected  = kb.get("pms_api_connected", False)

    # ── Determine reporting period ────────────────────────────────────────
    year, month = int(report_month[:4]), int(report_month[5:7])
    period_start = date(year, month, 1)
    next_month   = date(year + (month // 12), (month % 12) + 1, 1)
    period_end   = next_month - timedelta(days=1)

    # ── Determine attribution tier ────────────────────────────────────────
    tier = _load_attribution_tier(property_id)

    # ── Fetch GA4 analytics (Tier 1) ──────────────────────────────────────
    snapshot = fetch_property_analytics(
        property_id=property_id,
        slug=slug,
        period_start=period_start,
        period_end=period_end,
        avg_nightly_rate=avg_rate,
        attribution_tier=tier,
    )

    # ── Enrich with social performance data ───────────────────────────────
    snapshot = _enrich_social_data(snapshot, property_id, period_start, period_end)

    # ── Tier 2: Pixel conversions ─────────────────────────────────────────
    if tier in (AttributionTier.TIER_2_PIXEL, AttributionTier.TIER_3_PMS_API):
        pixel_conversions = fetch_pixel_conversions(property_id, period_start, period_end)
        if pixel_conversions:
            snapshot = apply_pixel_data_to_snapshot(snapshot, pixel_conversions)

    # ── Tier 3: PMS API reconciliation ────────────────────────────────────
    if tier == AttributionTier.TIER_3_PMS_API and pms_type and pms_connected:
        pms_credentials = _load_pms_credentials(property_id, pms_type)
        if pms_credentials:
            pms_reservations = fetch_pms_reservations(
                property_id, pms_type, pms_credentials, period_start, period_end
            )
            if pms_reservations:
                snapshot = apply_pms_data_to_snapshot(snapshot, pms_reservations)

    # ── Month-over-month comparison ────────────────────────────────────────
    snapshot = _add_mom_comparison(snapshot, property_id, period_start)

    # ── Generate report ────────────────────────────────────────────────────
    report = generate_monthly_report(
        snapshot=snapshot,
        property_name=property_name,
        property_id=property_id,
        client_id=client_id,
        report_month=report_month,
        anthropic_client=_get_anthropic(),
    )

    # ── Save report and snapshot to Supabase ──────────────────────────────
    _save_snapshot(snapshot)
    _save_report(report)

    logger.info(
        f"[Agent 7] Monthly report complete: property={property_id}, "
        f"month={report_month}, pdf={'yes' if report.pdf_r2_url else 'no'}"
    )
    return report


def run_all_monthly_reports(report_month: str) -> dict:
    """
    Run the monthly report job for all active properties.
    Called by cron on the last day of each month.
    Returns summary of results.
    """
    logger.info(f"[Agent 7] Running monthly reports for all properties: {report_month}")

    active_properties = _load_all_active_properties()
    results = {"success": 0, "failed": 0, "skipped": 0}

    for row in active_properties:
        property_id = row["property_id"]
        client_id   = row["client_id"]
        try:
            report = run_monthly_report_job(property_id, client_id, report_month)
            if report:
                results["success"] += 1
            else:
                results["skipped"] += 1
        except Exception as exc:
            logger.error(f"[Agent 7] Monthly report failed for {property_id}: {exc}")
            results["failed"] += 1

    logger.info(f"[Agent 7] Monthly reports complete: {results}")
    return results


# ── Helpers ───────────────────────────────────────────────────────────────

def _determine_attribution_tier(kb: dict) -> str:
    """Determine the starting attribution tier for a new property."""
    if kb.get("pms_api_connected"):
        return AttributionTier.TIER_3_PMS_API
    # New properties start on Tier 1 — pixel not yet installed
    return AttributionTier.TIER_1_UTM


def _load_attribution_tier(property_id: str) -> AttributionTier:
    """Load current attribution tier from Supabase."""
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("property_analytics_config")
            .select("attribution_tier")
            .eq("property_id", property_id)
            .single()
            .execute()
        )
        tier_str = (result.data or {}).get("attribution_tier", "tier_1_utm")
        return AttributionTier(tier_str)
    except Exception:
        return AttributionTier.TIER_1_UTM


def _register_property_analytics(
    property_id: str,
    client_id: str,
    slug: str,
    page_url: str,
    tier: str,
    pixel_snippet: str,
) -> None:
    """Register a new property in the analytics system."""
    try:
        from core.supabase_store import get_supabase
        get_supabase().table("property_analytics_config").upsert(
            {
                "property_id": property_id,
                "client_id": client_id,
                "slug": slug,
                "page_url": page_url,
                "attribution_tier": tier,
                "pixel_snippet": pixel_snippet,
                "pixel_installed": False,
                "tracking_active": True,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="property_id",
        ).execute()
    except Exception as exc:
        logger.error(f"[Agent 7] Failed to register analytics config: {exc}")


def _enrich_social_data(
    snapshot: AnalyticsSnapshot,
    property_id: str,
    period_start: date,
    period_end: date,
) -> AnalyticsSnapshot:
    """Add social post counts and engagement from Supabase."""
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("social_posts")
            .select("platform, views, likes, shares, comments")
            .eq("property_id", property_id)
            .eq("status", "published")
            .gte("published_at", period_start.isoformat())
            .lte("published_at", period_end.isoformat())
            .execute()
        )
        posts = result.data or []
        snapshot.total_social_posts_published = len(posts)
        snapshot.total_social_impressions = sum(p.get("views", 0) for p in posts)
        snapshot.total_social_engagement = sum(
            p.get("likes", 0) + p.get("shares", 0) + p.get("comments", 0)
            for p in posts
        )
    except Exception as exc:
        logger.warning(f"[Agent 7] Social data enrichment failed: {exc}")
    return snapshot


def _add_mom_comparison(
    snapshot: AnalyticsSnapshot,
    property_id: str,
    current_period_start: date,
) -> AnalyticsSnapshot:
    """Add month-over-month percentage changes."""
    prev_month_end   = current_period_start - timedelta(days=1)
    prev_month_start = date(prev_month_end.year, prev_month_end.month, 1)
    prev_month_str   = prev_month_start.strftime("%Y-%m")

    prev = _load_snapshot(property_id, prev_month_str)
    if not prev:
        return snapshot

    def pct_change(current: int, previous: int) -> Optional[float]:
        if not previous:
            return None
        return round((current - previous) / previous * 100, 1)

    snapshot.sessions_mom_pct = pct_change(snapshot.total_sessions, prev.get("total_sessions", 0))
    snapshot.clicks_mom_pct   = pct_change(snapshot.total_booking_site_clicks, prev.get("total_booking_site_clicks", 0))
    return snapshot


def _load_property_kb(property_id: str) -> dict:
    """Load property knowledge base from Supabase."""
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("property_knowledge_bases")
            .select("data")
            .eq("property_id", property_id)
            .single()
            .execute()
        )
        return (result.data or {}).get("data", {})
    except Exception as exc:
        logger.error(f"[Agent 7] KB load failed for {property_id}: {exc}")
        return {}


def _load_pms_credentials(property_id: str, pms_type: str) -> dict:
    """Load PMS OAuth credentials from Supabase (encrypted at rest)."""
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("pms_credentials")
            .select("credentials")
            .eq("property_id", property_id)
            .eq("pms_type", pms_type)
            .single()
            .execute()
        )
        return (result.data or {}).get("credentials", {})
    except Exception:
        return {}


def _load_snapshot(property_id: str, month_str: str) -> dict:
    """Load a previous month's analytics snapshot."""
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("analytics_snapshots")
            .select("data")
            .eq("property_id", property_id)
            .eq("report_month", month_str)
            .single()
            .execute()
        )
        return (result.data or {}).get("data", {})
    except Exception:
        return {}


def _load_all_active_properties() -> list[dict]:
    """Load all active properties for monthly report batch."""
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("landing_pages")
            .select("property_id, client_id: property_knowledge_bases(client_id)")
            .eq("status", "deployed")
            .execute()
        )
        return result.data or []
    except Exception as exc:
        logger.error(f"[Agent 7] Failed to load active properties: {exc}")
        return []


def _save_snapshot(snapshot: AnalyticsSnapshot) -> None:
    """Save analytics snapshot to Supabase."""
    try:
        from core.supabase_store import get_supabase
        month_str = snapshot.period_start[:7]
        get_supabase().table("analytics_snapshots").upsert(
            {
                "property_id": snapshot.property_id,
                "report_month": month_str,
                "data": snapshot.to_dict(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="property_id,report_month",
        ).execute()
    except Exception as exc:
        logger.error(f"[Agent 7] Snapshot save failed: {exc}")


def _save_report(report: MonthlyReport) -> None:
    """Save monthly report record to Supabase."""
    try:
        from core.supabase_store import get_supabase
        get_supabase().table("monthly_reports").upsert(
            report.to_dict(),
            on_conflict="property_id,report_month",
        ).execute()
    except Exception as exc:
        logger.error(f"[Agent 7] Report save failed: {exc}")


def _extract_float(kb: dict, key: str) -> Optional[float]:
    f = kb.get(key)
    val = f.get("value") if isinstance(f, dict) else f
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None
