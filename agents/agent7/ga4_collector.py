"""
TS-16 — GA4 + Segment Analytics Collector
Tier 1: UTM click tracking — universal, all properties, day one.

Pulls session data, traffic source breakdown, and booking site
click events from GA4 Data API for a given property and period.

Segment handles cross-domain attribution:
  Instagram post → Staylio landing page → PMC booking site
  Tracked as a single attributed journey, not three separate sessions.

GA4 is the primary data store for all page-level analytics.
Segment routes events to GA4 + PostgreSQL simultaneously,
enabling both real-time dashboard queries and monthly reporting.

Tier 1 commission savings methodology (TS-16 open question, resolved):
  estimated_bookings = clicks × industry_booking_rate
  industry_booking_rate = 0.02 (2% click-to-booking — conservative STR estimate)
  estimated_revenue = bookings × avg_nightly_rate × avg_stay_nights
  commission_saved = estimated_revenue × AIRBNB_FEE_RATE

This is clearly labeled "estimated" in the dashboard and report.
Actual numbers require Tier 2 pixel or Tier 3 PMS API.
"""

import logging
import os
from datetime import date, timedelta
from typing import Optional

import httpx

from agents.agent7.models import (
    AIRBNB_FEE_RATE,
    AnalyticsSnapshot,
    AttributionTier,
    TrafficBreakdown,
    TrafficSource,
)

logger = logging.getLogger(__name__)

GA4_PROPERTY_ID       = os.environ.get("GA4_PROPERTY_ID", "")
GA4_API_BASE          = "https://analyticsdata.googleapis.com/v1beta"
GOOGLE_SERVICE_ACCOUNT = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")

SEGMENT_WRITE_KEY     = os.environ.get("SEGMENT_WRITE_KEY", "")
SEGMENT_API_BASE      = "https://api.segment.io/v1"

# Conservative STR click-to-booking rate for Tier 1 estimates
INDUSTRY_BOOKING_RATE   = 0.02   # 2%
AVG_STAY_NIGHTS_DEFAULT = 4.0    # nights, for commission calc fallback


def fetch_property_analytics(
    property_id: str,
    slug: str,
    period_start: date,
    period_end: date,
    avg_nightly_rate: Optional[float] = None,
    attribution_tier: AttributionTier = AttributionTier.TIER_1_UTM,
) -> AnalyticsSnapshot:
    """
    Fetch GA4 analytics for a property over the given period.
    Returns an AnalyticsSnapshot populated with Tier 1 data.
    Tier 2/3 data is added by separate functions below.

    Args:
        property_id:     Property UUID
        slug:            URL slug (matches utm_campaign in GA4)
        period_start:    Start date (inclusive)
        period_end:      End date (inclusive)
        avg_nightly_rate: Used in commission savings estimate
        attribution_tier: Current tier for this property
    """
    snapshot = AnalyticsSnapshot(
        property_id=property_id,
        period_start=period_start.isoformat(),
        period_end=period_end.isoformat(),
        attribution_tier=attribution_tier,
    )

    if not GA4_PROPERTY_ID:
        logger.warning("[TS-16] GA4 property ID not configured — returning empty snapshot")
        return snapshot

    # ── Fetch core metrics from GA4 ───────────────────────────────────────
    try:
        core_data = _ga4_report(
            dimensions=["sessionSource", "sessionMedium", "sessionCampaign"],
            metrics=["sessions", "totalUsers", "eventCount"],
            filters=_property_filter(slug),
            start_date=period_start,
            end_date=period_end,
        )
        _apply_core_metrics(snapshot, core_data)
    except Exception as exc:
        logger.error(f"[TS-16] GA4 core metrics fetch failed for {property_id}: {exc}")

    # ── Fetch booking site click events ──────────────────────────────────
    try:
        click_data = _ga4_report(
            dimensions=["sessionSource", "eventName"],
            metrics=["eventCount"],
            filters=_booking_click_filter(slug),
            start_date=period_start,
            end_date=period_end,
        )
        _apply_click_data(snapshot, click_data)
    except Exception as exc:
        logger.error(f"[TS-16] GA4 click data fetch failed for {property_id}: {exc}")

    # ── Tier 1 commission savings estimate ────────────────────────────────
    if avg_nightly_rate and snapshot.total_booking_site_clicks > 0:
        snapshot.estimated_bookings_from_clicks = round(
            snapshot.total_booking_site_clicks * INDUSTRY_BOOKING_RATE, 1
        )
        est_revenue = (
            snapshot.estimated_bookings_from_clicks
            * avg_nightly_rate
            * AVG_STAY_NIGHTS_DEFAULT
        )
        snapshot.estimated_commission_saved = round(est_revenue * AIRBNB_FEE_RATE, 2)
        snapshot.commission_calc_methodology = "estimated"

    # ── Top social posts by click performance ────────────────────────────
    try:
        top_posts = _fetch_top_posts(slug, period_start, period_end)
        snapshot.top_posts = top_posts
    except Exception as exc:
        logger.warning(f"[TS-16] Top posts fetch failed for {property_id}: {exc}")

    logger.info(
        f"[TS-16] Analytics fetched for property {property_id}: "
        f"sessions={snapshot.total_sessions}, "
        f"clicks={snapshot.total_booking_site_clicks}, "
        f"est_savings=${snapshot.estimated_commission_saved:.2f}"
    )
    return snapshot


def track_page_event(
    property_id: str,
    event_name: str,
    properties: dict,
    anonymous_id: str,
) -> None:
    """
    Send a tracking event to Segment (which routes to GA4 + PostgreSQL).
    Called from the landing page JavaScript via a Cloudflare Worker proxy.
    Not called directly from Python — documented here for completeness.
    """
    if not SEGMENT_WRITE_KEY:
        return
    try:
        import base64
        auth = base64.b64encode(f"{SEGMENT_WRITE_KEY}:".encode()).decode()
        with httpx.Client(timeout=10) as client:
            client.post(
                f"{SEGMENT_API_BASE}/track",
                headers={
                    "Authorization": f"Basic {auth}",
                    "Content-Type": "application/json",
                },
                json={
                    "event": event_name,
                    "anonymousId": anonymous_id,
                    "properties": {**properties, "property_id": property_id},
                },
            )
    except Exception as exc:
        logger.warning(f"Segment track failed: {exc}")


# ── GA4 API helpers ───────────────────────────────────────────────────────

def _ga4_report(
    dimensions: list[str],
    metrics: list[str],
    filters: dict,
    start_date: date,
    end_date: date,
) -> dict:
    """Run a GA4 Data API report request."""
    token = _get_ga4_access_token()
    if not token:
        return {}

    payload = {
        "dateRanges": [{"startDate": start_date.isoformat(), "endDate": end_date.isoformat()}],
        "dimensions": [{"name": d} for d in dimensions],
        "metrics": [{"name": m} for m in metrics],
        "dimensionFilter": filters,
        "limit": 100,
    }

    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{GA4_API_BASE}/properties/{GA4_PROPERTY_ID}:runReport",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


def _get_ga4_access_token() -> Optional[str]:
    """Get a Google OAuth access token for GA4 API calls."""
    if not GOOGLE_SERVICE_ACCOUNT:
        return None
    try:
        import json as json_module
        import time
        import jwt   # pip install PyJWT
        sa = json_module.loads(GOOGLE_SERVICE_ACCOUNT)
        now = int(time.time())
        claim = {
            "iss": sa["client_email"],
            "scope": "https://www.googleapis.com/auth/analytics.readonly",
            "aud": "https://oauth2.googleapis.com/token",
            "exp": now + 3600,
            "iat": now,
        }
        signed = jwt.encode(claim, sa["private_key"], algorithm="RS256")
        with httpx.Client(timeout=15) as client:
            resp = client.post(
                "https://oauth2.googleapis.com/token",
                data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": signed},
            )
            resp.raise_for_status()
            return resp.json().get("access_token")
    except Exception as exc:
        logger.error(f"GA4 token fetch failed: {exc}")
        return None


def _property_filter(slug: str) -> dict:
    """GA4 dimension filter: sessions where utm_campaign = property slug."""
    return {
        "filter": {
            "fieldName": "sessionCampaign",
            "stringFilter": {"matchType": "EXACT", "value": slug},
        }
    }


def _booking_click_filter(slug: str) -> dict:
    """GA4 filter for booking_site_click events for this property."""
    return {
        "andGroup": {
            "expressions": [
                {"filter": {"fieldName": "eventName",
                            "stringFilter": {"matchType": "EXACT", "value": "booking_site_click"}}},
                {"filter": {"fieldName": "sessionCampaign",
                            "stringFilter": {"matchType": "EXACT", "value": slug}}},
            ]
        }
    }


def _apply_core_metrics(snapshot: AnalyticsSnapshot, data: dict) -> None:
    """Parse GA4 report rows into core snapshot metrics."""
    source_map: dict[str, TrafficBreakdown] = {}

    for row in data.get("rows", []):
        dims = [d.get("value", "") for d in row.get("dimensionValues", [])]
        vals = [m.get("value", "0") for m in row.get("metricValues", [])]

        source_str = dims[0] if dims else "other"
        sessions   = int(vals[0]) if vals else 0
        users      = int(vals[1]) if len(vals) > 1 else 0

        traffic_source = _map_source(source_str)
        if traffic_source not in source_map:
            source_map[traffic_source] = TrafficBreakdown(
                source=traffic_source,
                sessions=0,
                clicks_to_booking_site=0,
            )
        source_map[traffic_source].sessions += sessions
        snapshot.total_sessions += sessions
        snapshot.unique_visitors += users

    snapshot.traffic_by_source = list(source_map.values())


def _apply_click_data(snapshot: AnalyticsSnapshot, data: dict) -> None:
    """Parse booking click event data into snapshot."""
    total_clicks = 0
    source_clicks: dict[str, int] = {}

    for row in data.get("rows", []):
        dims = [d.get("value", "") for d in row.get("dimensionValues", [])]
        vals = [m.get("value", "0") for m in row.get("metricValues", [])]
        source_str = dims[0] if dims else "other"
        clicks = int(vals[0]) if vals else 0
        total_clicks += clicks
        source_clicks[source_str] = source_clicks.get(source_str, 0) + clicks

    snapshot.total_booking_site_clicks = total_clicks
    if snapshot.total_sessions > 0:
        snapshot.overall_ctr = round(total_clicks / snapshot.total_sessions, 4)

    # Apply clicks to traffic breakdown
    for tb in snapshot.traffic_by_source:
        src_str = tb.source.value
        tb.clicks_to_booking_site = source_clicks.get(src_str, 0)


def _fetch_top_posts(slug: str, start: date, end: date) -> list[dict]:
    """Fetch top social posts by booking clicks for this period."""
    data = _ga4_report(
        dimensions=["sessionSource", "sessionMedium", "customEvent:utm_content"],
        metrics=["eventCount"],
        filters=_booking_click_filter(slug),
        start_date=start,
        end_date=end,
    )
    posts = []
    for row in sorted(
        data.get("rows", []),
        key=lambda r: int((r.get("metricValues") or [{"value": "0"}])[0].get("value", "0")),
        reverse=True,
    )[:5]:
        dims = [d.get("value", "") for d in row.get("dimensionValues", [])]
        vals = [m.get("value", "0") for m in row.get("metricValues", [])]
        posts.append({
            "platform": dims[0] if dims else "",
            "post_id": dims[2] if len(dims) > 2 else "",
            "clicks": int(vals[0]) if vals else 0,
        })
    return posts


def _map_source(ga4_source: str) -> TrafficSource:
    """Map GA4 session source string to TrafficSource enum."""
    mapping = {
        "instagram": TrafficSource.INSTAGRAM,
        "tiktok": TrafficSource.TIKTOK,
        "facebook": TrafficSource.FACEBOOK,
        "pinterest": TrafficSource.PINTEREST,
        "google": TrafficSource.ORGANIC_SEARCH,
        "(direct)": TrafficSource.DIRECT,
        "meta": TrafficSource.PAID_META,
    }
    return mapping.get(ga4_source.lower(), TrafficSource.OTHER)
