"""
Agent 7 — Analytics & Reporting Data Models

AttributionTier:   Which attribution method is active for a property
AnalyticsSnapshot: Aggregated metrics for a period (used in reports)
ConversionEvent:   A confirmed booking from Tier 2 pixel or Tier 3 PMS
MonthlyReport:     The complete report package for one property one month
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AttributionTier(str, Enum):
    """
    Which attribution tier a property is on.
    From TS-16 — each tier provides increasing accuracy.
    Staylio is transparent with PMCs about what each tier can and cannot prove.
    """
    TIER_1_UTM       = "tier_1_utm"        # Clicks only — all properties day one
    TIER_2_PIXEL     = "tier_2_pixel"      # Confirmed bookings via conversion pixel
    TIER_3_PMS_API   = "tier_3_pms_api"   # Full reconciliation — Portfolio tier only
    TIER_4_SELF_REPORT = "tier_4_self_report"  # Manual entry — lowest accuracy


class TrafficSource(str, Enum):
    INSTAGRAM = "instagram"
    TIKTOK    = "tiktok"
    FACEBOOK  = "facebook"
    PINTEREST = "pinterest"
    ORGANIC_SEARCH = "organic_search"
    DIRECT    = "direct"
    PAID_META = "paid_meta"
    PAID_TIKTOK = "paid_tiktok"
    REFERRAL  = "referral"
    OTHER     = "other"


# Airbnb host-only fee rate (as of October 2025, per TS-16)
AIRBNB_FEE_RATE = 0.155   # 15.5%
VRBO_FEE_RATE   = 0.05    # 5.0%
BOOKING_COM_FEE_RATE = 0.15  # 15.0% average


@dataclass
class TrafficBreakdown:
    source: TrafficSource
    sessions: int
    clicks_to_booking_site: int
    avg_session_duration_seconds: float = 0.0

    def click_through_rate(self) -> float:
        if self.sessions == 0:
            return 0.0
        return round(self.clicks_to_booking_site / self.sessions, 4)


@dataclass
class ConversionEvent:
    """
    A confirmed booking conversion.
    Tier 2: from conversion pixel on booking confirmation page.
    Tier 3: from PMS API reservation record.
    """
    property_id: str
    booking_date: str              # ISO date when booking was made
    stay_start_date: Optional[str] = None
    stay_end_date: Optional[str] = None
    stay_nights: Optional[int] = None
    booking_value: Optional[float] = None      # Revenue from this booking
    source: AttributionTier = AttributionTier.TIER_2_PIXEL
    utm_source: Optional[str] = None           # Which platform drove this booking
    utm_campaign: Optional[str] = None
    utm_content: Optional[str] = None
    # Tier 3 only
    pms_reservation_id: Optional[str] = None
    cancellation_status: Optional[str] = None  # "confirmed" | "cancelled"

    def ota_fee_saved(self, ota_type: str = "airbnb") -> float:
        """Estimated OTA fee this direct booking avoided."""
        if not self.booking_value:
            return 0.0
        rates = {"airbnb": AIRBNB_FEE_RATE, "vrbo": VRBO_FEE_RATE, "booking_com": BOOKING_COM_FEE_RATE}
        return round(self.booking_value * rates.get(ota_type, AIRBNB_FEE_RATE), 2)


@dataclass
class AnalyticsSnapshot:
    """
    Aggregated metrics for a property over a reporting period.
    Built from GA4 + Segment data, enhanced with pixel/PMS data if available.
    """
    property_id: str
    period_start: str        # ISO date
    period_end: str
    attribution_tier: AttributionTier

    # Tier 1 — UTM click data (always available)
    total_sessions: int = 0
    unique_visitors: int = 0
    total_booking_site_clicks: int = 0
    overall_ctr: float = 0.0

    # Traffic breakdown by source
    traffic_by_source: list[TrafficBreakdown] = field(default_factory=list)

    # Top performing social posts (post_id, clicks, platform)
    top_posts: list[dict] = field(default_factory=list)

    # Tier 1 estimated commission savings
    # Methodology: clicks × avg_booking_rate × avg_nightly_rate × avg_stay_nights × OTA_fee_rate
    estimated_bookings_from_clicks: float = 0.0
    estimated_commission_saved: float = 0.0
    commission_calc_methodology: str = "estimated"   # "estimated" | "confirmed" | "reconciled"

    # Tier 2 — pixel confirmed bookings
    confirmed_bookings: int = 0
    confirmed_revenue: float = 0.0
    confirmed_commission_saved: float = 0.0
    pixel_active: bool = False

    # Tier 3 — PMS API reconciled data
    pms_confirmed_bookings: int = 0
    pms_total_revenue: float = 0.0
    pms_cancellation_rate: float = 0.0
    pms_avg_lead_time_days: float = 0.0
    pms_connected: bool = False

    # Tier 4 — self-reported
    self_reported_revenue: float = 0.0
    self_reported_note: str = ""

    # Social performance
    total_social_posts_published: int = 0
    total_social_impressions: int = 0
    total_social_engagement: int = 0

    # Month-over-month
    sessions_mom_pct: Optional[float] = None
    clicks_mom_pct: Optional[float] = None
    revenue_mom_pct: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "property_id": self.property_id,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "attribution_tier": self.attribution_tier,
            "total_sessions": self.total_sessions,
            "unique_visitors": self.unique_visitors,
            "total_booking_site_clicks": self.total_booking_site_clicks,
            "overall_ctr": self.overall_ctr,
            "traffic_by_source": [
                {
                    "source": t.source,
                    "sessions": t.sessions,
                    "clicks": t.clicks_to_booking_site,
                    "ctr": t.click_through_rate(),
                }
                for t in self.traffic_by_source
            ],
            "top_posts": self.top_posts,
            "estimated_bookings_from_clicks": self.estimated_bookings_from_clicks,
            "estimated_commission_saved": self.estimated_commission_saved,
            "commission_calc_methodology": self.commission_calc_methodology,
            "confirmed_bookings": self.confirmed_bookings,
            "confirmed_revenue": self.confirmed_revenue,
            "confirmed_commission_saved": self.confirmed_commission_saved,
            "pixel_active": self.pixel_active,
            "pms_confirmed_bookings": self.pms_confirmed_bookings,
            "pms_total_revenue": self.pms_total_revenue,
            "pms_cancellation_rate": self.pms_cancellation_rate,
            "pms_connected": self.pms_connected,
            "self_reported_revenue": self.self_reported_revenue,
            "total_social_posts_published": self.total_social_posts_published,
            "total_social_impressions": self.total_social_impressions,
            "sessions_mom_pct": self.sessions_mom_pct,
            "clicks_mom_pct": self.clicks_mom_pct,
            "revenue_mom_pct": self.revenue_mom_pct,
        }


@dataclass
class MonthlyReport:
    """
    The complete monthly report package for one property.
    Contains analytics snapshot + AI-generated narrative + PDF path.
    """
    property_id: str
    client_id: str
    report_month: str       # "2026-03" format
    analytics: Optional[AnalyticsSnapshot] = None
    narrative_html: Optional[str] = None    # Claude-generated HTML report body
    pdf_r2_url: Optional[str] = None        # R2 URL of rendered PDF
    email_delivered: bool = False
    generated_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "property_id": self.property_id,
            "client_id": self.client_id,
            "report_month": self.report_month,
            "analytics": self.analytics.to_dict() if self.analytics else None,
            "pdf_r2_url": self.pdf_r2_url,
            "email_delivered": self.email_delivered,
            "generated_at": self.generated_at,
        }
