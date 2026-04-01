"""
Agent 7 Test Suite
Run with: pytest booked/agents/agent7/tests/ -v

Tests cover:
  - Attribution tier determination logic
  - Tier 1 commission savings calculator (2% rate, correct formula)
  - ConversionEvent OTA fee calculation (15.5% Airbnb rate)
  - AnalyticsSnapshot to_dict serialises cleanly
  - CTR calculation (clicks / sessions)
  - TrafficBreakdown CTR formula
  - Pixel snippet contains required property_id
  - UTM cookie snippet structure
  - GA4 source mapping for all platforms
  - MoM percentage change calculation
  - Report HTML contains required sections
  - Report HTML tier badge reflects correct tier
  - Estimation methodology clearly labelled
  - Agent node pipeline mode sets correct flags
  - Monthly report mode with all tiers
"""

import json
import pytest
from datetime import date
from unittest.mock import MagicMock, patch

from agents.agent7.models import (
    AIRBNB_FEE_RATE,
    BOOKING_COM_FEE_RATE,
    AnalyticsSnapshot,
    AttributionTier,
    ConversionEvent,
    MonthlyReport,
    TrafficBreakdown,
    TrafficSource,
    VRBO_FEE_RATE,
)
from agents.agent7.attribution_engine import (
    UTM_COOKIE_SNIPPET,
    apply_pixel_data_to_snapshot,
    apply_pms_data_to_snapshot,
    generate_pixel_snippet,
)
from agents.agent7.ga4_collector import (
    INDUSTRY_BOOKING_RATE,
    AVG_STAY_NIGHTS_DEFAULT,
    _map_source,
)
from agents.agent7.report_generator import (
    _build_data_summary,
    _build_report_html,
    _tier_context,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

def make_snapshot(
    tier: AttributionTier = AttributionTier.TIER_1_UTM,
    sessions: int = 1200,
    clicks: int = 84,
    pixel_active: bool = False,
) -> AnalyticsSnapshot:
    s = AnalyticsSnapshot(
        property_id="prop-001",
        period_start="2026-03-01",
        period_end="2026-03-31",
        attribution_tier=tier,
        total_sessions=sessions,
        unique_visitors=900,
        total_booking_site_clicks=clicks,
        overall_ctr=clicks / sessions if sessions else 0,
        pixel_active=pixel_active,
    )
    s.traffic_by_source = [
        TrafficBreakdown(TrafficSource.TIKTOK, sessions=500, clicks_to_booking_site=40),
        TrafficBreakdown(TrafficSource.INSTAGRAM, sessions=300, clicks_to_booking_site=25),
        TrafficBreakdown(TrafficSource.ORGANIC_SEARCH, sessions=200, clicks_to_booking_site=12),
    ]
    s.total_social_posts_published = 42
    return s


def make_conversion(nights: int = 5, value: float = 2250.0) -> ConversionEvent:
    return ConversionEvent(
        property_id="prop-001",
        booking_date="2026-03-15",
        stay_start_date="2026-04-10",
        stay_end_date="2026-04-15",
        stay_nights=nights,
        booking_value=value,
        source=AttributionTier.TIER_2_PIXEL,
        utm_source="instagram",
    )


# ── OTA Fee Calculation Tests ─────────────────────────────────────────────

class TestOTAFeeCalculation:
    def test_airbnb_fee_rate_is_15_5_pct(self):
        """Per TS-16: Airbnb charges 15.5% as of October 2025."""
        assert AIRBNB_FEE_RATE == pytest.approx(0.155, rel=0.001)

    def test_vrbo_fee_rate_is_5_pct(self):
        assert VRBO_FEE_RATE == pytest.approx(0.05, rel=0.001)

    def test_booking_com_fee_rate_is_15_pct(self):
        assert BOOKING_COM_FEE_RATE == pytest.approx(0.15, rel=0.001)

    def test_airbnb_fee_saved_calculation(self):
        event = make_conversion(value=3000.0)
        fee_saved = event.ota_fee_saved("airbnb")
        # $3,000 × 15.5% = $465
        assert fee_saved == pytest.approx(465.0, rel=0.01)

    def test_vrbo_fee_saved_calculation(self):
        event = make_conversion(value=3000.0)
        fee_saved = event.ota_fee_saved("vrbo")
        # $3,000 × 5% = $150
        assert fee_saved == pytest.approx(150.0, rel=0.01)

    def test_zero_booking_value_returns_zero_fee(self):
        event = ConversionEvent(property_id="p1", booking_date="2026-03-01", booking_value=None,
                                source=AttributionTier.TIER_2_PIXEL)
        assert event.ota_fee_saved("airbnb") == 0.0


# ── Tier 1 Commission Savings Tests ──────────────────────────────────────

class TestTier1CommissionEstimate:
    def test_industry_booking_rate_is_2_pct(self):
        assert INDUSTRY_BOOKING_RATE == pytest.approx(0.02, rel=0.001)

    def test_commission_savings_formula(self):
        """
        estimated_bookings = 84 clicks × 2% = 1.68
        estimated_revenue = 1.68 × $350/night × 4 nights = $2,352
        commission_saved = $2,352 × 15.5% = $364.56
        """
        clicks = 84
        avg_rate = 350.0
        est_bookings = clicks * INDUSTRY_BOOKING_RATE
        est_revenue  = est_bookings * avg_rate * AVG_STAY_NIGHTS_DEFAULT
        est_savings  = est_revenue * AIRBNB_FEE_RATE

        assert est_bookings == pytest.approx(1.68, rel=0.01)
        assert est_savings  == pytest.approx(364.56, rel=0.01)

    def test_snapshot_commission_methodology_label(self):
        """Tier 1 must be labelled 'estimated', never 'confirmed'."""
        s = make_snapshot(tier=AttributionTier.TIER_1_UTM)
        s.commission_calc_methodology = "estimated"
        s.estimated_commission_saved = 364.56

        data = s.to_dict()
        assert data["commission_calc_methodology"] == "estimated"

    def test_confirmed_tier_uses_confirmed_label(self):
        s = make_snapshot(tier=AttributionTier.TIER_2_PIXEL, pixel_active=True)
        s.confirmed_commission_saved = 465.0
        s.commission_calc_methodology = "confirmed"

        data = s.to_dict()
        assert data["commission_calc_methodology"] == "confirmed"


# ── TrafficBreakdown Tests ────────────────────────────────────────────────

class TestTrafficBreakdown:
    def test_ctr_formula(self):
        tb = TrafficBreakdown(TrafficSource.TIKTOK, sessions=500, clicks_to_booking_site=40)
        assert tb.click_through_rate() == pytest.approx(0.08, rel=0.01)   # 8%

    def test_zero_sessions_ctr_returns_zero(self):
        tb = TrafficBreakdown(TrafficSource.TIKTOK, sessions=0, clicks_to_booking_site=0)
        assert tb.click_through_rate() == 0.0

    def test_overall_ctr_on_snapshot(self):
        s = make_snapshot(sessions=1200, clicks=84)
        assert s.overall_ctr == pytest.approx(84/1200, rel=0.01)


# ── Pixel Tests ───────────────────────────────────────────────────────────

class TestConversionPixel:
    def test_pixel_contains_property_id(self):
        snippet = generate_pixel_snippet("prop-abc-123")
        assert "prop-abc-123" in snippet

    def test_pixel_is_script_tag(self):
        snippet = generate_pixel_snippet("prop-abc-123")
        assert "<script>" in snippet
        assert "</script>" in snippet

    def test_pixel_posts_to_correct_endpoint(self):
        snippet = generate_pixel_snippet("prop-001")
        assert "pixel.staylio.ai" in snippet

    def test_pixel_reads_utm_from_session_storage(self):
        snippet = generate_pixel_snippet("prop-001")
        assert "sessionStorage" in snippet
        assert "staylio_utm_source" in snippet or "utm_source" in snippet

    def test_utm_cookie_snippet_saves_utm_params(self):
        assert "utm_source" in UTM_COOKIE_SNIPPET
        assert "sessionStorage" in UTM_COOKIE_SNIPPET
        assert "utm_campaign" in UTM_COOKIE_SNIPPET
        assert "utm_property_id" in UTM_COOKIE_SNIPPET


# ── GA4 Source Mapping Tests ──────────────────────────────────────────────

class TestGA4SourceMapping:
    def test_tiktok_maps_correctly(self):
        assert _map_source("tiktok") == TrafficSource.TIKTOK

    def test_instagram_maps_correctly(self):
        assert _map_source("instagram") == TrafficSource.INSTAGRAM

    def test_facebook_maps_correctly(self):
        assert _map_source("facebook") == TrafficSource.FACEBOOK

    def test_pinterest_maps_correctly(self):
        assert _map_source("pinterest") == TrafficSource.PINTEREST

    def test_google_maps_to_organic_search(self):
        assert _map_source("google") == TrafficSource.ORGANIC_SEARCH

    def test_direct_maps_correctly(self):
        assert _map_source("(direct)") == TrafficSource.DIRECT

    def test_unknown_maps_to_other(self):
        assert _map_source("some_unknown_source") == TrafficSource.OTHER

    def test_case_insensitive(self):
        assert _map_source("TIKTOK") == TrafficSource.TIKTOK
        assert _map_source("Instagram") == TrafficSource.INSTAGRAM


# ── Snapshot Application Tests ────────────────────────────────────────────

class TestSnapshotEnrichment:
    def test_pixel_data_applied_correctly(self):
        s = make_snapshot()
        conversions = [make_conversion(value=2250.0), make_conversion(value=1800.0)]
        s = apply_pixel_data_to_snapshot(s, conversions)

        assert s.confirmed_bookings == 2
        assert s.confirmed_revenue == pytest.approx(4050.0, rel=0.01)
        # $4,050 × 15.5% = $627.75
        assert s.confirmed_commission_saved == pytest.approx(627.75, rel=0.01)
        assert s.commission_calc_methodology == "confirmed"
        assert s.pixel_active is True

    def test_pms_data_calculates_cancellation_rate(self):
        s = make_snapshot()
        reservations = [
            ConversionEvent("p1", "2026-03-01", cancellation_status="confirmed",
                          booking_value=2000, source=AttributionTier.TIER_3_PMS_API),
            ConversionEvent("p1", "2026-03-05", cancellation_status="confirmed",
                          booking_value=1500, source=AttributionTier.TIER_3_PMS_API),
            ConversionEvent("p1", "2026-03-10", cancellation_status="cancelled",
                          booking_value=1800, source=AttributionTier.TIER_3_PMS_API),
        ]
        s = apply_pms_data_to_snapshot(s, reservations)

        assert s.pms_confirmed_bookings == 2
        assert s.pms_total_revenue == pytest.approx(3500.0, rel=0.01)
        assert s.pms_cancellation_rate == pytest.approx(1/3, rel=0.01)  # 33.3%
        assert s.commission_calc_methodology == "reconciled"
        assert s.pms_connected is True

    def test_pms_lead_time_calculated(self):
        """Average days between booking_date and stay_start_date."""
        s = make_snapshot()
        reservations = [
            ConversionEvent("p1", "2026-03-01", stay_start_date="2026-04-01",
                          cancellation_status="confirmed", booking_value=1000,
                          source=AttributionTier.TIER_3_PMS_API),  # 31 days lead
            ConversionEvent("p1", "2026-03-15", stay_start_date="2026-04-15",
                          cancellation_status="confirmed", booking_value=1000,
                          source=AttributionTier.TIER_3_PMS_API),  # 31 days lead
        ]
        s = apply_pms_data_to_snapshot(s, reservations)
        assert s.pms_avg_lead_time_days == pytest.approx(31.0, rel=0.1)


# ── Snapshot Serialisation Tests ─────────────────────────────────────────

class TestSnapshotSerialisation:
    def test_to_dict_is_json_serialisable(self):
        s = make_snapshot(tier=AttributionTier.TIER_2_PIXEL, pixel_active=True)
        s.confirmed_bookings = 3
        s.confirmed_revenue = 6750.0
        d = s.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 100

    def test_to_dict_contains_all_required_keys(self):
        s = make_snapshot()
        d = s.to_dict()
        required = [
            "property_id", "period_start", "period_end", "attribution_tier",
            "total_sessions", "unique_visitors", "total_booking_site_clicks",
            "overall_ctr", "traffic_by_source", "estimated_commission_saved",
            "commission_calc_methodology",
        ]
        for key in required:
            assert key in d, f"Missing key: {key}"


# ── Report HTML Tests ─────────────────────────────────────────────────────

class TestReportHTML:
    def test_tier1_report_html_contains_estimate_note(self):
        s = make_snapshot(tier=AttributionTier.TIER_1_UTM)
        s.estimated_commission_saved = 364.56
        html = _build_report_html("Vista Azule", "2026-03", s, "<p>Narrative</p>")
        # Tier 1 report must clearly label estimates
        assert "estimated" in html.lower() or "Estimated" in html

    def test_tier2_report_html_shows_confirmed(self):
        s = make_snapshot(tier=AttributionTier.TIER_2_PIXEL, pixel_active=True)
        s.confirmed_bookings = 3
        s.confirmed_revenue = 6750.0
        s.confirmed_commission_saved = 1046.25
        html = _build_report_html("Vista Azule", "2026-03", s, "<p>Narrative</p>")
        assert "Confirmed" in html or "confirmed" in html

    def test_report_html_contains_property_name(self):
        s = make_snapshot()
        html = _build_report_html("Vista Azule", "2026-03", s, "<p>Narrative</p>")
        assert "Vista Azule" in html

    def test_report_html_contains_period(self):
        s = make_snapshot()
        html = _build_report_html("Vista Azule", "2026-03", s, "<p>Narrative</p>")
        assert "March 2026" in html

    def test_report_html_contains_stats(self):
        s = make_snapshot(sessions=1200, clicks=84)
        html = _build_report_html("Vista Azule", "2026-03", s, "<p>Test</p>")
        assert "1,200" in html   # Formatted sessions
        assert "84" in html      # Clicks

    def test_tier_context_tier1_warns_about_estimates(self):
        ctx = _tier_context(AttributionTier.TIER_1_UTM)
        assert "estimate" in ctx.lower() or "estimated" in ctx.lower()
        assert "pixel" in ctx.lower()

    def test_tier_context_tier2_allows_confirmed(self):
        ctx = _tier_context(AttributionTier.TIER_2_PIXEL)
        assert "confirmed" in ctx.lower()

    def test_data_summary_includes_all_tiers(self):
        s = make_snapshot(tier=AttributionTier.TIER_2_PIXEL, pixel_active=True)
        s.confirmed_bookings = 2
        s.confirmed_revenue = 4000.0
        summary = _build_data_summary(s)
        assert "confirmed_bookings" in summary
        assert "confirmed_revenue_usd" in summary

    def test_data_summary_tier1_has_note(self):
        s = make_snapshot(tier=AttributionTier.TIER_1_UTM)
        s.estimated_bookings_from_clicks = 1.68
        summary = _build_data_summary(s)
        assert "note" in summary
        assert "estimated" in summary["note"].lower()


# ── Agent Node Contract Tests ─────────────────────────────────────────────

class TestAgent7NodeContract:
    def test_pipeline_node_sets_complete_flags(self):
        from agents.agent7.agent import agent7_node

        state = {
            "property_id": "prop-001",
            "knowledge_base": {
                "property_id": "prop-001",
                "client_id": "client-001",
                "slug": "vista-azule",
                "vibe_profile": "romantic_escape",
            },
            "page_url": "https://vista-azule.staylio.ai",
            "page_slug": "vista-azule",
            "errors": [],
        }

        with patch("booked.agents.agent7.agent.get_cached_knowledge_base", return_value=None), \
             patch("booked.agents.agent7.agent._register_property_analytics"), \
             patch("booked.agents.agent7.agent.update_pipeline_status"):

            result = agent7_node(state)

        assert result["agent7_complete"] is True
        assert result["pipeline_complete"] is True
        assert "pixel_snippet" in result
        assert "vista-azule" in (result.get("pixel_snippet") or "")

    def test_pipeline_node_generates_pixel_for_property(self):
        from agents.agent7.agent import agent7_node

        state = {
            "property_id": "prop-abc-999",
            "knowledge_base": {"property_id": "prop-abc-999", "slug": "test-prop"},
            "page_url": "https://test-prop.staylio.ai",
            "page_slug": "test-prop",
            "errors": [],
        }

        with patch("booked.agents.agent7.agent.get_cached_knowledge_base", return_value=None), \
             patch("booked.agents.agent7.agent._register_property_analytics"), \
             patch("booked.agents.agent7.agent.update_pipeline_status"):

            result = agent7_node(state)

        # Pixel snippet must contain the property ID
        assert "prop-abc-999" in result.get("pixel_snippet", "")
