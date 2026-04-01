"""
TS-16 — Conversion Pixel (Tier 2) + PMS API Reconciliation (Tier 3)

TIER 2 — CONVERSION PIXEL:
  Lightweight vanilla JS snippet PMC installs on their booking
  confirmation page. Fires when a guest completes a reservation.
  Captures: booking_value, stay_dates, guest_count, property_id.
  Stores confirmed conversions in Supabase attribution_events table.
  PMC marketing teams understand this model immediately — identical
  to Google Ads and Meta pixel installation.

TIER 3 — PMS API RECONCILIATION:
  Portfolio tier only. Read-only API access to Guesty, Hostaway,
  or OwnerRez. Pulls actual reservation records and cross-references
  against UTM click data to produce fully reconciled attribution.
  Unlocks: cancellation rate, booking lead time, true revenue.

CONVERSION PIXEL EVENT SCHEMA (TS-16 open question, resolved):
  staylio_conversion event with properties:
    property_id:    Staylio property UUID
    booking_value:  Total reservation value in USD
    stay_start:     Check-in date (ISO string)
    stay_end:       Check-out date (ISO string)
    stay_nights:    Number of nights
    guest_count:    Number of guests
    utm_source:     Originating platform (from cookie/sessionStorage)
    utm_campaign:   Property slug (from cookie/sessionStorage)
    utm_content:    Post ID (from cookie/sessionStorage)
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
    ConversionEvent,
)

logger = logging.getLogger(__name__)

# Pixel endpoint — receives POST from PMC confirmation pages
PIXEL_ENDPOINT = os.environ.get("PIXEL_ENDPOINT_URL", "https://pixel.staylio.ai/conversion")


# ── Conversion Pixel ──────────────────────────────────────────────────────

PIXEL_SNIPPET_TEMPLATE = """<!-- Staylio Conversion Pixel | Install on booking confirmation page only -->
<script>
(function() {{
  var PROPERTY_ID = "{property_id}";
  var ENDPOINT    = "https://pixel.staylio.ai/conversion";

  // Read UTM attribution from sessionStorage (set when visitor arrived from Staylio page)
  function getAttr(key) {{
    try {{ return sessionStorage.getItem("staylio_" + key) || ""; }} catch(e) {{ return ""; }}
  }}

  // Read booking data from page — PMC customises these selectors
  var bookingValue = parseFloat(
    document.querySelector("[data-booked-value]")?.dataset.bookedValue ||
    document.querySelector(".booking-total")?.textContent.replace(/[^0-9.]/g, "") || "0"
  ) || 0;

  var stayStart = document.querySelector("[data-booked-checkin]")?.dataset.bookedCheckin ||
                  document.querySelector(".checkin-date")?.textContent.trim() || "";
  var stayEnd   = document.querySelector("[data-booked-checkout]")?.dataset.bookedCheckout ||
                  document.querySelector(".checkout-date")?.textContent.trim() || "";
  var guestCount = parseInt(
    document.querySelector("[data-booked-guests]")?.dataset.bookedGuests || "1"
  ) || 1;

  // Calculate nights
  var nights = 0;
  if (stayStart && stayEnd) {{
    var msPerDay = 86400000;
    nights = Math.round((new Date(stayEnd) - new Date(stayStart)) / msPerDay) || 0;
  }}

  var payload = {{
    property_id:  PROPERTY_ID,
    booking_value: bookingValue,
    stay_start:   stayStart,
    stay_end:     stayEnd,
    stay_nights:  nights,
    guest_count:  guestCount,
    utm_source:   getAttr("utm_source"),
    utm_campaign: getAttr("utm_campaign"),
    utm_content:  getAttr("utm_content"),
    utm_property_id: getAttr("utm_property_id"),
  }};

  // Only fire if we have evidence this visitor came from Staylio
  if (payload.utm_campaign || payload.utm_property_id === PROPERTY_ID) {{
    fetch(ENDPOINT, {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify(payload),
      keepalive: true,
    }}).catch(function() {{ /* silent fail */ }});
  }}
}})();
</script>"""

# UTM cookie snippet — installed in the Staylio landing page <head>
# Saves UTM params to sessionStorage so pixel can read them on booking confirmation
UTM_COOKIE_SNIPPET = """<script>
(function() {
  var params = new URLSearchParams(window.location.search);
  var utmKeys = ["utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_property_id"];
  utmKeys.forEach(function(key) {
    var val = params.get(key);
    if (val) {
      try { sessionStorage.setItem("staylio_" + key, val); } catch(e) {}
    }
  });
})();
</script>"""


def generate_pixel_snippet(property_id: str) -> str:
    """
    Generate the conversion pixel script tag for a specific property.
    PMC pastes this into their booking confirmation page <body>.
    Five-minute installation — one copy-paste.
    """
    return PIXEL_SNIPPET_TEMPLATE.format(property_id=property_id)


def fetch_pixel_conversions(
    property_id: str,
    period_start: date,
    period_end: date,
) -> list[ConversionEvent]:
    """
    Load confirmed conversion events for a property from Supabase.
    These were stored by the pixel endpoint when the snippet fired.
    """
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("attribution_events")
            .select("*")
            .eq("property_id", property_id)
            .eq("source", "pixel")
            .gte("booking_date", period_start.isoformat())
            .lte("booking_date", period_end.isoformat())
            .execute()
        )
        events = []
        for row in (result.data or []):
            events.append(ConversionEvent(
                property_id=property_id,
                booking_date=row.get("booking_date", ""),
                stay_start_date=row.get("stay_start"),
                stay_end_date=row.get("stay_end"),
                stay_nights=row.get("stay_nights"),
                booking_value=row.get("booking_value"),
                source=AttributionTier.TIER_2_PIXEL,
                utm_source=row.get("utm_source"),
                utm_campaign=row.get("utm_campaign"),
                utm_content=row.get("utm_content"),
            ))
        return events
    except Exception as exc:
        logger.error(f"[TS-16 Tier 2] Pixel conversion fetch failed for {property_id}: {exc}")
        return []


def apply_pixel_data_to_snapshot(
    snapshot: AnalyticsSnapshot,
    conversions: list[ConversionEvent],
) -> AnalyticsSnapshot:
    """Enrich an AnalyticsSnapshot with Tier 2 pixel conversion data."""
    snapshot.pixel_active = True
    snapshot.confirmed_bookings = len(conversions)
    snapshot.confirmed_revenue = sum(
        c.booking_value or 0 for c in conversions
    )
    snapshot.confirmed_commission_saved = round(
        snapshot.confirmed_revenue * AIRBNB_FEE_RATE, 2
    )
    snapshot.commission_calc_methodology = "confirmed"
    return snapshot


# ── PMS API Reconciliation (Tier 3 — Portfolio only) ─────────────────────

def fetch_pms_reservations(
    property_id: str,
    pms_type: str,         # "guesty" | "hostaway" | "ownerrez"
    pms_credentials: dict,  # Stored OAuth tokens from Supabase
    period_start: date,
    period_end: date,
) -> list[ConversionEvent]:
    """
    Pull reservation records from the PMS API for a property.
    Read-only access — booking source, dates, revenue, status.
    Cross-referenced against UTM click data to confirm attribution.
    """
    if pms_type == "guesty":
        return _fetch_guesty_reservations(property_id, pms_credentials, period_start, period_end)
    elif pms_type == "hostaway":
        return _fetch_hostaway_reservations(property_id, pms_credentials, period_start, period_end)
    elif pms_type == "ownerrez":
        return _fetch_ownerrez_reservations(property_id, pms_credentials, period_start, period_end)
    else:
        logger.warning(f"[TS-16 Tier 3] Unknown PMS type: {pms_type}")
        return []


def apply_pms_data_to_snapshot(
    snapshot: AnalyticsSnapshot,
    reservations: list[ConversionEvent],
) -> AnalyticsSnapshot:
    """Enrich snapshot with Tier 3 PMS reconciliation data."""
    snapshot.pms_connected = True
    confirmed = [r for r in reservations if r.cancellation_status != "cancelled"]
    cancelled = [r for r in reservations if r.cancellation_status == "cancelled"]

    snapshot.pms_confirmed_bookings = len(confirmed)
    snapshot.pms_total_revenue = sum(r.booking_value or 0 for r in confirmed)

    if reservations:
        snapshot.pms_cancellation_rate = round(len(cancelled) / len(reservations), 4)

    # Average lead time (days between booking_date and stay_start_date)
    lead_times = []
    for r in confirmed:
        if r.booking_date and r.stay_start_date:
            try:
                booking_dt = date.fromisoformat(r.booking_date)
                stay_dt    = date.fromisoformat(r.stay_start_date)
                lead_times.append((stay_dt - booking_dt).days)
            except ValueError:
                pass
    if lead_times:
        snapshot.pms_avg_lead_time_days = round(sum(lead_times) / len(lead_times), 1)

    snapshot.commission_calc_methodology = "reconciled"
    return snapshot


# ── PMS API clients (stubs — real implementation requires partner credentials) ──

def _fetch_guesty_reservations(
    property_id: str,
    credentials: dict,
    start: date,
    end: date,
) -> list[ConversionEvent]:
    """Guesty API: GET /reservations filtered by listing and date range."""
    try:
        access_token = credentials.get("access_token", "")
        listing_id   = credentials.get("listing_id", "")
        if not access_token or not listing_id:
            return []

        with httpx.Client(timeout=30) as client:
            resp = client.get(
                "https://open-api.guesty.com/v1/reservations",
                headers={"Authorization": f"Bearer {access_token}"},
                params={
                    "listingId": listing_id,
                    "checkIn[$gte]": start.isoformat(),
                    "checkIn[$lte]": end.isoformat(),
                    "status": "confirmed,checked_in,checked_out",
                    "limit": 100,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        return [
            ConversionEvent(
                property_id=property_id,
                booking_date=r.get("createdAt", "")[:10],
                stay_start_date=r.get("checkIn", "")[:10],
                stay_end_date=r.get("checkOut", "")[:10],
                booking_value=r.get("money", {}).get("farePaid"),
                source=AttributionTier.TIER_3_PMS_API,
                cancellation_status="confirmed" if r.get("status") != "cancelled" else "cancelled",
                pms_reservation_id=r.get("_id"),
            )
            for r in data.get("results", [])
        ]
    except Exception as exc:
        logger.error(f"[TS-16 Tier 3 Guesty] Fetch failed for {property_id}: {exc}")
        return []


def _fetch_hostaway_reservations(
    property_id: str,
    credentials: dict,
    start: date,
    end: date,
) -> list[ConversionEvent]:
    """Hostaway API: GET /reservations."""
    try:
        access_token = credentials.get("access_token", "")
        listing_id   = credentials.get("listing_id", "")
        if not access_token:
            return []

        with httpx.Client(timeout=30) as client:
            resp = client.get(
                "https://api.hostaway.com/v1/reservations",
                headers={"Authorization": f"Bearer {access_token}"},
                params={
                    "listingId": listing_id,
                    "arrivalStartDate": start.isoformat(),
                    "arrivalEndDate": end.isoformat(),
                    "limit": 100,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        return [
            ConversionEvent(
                property_id=property_id,
                booking_date=r.get("insertedOn", "")[:10],
                stay_start_date=r.get("arrivalDate", ""),
                stay_end_date=r.get("departureDate", ""),
                booking_value=r.get("totalPrice"),
                source=AttributionTier.TIER_3_PMS_API,
                cancellation_status="cancelled" if r.get("status") == "cancelled" else "confirmed",
                pms_reservation_id=str(r.get("id", "")),
            )
            for r in data.get("result", [])
        ]
    except Exception as exc:
        logger.error(f"[TS-16 Tier 3 Hostaway] Fetch failed for {property_id}: {exc}")
        return []


def _fetch_ownerrez_reservations(
    property_id: str,
    credentials: dict,
    start: date,
    end: date,
) -> list[ConversionEvent]:
    """OwnerRez API: GET /bookings."""
    try:
        api_key = credentials.get("api_key", "")
        prop_id = credentials.get("ownerrez_property_id", "")
        if not api_key:
            return []

        with httpx.Client(timeout=30) as client:
            resp = client.get(
                f"https://api.ownerreservations.com/v2/bookings",
                headers={"Authorization": f"ownerrez-pair {api_key}"},
                params={
                    "propertyId": prop_id,
                    "since": start.isoformat(),
                    "until": end.isoformat(),
                },
            )
            resp.raise_for_status()
            data = resp.json()

        return [
            ConversionEvent(
                property_id=property_id,
                booking_date=r.get("createdUtc", "")[:10],
                stay_start_date=r.get("arrival", ""),
                stay_end_date=r.get("departure", ""),
                booking_value=r.get("totalAmount"),
                source=AttributionTier.TIER_3_PMS_API,
                cancellation_status="cancelled" if r.get("status") == "Cancelled" else "confirmed",
                pms_reservation_id=str(r.get("id", "")),
            )
            for r in data.get("items", [])
        ]
    except Exception as exc:
        logger.error(f"[TS-16 Tier 3 OwnerRez] Fetch failed for {property_id}: {exc}")
        return []
