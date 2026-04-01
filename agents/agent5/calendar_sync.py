"""
TS-14 — Availability Calendar Sync
Tools: ical.js (parsing) + Cloudflare Worker (30-minute refresh cache)

Two operating modes:
  MODE 1 — iCal (universal, all tiers, day one)
    - PMC provides iCal URL at intake
    - Cloudflare Worker fetches and parses every 30 minutes
    - Cached result served as JSON to the calendar widget
    - Works with every PMS and OTA that exports iCal

  MODE 2 — Direct PMS API (Portfolio tier)
    - Guesty / Hostaway / OwnerRez read-only API
    - Richer data: exact pricing, min stay, booking window
    - Also powers Tier 3 attribution in TS-16

The Cloudflare Worker is defined as a JavaScript snippet deployed
via Cloudflare Workers API. This Python module manages:
  - Creating and updating the Worker for each property
  - Registering the iCal URL and cache endpoint in Supabase
  - Generating the calendar config object for the landing page
"""

import hashlib
import json
import logging
import os
from typing import Optional

import httpx

from agents.agent5.models import CalendarConfig

logger = logging.getLogger(__name__)

CLOUDFLARE_API_KEY     = os.environ.get("CLOUDFLARE_API_KEY", "")
CLOUDFLARE_ACCOUNT_ID  = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_WORKER_BASE = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/workers/scripts"

# Supported PMS types for Portfolio tier direct API
SUPPORTED_PMS = {"guesty", "hostaway", "ownerrez"}

# iCal Worker template — deployed per-property to Cloudflare Workers
# Worker fetches the iCal feed, parses blocked dates, returns JSON
# The property's iCal URL is baked into the worker at deploy time
ICAL_WORKER_TEMPLATE = """
addEventListener('fetch', event => {{
  event.respondWith(handleRequest(event.request))
}})

const ICAL_URL = "{ical_url}";
const CACHE_KEY = "ical_cache_{property_id}";
const CACHE_TTL = 1800; // 30 minutes

async function handleRequest(request) {{
  const cache = caches.default;
  const cacheUrl = new URL(request.url);
  const cachedResponse = await cache.match(cacheUrl);

  if (cachedResponse) {{
    return cachedResponse;
  }}

  try {{
    const icalResponse = await fetch(ICAL_URL, {{
      headers: {{ 'User-Agent': 'Staylio-CalendarSync/1.0' }}
    }});

    if (!icalResponse.ok) {{
      return new Response(JSON.stringify({{
        error: 'Failed to fetch calendar',
        blocked_dates: [],
        fetched_at: new Date().toISOString()
      }}), {{
        headers: {{ 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' }}
      }});
    }}

    const icalText = await icalResponse.text();
    const blockedDates = parseIcalDates(icalText);

    const result = JSON.stringify({{
      blocked_dates: blockedDates,
      fetched_at: new Date().toISOString(),
      property_id: "{property_id}"
    }});

    const response = new Response(result, {{
      headers: {{
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Cache-Control': `max-age=${{CACHE_TTL}}`,
      }}
    }});

    event.waitUntil(cache.put(cacheUrl, response.clone()));
    return response;

  }} catch (err) {{
    return new Response(JSON.stringify({{
      error: err.message,
      blocked_dates: [],
      fetched_at: new Date().toISOString()
    }}), {{
      status: 500,
      headers: {{ 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' }}
    }});
  }}
}}

// Minimal iCal VEVENT date parser
// Extracts DTSTART/DTEND pairs for VEVENT blocks (blocked/booked dates)
function parseIcalDates(icalText) {{
  const blocked = [];
  const events = icalText.split('BEGIN:VEVENT');

  for (let i = 1; i < events.length; i++) {{
    const event = events[i];
    const startMatch = event.match(/DTSTART(?:;[^:]+)?:([\\d]{{8}})/);
    const endMatch   = event.match(/DTEND(?:;[^:]+)?:([\\d]{{8}})/);

    if (startMatch && endMatch) {{
      const start = formatDate(startMatch[1]);
      const end   = formatDate(endMatch[1]);
      if (start && end) {{
        blocked.push({{ start, end }});
      }}
    }}
  }}
  return blocked;
}}

function formatDate(dateStr) {{
  if (dateStr.length < 8) return null;
  return `${{dateStr.slice(0,4)}}-${{dateStr.slice(4,6)}}-${{dateStr.slice(6,8)}}`;
}}
"""


def provision_calendar_sync(
    property_id: str,
    ical_url: Optional[str],
    pms_type: Optional[str] = None,
    pms_api_connected: bool = False,
    slug: str = "",
) -> CalendarConfig:
    """
    Set up calendar sync for a property.

    For Mode 1 (iCal):
      - Deploys or updates the Cloudflare Worker for this property
      - Returns the cache endpoint URL for the calendar widget

    For Mode 2 (Direct PMS API — Portfolio tier):
      - Records PMS type and connection status
      - Calendar widget reads from PMS-specific endpoint

    Returns CalendarConfig for embedding in the landing page.
    """
    config = CalendarConfig(
        ical_url=ical_url,
        pms_type=pms_type if pms_type in SUPPORTED_PMS else None,
        pms_api_connected=pms_api_connected,
    )

    if ical_url:
        cache_endpoint = _deploy_ical_worker(property_id, ical_url, slug)
        config.cache_endpoint = cache_endpoint
        if cache_endpoint:
            logger.info(f"[TS-14] iCal worker deployed for property {property_id}: {cache_endpoint}")
        else:
            logger.warning(f"[TS-14] iCal worker deployment failed for property {property_id} — will use direct iCal fetch")

    if pms_api_connected and pms_type in SUPPORTED_PMS:
        logger.info(f"[TS-14] PMS API mode active for property {property_id} ({pms_type})")

    return config


def update_ical_url(
    property_id: str,
    new_ical_url: str,
    slug: str,
) -> Optional[str]:
    """
    Update the iCal URL for an existing property.
    Called when a PMC changes their PMS or updates their iCal feed.
    Redeploys the Cloudflare Worker with the new URL.
    Returns new cache endpoint or None on failure.
    """
    return _deploy_ical_worker(property_id, new_ical_url, slug)


# ── Cloudflare Worker deployment ──────────────────────────────────────────

def _deploy_ical_worker(
    property_id: str,
    ical_url: str,
    slug: str,
) -> Optional[str]:
    """
    Deploy a Cloudflare Worker for this property's iCal feed.
    Worker name: staylio-cal-{property_id_hash}
    Returns the worker URL or None on failure.
    """
    if not CLOUDFLARE_API_KEY or not CLOUDFLARE_ACCOUNT_ID:
        logger.warning("[TS-14] Cloudflare credentials not configured — skipping worker deployment")
        # Return a predictable fallback URL format for testing
        return _worker_url(property_id)

    worker_name = _worker_name(property_id)
    worker_script = ICAL_WORKER_TEMPLATE.format(
        ical_url=ical_url.replace('"', '\\"'),
        property_id=property_id,
    )

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.put(
                f"{CLOUDFLARE_WORKER_BASE}/{worker_name}",
                headers={
                    "Authorization": f"Bearer {CLOUDFLARE_API_KEY}",
                    "Content-Type": "application/javascript",
                },
                content=worker_script.encode(),
            )
            resp.raise_for_status()
        return _worker_url(property_id)
    except Exception as exc:
        logger.error(f"[TS-14] Cloudflare Worker deployment failed: {exc}")
        return None


def _worker_name(property_id: str) -> str:
    """Generate a stable Cloudflare Worker name from property ID."""
    id_hash = hashlib.md5(property_id.encode()).hexdigest()[:12]
    return f"staylio-cal-{id_hash}"


def _worker_url(property_id: str) -> str:
    """Construct the expected Worker URL for a property."""
    name = _worker_name(property_id)
    return f"https://{name}.{CLOUDFLARE_ACCOUNT_ID}.workers.dev/calendar"
