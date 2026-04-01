"""
TS-24 — A/B Testing & Page Optimization
Tool: GrowthBook (open source, self-hosted)

GrowthBook runs A/B tests on landing page elements to continuously
improve booking site click-through rates.

Architecture:
  - GrowthBook SDK injected as JavaScript in every landing page
  - Experiments defined in GrowthBook dashboard
  - Variant assignment happens client-side on page load
  - Results feed back to GrowthBook via event tracking
  - Agent 5 auto-deploys the winning variant when significance reached

First batch of experiments (from TS-24):
  1. Hero headline variants (highest-leverage test)
  2. CTA button copy (Check Availability vs Book Now vs See Dates)
  3. Hero image category selection
  4. Social proof placement (guest book above vs below features)

Schedule: build Weeks 7-8, requires live pages with real traffic.
The SDK injection is built at page generation time from Phase 1
so experiments can activate without page rebuilds.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

GROWTHBOOK_API_HOST = os.environ.get("GROWTHBOOK_API_HOST", "http://localhost:3100")
GROWTHBOOK_CLIENT_KEY = os.environ.get("GROWTHBOOK_CLIENT_KEY", "")

# Minimum traffic before an experiment is meaningful
MIN_SESSIONS_FOR_EXPERIMENT = 100

# Confidence level at which winner is auto-deployed
AUTO_DEPLOY_CONFIDENCE = 0.95


def generate_growthbook_snippet(
    property_id: str,
    slug: str,
    hero_headline_variants: Optional[list[str]] = None,
) -> str:
    """
    Generate the GrowthBook SDK JavaScript snippet to embed in the landing page.

    The snippet:
    1. Loads the GrowthBook SDK
    2. Initialises with the property's experiment configuration
    3. Applies variant assignments on page load
    4. Tracks conversion events (booking site clicks)

    Returns an HTML <script> string for embedding before </body>.
    """
    if not GROWTHBOOK_CLIENT_KEY:
        # Return a no-op snippet — page works without GrowthBook
        return "<!-- GrowthBook: not configured -->"

    # Headline variants for the first experiment batch
    # If Claude generated alternatives, use them; otherwise use defaults
    headline_variants = hero_headline_variants or []

    snippet = f"""
<script type="module">
  import {{ GrowthBook }} from "https://cdn.jsdelivr.net/npm/@growthbook/growthbook@latest/dist/bundles/esm.min.js";

  const gb = new GrowthBook({{
    apiHost: "{GROWTHBOOK_API_HOST}",
    clientKey: "{GROWTHBOOK_CLIENT_KEY}",
    // Track experiment impressions and conversions
    trackingCallback: (experiment, result) => {{
      // Send to GA4 via gtag
      if (typeof gtag !== "undefined") {{
        gtag("event", "experiment_impression", {{
          experiment_id: experiment.key,
          variant_id: result.variationId,
          property_id: "{property_id}",
        }});
      }}
    }},
    attributes: {{
      property_id: "{property_id}",
      property_slug: "{slug}",
    }},
  }});

  await gb.loadFeatures({{ timeout: 1000 }});

  // ── Experiment 1: Hero Headline ───────────────────────────────────────
  const headlineResult = gb.run({{
    key: "hero-headline-{slug}",
    variations: [0, 1, 2],  // 0=control, 1=variant_a, 2=variant_b
  }});

  const headlineEl = document.getElementById("hero-headline");
  if (headlineEl && headlineResult.value > 0) {{
    const variants = {headline_variants!r};
    if (variants[headlineResult.value - 1]) {{
      headlineEl.textContent = variants[headlineResult.value - 1];
    }}
  }}

  // ── Experiment 2: CTA Button Copy ────────────────────────────────────
  const ctaResult = gb.run({{
    key: "cta-copy-{slug}",
    variations: ["Check Availability", "Book Now", "See Available Dates"],
  }});

  document.querySelectorAll(".staylio-cta-btn").forEach(btn => {{
    btn.textContent = ctaResult.value;
  }});

  // ── Conversion tracking: booking site click ───────────────────────────
  document.querySelectorAll(".staylio-cta-btn").forEach(btn => {{
    btn.addEventListener("click", () => {{
      gb.track("booking_site_click", {{
        property_id: "{property_id}",
        headline_variant: headlineResult.variationId,
        cta_variant: ctaResult.value,
      }});
      if (typeof gtag !== "undefined") {{
        gtag("event", "booking_site_click", {{
          property_id: "{property_id}",
        }});
      }}
    }});
  }});
</script>
"""
    return snippet


def get_winning_variants(property_id: str, slug: str) -> dict[str, str]:
    """
    Query GrowthBook API for winning variants of a property's experiments.
    Returns dict of {experiment_key: winning_value} for auto-deployment.

    Called by Agent 5 during page rebuilds to apply confirmed winners.
    If no experiment has reached significance, returns empty dict.
    """
    if not GROWTHBOOK_CLIENT_KEY:
        return {}

    try:
        import httpx
        with httpx.Client(timeout=10) as client:
            resp = client.get(
                f"{GROWTHBOOK_API_HOST}/api/v1/experiments",
                headers={"Authorization": f"Bearer {GROWTHBOOK_CLIENT_KEY}"},
                params={"project": slug},
            )
            resp.raise_for_status()
            experiments = resp.json().get("experiments", [])

        winners = {}
        for exp in experiments:
            if exp.get("status") == "stopped" and exp.get("winner") is not None:
                winners[exp["trackingKey"]] = exp["winner"]
        return winners

    except Exception as exc:
        logger.warning(f"[TS-24] GrowthBook API unavailable for property {property_id}: {exc}")
        return {}
