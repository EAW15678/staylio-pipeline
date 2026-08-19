"""
GrowthBook A/B testing snippet for property landing pages.

Created 2026-08-19 (PAGE-2). Logic-preserving move from
agents/agent5/ab_testing.py. No database access, no agents/ imports.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

GROWTHBOOK_API_HOST = os.environ.get("GROWTHBOOK_API_HOST", "http://localhost:3100")
GROWTHBOOK_CLIENT_KEY = os.environ.get("GROWTHBOOK_CLIENT_KEY", "")

MIN_SESSIONS_FOR_EXPERIMENT = 100
AUTO_DEPLOY_CONFIDENCE = 0.95


def generate_growthbook_snippet(
    property_id: str,
    slug: str,
    hero_headline_variants: Optional[list] = None,
) -> str:
    """Generate the GrowthBook SDK JavaScript snippet to embed in the landing page.

    Returns an HTML <script> string for embedding before </body>.
    """
    if not GROWTHBOOK_CLIENT_KEY:
        return "<!-- GrowthBook: not configured -->"

    headline_variants = hero_headline_variants or []

    snippet = f"""
<script type="module">
  import {{ GrowthBook }} from "https://cdn.jsdelivr.net/npm/@growthbook/growthbook@latest/dist/bundles/esm.min.js";

  const gb = new GrowthBook({{
    apiHost: "{GROWTHBOOK_API_HOST}",
    clientKey: "{GROWTHBOOK_CLIENT_KEY}",
    trackingCallback: (experiment, result) => {{
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
    variations: [0, 1, 2],
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
