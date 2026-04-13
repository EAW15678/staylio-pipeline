"""
Staylio Pipeline — Cost Emitter

Thin HTTP bridge from pipeline agents to the Staylio Cost Console.
Emits real-time cost estimates after LLM and media API calls.
Runs fire-and-forget in a daemon thread — never blocks the pipeline.

Required env vars:
  COST_CONSOLE_URL    Base URL of the cost console
  COST_CONSOLE_TOKEN  Bearer token matching CONSOLE_API_TOKEN on the console

If these vars are not set, all calls are silent no-ops.
"""

from __future__ import annotations

import logging
import os
import threading
from decimal import Decimal

logger = logging.getLogger(__name__)

# ── Pricing tables (keep in sync with staylio-cost-console/pipeline_emitter.py) ──

_LLM_PRICING: dict[str, tuple[Decimal, Decimal]] = {
    "claude-sonnet-4-6":          (Decimal("3.00"),  Decimal("15.00")),
    "claude-haiku-4-5-20251001":  (Decimal("0.80"),  Decimal("4.00")),
    "gpt-4o":                     (Decimal("2.50"),  Decimal("10.00")),
    "gpt-4o-mini":                (Decimal("0.15"),  Decimal("0.60")),
}

_ELEVENLABS_PRICE_PER_1K_CHARS = Decimal("0.18")
_RUNWAY_PRICE_PER_5S_CLIP      = Decimal("0.25")   # Gen-4 Turbo, 5s clip
_CREATOMATE_PRICE_PER_RENDER   = Decimal("0.10")


def _compute_llm_cost(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    inp, out = _LLM_PRICING.get(model, (Decimal("3.00"), Decimal("15.00")))
    return (
        Decimal(input_tokens)  / Decimal(1_000_000) * inp
        + Decimal(output_tokens) / Decimal(1_000_000) * out
    ).quantize(Decimal("0.000001"))


def _post_estimate(payload: dict) -> None:
    """Blocking POST — called from a daemon thread only."""
    base = os.environ.get("COST_CONSOLE_URL", "").rstrip("/")
    token = os.environ.get("COST_CONSOLE_TOKEN", "")
    if not base:
        return
    try:
        import requests
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        requests.post(f"{base}/events/estimate", json=payload, headers=headers, timeout=5)
    except Exception as exc:
        logger.debug(f"[cost_emitter] POST failed (non-fatal): {exc}")


def _emit(payload: dict) -> None:
    """Submit payload in a daemon thread — never blocks the pipeline."""
    threading.Thread(target=_post_estimate, args=(payload,), daemon=True).start()


# ── Public emitters ──────────────────────────────────────────────────────────

def emit_llm_cost(
    *,
    vendor: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    property_id: str | None = None,
    workflow_name: str | None = None,
    slot_name: str | None = None,
    job_id: str | None = None,
    generation_reason: str | None = None,
) -> Decimal:
    """
    Emit a cost estimate after any LLM call.
    Returns the estimated cost (so callers can log it if desired).
    """
    cost = _compute_llm_cost(model, input_tokens, output_tokens)
    _emit({
        "vendor_id": vendor,
        "model": model,
        "service_name": "chat_completions",
        "estimated_cost_usd": str(cost),
        "property_id": property_id,
        "workflow_name": workflow_name,
        "slot_name": slot_name,
        "job_id": job_id,
        "generation_reason": generation_reason,
        "environment": "production",
        "raw_payload_json": {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    })
    return cost


def emit_media_cost(
    *,
    vendor: str,
    service: str,
    units: int | float,
    unit_name: str,
    property_id: str | None = None,
    workflow_name: str | None = None,
    slot_name: str | None = None,
    job_id: str | None = None,
    generation_reason: str | None = None,
) -> Decimal:
    """
    Emit a cost estimate after ElevenLabs, Runway, or Creatomate calls.

    vendor / service   units        unit_name
    "elevenlabs"/"tts"  char count  "characters"
    "runway"/"gen4_5s"  5           "seconds"
    "creatomate"/"video_render"  1  "renders"
    """
    if vendor == "elevenlabs":
        cost = (Decimal(str(units)) / Decimal(1000)) * _ELEVENLABS_PRICE_PER_1K_CHARS
    elif vendor == "runway":
        cost = _RUNWAY_PRICE_PER_5S_CLIP  # flat per 5s clip
    elif vendor == "creatomate":
        cost = _CREATOMATE_PRICE_PER_RENDER * Decimal(str(units))
    else:
        cost = Decimal("0")
    cost = cost.quantize(Decimal("0.000001"))

    _emit({
        "vendor_id": vendor,
        "service_name": service,
        "estimated_cost_usd": str(cost),
        "property_id": property_id,
        "workflow_name": workflow_name,
        "slot_name": slot_name,
        "job_id": job_id,
        "generation_reason": generation_reason,
        "environment": "production",
        "raw_payload_json": {
            "service": service,
            "units": units,
            "unit_name": unit_name,
        },
    })
    return cost
