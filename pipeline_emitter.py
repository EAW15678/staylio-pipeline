"""
Staylio Pipeline — Cost Emitter (Self-Contained)

Emits real-time cost estimates from pipeline agents into:
    cost_console.operational_estimates
    cost_console.cost_attribution

Self-contained: no imports from the cost-console repo.
Uses COST_CONSOLE_DSN env var and direct psycopg inserts.

CANONICAL WORKFLOW NAMES (strict enforcement):
    listing_generation
    property_enrichment
    page_rendering
    video_generation

Invalid workflow names are logged and skipped. Never crashes the pipeline.

STABLE ID RULES:
    - Deterministic: same inputs always produce the same ID
    - Uses job_id when available, otherwise composite of business fields
    - No timestamps — retries produce the same ID
    - Use discriminator for multiple events within the same slot

USAGE IN AGENTS:

    from pipeline_emitter import emit_llm_cost, emit_media_cost, emit_storage_cost

    # Agent 2 — LLM call:
    emit_llm_cost(
        vendor="anthropic",
        model="claude-sonnet-4-20250514",
        input_tokens=4200,
        output_tokens=800,
        property_id=property_id,
        workflow_name="listing_generation",
        slot_name="hero_text",
        job_id=job_id,
        generation_reason="landing_page_copy",
    )

    # Agent 3 — multiple Runway clips for same slot:
    emit_media_cost(
        vendor="runway",
        service="gen4_turbo_5s",
        units=5,
        unit_name="seconds",
        property_id=property_id,
        workflow_name="video_generation",
        slot_name="hero_video",
        job_id=job_id,
        generation_reason="runway_clip",
        discriminator="clip_00",
    )

    # Agent 3 — R2 upload:
    emit_storage_cost(
        vendor="cloudflare_r2",
        operation="upload",
        bytes_transferred=file_size_bytes,
        property_id=property_id,
        workflow_name="video_generation",
        slot_name="hero_video",
        job_id=job_id,
        generation_reason="hero_video_upload",
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from decimal import Decimal

import psycopg

UTC = timezone.utc
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inline stable_id — deterministic UUID-shaped ID from business fields
# ---------------------------------------------------------------------------

def _stable_id(*parts: str) -> str:
    """Deterministic ID from identifying strings. Same inputs always produce same ID."""
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


# ---------------------------------------------------------------------------
# Canonical workflow names — strict enforcement
# ---------------------------------------------------------------------------

CANONICAL_WORKFLOWS = frozenset({
    "listing_generation",
    "property_enrichment",
    "page_rendering",
    "video_generation",
})


def _validate_workflow(workflow_name: str | None, caller: str) -> bool:
    """Return True if valid or None. Log warning and return False otherwise."""
    if workflow_name is None:
        return True
    if workflow_name in CANONICAL_WORKFLOWS:
        return True
    logger.warning(
        "[cost_emitter] %s called with non-canonical workflow_name=%r — skipping. "
        "Allowed: %s",
        caller, workflow_name, ", ".join(sorted(CANONICAL_WORKFLOWS)),
    )
    return False


# ---------------------------------------------------------------------------
# Pricing tables — update when vendors change rates
# ---------------------------------------------------------------------------

LLM_PRICING: dict[str, tuple[Decimal, Decimal]] = {
    "gpt-4o":                       (Decimal("2.50"),  Decimal("10.00")),
    "gpt-4o-mini":                  (Decimal("0.15"),  Decimal("0.60")),
    "claude-sonnet-4-6":            (Decimal("3.00"),  Decimal("15.00")),
    "claude-sonnet-4-20250514":     (Decimal("3.00"),  Decimal("15.00")),
    "claude-haiku-4-5-20251001":    (Decimal("0.80"),  Decimal("4.00")),
}

ELEVENLABS_PRICING: dict[str, Decimal] = {
    "default": Decimal("0.18"),
}

CREATOMATE_PRICING: dict[str, Decimal] = {
    "video_render": Decimal("0.10"),
    "default":      Decimal("0.05"),
}

CLAID_PRICING: dict[str, Decimal] = {
    "enhance":  Decimal("0.012"),
    "upscale":  Decimal("0.015"),
    "default":  Decimal("0.010"),
}

RUNWAY_PRICING: dict[str, Decimal] = {
    "gen4_turbo_5s": Decimal("0.25"),
    "default":       Decimal("0.05"),
}

GCV_PRICING: dict[str, Decimal] = {
    "label_detection": Decimal("1.50"),
    "default":         Decimal("1.50"),
}

R2_CLASS_A_PER_MILLION  = Decimal("4.50")
R2_CLASS_B_PER_MILLION  = Decimal("0.36")
R2_STORAGE_PER_GB_MONTH = Decimal("0.015")


# ---------------------------------------------------------------------------
# Direct DB insert — minimal, self-contained, non-blocking
# ---------------------------------------------------------------------------

def _safe_emit(
    estimate_id: str,
    vendor_id: str,
    service_name: str | None,
    model: str | None,
    estimated_cost_usd: Decimal,
    property_id: str | None,
    workflow_name: str | None,
    slot_name: str | None,
    job_id: str | None,
    environment: str,
    generation_reason: str | None,
    occurred_at: datetime,
    raw_payload_json: dict,
    notes: str | None = None,
) -> None:
    """Insert into operational_estimates and cost_attribution. Never raises."""
    dsn = os.environ.get("COST_CONSOLE_DSN")
    if not dsn:
        logger.warning("[cost_emitter] COST_CONSOLE_DSN not set — skipping emission")
        return

    try:
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO cost_console.operational_estimates (
                    estimate_id, vendor_id, service_name, model,
                    estimated_cost_usd, property_id, workflow_name,
                    slot_name, job_id, environment, generation_reason,
                    occurred_at, raw_payload_json
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                ) ON CONFLICT (estimate_id) DO NOTHING
                """,
                (
                    estimate_id, vendor_id, service_name, model,
                    str(estimated_cost_usd), property_id, workflow_name,
                    slot_name, job_id, environment, generation_reason,
                    occurred_at, json.dumps(raw_payload_json),
                ),
            )

            if property_id or job_id:
                attr_id = _stable_id("attr", estimate_id)
                cur.execute(
                    """
                    INSERT INTO cost_console.cost_attribution (
                        attribution_id, usage_event_id, cost_event_id,
                        property_id, workflow_name, slot_name, job_id,
                        environment, attribution_method, attribution_confidence, notes
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    ) ON CONFLICT (attribution_id) DO NOTHING
                    """,
                    (
                        attr_id, None, None,
                        property_id, workflow_name, slot_name, job_id,
                        environment, "pipeline_emitted", "high", notes,
                    ),
                )
    except Exception as exc:
        logger.warning("[cost_emitter] Failed to emit cost event: %s", exc)


# ---------------------------------------------------------------------------
# Public emitters
# ---------------------------------------------------------------------------

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
    environment: str = "production",
    generation_reason: str | None = None,
    discriminator: str | None = None,
) -> Decimal:
    """
    Emit a cost estimate after any LLM call.
    ID is deterministic — retries will not create duplicate rows.
    Use discriminator only when multiple LLM calls occur for the same
    property/workflow/slot/reason.
    """
    if not _validate_workflow(workflow_name, "emit_llm_cost"):
        return Decimal(0)

    input_price, output_price = LLM_PRICING.get(model, (Decimal("3.00"), Decimal("15.00")))
    cost = (
        Decimal(input_tokens) / Decimal(1_000_000) * input_price
        + Decimal(output_tokens) / Decimal(1_000_000) * output_price
    ).quantize(Decimal("0.000001"))

    eid = _stable_id(
        "llm",
        job_id or "",
        property_id or "",
        workflow_name or "",
        slot_name or "",
        generation_reason or "",
        discriminator or "",
        vendor,
        model,
    )

    _safe_emit(
        estimate_id=eid,
        vendor_id=vendor,
        service_name="chat_completions",
        model=model,
        estimated_cost_usd=cost,
        property_id=property_id,
        workflow_name=workflow_name,
        slot_name=slot_name,
        job_id=job_id,
        environment=environment,
        generation_reason=generation_reason,
        occurred_at=datetime.now(UTC),
        raw_payload_json={
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "discriminator": discriminator,
        },
        notes=f"{vendor}/{model} call",
    )
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
    environment: str = "production",
    generation_reason: str | None = None,
    discriminator: str | None = None,
) -> Decimal:
    """
    Emit a cost estimate after ElevenLabs, Creatomate, Claid, or Runway calls.
    Use discriminator when multiple calls occur for the same slot:
        discriminator="clip_00"          (Runway clip index)
        discriminator="vibe_match_16_9"  (Creatomate format variant)
    """
    if not _validate_workflow(workflow_name, "emit_media_cost"):
        return Decimal(0)

    if vendor == "elevenlabs":
        price_per_1k = ELEVENLABS_PRICING.get(service, ELEVENLABS_PRICING["default"])
        cost = (Decimal(str(units)) / Decimal(1000)) * price_per_1k
    elif vendor == "creatomate":
        price_per_unit = CREATOMATE_PRICING.get(service, CREATOMATE_PRICING["default"])
        cost = Decimal(str(units)) * price_per_unit
    elif vendor == "claid":
        price_per_unit = CLAID_PRICING.get(service, CLAID_PRICING["default"])
        cost = Decimal(str(units)) * price_per_unit
    elif vendor == "runway":
        price_per_unit = RUNWAY_PRICING.get(service, RUNWAY_PRICING["default"])
        cost = Decimal(str(units)) * price_per_unit
    elif vendor in ("google_vision", "gcv"):
        price_per_1k = GCV_PRICING.get(service, GCV_PRICING["default"])
        cost = (Decimal(str(units)) / Decimal(1000)) * price_per_1k
    else:
        cost = Decimal("0")

    cost = cost.quantize(Decimal("0.000001"))

    eid = _stable_id(
        "media",
        job_id or "",
        property_id or "",
        workflow_name or "",
        slot_name or "",
        generation_reason or "",
        discriminator or "",
        vendor,
        service,
    )

    _safe_emit(
        estimate_id=eid,
        vendor_id=vendor,
        service_name=service,
        model=service,
        estimated_cost_usd=cost,
        property_id=property_id,
        workflow_name=workflow_name,
        slot_name=slot_name,
        job_id=job_id,
        environment=environment,
        generation_reason=generation_reason,
        occurred_at=datetime.now(UTC),
        raw_payload_json={
            "service": service,
            "units": units,
            "unit_name": unit_name,
            "discriminator": discriminator,
        },
        notes=f"{vendor}/{service} call — {units} {unit_name}",
    )
    return cost


def emit_storage_cost(
    *,
    vendor: str,
    operation: str,
    bytes_transferred: int = 0,
    request_count: int = 1,
    property_id: str | None = None,
    workflow_name: str | None = None,
    slot_name: str | None = None,
    job_id: str | None = None,
    environment: str = "production",
    generation_reason: str | None = None,
    discriminator: str | None = None,
) -> Decimal:
    """
    Emit a cost estimate for R2 storage operations.
    Call after every upload, download, or list operation against Cloudflare R2.

    Class A ops (charged at $4.50/million): upload, put, list, delete, copy
    Class B ops (charged at $0.36/million): download, get, head
    Storage cost is also estimated from bytes_transferred as a GB-month fraction.
    """
    if not _validate_workflow(workflow_name, "emit_storage_cost"):
        return Decimal(0)

    class_a_ops = {"upload", "put", "list", "delete", "copy"}
    is_class_a = operation.lower() in class_a_ops
    unit_cost = R2_CLASS_A_PER_MILLION if is_class_a else R2_CLASS_B_PER_MILLION
    op_cost = (Decimal(request_count) / Decimal(1_000_000)) * unit_cost

    storage_cost = Decimal(0)
    if bytes_transferred > 0:
        gb = Decimal(bytes_transferred) / Decimal(1_073_741_824)
        storage_cost = gb * R2_STORAGE_PER_GB_MONTH

    cost = (op_cost + storage_cost).quantize(Decimal("0.000001"))

    eid = _stable_id(
        "storage",
        job_id or "",
        property_id or "",
        workflow_name or "",
        slot_name or "",
        generation_reason or "",
        discriminator or "",
        vendor,
        operation,
    )

    _safe_emit(
        estimate_id=eid,
        vendor_id=vendor,
        service_name=f"r2_{operation}",
        model=None,
        estimated_cost_usd=cost,
        property_id=property_id,
        workflow_name=workflow_name,
        slot_name=slot_name,
        job_id=job_id,
        environment=environment,
        generation_reason=generation_reason,
        occurred_at=datetime.now(UTC),
        raw_payload_json={
            "operation": operation,
            "bytes_transferred": bytes_transferred,
            "request_count": request_count,
            "discriminator": discriminator,
        },
        notes=f"R2 {operation}",
    )
    return cost
