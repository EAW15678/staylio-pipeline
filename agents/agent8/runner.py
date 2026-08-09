"""
agents/agent8/runner.py — Agent 8 cycle runner.

Entry point: run_cycle(property_id, cycle_month)

TRIGGER MODEL (read before wiring):
    run_cycle is MANUAL for now, invoked by hand. It must be callable
    by a scheduler later without changing shape.

    Do NOT wire this to pipeline completion. Property onboarding is a
    moment; the content engine is a cadence. The pipeline fires once
    per property; Agent 8 runs monthly, forever. A trigger hung off
    pipeline completion fires cycle one and nothing starts month two.

    The long-term trigger is Agent 6 — the social scheduler, currently
    switched off by design — firing Agent 8 monthly per property, with
    intensity varying by where the property sits in its lifecycle (paid
    placement is heavy in months 1-2 and tapers as SEO takes over).
    cycle_month already exists on concept_ledger, so the monthly cadence
    is expressible today even while invocation is manual.
"""

import logging
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)


def run_cycle(
    property_id: str,
    cycle_month: date,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> list[dict]:
    """
    Run the Agent 8 content cycle for a property.

    Currently runs Stage 2 (concept generation) only. Later stages
    plug in here as they are built — the runner grows with the stages
    rather than preceding them.

    Resumability: checks concept_ledger for existing active concepts
    for this (property_id, cycle_month). If found, returns them without
    regenerating. This is derived from the status columns and partial
    unique indexes already on the table — no parallel state store.

    Parameter semantics and cost:

        dry_run=False, force=False (default)
            If 8 active concepts exist: returns them. NO model call. $0.
            If not: generates 8, writes to DB. ~$0.03 (one Claude call).

        dry_run=True, force=False
            If 8 active concepts exist: returns them. NO model call. $0.
            If not: generates 8, returns WITHOUT writing. ~$0.03.

        dry_run=False, force=True
            Ignores existing concepts, regenerates, writes to DB
            (supersedes prior set). ~$0.03.

        dry_run=True, force=True
            Ignores existing concepts, regenerates, returns WITHOUT
            writing. ~$0.03. The "let me see a different set before
            committing" case — the only combination that costs money
            without producing a row.

    Args:
        property_id: UUID of the property.
        cycle_month: First day of the target month (e.g. date(2026, 9, 1)).
        dry_run: If True, return concepts without writing to DB.
        force: If True, ignore existing concepts and regenerate.

    Returns:
        List of concept dicts from Stage 2.
    """
    logger.info(
        "[Agent 8] run_cycle: property=%s cycle=%s dry_run=%s force=%s",
        property_id, cycle_month.isoformat(), dry_run, force,
    )

    # ── Stage 2: Concept generation ──────────────────────────────────────

    if not force:
        existing = _check_existing_concepts(property_id, cycle_month)
        if existing:
            logger.info(
                "[Agent 8] %d active concepts already exist for property=%s cycle=%s, skipping generation",
                len(existing), property_id, cycle_month.isoformat(),
            )
            return existing

    from agents.agent8.concept_generator import generate_concepts

    concepts = generate_concepts(
        property_id=property_id,
        cycle_month=cycle_month,
        dry_run=dry_run,
    )

    logger.info(
        "[Agent 8] run_cycle complete: %d concepts, property=%s cycle=%s",
        len(concepts), property_id, cycle_month.isoformat(),
    )

    return concepts


def _check_existing_concepts(
    property_id: str, cycle_month: date
) -> Optional[list[dict]]:
    """
    Check concept_ledger for active (non-superseded) concepts for this
    property and cycle. Returns the rows if a complete set of 8 exists,
    None otherwise.
    """
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("concept_ledger")
            .select("concept_id, concept_number, title, premise, status, concept_version, utm_content_slug")
            .eq("property_id", property_id)
            .eq("cycle_month", cycle_month.isoformat())
            .is_("superseded_at", "null")
            .execute()
        )
        if result.data and len(result.data) == 8:
            return result.data
        return None
    except Exception as exc:
        logger.warning("[Agent 8] Failed to check existing concepts: %s", exc)
        return None
