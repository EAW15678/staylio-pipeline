"""
Agent 8 — Stage 2: Concept Generator

Generates 8 social-content concepts per monthly cycle for a property.
Each concept is a premise — pre-visual, vendor-neutral, grounded in
the property's knowledge base material.

Standalone module. NOT wired into the pipeline graph.
Call directly: generate_concepts(property_id, cycle_month)

Concepts are written to public.concept_ledger with status='draft'.
Re-running for the same (property_id, cycle_month) supersedes the
prior active set and inserts at concept_version + 1, preserving
utm_content_slug as the stable join key for Zernio per-post metrics.
"""

import hashlib
import json
import logging
import os
import re
from datetime import date, datetime, timezone
from typing import Optional

import anthropic

from agents.agent8.retry import retry_with_backoff

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 4096


# ── KB loader (same as Agent 2/3) ───────────────────────────────────────

def _load_kb(property_id: str) -> Optional[dict]:
    """
    Load PropertyKnowledgeBase from Redis cache, matching the pattern
    used by Agent 2 (agents/agent2/agent.py:82) and Agent 3
    (agents/agent3/agent.py:91).
    """
    from core.pipeline_status import get_cached_knowledge_base
    return get_cached_knowledge_base(property_id)


def _load_kb_from_supabase(property_id: str) -> Optional[dict]:
    """
    Fallback: load KB from Supabase property_knowledge_bases table.
    Used when Redis cache is cold (property was built but cache expired).
    """
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("property_knowledge_bases")
            .select("knowledge_base")
            .eq("property_id", property_id)
            .limit(1)
            .execute()
        )
        if result.data and result.data[0].get("knowledge_base"):
            return result.data[0]["knowledge_base"]
        return None
    except Exception as exc:
        logger.error(f"[Agent 8] Supabase KB load failed for {property_id}: {exc}")
        return None


# ── Helpers ──────────────────────────────────────────────────────────────

def _val(kb: dict, key: str) -> str:
    """Extract a value from KB dict, handling PropertyField dicts."""
    f = kb.get(key)
    if isinstance(f, dict):
        return str(f.get("value") or "")
    return str(f or "")


def _extract_amenity_names(kb: dict) -> list[str]:
    """
    Extract the structured amenity list from the KB.
    KB stores amenities as list[PropertyField], serialized as
    [{"value": "Hot tub", "source": "airbnb", "confidence": 0.8}, ...].
    """
    amenities = kb.get("amenities") or []
    names = []
    for a in amenities:
        if isinstance(a, dict):
            v = a.get("value")
            if v:
                names.append(str(v))
        elif isinstance(a, str) and a:
            names.append(a)
    return names


def _extract_guest_book_entries(kb: dict) -> list[dict]:
    """Extract guest book entries from KB's guest_reviews list."""
    reviews = kb.get("guest_reviews") or []
    entries = []
    for r in reviews:
        if isinstance(r, dict) and r.get("is_guest_book"):
            entries.append({
                "text": r.get("text", ""),
                "reviewer_name": r.get("reviewer_name"),
                "stay_date": r.get("stay_date"),
            })
    return entries


def _build_utm_content_slug(
    property_slug: str, cycle_month: date, concept_number: int
) -> str:
    """
    Deterministic utm_content_slug for Zernio per-post metrics join.

    STABILITY REQUIREMENT: this slug is the join key Zernio returns
    through utm_content. It MUST be derived from (property_slug,
    cycle_month, concept_number) ONLY — never from the concept title,
    which changes on re-authoring. The slug is carried forward across
    concept_version bumps so metrics remain attributable.
    """
    month_str = cycle_month.strftime("%Y-%m")
    return f"{property_slug}_{month_str}_c{concept_number}"


# ── Prompt ───────────────────────────────────────────────────────────────

def _build_prompt(kb: dict, cycle_month: date) -> str:
    """Build the concept generation prompt from KB material."""
    vibe = kb.get("vibe_profile") or "unknown"
    name = _val(kb, "name")
    city = _val(kb, "city")
    state = _val(kb, "state")
    bedrooms = _val(kb, "bedrooms")
    bathrooms = _val(kb, "bathrooms")
    owner_story = kb.get("owner_story") or ""
    wow_factor = kb.get("wow_factor") or ""
    hidden_gems = kb.get("hidden_gems") or ""
    amenities = _extract_amenity_names(kb)
    guest_book = _extract_guest_book_entries(kb)

    amenity_str = ", ".join(amenities[:50]) if amenities else "None available"
    gb_str = ""
    if guest_book:
        gb_entries = []
        for gb in guest_book[:10]:
            text = gb["text"]
            who = gb.get("reviewer_name") or "Guest"
            gb_entries.append(f'  - "{text}" — {who}')
        gb_str = "\n".join(gb_entries)
    else:
        gb_str = "  None available"

    return f"""\
You are a social-content strategist for a vacation rental property.

Property:
  Name: {name}
  Location: {city}, {state}
  Layout: {bedrooms} bedrooms, {bathrooms} bathrooms
  Vibe profile: {vibe}
  Owner story: {owner_story[:500] if owner_story else 'Not provided'}
  Wow factor: {wow_factor[:300] if wow_factor else 'Not provided'}
  Hidden gems: {hidden_gems[:300] if hidden_gems else 'Not provided'}

Amenities (confirmed — do NOT fabricate any not on this list):
  {amenity_str}

Guest book entries (VERBATIM — do not reword these):
{gb_str}

Month: {cycle_month.strftime('%B %Y')}

Generate exactly 8 social-content concepts for this property's monthly cycle.

HARD RULES:
1. GUEST-BOOK TEXT IS VERBATIM. If a concept references a guest-book quote, use
   the exact text as written above. Trimming to a phrase boundary is permitted;
   rewording is NOT. Preserve spelling quirks and the guest's voice.
2. NO FABRICATED AMENITIES. A concept may only reference features listed above.
3. VIBE IS A PRIOR, NOT A TEMPLATE. The vibe is a starting hypothesis, not a
   format to fill. Eight concepts must be genuinely distinct from each other in
   angle, audience appeal, and emotional register.
4. NO SHOT SPEC. Concepts are pre-visual and vendor-neutral. No camera moves,
   no clip counts, no transitions, no music suggestions. Stage 3 decides those.
5. NO OTA LINKS. Do not reference Airbnb, VRBO, Booking.com, or any OTA by name.

For each concept return:
- title: a short, compelling working title (5-10 words)
- premise: 2-3 sentences describing the content idea, the angle, and why it
  would resonate with the target audience. Be specific to THIS property.
- source_material: an array of strings identifying which KB material this concept
  draws on (e.g. "guest_book: The Williamson Family", "amenity: Hot tub",
  "wow_factor", "hidden_gems", "owner_story"). Empty array if original.

Return ONLY valid JSON — no markdown, no prose, no code fences:
{{
  "concepts": [
    {{
      "title": "...",
      "premise": "...",
      "source_material": ["..."]
    }}
  ]
}}"""


# ── Claude call with retry ───────────────────────────────────────────────

def _call_claude(prompt: str) -> Optional[dict]:
    """
    Call Claude Sonnet with retry and backoff via agents.agent8.retry.

    Retries on API errors and rate limits. Concept-specific validation
    (exactly 8 concepts, valid JSON) is checked after the call; a wrong
    count raises ValueError to trigger a retry.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def _attempt() -> dict:
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
        concepts = parsed.get("concepts")
        if not isinstance(concepts, list) or len(concepts) != 8:
            count = len(concepts) if isinstance(concepts, list) else 0
            raise ValueError(f"Expected 8 concepts, got {count}")
        return parsed

    try:
        return retry_with_backoff(
            fn=_attempt,
            max_retries=3,
            backoff_base=2.0,
            retryable=(
                anthropic.RateLimitError,
                anthropic.APIError,
                json.JSONDecodeError,
                KeyError,
                ValueError,
            ),
            label="Claude concept generation",
        )
    except Exception as exc:
        logger.error("[Agent 8] Claude call failed after retries: %s", exc)
        return None


# ── DB write ─────────────────────────────────────────────────────────────

def _supersede_and_insert(
    sb, property_id: str, cycle_month: date,
    rows: list[dict],
) -> list[dict]:
    """
    Supersede existing active concepts and insert new ones.

    1. Find current active rows for (property_id, cycle_month)
    2. Set superseded_at on them
    3. Determine next concept_version
    4. Insert new rows with the same utm_content_slug values

    Ideally this would be a single Supabase RPC for atomicity.
    RPC DDL is provided in the report for review — not applied here.
    """
    # Find existing active rows
    existing = (
        sb.table("concept_ledger")
        .select("concept_id, concept_number, concept_version, utm_content_slug")
        .eq("property_id", property_id)
        .eq("cycle_month", cycle_month.isoformat())
        .is_("superseded_at", "null")
        .execute()
    )

    next_version = 1
    slug_map: dict[int, str] = {}  # concept_number -> existing slug

    if existing.data:
        for row in existing.data:
            v = row.get("concept_version", 1)
            if v >= next_version:
                next_version = v + 1
            slug_map[row["concept_number"]] = row["utm_content_slug"]

        # Supersede existing active rows
        now_iso = datetime.now(timezone.utc).isoformat()
        ids = [row["concept_id"] for row in existing.data]
        sb.table("concept_ledger").update(
            {"superseded_at": now_iso}
        ).in_("concept_id", ids).execute()

    # Assign version and carry forward slugs
    for row in rows:
        row["concept_version"] = next_version
        cn = row["concept_number"]
        if cn in slug_map:
            row["utm_content_slug"] = slug_map[cn]

    # Insert new rows
    sb.table("concept_ledger").insert(rows).execute()

    return rows


# ── Public entry point ───────────────────────────────────────────────────

def generate_concepts(
    property_id: str,
    cycle_month: date,
    *,
    dry_run: bool = False,
) -> list[dict]:
    """
    Generate 8 social-content concepts for a property's monthly cycle.

    Args:
        property_id: UUID of the property
        cycle_month: First day of the target month (e.g. date(2026, 9, 1))
        dry_run: If True, return concepts without writing to DB

    Returns:
        List of 8 concept dicts with title, premise, source_material,
        utm_content_slug, concept_number, concept_version.
    """
    logger.info(
        "[Agent 8] Generating concepts for property=%s cycle=%s dry_run=%s",
        property_id, cycle_month.isoformat(), dry_run,
    )

    # Load KB — Redis first, Supabase fallback
    kb = _load_kb(property_id)
    if not kb:
        logger.info("[Agent 8] Redis cache miss, trying Supabase for %s", property_id)
        kb = _load_kb_from_supabase(property_id)
    if not kb:
        raise ValueError(f"Knowledge base not found for property {property_id}")

    property_slug = kb.get("slug") or property_id[:8]
    vibe = kb.get("vibe_profile") or "unknown"

    # Generate concepts via Claude
    prompt = _build_prompt(kb, cycle_month)
    result = _call_claude(prompt)
    if not result or not result.get("concepts"):
        raise RuntimeError("Claude failed to generate concepts after retries")

    concepts = result["concepts"]
    if len(concepts) != 8:
        raise RuntimeError(f"Expected 8 concepts, got {len(concepts)}")

    # Build DB rows
    rows = []
    for i, concept in enumerate(concepts):
        cn = i + 1
        rows.append({
            "property_id": property_id,
            "cycle_month": cycle_month.isoformat(),
            "concept_number": cn,
            "concept_version": 1,  # overridden by _supersede_and_insert
            "utm_content_slug": _build_utm_content_slug(property_slug, cycle_month, cn),
            "title": concept.get("title", f"Concept {cn}"),
            "premise": concept.get("premise", ""),
            "vibe_prior": vibe,
            "evidence_basis": [],  # cold start
            "source_material": concept.get("source_material", []),
            "target_platforms": ["tiktok", "pinterest", "facebook", "instagram"],
            "status": "draft",
            "created_by_agent": "agent8_stage2",
        })

    logger.info(
        "[Agent 8] Generated %d concepts: %s",
        len(rows),
        [r["title"] for r in rows],
    )

    if dry_run:
        return rows

    # Write to concept_ledger (supersede + insert)
    from core.supabase_store import get_supabase
    sb = get_supabase()
    written = _supersede_and_insert(sb, property_id, cycle_month, rows)

    logger.info(
        "[Agent 8] Wrote %d concepts to concept_ledger for property=%s cycle=%s v%d",
        len(written), property_id, cycle_month.isoformat(),
        written[0].get("concept_version", 1) if written else 0,
    )

    return written
