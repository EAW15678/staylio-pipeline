"""
LLM Vision Curation — Agent 3 Step 5b

One-time visual curation of all retained property images using Claude vision.

Strategy: Build PIL contact-sheet grids (3×4 = 12 images per sheet, each
thumbnail ≤320×240 px). Send all sheets in 1–2 Claude API calls. Claude
compares images against each other and returns a structured JSON curation
covering every retained photo.

Idempotency: keyed on SHA-256(sorted(asset_url_original)). If a 'complete'
record exists for (property_id, image_set_hash), results are returned
immediately with no LLM calls.

Entry point:
    run_llm_vision_curation(assets, enhanced_bytes_map, property_id, kb)
    → dict | None

Returns the curation dict (same shape as property_image_curations.per_image_results
+ property_recommendations) embedded under 'image_curation' in the visual_media
Redis payload. Returns None on any failure — pipeline continues normally.
"""

import base64
import hashlib
import io
import json
import logging
import math
import os
import re
from typing import Optional

import anthropic
import httpx
from PIL import Image, ImageDraw, ImageFont

from core.pipeline_status import cache_knowledge_base, get_cached_knowledge_base

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

_MODEL              = "claude-sonnet-4-6"
_CELL_W             = 320       # px per thumbnail cell
_CELL_H             = 240       # px per thumbnail cell (landscape-friendly)
_CELL_PAD           = 6         # px between cells
_GRID_COLS          = 3
_GRID_ROWS          = 4
_IMAGES_PER_SHEET   = _GRID_COLS * _GRID_ROWS   # 12
_MAX_IMAGES_TO_CURATE = 80      # hard cap — cost control
_SHEETS_PER_CALL    = 6         # max contact-sheet images per API call (within token budget)
_SUPABASE_TABLE     = "property_image_curations"
_REDIS_TTL          = 7 * 24 * 3600  # 7 days

# Must match _CATEGORY_MODULES display labels in agent5/page_builder.py exactly
_PHOTO_TOUR_SECTIONS = [
    "Exterior & Views",
    "Outdoor & Pool",
    "Living Room",
    "Kitchen",
    "Bedrooms",
    "Bathrooms",
    "Amenities & Extras",
]

_VALID_CATEGORIES = frozenset({
    "exterior", "view", "pool_hot_tub", "outdoor_entertaining",
    "living_room", "kitchen", "master_bedroom", "standard_bedroom",
    "bathroom", "game_entertainment", "local_area", "uncategorised",
})

_VALID_ROLES = frozenset({"hero", "supporting", "gallery_only", "exclude"})

_SHEET_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ── Public entry point ────────────────────────────────────────────────────────

def run_llm_vision_curation(
    assets: list,
    enhanced_bytes_map: dict,
    property_id: str,
    kb: dict,
) -> Optional[dict]:
    """
    Run one-time LLM vision curation for a property's image set.

    Args:
        assets:             List of MediaAsset objects (all retained photos).
        enhanced_bytes_map: {asset_url_original: enhanced_bytes} — in-memory
                            bytes for freshly Claid-enhanced images.
        property_id:        Property UUID string.
        kb:                 Knowledge base dict (for vibe context in prompt).

    Returns:
        dict with keys 'images' and 'property' (the full curation result),
        or None if curation failed or was skipped.

    Side-effects:
        - Writes to Supabase property_image_curations table.
        - Caches result in Redis under {property_id}:image_curation.
        - Does NOT modify any asset field (no side-effects on assets list).
    """
    if not assets:
        return None

    try:
        # Limit to _MAX_IMAGES_TO_CURATE — prioritise enhanced assets
        candidate_assets = sorted(
            assets,
            key=lambda a: (0 if a.asset_url_enhanced else 1),
        )[:_MAX_IMAGES_TO_CURATE]

        image_set_hash = _compute_image_set_hash(candidate_assets)

        # ── Cache check ───────────────────────────────────────────────────
        cached = _load_cached_curation(property_id, image_set_hash)
        if cached:
            logger.info(
                "[LLM Curator] Cache hit — reusing curation for property %s "
                "(hash=%s, %d images)",
                property_id, image_set_hash[:12], len(candidate_assets),
            )
            return cached

        logger.info(
            "[LLM Curator] Running LLM vision curation for property %s "
            "(%d images, hash=%s)",
            property_id, len(candidate_assets), image_set_hash[:12],
        )

        # ── Build index: cell_label → asset_id (R2 original URL) ─────────
        index_map: dict[str, str] = {}   # "A01" → asset_url_original
        asset_list: list = []            # ordered, parallel with index_map
        for i, asset in enumerate(candidate_assets):
            sheet_idx  = i // _IMAGES_PER_SHEET
            cell_idx   = i %  _IMAGES_PER_SHEET
            sheet_letter = _SHEET_LETTERS[sheet_idx % len(_SHEET_LETTERS)]
            label = f"{sheet_letter}{cell_idx + 1:02d}"
            index_map[label] = asset.asset_url_original
            asset_list.append(asset)

        # ── Fetch image bytes for contact sheet building ──────────────────
        bytes_by_original: dict[str, bytes] = {}
        for asset in asset_list:
            b = _fetch_image_bytes(asset, enhanced_bytes_map)
            if b:
                bytes_by_original[asset.asset_url_original] = b

        if not bytes_by_original:
            logger.warning(
                "[LLM Curator] No image bytes available for property %s — skipping curation",
                property_id,
            )
            return None

        # ── Build contact sheets ──────────────────────────────────────────
        n_sheets = math.ceil(len(asset_list) / _IMAGES_PER_SHEET)
        sheet_jpeg_list: list[bytes] = []  # one JPEG per sheet
        for sheet_idx in range(n_sheets):
            start = sheet_idx * _IMAGES_PER_SHEET
            sheet_assets = asset_list[start: start + _IMAGES_PER_SHEET]
            sheet_letter = _SHEET_LETTERS[sheet_idx % len(_SHEET_LETTERS)]
            cells: list[tuple[str, Optional[bytes]]] = []
            for cell_idx, asset in enumerate(sheet_assets):
                label = f"{sheet_letter}{cell_idx + 1:02d}"
                img_bytes = bytes_by_original.get(asset.asset_url_original)
                cells.append((label, img_bytes))
            sheet_jpeg = _build_contact_sheet(cells, sheet_letter)
            sheet_jpeg_list.append(sheet_jpeg)

        # ── Call LLM ──────────────────────────────────────────────────────
        result = _run_curation_call(
            sheet_jpeg_list=sheet_jpeg_list,
            index_map=index_map,
            kb=kb,
            property_id=property_id,
        )

        if result is None:
            logger.warning(
                "[LLM Curator] LLM call failed for property %s — pipeline continues with GCV path",
                property_id,
            )
            _mark_failed(property_id, image_set_hash, len(candidate_assets))
            return None

        # ── Persist to Supabase ───────────────────────────────────────────
        _save_curation(
            property_id=property_id,
            image_set_hash=image_set_hash,
            per_image_results=result["images"],
            property_recommendations=result["property"],
            asset_count=len(candidate_assets),
        )

        # ── Cache in Redis ────────────────────────────────────────────────
        cache_payload = {
            "status": "complete",
            "image_set_hash": image_set_hash,
            "images": result["images"],
            "property": result["property"],
        }
        cache_knowledge_base(
            f"{property_id}:image_curation",
            cache_payload,
            ttl_seconds=_REDIS_TTL,
        )

        logger.info(
            "[LLM Curator] Curation complete for property %s — "
            "%d images analysed, hero=%s",
            property_id,
            len(result["images"]),
            result.get("property", {}).get("page_hero"),
        )
        return cache_payload

    except Exception as exc:
        logger.error(
            "[LLM Curator] Unexpected error for property %s: %s — skipping curation",
            property_id, exc, exc_info=True,
        )
        return None


# ── Hash ──────────────────────────────────────────────────────────────────────

def _compute_image_set_hash(assets: list) -> str:
    """SHA-256 of sorted R2 original URLs. Stable across runs for same image set."""
    urls = sorted(a.asset_url_original for a in assets if a.asset_url_original)
    digest = hashlib.sha256("\n".join(urls).encode()).hexdigest()
    return digest


# ── Cache load ────────────────────────────────────────────────────────────────

def _load_cached_curation(property_id: str, image_set_hash: str) -> Optional[dict]:
    """
    Check Redis first, then Supabase for a complete curation record.
    Returns the curation dict if found and complete, else None.
    """
    # Redis (fast path)
    redis_key = f"{property_id}:image_curation"
    cached = get_cached_knowledge_base(redis_key)
    if cached and cached.get("status") == "complete":
        if cached.get("image_set_hash") == image_set_hash:
            return cached

    # Supabase (authoritative)
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table(_SUPABASE_TABLE)
            .select("per_image_results,property_recommendations,image_set_hash")
            .eq("property_id", property_id)
            .eq("image_set_hash", image_set_hash)
            .eq("status", "complete")
            .limit(1)
            .execute()
        )
        if result.data:
            row = result.data[0]
            payload = {
                "status": "complete",
                "image_set_hash": image_set_hash,
                "images": row.get("per_image_results") or [],
                "property": row.get("property_recommendations") or {},
            }
            # Backfill Redis
            cache_knowledge_base(redis_key, payload, ttl_seconds=_REDIS_TTL)
            return payload
    except Exception as exc:
        logger.warning("[LLM Curator] Supabase cache lookup failed: %s", exc)

    return None


# ── Image fetching ────────────────────────────────────────────────────────────

def _fetch_image_bytes(asset, enhanced_bytes_map: dict) -> Optional[bytes]:
    """
    Return image bytes for a MediaAsset, in priority order:
      1. enhanced_bytes_map (in-memory from this run)
      2. HTTP fetch of asset_url_enhanced
      3. HTTP fetch of asset_url_original
    Returns None if all sources fail.
    """
    # In-memory from this run
    b = enhanced_bytes_map.get(asset.asset_url_original)
    if b:
        return b

    # Download enhanced URL (cached from previous run)
    for url in [asset.asset_url_enhanced, asset.asset_url_original]:
        if not url:
            continue
        try:
            resp = httpx.get(url, timeout=15, follow_redirects=True)
            if resp.status_code == 200 and resp.content:
                return resp.content
        except Exception:
            pass

    return None


# ── Contact sheet builder ─────────────────────────────────────────────────────

def _build_contact_sheet(
    cells: list[tuple[str, Optional[bytes]]],
    sheet_letter: str,
) -> bytes:
    """
    Compose a PIL contact sheet from (label, image_bytes) pairs.
    Cells with None bytes render as a grey placeholder.
    Returns JPEG bytes.

    Layout:
      _GRID_COLS × _GRID_ROWS cells
      Each cell is _CELL_W × _CELL_H px
      _CELL_PAD px gap between cells
    """
    n_cells = len(cells)
    n_cols  = _GRID_COLS
    n_rows  = math.ceil(n_cells / n_cols)

    sheet_w = n_cols * _CELL_W + (n_cols + 1) * _CELL_PAD
    sheet_h = n_rows * _CELL_H + (n_rows + 1) * _CELL_PAD + 30  # +30 for header row

    canvas = Image.new("RGB", (sheet_w, sheet_h), color=(240, 240, 240))
    draw   = ImageDraw.Draw(canvas)

    # Sheet header
    try:
        header_font = ImageFont.load_default(size=18)
    except TypeError:
        header_font = ImageFont.load_default()
    draw.text(
        (8, 6),
        f"Sheet {sheet_letter}  ({n_cells} images)",
        fill=(60, 60, 60),
        font=header_font,
    )

    # Cell font for labels
    try:
        label_font = ImageFont.load_default(size=16)
    except TypeError:
        label_font = ImageFont.load_default()

    for idx, (label, img_bytes) in enumerate(cells):
        col = idx % n_cols
        row = idx // n_cols
        x0  = _CELL_PAD + col * (_CELL_W + _CELL_PAD)
        y0  = 30 + _CELL_PAD + row * (_CELL_H + _CELL_PAD)

        # Place image or placeholder
        if img_bytes:
            try:
                thumb = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                thumb.thumbnail((_CELL_W, _CELL_H), Image.LANCZOS)
                # Centre within cell
                off_x = (_CELL_W  - thumb.width)  // 2
                off_y = (_CELL_H - thumb.height) // 2
                canvas.paste(thumb, (x0 + off_x, y0 + off_y))
            except Exception:
                draw.rectangle([x0, y0, x0 + _CELL_W, y0 + _CELL_H], fill=(200, 200, 200))
        else:
            draw.rectangle([x0, y0, x0 + _CELL_W, y0 + _CELL_H], fill=(200, 200, 200))

        # Label badge (white rect + dark text in top-left of cell)
        badge_w, badge_h = 40, 22
        draw.rectangle([x0, y0, x0 + badge_w, y0 + badge_h], fill=(255, 255, 255))
        draw.text((x0 + 3, y0 + 3), label, fill=(20, 20, 20), font=label_font)

    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


# ── LLM call ─────────────────────────────────────────────────────────────────

def _run_curation_call(
    sheet_jpeg_list: list[bytes],
    index_map: dict[str, str],
    kb: dict,
    property_id: str,
) -> Optional[dict]:
    """
    Send all contact sheets to Claude in one or two API calls.
    Parses and validates the JSON response.
    Returns {images: [...], property: {...}} or None on failure.
    """
    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    except Exception as exc:
        logger.error("[LLM Curator] Failed to create Anthropic client: %s", exc)
        return None

    system_prompt = _build_system_prompt()

    # Split sheets across calls if needed (≤ _SHEETS_PER_CALL per call)
    # For 1–6 sheets: single call. For 7–12: two calls.
    all_images_parsed: list[dict] = []
    property_recs: dict = {}

    # If sheets fit in one call, do everything in one request.
    # If not, do per-image analysis in batches then a text-only synthesis pass.
    if len(sheet_jpeg_list) <= _SHEETS_PER_CALL:
        result = _single_call(client, system_prompt, sheet_jpeg_list, index_map, kb)
        if result is None:
            return None
        return result
    else:
        # Batch 1: first N sheets — per-image analysis only
        # Batch 2: remaining sheets — per-image analysis only
        # Synthesis: text-only pass combining all per-image results
        partial_results: list[dict] = []
        for batch_start in range(0, len(sheet_jpeg_list), _SHEETS_PER_CALL):
            batch = sheet_jpeg_list[batch_start: batch_start + _SHEETS_PER_CALL]
            batch_index = {
                k: v for k, v in index_map.items()
                if ord(k[0]) - ord("A") in range(
                    batch_start, min(batch_start + _SHEETS_PER_CALL, len(sheet_jpeg_list))
                )
            }
            partial = _batch_call(client, system_prompt, batch, batch_index, kb)
            if partial:
                partial_results.extend(partial)

        if not partial_results:
            return None

        property_recs = _synthesis_call(client, partial_results, index_map, kb)
        return {
            "images": partial_results,
            "property": property_recs or {},
        }


def _single_call(
    client: anthropic.Anthropic,
    system_prompt: str,
    sheet_jpeg_list: list[bytes],
    index_map: dict[str, str],
    kb: dict,
) -> Optional[dict]:
    """Single API call: all sheets + full JSON response including property-level."""
    content: list[dict] = []

    for sheet_bytes in sheet_jpeg_list:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(sheet_bytes).decode(),
            },
        })

    content.append({
        "type": "text",
        "text": _build_full_prompt(index_map, kb),
    })

    try:
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=8192,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )
        raw = resp.content[0].text
        return _parse_and_validate(raw, index_map)
    except Exception as exc:
        logger.error("[LLM Curator] Single call failed: %s", exc)
        return None


def _batch_call(
    client: anthropic.Anthropic,
    system_prompt: str,
    sheet_batch: list[bytes],
    batch_index: dict[str, str],
    kb: dict,
) -> list[dict]:
    """Batch call: returns only per-image results (no property-level)."""
    content: list[dict] = []
    for sheet_bytes in sheet_batch:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(sheet_bytes).decode(),
            },
        })
    content.append({
        "type": "text",
        "text": _build_per_image_only_prompt(batch_index, kb),
    })
    try:
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=6000,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )
        raw = resp.content[0].text
        parsed = _parse_and_validate(raw, batch_index)
        return parsed["images"] if parsed else []
    except Exception as exc:
        logger.error("[LLM Curator] Batch call failed: %s", exc)
        return []


def _synthesis_call(
    client: anthropic.Anthropic,
    per_image_results: list[dict],
    index_map: dict[str, str],
    kb: dict,
) -> dict:
    """Text-only synthesis: produces property-level recommendations from per-image results."""
    prompt = _build_synthesis_prompt(per_image_results, kb)
    try:
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text
        parsed = _extract_json(raw)
        if parsed and isinstance(parsed, dict):
            return _validate_property_recs(parsed, {r["asset_id"] for r in per_image_results})
    except Exception as exc:
        logger.error("[LLM Curator] Synthesis call failed: %s", exc)
    return {}


# ── Prompts ───────────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    return """\
You are a professional vacation rental visual merchandising specialist.
You curate property photo libraries to maximise guest conversion.

Your job: analyse contact sheets of property photos, identify every image's \
room type, group duplicates, select the best images for conversion, and \
output a structured JSON curation.

Rules:
- Each contact sheet thumbnail is labelled (e.g. A01, A02, B03).
- You will be given an "image_index" mapping label → asset_id.
- Every label in the image_index MUST appear in your output.
- Output ONLY valid JSON — no markdown, no prose, no code fences.
- Never invent asset_ids; use only the exact strings from image_index values.
"""


def _build_full_prompt(index_map: dict[str, str], kb: dict) -> str:
    """Full prompt: per-image analysis + property-level recommendations."""
    context = _property_context(kb)
    schema_example = _schema_example()
    categories = ", ".join(sorted(_VALID_CATEGORIES))
    sections = ", ".join(f'"{s}"' for s in _PHOTO_TOUR_SECTIONS)

    return f"""\
Property context:
{context}

Image index (label → asset_id):
{json.dumps(index_map, indent=2)}

Task:
1. For EVERY label in the image_index, produce one entry in "images".
2. Identify the correct room/space category.
3. Group images that show the same room from different angles — assign them \
   the same duplicate_group string (e.g. "dg_kitchen_1"). Use null if unique.
4. Within each duplicate group, mark exactly one image as is_primary_in_group=true \
   (the best angle/quality). All others: false.
5. Assign rank_within_category: 1 = best photo in that category, 2 = second best, etc.
6. Assign role:
   - "hero"         → the single best overall page hero
   - "supporting"   → good image worth showing in the photo tour
   - "gallery_only" → acceptable for full gallery but not photo tour
   - "exclude"      → blurry, dark, duplicate closeup, or irrelevant — hide completely
7. For images marked "exclude", set reason briefly (e.g. "blurry", "dark", "duplicate_inferior").
8. Write a concise, descriptive alt text for every non-excluded image.
9. In "property", choose the best overall page_hero (exactly one asset_id).
10. Build photo_tour_sections — use ONLY these section names: {sections}
    Each section: 1 hero + up to 2 supporting. Only include sections with qualifying images.
11. Set category_order (ordered list of section names, best-first for this property).
12. Infer vibe (1 sentence: what feeling this property evokes).
13. Set merchandising_strategy.prioritize (what to lead with) and deprioritize.

Valid categories: {categories}

Return ONLY this JSON structure (no markdown, no prose):

{schema_example}
"""


def _build_per_image_only_prompt(batch_index: dict[str, str], kb: dict) -> str:
    """Prompt for batch call — per-image analysis only, no property-level."""
    context = _property_context(kb)
    categories = ", ".join(sorted(_VALID_CATEGORIES))

    return f"""\
Property context:
{context}

Image index (label → asset_id):
{json.dumps(batch_index, indent=2)}

Task: For EVERY label in the image_index, produce one entry in "images".
Analyse room type, duplicates, quality, and conversion value.

Valid categories: {categories}
Valid roles: hero, supporting, gallery_only, exclude

Return ONLY valid JSON:
{{
  "images": [
    {{
      "asset_id": "<exact string from image_index>",
      "llm_category": "<category>",
      "room_subtype": "<specific subtype or null>",
      "duplicate_group": "<shared string for same room or null>",
      "is_primary_in_group": true,
      "rank_within_category": 1,
      "role": "supporting",
      "reason": "<only if exclude>",
      "alt": "<descriptive alt text>"
    }}
  ]
}}
"""


def _build_synthesis_prompt(per_image_results: list[dict], kb: dict) -> str:
    """Prompt for text-only synthesis call."""
    context = _property_context(kb)
    sections = ", ".join(f'"{s}"' for s in _PHOTO_TOUR_SECTIONS)

    return f"""\
Property context:
{context}

Per-image curation results (already analysed):
{json.dumps(per_image_results, indent=2)}

Task: Based on the per-image results above, produce property-level recommendations.
Use ONLY asset_ids from the per-image results. Use ONLY these section names: {sections}

Return ONLY valid JSON:
{{
  "page_hero": "<asset_id of best overall hero>",
  "photo_tour_sections": [
    {{
      "section": "<section name>",
      "hero": "<asset_id>",
      "supporting": ["<asset_id>", "<asset_id>"]
    }}
  ],
  "category_order": ["<section name>", ...],
  "vibe": "<one sentence>",
  "merchandising_strategy": {{
    "prioritize": ["<what to lead with>"],
    "deprioritize": ["<what to minimise>"]
  }}
}}
"""


def _property_context(kb: dict) -> str:
    """Extract vibe context from KB for the prompt."""
    def _val(key: str) -> str:
        f = kb.get(key)
        if isinstance(f, dict):
            return str(f.get("value") or "")
        return str(f or "")

    lines = [
        f"- Vibe profile: {kb.get('vibe_profile') or 'unknown'}",
        f"- Property name: {_val('name')}",
        f"- Location: {_val('city')}, {_val('state')}",
    ]
    if kb.get("owner_story"):
        lines.append(f"- Owner story: {str(kb['owner_story'])[:300]}")
    wow = kb.get("wow_factor") or kb.get("unique_selling_points")
    if wow:
        lines.append(f"- Wow factor / USP: {str(wow)[:200]}")
    gems = kb.get("hidden_gems") or kb.get("local_secrets")
    if gems:
        lines.append(f"- Hidden gems: {str(gems)[:200]}")
    return "\n".join(lines)


def _schema_example() -> str:
    return """\
{
  "images": [
    {
      "asset_id": "<exact string from image_index — never invented>",
      "llm_category": "kitchen",
      "room_subtype": "island_kitchen",
      "duplicate_group": "dg_kitchen_1",
      "is_primary_in_group": true,
      "rank_within_category": 1,
      "role": "supporting",
      "reason": null,
      "alt": "Bright open-plan kitchen with marble island and ocean views"
    }
  ],
  "property": {
    "page_hero": "<asset_id>",
    "photo_tour_sections": [
      {
        "section": "Kitchen",
        "hero": "<asset_id>",
        "supporting": ["<asset_id>", "<asset_id>"]
      }
    ],
    "category_order": ["Exterior & Views", "Kitchen", "Bedrooms"],
    "vibe": "A light-filled coastal retreat with sweeping ocean views.",
    "merchandising_strategy": {
      "prioritize": ["ocean view shots", "pool at golden hour"],
      "deprioritize": ["bathroom closeups", "utility areas"]
    }
  }
}"""


# ── JSON parsing + validation ─────────────────────────────────────────────────

def _parse_and_validate(raw: str, index_map: dict[str, str]) -> Optional[dict]:
    """
    Extract JSON from raw LLM output, validate structure, fill defaults
    for invalid/missing fields. Returns {images, property} or None.
    """
    parsed = _extract_json(raw)
    if not parsed or not isinstance(parsed, dict):
        logger.warning("[LLM Curator] Could not extract valid JSON from LLM response")
        return None

    images_raw = parsed.get("images")
    if not isinstance(images_raw, list):
        logger.warning("[LLM Curator] LLM response missing 'images' list")
        return None

    known_asset_ids = set(index_map.values())
    valid_images: list[dict] = []

    for img in images_raw:
        if not isinstance(img, dict):
            continue
        asset_id = img.get("asset_id") or ""
        if asset_id not in known_asset_ids:
            # LLM invented an asset_id — discard this entry
            continue
        category = img.get("llm_category") or "uncategorised"
        if category not in _VALID_CATEGORIES:
            category = "uncategorised"
        role = img.get("role") or "gallery_only"
        if role not in _VALID_ROLES:
            role = "gallery_only"
        rank = img.get("rank_within_category")
        try:
            rank = max(1, int(rank)) if rank is not None else 99
        except (TypeError, ValueError):
            rank = 99

        valid_images.append({
            "asset_id":            asset_id,
            "llm_category":        category,
            "room_subtype":        img.get("room_subtype") or None,
            "duplicate_group":     img.get("duplicate_group") or None,
            "is_primary_in_group": bool(img.get("is_primary_in_group", True)),
            "rank_within_category": rank,
            "role":                role,
            "reason":              img.get("reason") or None,
            "alt":                 str(img.get("alt") or "")[:200],
        })

    # Ensure every asset_id in the index has an entry (fill missing with defaults)
    seen_ids = {img["asset_id"] for img in valid_images}
    for label, asset_id in index_map.items():
        if asset_id not in seen_ids:
            valid_images.append({
                "asset_id":            asset_id,
                "llm_category":        "uncategorised",
                "room_subtype":        None,
                "duplicate_group":     None,
                "is_primary_in_group": True,
                "rank_within_category": 99,
                "role":                "gallery_only",
                "reason":              "not_analysed",
                "alt":                 "",
            })

    # Validate property-level
    property_raw = parsed.get("property") or {}
    property_recs = _validate_property_recs(property_raw, known_asset_ids)

    return {"images": valid_images, "property": property_recs}


def _validate_property_recs(raw: dict, valid_asset_ids: set) -> dict:
    """Validate and sanitise property-level recommendations."""
    page_hero = raw.get("page_hero") or ""
    if page_hero not in valid_asset_ids:
        page_hero = ""

    sections_raw = raw.get("photo_tour_sections") or []
    sections: list[dict] = []
    for sec in sections_raw:
        if not isinstance(sec, dict):
            continue
        section_name = sec.get("section") or ""
        if section_name not in _PHOTO_TOUR_SECTIONS:
            continue
        hero_id = sec.get("hero") or ""
        if hero_id not in valid_asset_ids:
            continue
        supporting = [
            sid for sid in (sec.get("supporting") or [])
            if sid in valid_asset_ids and sid != hero_id
        ][:2]
        sections.append({
            "section": section_name,
            "hero": hero_id,
            "supporting": supporting,
        })

    cat_order = [
        s for s in (raw.get("category_order") or [])
        if s in _PHOTO_TOUR_SECTIONS
    ]

    return {
        "page_hero":           page_hero,
        "photo_tour_sections": sections,
        "category_order":      cat_order,
        "vibe":                str(raw.get("vibe") or "")[:300],
        "merchandising_strategy": {
            "prioritize":   [str(x) for x in (raw.get("merchandising_strategy", {}).get("prioritize") or [])[:5]],
            "deprioritize": [str(x) for x in (raw.get("merchandising_strategy", {}).get("deprioritize") or [])[:5]],
        },
    }


def _extract_json(text: str) -> Optional[dict]:
    """Extract a JSON object from LLM text that may include prose or markdown fences."""
    # Try parsing the whole string first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Find the outermost JSON object
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ── Persistence ───────────────────────────────────────────────────────────────

def _save_curation(
    property_id: str,
    image_set_hash: str,
    per_image_results: list[dict],
    property_recommendations: dict,
    asset_count: int,
) -> None:
    """Upsert curation record to Supabase. Non-fatal on failure."""
    from datetime import datetime, timezone
    try:
        from core.supabase_store import get_supabase
        get_supabase().table(_SUPABASE_TABLE).upsert(
            {
                "property_id":              property_id,
                "image_set_hash":           image_set_hash,
                "status":                   "complete",
                "asset_count":              asset_count,
                "per_image_results":        per_image_results,
                "property_recommendations": property_recommendations,
                "curation_model":           _MODEL,
                "completed_at":             datetime.now(timezone.utc).isoformat(),
            },
            on_conflict="property_id,image_set_hash",
        ).execute()
        logger.info(
            "[LLM Curator] Saved curation to Supabase (property=%s, hash=%s)",
            property_id, image_set_hash[:12],
        )
    except Exception as exc:
        logger.error("[LLM Curator] Supabase save failed: %s — result still cached in Redis", exc)


def _mark_failed(property_id: str, image_set_hash: str, asset_count: int) -> None:
    """Mark a curation attempt as failed so it can be retried next run."""
    try:
        from core.supabase_store import get_supabase
        get_supabase().table(_SUPABASE_TABLE).upsert(
            {
                "property_id":    property_id,
                "image_set_hash": image_set_hash,
                "status":         "failed",
                "asset_count":    asset_count,
                "curation_model": _MODEL,
            },
            on_conflict="property_id,image_set_hash",
        ).execute()
    except Exception:
        pass  # Non-fatal — failure mode is already logged above
