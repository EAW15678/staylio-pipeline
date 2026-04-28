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
_MAX_IMAGES_TO_CURATE = 120     # hard cap — cost control (raised from 80)
_SHEETS_PER_CALL    = 4         # max contact sheets per API call (4 × 12 = 48 images → ~2 K output tokens)
_SUPABASE_TABLE     = "property_image_curations"
_REDIS_TTL          = 7 * 24 * 3600  # 7 days

# ── Curation version ──────────────────────────────────────────────────────────
# Bump this string whenever the taxonomy or curation rules change.
# It is prepended to the URL list before hashing, so existing cached curations
# (keyed on old hash) are automatically bypassed and a fresh LLM call is made.
_CURATION_VERSION = "curation_v2_no_primary_flat_gallery"

# ── Canonical section taxonomy ─────────────────────────────────────────────────
# Single source of truth. agent5/page_builder.py imports CURATED_SECTION_NAMES.
# gcv_categories drives _derive_curated_section_fallback for uncapped/unanalysed images.
CURATED_SECTIONS: list[dict] = [
    {
        "name":           "Exterior",
        "description":    "Building exterior, facade, driveway, landscaping, entrance, views",
        "gcv_categories": ["exterior", "view"],
    },
    {
        "name":           "Pool",
        "description":    "Pool, hot tub, spa, water features",
        "gcv_categories": ["pool_hot_tub"],
    },
    {
        "name":           "Living Areas",
        "description":    "Living room, great room, lounge, sitting area, game room",
        "gcv_categories": ["living_room", "game_entertainment"],
    },
    {
        "name":           "Kitchen",
        "description":    "Kitchen, island, dining area",
        "gcv_categories": ["kitchen"],
    },
    {
        "name":           "Bedrooms",
        "description":    "All sleeping rooms — beds, bunk rooms, guest rooms, suites. Prioritise diverse coverage across distinct rooms.",
        "gcv_categories": ["master_bedroom", "standard_bedroom"],
    },
    {
        "name":           "Bathrooms",
        "description":    "All bathrooms — vanity, shower, tub, toilet. Prioritise diverse coverage across distinct bathrooms.",
        "gcv_categories": ["bathroom"],
    },
    {
        "name":           "Extras",
        "description":    "Gym, office, laundry, garage, outdoor entertaining areas",
        "gcv_categories": ["outdoor_entertaining", "local_area", "uncategorised"],
    },
]

CURATED_SECTION_NAMES: list[str] = [s["name"] for s in CURATED_SECTIONS]
_CURATED_SECTION_NAMES_SET: frozenset = frozenset(CURATED_SECTION_NAMES)

_VALID_CATEGORIES = frozenset({
    "exterior", "view", "pool_hot_tub", "outdoor_entertaining",
    "living_room", "kitchen", "master_bedroom", "standard_bedroom",
    "bathroom", "game_entertainment", "local_area", "uncategorised",
})

_VALID_ROLES = frozenset({"hero", "supporting", "gallery_only", "exclude"})

_SHEET_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ── Classification rules injected into every LLM prompt ──────────────────────
# Shared by _build_full_prompt and _build_per_image_only_prompt.
# Edit here; both prompts update automatically.
_CLASSIFICATION_RULES = """\
HARD CLASSIFICATION RULES — apply BEFORE assigning any curated_section:

  Bedrooms
    MUST contain a clearly visible bed.
    Reject: living rooms, lounges, rooms without a bed.

  Bathrooms
    MUST contain at least one of: sink/vanity, toilet, shower, bathtub.
    Reject: wet bars, kitchens, closets.

  Pool
    MUST contain a visible swimming pool.
    Reject: patios or decks without a pool, hot-tub-only shots (assign those to Extras).

  Kitchen
    MUST clearly show a stove, kitchen island, or kitchen cabinetry.
    Reject: dining-area-only shots (unless the kitchen is also clearly visible), living rooms.

  Living Areas
    Must show couches, seating, TV, fireplace, or a gathering/lounge space.
    Reject: bedrooms, kitchens.

  Exterior
    Must show the outside structure of the home or strong exterior elements.
    Reject: interior shots, pool-only shots (pool shots → Pool section).

  Extras
    Use ONLY when the image does not clearly belong to another section.
    Examples: outdoor fireplace, outdoor grilling area, elevator, parking,
    baby crib/gate, beach supplies, golf cart/LSV, hot tub (without pool),
    children's play space, activity/event images.

OPEN-CONCEPT RULE:
  If an image shows both kitchen and living areas, assign based on the PRIMARY
  visual focus of the image. Do NOT assign the same image to both sections.

DUPLICATE SUPPRESSION — within EACH section:
  Do NOT select multiple images that show:
  * the same angle
  * minor variations of the same shot
  If two images are very similar in orientation, keep only the best one.
  Assign the inferior one role="exclude" with reason="duplicate_inferior".

DIVERSITY REQUIREMENT — within EACH section:
  Prioritise different spaces over different angles of the same space.
  - Bedrooms: prefer distinct bedrooms over multiple shots of the same bed.
  - Bathrooms: prefer one vanity shot + one shower + one tub rather than
    three angles of the same fixture.
  - Living Areas: show different areas (living room, dining, lounge, sitting room).

CROSS-SECTION UNIQUENESS:
  Each image belongs to EXACTLY ONE section. An image MUST NOT appear in
  multiple sections. If an image is borderline, assign it to the single
  best-matching section and leave it out of all others.

PRIORITY ORDER within each section:
  1. Correct classification (must pass hard rules above)
  2. Unique space representation (different room/area)
  3. Visual quality and aesthetic appeal

SELF-CHECK before finalising output:
  Verify that every image assigned to Bedrooms shows a bed.
  Verify that every image assigned to Bathrooms shows bathroom fixtures.
  Verify that every image assigned to Pool shows a pool.
  If any image fails its hard rule → reassign to the correct section or exclude.
"""


# ── Public entry point ────────────────────────────────────────────────────────

def run_llm_vision_curation(
    assets: list,
    enhanced_bytes_map: dict,
    property_id: str,
    kb: dict,
    source_hero_url: Optional[str] = None,
) -> Optional[dict]:
    """
    Run one-time LLM vision curation for a property's image set.

    Args:
        assets:             List of MediaAsset objects (all retained photos).
        enhanced_bytes_map: {asset_url_original: enhanced_bytes} — in-memory
                            bytes for freshly Claid-enhanced images.
        property_id:        Property UUID string.
        kb:                 Knowledge base dict (for vibe context in prompt).
        source_hero_url:    GCV-selected hero URL. Treated as the default page
                            hero — LLM should only override if clearly inferior.
                            Matching contact sheet cell is labelled with "*".

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
        # The source hero cell is annotated with "*" so the LLM can see it.
        # The index_map value remains the plain asset_url_original (no "*").
        index_map: dict[str, str] = {}   # "A01" → asset_url_original
        index_map_display: dict[str, str] = {}  # "A01*" → asset_url_original (for prompt)
        asset_list: list = []            # ordered, parallel with index_map
        source_hero_label: Optional[str] = None

        for i, asset in enumerate(candidate_assets):
            sheet_idx  = i // _IMAGES_PER_SHEET
            cell_idx   = i %  _IMAGES_PER_SHEET
            sheet_letter = _SHEET_LETTERS[sheet_idx % len(_SHEET_LETTERS)]
            label = f"{sheet_letter}{cell_idx + 1:02d}"
            index_map[label] = asset.asset_url_original
            # Mark source hero with "*" in display map and contact sheet
            display_label = label
            if (source_hero_url and
                    asset.asset_url_original == source_hero_url or
                    (asset.asset_url_enhanced and asset.asset_url_enhanced == source_hero_url)):
                display_label = f"{label}*"
                source_hero_label = label
            index_map_display[display_label] = asset.asset_url_original
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
        # Use display labels (with * for source hero) in the sheet cells.
        # Build a reverse map: asset_url_original → display_label for cell lookup.
        orig_to_display = {v: k for k, v in index_map_display.items()}

        n_sheets = math.ceil(len(asset_list) / _IMAGES_PER_SHEET)
        sheet_jpeg_list: list[bytes] = []  # one JPEG per sheet
        for sheet_idx in range(n_sheets):
            start = sheet_idx * _IMAGES_PER_SHEET
            sheet_assets = asset_list[start: start + _IMAGES_PER_SHEET]
            sheet_letter = _SHEET_LETTERS[sheet_idx % len(_SHEET_LETTERS)]
            cells: list[tuple[str, Optional[bytes]]] = []
            for asset in sheet_assets:
                display_label = orig_to_display.get(asset.asset_url_original,
                                                    f"{sheet_letter}??")
                img_bytes = bytes_by_original.get(asset.asset_url_original)
                cells.append((display_label, img_bytes))
            sheet_jpeg = _build_contact_sheet(cells, sheet_letter)
            sheet_jpeg_list.append(sheet_jpeg)

        if source_hero_label:
            logger.info(
                "[LLM Curator] Source hero annotated as %s* in contact sheets",
                source_hero_label,
            )

        # ── Build label → caption map ─────────────────────────────────────
        # Maps clean label (e.g. "A01") → source caption from the listing.
        # VRBO populates captions (e.g. "Master bedroom with king bed").
        # Airbnb and PMC entries are None and are excluded from the prompt.
        url_to_asset = {a.asset_url_original: a for a in candidate_assets}
        caption_map: dict[str, str] = {}
        for label, orig_url in index_map.items():
            asset = url_to_asset.get(orig_url)
            caption = getattr(asset, "source_caption", None) if asset else None
            if caption:
                caption_map[label.rstrip("*")] = caption

        # ── Call LLM ──────────────────────────────────────────────────────
        result = _run_curation_call(
            sheet_jpeg_list=sheet_jpeg_list,
            index_map=index_map,
            index_map_display=index_map_display,
            kb=kb,
            property_id=property_id,
            source_hero_url=source_hero_url,
            source_hero_label=source_hero_label,
            caption_map=caption_map,
        )

        if result is None:
            logger.warning(
                "[LLM Curator] LLM call failed for property %s — pipeline continues with GCV path",
                property_id,
            )
            _mark_failed(property_id, image_set_hash, len(candidate_assets))
            return None

        # ── Post-process: fill missing curated_section + log summary ──────
        result["images"] = _derive_curated_section_fallback(result["images"])
        _log_curation_summary(result, source_hero_url)

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
    """SHA-256 of curation version + sorted R2 original URLs.
    Bumping _CURATION_VERSION invalidates all existing cached curations."""
    urls = sorted(a.asset_url_original for a in assets if a.asset_url_original)
    payload = _CURATION_VERSION + "\n" + "\n".join(urls)
    digest = hashlib.sha256(payload.encode()).hexdigest()
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
    index_map_display: dict[str, str],
    kb: dict,
    property_id: str,
    source_hero_url: Optional[str] = None,
    source_hero_label: Optional[str] = None,
    caption_map: Optional[dict] = None,
) -> Optional[dict]:
    """
    Send all contact sheets to Claude in one or two API calls.
    Parses and validates the JSON response.
    Returns {images: [...], property: {...}} or None on failure.
    """
    caption_map = caption_map or {}

    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    except Exception as exc:
        logger.error("[LLM Curator] Failed to create Anthropic client: %s", exc)
        return None

    system_prompt = _build_system_prompt()

    # If sheets fit in one call, do everything in one request.
    # If not, do per-image analysis in batches then a text-only synthesis pass.
    if len(sheet_jpeg_list) <= _SHEETS_PER_CALL:
        result = _single_call(
            client, system_prompt, sheet_jpeg_list, index_map, index_map_display,
            kb, source_hero_url, source_hero_label, caption_map,
        )
        if result is None:
            return None
        return result
    else:
        partial_results: list[dict] = []
        for batch_start in range(0, len(sheet_jpeg_list), _SHEETS_PER_CALL):
            batch = sheet_jpeg_list[batch_start: batch_start + _SHEETS_PER_CALL]
            batch_index = {
                k: v for k, v in index_map.items()
                if ord(k[0]) - ord("A") in range(
                    batch_start, min(batch_start + _SHEETS_PER_CALL, len(sheet_jpeg_list))
                )
            }
            batch_index_display = {
                k: v for k, v in index_map_display.items()
                if ord(k[0]) - ord("A") in range(
                    batch_start, min(batch_start + _SHEETS_PER_CALL, len(sheet_jpeg_list))
                )
            }
            # Filter caption_map to labels in this batch
            batch_caption_map = {
                k: v for k, v in caption_map.items()
                if k in {lbl.rstrip("*") for lbl in batch_index_display}
            }
            partial = _batch_call(
                client, system_prompt, batch, batch_index, batch_index_display, kb,
                batch_caption_map,
            )
            if partial:
                partial_results.extend(partial)

        if not partial_results:
            return None

        property_recs = _synthesis_call(
            client, partial_results, index_map, kb, source_hero_url, source_hero_label,
        )
        return {
            "images": partial_results,
            "property": property_recs or {},
        }


def _single_call(
    client: anthropic.Anthropic,
    system_prompt: str,
    sheet_jpeg_list: list[bytes],
    index_map: dict[str, str],
    index_map_display: dict[str, str],
    kb: dict,
    source_hero_url: Optional[str] = None,
    source_hero_label: Optional[str] = None,
    caption_map: Optional[dict] = None,
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
        "text": _build_full_prompt(
            index_map, index_map_display, kb, source_hero_url, source_hero_label,
            caption_map or {},
        ),
    })

    try:
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=8192,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )
        _log_token_usage(resp, "single")
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
    batch_index_display: dict[str, str],
    kb: dict,
    caption_map: Optional[dict] = None,
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
        "text": _build_per_image_only_prompt(batch_index, batch_index_display, kb, caption_map or {}),
    })
    try:
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=8192,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )
        _log_token_usage(resp, "batch")
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
    source_hero_url: Optional[str] = None,
    source_hero_label: Optional[str] = None,
) -> dict:
    """Text-only synthesis: produces property-level recommendations from per-image results."""
    # Build reverse map: URL → label (stripped of "*")
    url_to_label = {v: k.rstrip("*") for k, v in index_map.items()}
    label_to_url = {k.rstrip("*"): v for k, v in index_map.items()}

    # Compact summary for prompt — labels only, non-excluded images
    compact_summary = [
        {
            "label":           url_to_label.get(img["asset_id"], img["asset_id"]),
            "role":            img.get("role"),
            "curated_section": img.get("curated_section"),
            "llm_category":    img.get("llm_category"),
        }
        for img in per_image_results
        if img.get("role") != "exclude"
    ]

    prompt = _build_synthesis_prompt(compact_summary, kb, source_hero_label)
    try:
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )
        _log_token_usage(resp, "synthesis")
        raw = resp.content[0].text
        parsed = _extract_json(raw)
        if parsed and isinstance(parsed, dict):
            # Remap labels → URLs in synthesis output before validation
            def _resolve_label(val: str) -> str:
                return label_to_url.get((val or "").rstrip("*"), "")

            parsed["page_hero"] = _resolve_label(parsed.get("page_hero") or "")
            remapped_sections = []
            for sec in (parsed.get("photo_tour_sections") or []):
                if not isinstance(sec, dict):
                    continue
                sec["hero"] = _resolve_label(sec.get("hero") or "")
                sec["supporting"] = [
                    _resolve_label(s) for s in (sec.get("supporting") or [])
                ]
                remapped_sections.append(sec)
            parsed["photo_tour_sections"] = remapped_sections
            return _validate_property_recs(parsed, set(index_map.values()))
    except Exception as exc:
        logger.error("[LLM Curator] Synthesis call failed: %s", exc)
    return {}


def _log_token_usage(resp, call_type: str) -> None:
    """Log token counts and estimated cost after an API call."""
    try:
        usage = resp.usage
        input_tokens  = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        # Sonnet 4.6: $3/M input, $15/M output
        cost_usd = (input_tokens * 3 + output_tokens * 15) / 1_000_000
        logger.info(
            "[LLM Curator] %s call — input_tokens=%d, output_tokens=%d, "
            "estimated_cost=USD %.4f",
            call_type, input_tokens, output_tokens, cost_usd,
        )
    except Exception:
        pass   # non-fatal


# ── Prompts ───────────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    return """\
You are a professional vacation rental visual merchandising specialist.
You curate property photo libraries to maximise guest conversion.

Your job: analyse contact sheets of property photos, identify every image's \
room type, group duplicates, select the best images for conversion, and \
output a structured JSON curation.

Rules:
- Each contact sheet thumbnail is labelled with a short code (e.g. A01, A02, B03).
- You will be given the list of labels to analyse.
- Use the label (e.g. "A01") as the value for "asset_id" in your output — NOT the full URL.
- Every label in the list MUST appear exactly once in your output.
- Output ONLY valid JSON — no markdown, no prose, no code fences.
- Never invent labels; use only the exact codes shown.
"""


def _build_captions_block(caption_map: Optional[dict]) -> str:
    """
    Build the source-captions section for LLM prompts.
    Returns an empty string when no captions are available (Airbnb / PMC properties).
    Only labels with non-empty captions are included.
    """
    if not caption_map:
        return ""
    lines = [
        f'  {label}: "{caption}"'
        for label, caption in sorted(caption_map.items())
        if caption
    ]
    if not lines:
        return ""
    return (
        "\nSource captions from listing (use these to identify room type "
        "and primary/secondary designation):\n" + "\n".join(lines) + "\n"
    )


def _build_full_prompt(
    index_map: dict[str, str],
    index_map_display: dict[str, str],
    kb: dict,
    source_hero_url: Optional[str] = None,
    source_hero_label: Optional[str] = None,
    caption_map: Optional[dict] = None,
) -> str:
    """Full prompt: per-image analysis + property-level recommendations."""
    context = _property_context(kb)
    schema_example = _schema_example()
    categories = ", ".join(sorted(_VALID_CATEGORIES))
    sections_list = "\n".join(
        f'  - "{s["name"]}": {s["description"]}' for s in CURATED_SECTIONS
    )
    section_names = ", ".join(f'"{s["name"]}"' for s in CURATED_SECTIONS)

    hero_instruction = ""
    if source_hero_url and source_hero_label:
        hero_instruction = f"""
Source hero note:
The image labelled {source_hero_label}* is the current property hero selected by the \
listing platform. Treat it as the default page_hero unless you identify a clearly superior \
image elsewhere. If you override it, still include it as "supporting" or "gallery_only" — \
never exclude it solely because another image is better.
"""

    captions_block = _build_captions_block(caption_map)
    labels_list = list(index_map_display.keys())

    return f"""\
Property context:
{context}
{hero_instruction}
Image labels to analyse (use these exact codes as "asset_id" in your output):
{json.dumps(labels_list)}
{captions_block}
Task:
1. For EVERY label in the list above, produce one entry in "images".
   Use the label (e.g. "A01") as the value for asset_id — NOT a URL.
2. Identify the correct room/space category (llm_category).
3. Group images showing the same room from different angles — assign the same
   duplicate_group string (e.g. "dg_kitchen_1"). Use null if unique.
4. Within each duplicate group, mark exactly one image is_primary_in_group=true
   (best angle/quality). All others: false.
5. Assign rank_within_category: 1 = best in that category, 2 = second best, etc.
6. Assign role:
   - "hero"         → the single best overall page hero (exactly one image)
   - "supporting"   → good image worth featuring in the photo tour
   - "gallery_only" → acceptable for full gallery but not photo tour
   - "exclude"      → blurry, dark, duplicate inferior angle, or irrelevant
7. For "exclude" images, set reason briefly (e.g. "blurry", "dark", "duplicate_inferior").
8. Write a concise alt text for every non-excluded image.
9. Assign curated_section to every non-excluded image. Available sections:
{sections_list}
   Set curated_section to null for excluded images.
   If bedroom_count is in the layout, aim to represent up to that many distinct bedrooms.
   If bathroom_count is in the layout, aim to represent up to that many distinct bathrooms.

{_CLASSIFICATION_RULES}
10. In "property", choose the best overall page_hero (label, e.g. "A01").
11. Build photo_tour_sections using ONLY these section names: {section_names}
    Each section: 1 hero label + up to 2 supporting labels. Only include sections with qualifying images.
    For Bedrooms and Bathrooms: choose supporting images from DIFFERENT rooms/angles.
12. Set category_order (section names ordered best-first for this property).
13. Infer vibe (1 sentence describing the feeling this property evokes).
14. Set merchandising_strategy.prioritize and deprioritize.

Valid llm_category values: {categories}

Return ONLY valid JSON — no markdown, no prose, no code fences:

{schema_example}
"""


def _build_per_image_only_prompt(
    batch_index: dict[str, str],
    batch_index_display: dict[str, str],
    kb: dict,
    caption_map: Optional[dict] = None,
) -> str:
    """Prompt for batch call — per-image analysis only, no property-level."""
    context = _property_context(kb)
    categories = ", ".join(sorted(_VALID_CATEGORIES))
    sections_list = "\n".join(
        f'  - "{s["name"]}": {s["description"]}' for s in CURATED_SECTIONS
    )

    captions_block = _build_captions_block(caption_map)
    labels_list = list(batch_index_display.keys())

    return f"""\
Property context:
{context}

Image labels to analyse (use these exact codes as "asset_id"):
{json.dumps(labels_list)}
{captions_block}

Task: For EVERY label above, produce one entry in "images".
Use the label (e.g. "A01") as asset_id — NOT a URL.
Analyse room type, duplicates, quality, and conversion value.
Assign curated_section to every non-excluded image. Available sections:
{sections_list}
Set curated_section to null for excluded images.

{_CLASSIFICATION_RULES}

Valid llm_category values: {categories}
Valid roles: hero, supporting, gallery_only, exclude

Return ONLY valid JSON:
{{
  "images": [
    {{
      "asset_id": "A01",
      "llm_category": "kitchen",
      "curated_section": "Kitchen",
      "room_subtype": "island_kitchen",
      "duplicate_group": "dg_kitchen_1",
      "is_primary_in_group": true,
      "rank_within_category": 1,
      "role": "supporting",
      "reason": null,
      "alt": "Open-plan kitchen with marble island"
    }}
  ]
}}
"""


def _build_synthesis_prompt(
    compact_summary: list[dict],
    kb: dict,
    source_hero_label: Optional[str] = None,
) -> str:
    """
    Prompt for text-only synthesis call.
    compact_summary: [{label, role, curated_section, llm_category}, ...] — non-excluded only.
    """
    context = _property_context(kb)
    section_names = ", ".join(f'"{s["name"]}"' for s in CURATED_SECTIONS)

    hero_instruction = ""
    if source_hero_label:
        hero_instruction = f"""
Source hero note: Label {source_hero_label}* is the current listing platform hero. \
Use it as page_hero unless a clearly superior image exists. If you override it, \
ensure it still appears as supporting or gallery_only.
"""

    return f"""\
Property context:
{context}
{hero_instruction}
Per-image curation summary (label, role, curated_section for non-excluded images):
{json.dumps(compact_summary)}

Task: Produce property-level recommendations.
Use ONLY labels from the summary above (e.g. "A01") in page_hero, hero, and supporting.
Use ONLY these section names: {section_names}

Return ONLY valid JSON:
{{
  "page_hero": "A01",
  "photo_tour_sections": [
    {{
      "section": "Exterior",
      "hero": "A01",
      "supporting": ["A02", "A03"]
    }}
  ],
  "category_order": ["Exterior", "Pool", "Living Areas"],
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
    beds  = _val("bedrooms")
    baths = _val("bathrooms")
    if beds or baths:
        lines.append(f"- Layout: {beds} bedrooms, {baths} bathrooms")
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
      "asset_id": "A01",
      "llm_category": "kitchen",
      "curated_section": "Kitchen",
      "room_subtype": "island_kitchen",
      "duplicate_group": "dg_kitchen_1",
      "is_primary_in_group": true,
      "rank_within_category": 1,
      "role": "supporting",
      "reason": null,
      "alt": "Open-plan kitchen with marble island and ocean views"
    }
  ],
  "property": {
    "page_hero": "A01",
    "photo_tour_sections": [
      {
        "section": "Kitchen",
        "hero": "A01",
        "supporting": ["A02", "A03"]
      }
    ],
    "category_order": ["Exterior", "Pool", "Living Areas", "Kitchen", "Bedrooms", "Bathrooms"],
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

    known_asset_ids = set(index_map.values())   # full URLs
    # Label → URL map (strip "*" from annotated source-hero label)
    label_to_url: dict[str, str] = {k.rstrip("*"): v for k, v in index_map.items()}
    valid_images: list[dict] = []

    for img in images_raw:
        if not isinstance(img, dict):
            continue
        asset_id_raw = (img.get("asset_id") or "").rstrip("*")
        # Accept label (e.g. "A01") or full URL (backward compat)
        if asset_id_raw in label_to_url:
            asset_id = label_to_url[asset_id_raw]
        elif asset_id_raw in known_asset_ids:
            asset_id = asset_id_raw  # already a URL
        else:
            # LLM invented a value — discard
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

        curated_section = img.get("curated_section") or None
        if curated_section is not None and curated_section not in _CURATED_SECTION_NAMES_SET:
            logger.debug(
                "[LLM Curator] Invalid curated_section %r for %s — clearing",
                curated_section, asset_id[-30:],
            )
            curated_section = None

        valid_images.append({
            "asset_id":            asset_id,
            "llm_category":        category,
            "curated_section":     curated_section,
            "room_subtype":        img.get("room_subtype") or None,
            "duplicate_group":     img.get("duplicate_group") or None,
            "is_primary_in_group": bool(img.get("is_primary_in_group", True)),
            "rank_within_category": rank,
            "role":                role,
            "reason":              img.get("reason") or None,
            "alt":                 str(img.get("alt") or "")[:80],
        })

    # Ensure every asset_id in the index has an entry (fill missing with defaults)
    seen_ids = {img["asset_id"] for img in valid_images}
    for label, asset_id in index_map.items():
        if asset_id not in seen_ids:
            valid_images.append({
                "asset_id":            asset_id,
                "llm_category":        "uncategorised",
                "curated_section":     None,   # filled by _derive_curated_section_fallback
                "room_subtype":        None,
                "duplicate_group":     None,
                "is_primary_in_group": True,
                "rank_within_category": 99,
                "role":                "gallery_only",
                "reason":              "not_analysed",
                "alt":                 "",
            })

    # Validate property-level (LLM may have used labels or URLs in page_hero/supporting)
    property_raw = parsed.get("property") or {}
    property_recs = _validate_property_recs(property_raw, known_asset_ids, label_to_url)

    return {"images": valid_images, "property": property_recs}


def _validate_property_recs(
    raw: dict,
    valid_asset_ids: set,
    label_to_url: Optional[dict] = None,
) -> dict:
    """
    Validate and sanitise property-level recommendations.
    label_to_url: if provided, resolves label strings (e.g. "A01") to full URLs
                  before validation. Supports both labels and URLs in LLM output.
    """
    label_to_url = label_to_url or {}

    def _resolve(val: str) -> str:
        """Map label → URL, or return val unchanged if already a URL."""
        v = (val or "").rstrip("*")
        return label_to_url.get(v, v)

    page_hero = _resolve(raw.get("page_hero") or "")
    if page_hero not in valid_asset_ids:
        page_hero = ""

    sections_raw = raw.get("photo_tour_sections") or []
    sections: list[dict] = []
    for sec in sections_raw:
        if not isinstance(sec, dict):
            continue
        section_name = sec.get("section") or ""
        if section_name not in _CURATED_SECTION_NAMES_SET:
            logger.warning(
                "[LLM Curator] Unknown photo_tour section name %r — skipped", section_name,
            )
            continue
        hero_id = _resolve(sec.get("hero") or "")
        if hero_id not in valid_asset_ids:
            continue
        supporting = [
            sid for sid in (_resolve(s) for s in (sec.get("supporting") or []))
            if sid in valid_asset_ids and sid != hero_id
        ][:2]
        sections.append({
            "section": section_name,
            "hero": hero_id,
            "supporting": supporting,
        })

    cat_order = [
        s for s in (raw.get("category_order") or [])
        if s in _CURATED_SECTION_NAMES_SET
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


def _derive_curated_section_fallback(images: list[dict]) -> list[dict]:
    """
    Fill curated_section=None for non-excluded images using CURATED_SECTIONS gcv_categories.

    Limitations (by design):
      - bathroom → "Bathrooms" (cannot distinguish Primary Bathroom without vision context)
      - master_bedroom → "Bedrooms" (cannot distinguish Primary Bedroom without vision context)
      - Images the LLM analysed and explicitly assigned a section are never overwritten.
      - Excluded images always get curated_section=None.
    """
    # Build gcv_category → first matching section name (first-match wins)
    gcv_to_section: dict[str, str] = {}
    for sec in CURATED_SECTIONS:
        for gcv_cat in sec["gcv_categories"]:
            if gcv_cat not in gcv_to_section:
                gcv_to_section[gcv_cat] = sec["name"]

    n_filled = 0
    for img in images:
        if img.get("role") == "exclude":
            img["curated_section"] = None
            continue
        if img.get("curated_section"):
            continue   # LLM already assigned — do not overwrite
        derived = gcv_to_section.get(img.get("llm_category") or "")
        img["curated_section"] = derived
        if derived:
            n_filled += 1

    if n_filled:
        logger.info(
            "[LLM Curator] Fallback curated_section derived for %d images", n_filled,
        )
    return images


def _log_curation_summary(result: dict, source_hero_url: Optional[str]) -> None:
    """Log role counts, section counts, and source hero preservation."""
    images = result.get("images") or []
    by_role: dict[str, int] = {}
    by_section: dict[str, int] = {}
    for img in images:
        role = img.get("role") or "unknown"
        by_role[role] = by_role.get(role, 0) + 1
        sec = img.get("curated_section")
        if sec and role != "exclude":
            by_section[sec] = by_section.get(sec, 0) + 1

    llm_hero = (result.get("property") or {}).get("page_hero") or ""
    if source_hero_url:
        hero_preserved = (llm_hero == source_hero_url)
        hero_note = f"preserved={hero_preserved}"
    else:
        hero_note = "no source hero provided"

    logger.info(
        "[LLM Curator] Summary — total=%d | roles: %s | sections: %s | "
        "page_hero: %s",
        len(images),
        ", ".join(f"{r}={n}" for r, n in sorted(by_role.items())),
        ", ".join(f"{s}={n}" for s, n in sorted(by_section.items())),
        hero_note,
    )


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
