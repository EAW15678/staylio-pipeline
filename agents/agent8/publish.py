"""
Agent 8 — Stage 9: Adapt and Publish

Adapts the 9:16 master video (from Stage 7) into platform-specific
variants for TikTok, Instagram, Facebook, and Pinterest, generates
platform-aware captions, runs governance checks, and uploads to R2.

Pinterest gets a 2:3 crop (bottom 300px removed). All others receive the
master unchanged. Captions are generated in a single Claude call per
concept (4 platforms at once).

Standalone module. NOT wired into the pipeline graph.
Call directly: adapt_variants(assembly_id)

SAFE-ZONE NOTE:
The universal safe zone is 900x1400 centered in 1080x1920:
  - Safe zone bottom: 1660px from top of master
  - Pinterest crop: removes bottom 300px -> new height 1620
  - Safe zone bottom (1660) > Pinterest height (1620) -> 40px strip clipped
The 1620-1660 strip IS affected by the Pinterest crop. Stage 7's safe zone
should use 1620 as the effective bottom (matching Pinterest's crop), not 1920.
That is a Stage 7 change. This stage warns if any overlay lands in the strip.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from typing import Optional

import anthropic

from agents.agent8.creative_director import (
    validate_no_guest_names,
    validate_no_ota,
    _OTA_NAMES,
)
from agents.agent8.compliance import (
    check_no_numeric_ratings,
    _NUMERIC_RATING_PATTERNS,
)
from agents.agent8.assembly import (
    SAFE_TOP,
    SAFE_BOTTOM,
    MASTER_W,
    MASTER_H,
)
from agents.agent8.retry import retry_with_backoff

logger = logging.getLogger(__name__)

# ── FFmpeg availability — FAIL LOUDLY ────────────────────────────────────
if not shutil.which("ffmpeg"):
    raise RuntimeError(
        "FFmpeg is NOT installed. Stage 9 requires it for platform adaptation. "
        "Check that railpack.json declares ffmpeg in deploy.aptPackages."
    )

# ── Platform specifications ──────────────────────────────────────────────

PLATFORMS = ["tiktok", "instagram", "facebook", "pinterest"]

PLATFORM_SPECS = {
    "tiktok":    {"aspect": "9:16", "width": 1080, "height": 1920, "crop": False},
    "instagram": {"aspect": "9:16", "width": 1080, "height": 1920, "crop": False},
    "facebook":  {"aspect": "9:16", "width": 1080, "height": 1920, "crop": False},
    "pinterest": {"aspect": "2:3",  "width": 1080, "height": 1620, "crop": True, "crop_px": 300},
}

# Pinterest crop removes bottom 300px
_PINTEREST_CROP_TOP = 1620  # 1920 - 300

# Safe zone strip that Pinterest crop clips (1620-1660, 40px)
_PINTEREST_SAFE_ZONE_CLIP_TOP = _PINTEREST_CROP_TOP  # 1620
_PINTEREST_SAFE_ZONE_CLIP_BOTTOM = SAFE_BOTTOM        # 1660

_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 2048


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════


def _load_assembled_video(assembly_id: str) -> Optional[dict]:
    """Load assembled_videos row by assembly_id."""
    try:
        from core.supabase_store import get_supabase
        sb = get_supabase()
        result = (
            sb.table("assembled_videos")
            .select("*")
            .eq("assembly_id", assembly_id)
            .is_("superseded_at", "null")
            .limit(1)
            .execute()
        )
        if not result.data:
            # Try with 'id' column
            result = (
                sb.table("assembled_videos")
                .select("*")
                .eq("id", assembly_id)
                .is_("superseded_at", "null")
                .limit(1)
                .execute()
            )
        return result.data[0] if result.data else None
    except Exception as exc:
        logger.error("[Agent 8 Stage 9] Failed to load assembly %s: %s", assembly_id, exc)
        return None


def _load_concept(concept_id: str) -> Optional[dict]:
    """Load concept from concept_ledger by concept_id."""
    try:
        from core.supabase_store import get_supabase
        sb = get_supabase()
        result = (
            sb.table("concept_ledger")
            .select("*")
            .eq("concept_id", concept_id)
            .is_("superseded_at", "null")
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None
    except Exception as exc:
        logger.error("[Agent 8 Stage 9] Failed to load concept %s: %s", concept_id, exc)
        return None


def _load_existing_variant(
    assembly_id: str, platform: str,
) -> Optional[dict]:
    """Check if an active variant already exists for this assembly+platform."""
    try:
        from core.supabase_store import get_supabase
        sb = get_supabase()
        result = (
            sb.table("video_variants")
            .select("*")
            .eq("assembly_id", assembly_id)
            .eq("platform", platform)
            .eq("status", "draft")
            .is_("superseded_at", "null")
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None
    except Exception as exc:
        logger.warning("[Agent 8 Stage 9] Variant check failed: %s", exc)
        return None


def _resolve_link_url(property_id: str) -> str | None:
    """Resolve per-property from landing_pages.page_url. Never a constant."""
    from core.supabase_store import get_supabase
    sb = get_supabase()
    result = (
        sb.table("landing_pages")
        .select("page_url")
        .eq("property_id", property_id)
        .limit(1)
        .execute()
    )
    if result.data:
        return result.data[0].get("page_url")
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Safe zone verification for Pinterest crop
# ═══════════════════════════════════════════════════════════════════════════


def _check_pinterest_safe_zone(assembly: dict) -> list[str]:
    """
    Check if any overlay in the assembly lands in the 1620-1660 strip
    that Pinterest's crop will clip. Returns list of warnings.
    """
    warnings: list[str] = []
    overlays = assembly.get("overlay_payload") or []
    for i, ov in enumerate(overlays):
        y = ov.get("y")
        h = ov.get("height")
        if y is not None and h is not None:
            try:
                y_val = int(y)
                h_val = int(h)
                ov_bottom = y_val + h_val
                if ov_bottom > _PINTEREST_SAFE_ZONE_CLIP_TOP:
                    warnings.append(
                        f"Overlay {i} extends to y={ov_bottom}, which is in the "
                        f"Pinterest crop zone (1620-1920). Bottom 40px of safe zone "
                        f"(1620-1660) will be clipped."
                    )
            except (ValueError, TypeError):
                pass

        # Also check by grid_region
        region = ov.get("grid_region", "")
        if "bottom" in region.lower():
            warnings.append(
                f"Overlay {i} uses grid_region '{region}' — bottom row of safe zone "
                f"(1620-1660 strip) is partially clipped by Pinterest's 2:3 crop."
            )

    return warnings


# ═══════════════════════════════════════════════════════════════════════════
# FFmpeg adaptation
# ═══════════════════════════════════════════════════════════════════════════


def _crop_for_pinterest(video_bytes: bytes) -> bytes:
    """
    Crop bottom 300px from a 1080x1920 master to produce 1080x1620 (2:3).
    Uses ffmpeg -vf crop=1080:1620:0:0
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_in:
        f_in.write(video_bytes)
        in_path = f_in.name

    out_path = in_path.replace(".mp4", "_pinterest.mp4")

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", in_path,
            "-vf", "crop=1080:1620:0:0",
            "-c:a", "copy",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg Pinterest crop failed (rc={result.returncode}): "
                f"{result.stderr.decode(errors='replace')[:500]}"
            )
        with open(out_path, "rb") as f_out:
            return f_out.read()
    finally:
        for p in [in_path, out_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# Caption generation
# ═══════════════════════════════════════════════════════════════════════════


def _generate_captions(title: str, premise: str) -> dict[str, str]:
    """
    One Claude Sonnet call per concept — generates all 4 platform captions.
    Returns dict mapping platform name to caption text.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    prompt = f"""\
For this vacation rental social media concept:
Title: {title}
Premise: {premise}

Write 4 platform-specific captions:
1. TikTok (max 150 chars, punchy, trend-aware)
2. Instagram (max 2200 chars, storytelling, hashtag-ready)
3. Facebook (max 500 chars, community-oriented)
4. Pinterest (max 500 chars, aspirational, search-friendly)

Rules:
- NO guest names
- NO OTA references (Airbnb, VRBO, Booking.com)
- NO numeric ratings
- Guest quotes follow social scope: corrected misspellings OK, truncation OK, meaning cannot change

Return ONLY valid JSON — no markdown, no prose, no code fences:
{{
  "tiktok": "...",
  "instagram": "...",
  "facebook": "...",
  "pinterest": "..."
}}"""

    def _attempt() -> dict[str, str]:
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
        for platform in PLATFORMS:
            if platform not in parsed:
                raise ValueError(f"Missing caption for {platform}")
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
            label="Claude caption generation",
        )
    except Exception as exc:
        logger.error("[Agent 8 Stage 9] Caption generation failed: %s", exc)
        raise


# ═══════════════════════════════════════════════════════════════════════════
# Governance checks on captions
# ═══════════════════════════════════════════════════════════════════════════


def _check_caption_governance(
    caption: str, platform: str, kb: dict | None,
) -> list[dict]:
    """
    Run governance checks on a single caption:
    - no_guest_names
    - no_ota
    - no_numeric_ratings
    Returns list of violation dicts.
    """
    violations: list[dict] = []

    # Build a spec-like dict for validate_no_guest_names
    if kb:
        spec_like = {"narration_brief": caption, "overlay_register": []}
        name_violations = validate_no_guest_names(spec_like, kb)
        for v in name_violations:
            violations.append({
                "rule": "no_guest_names",
                "platform": platform,
                "detail": v["detail"],
            })

    # OTA check
    ota_spec = {"narration_brief": caption, "overlay_register": []}
    ota_violations = validate_no_ota(ota_spec)
    for v in ota_violations:
        violations.append({
            "rule": "no_ota",
            "platform": platform,
            "detail": v["detail"],
        })

    # Numeric ratings check
    subject_like = {"caption": caption}
    rating_result = check_no_numeric_ratings(subject_like)
    if rating_result["severity"] == "fail":
        violations.append({
            "rule": "no_numeric_ratings",
            "platform": platform,
            "detail": rating_result["detail"],
        })

    return violations


# ═══════════════════════════════════════════════════════════════════════════
# UTM content slug
# ═══════════════════════════════════════════════════════════════════════════

# UTM CONTENT COLLISION — two schemes exist in the codebase:
# 1. Agent 6 utm_generator.py: video_type_week_seq (e.g. "vibe_match_w01_01")
# 2. Agent 8 concept_ledger.utm_content_slug: property_cycle_concept
#    (e.g. "vista-azule_2026-09_c1")
#
# This stage writes the Agent 8 scheme plus platform:
#   "vista-azule_2026-09_c1_tiktok"
#
# This is the ONLY path from a published post back to a booking (SOC-08).
# Erick decides which scheme wins. DO NOT change Agent 6.


def _build_variant_utm_content(concept_slug: str, platform: str) -> str:
    """
    Build utm_content for a variant: {concept_slug}_{platform}
    e.g. "vista-azule_2026-09_c1_tiktok"
    """
    return f"{concept_slug}_{platform}"


# ═══════════════════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════════════════


def _persist_variant(variant: dict) -> None:
    """Write variant to video_variants table."""
    try:
        from core.supabase_store import get_supabase
        sb = get_supabase()
        sb.table("video_variants").insert(variant).execute()
    except Exception as exc:
        logger.error("[Agent 8 Stage 9] Persist variant failed: %s", exc)


def _load_kb(property_id: str) -> dict | None:
    """Load KB — Redis first, Supabase fallback."""
    from core.pipeline_status import get_cached_knowledge_base
    kb = get_cached_knowledge_base(property_id)
    if kb:
        return kb
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
    except Exception as exc:
        logger.warning("[Agent 8 Stage 9] KB load failed: %s", exc)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════


def adapt_variants(
    assembly_id: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> list[dict]:
    """
    Adapt a 9:16 master video into platform-specific variants.

    For each of the 4 platforms:
      - Pinterest: crop bottom 300px via FFmpeg (2:3)
      - TikTok/Instagram/Facebook: copy master unchanged (9:16)
      - Generate platform-adapted caption
      - Run governance checks on caption
      - Build utm_content slug
      - Resolve link_url per property
      - Upload adapted file to R2
      - Status = "draft" (cannot be "ready" without compliance check)

    Args:
        assembly_id: UUID of the assembled_videos row.
        dry_run: If True, return variant specs without uploading or persisting.
        force: If True, ignore existing variants and recreate.

    Returns:
        List of variant dicts (one per platform).
    """
    logger.info(
        "[Agent 8 Stage 9] adapt_variants: assembly_id=%s dry_run=%s force=%s",
        assembly_id, dry_run, force,
    )

    # ── Load assembled video ───────────────────────────────────────────
    assembly = _load_assembled_video(assembly_id)
    if not assembly:
        raise ValueError(f"Assembled video {assembly_id} not found or not active")

    if assembly.get("status") != "ready":
        raise ValueError(
            f"Assembled video {assembly_id} status is '{assembly.get('status')}', "
            f"expected 'ready'"
        )

    property_id = assembly["property_id"]
    concept_id = assembly.get("concept_id")
    r2_url = assembly.get("r2_url")
    if not r2_url:
        raise ValueError(f"Assembled video {assembly_id} has no r2_url")

    # ── Load concept for caption generation ────────────────────────────
    concept = _load_concept(concept_id) if concept_id else None
    title = (concept.get("title") or "Vacation Rental") if concept else "Vacation Rental"
    premise = (concept.get("premise") or "") if concept else ""
    concept_slug = (concept.get("utm_content_slug") or "") if concept else ""

    # ── Load KB for governance checks ──────────────────────────────────
    kb = _load_kb(property_id)

    # ── Resolve link URL per property ──────────────────────────────────
    link_url = _resolve_link_url(property_id)

    # ── Check Pinterest safe zone ──────────────────────────────────────
    safe_zone_warnings = _check_pinterest_safe_zone(assembly)
    for warning in safe_zone_warnings:
        logger.warning("[Agent 8 Stage 9] Pinterest safe zone: %s", warning)

    # ── Download master video bytes ────────────────────────────────────
    master_bytes: bytes | None = None
    if not dry_run:
        try:
            from core.r2_storage import download_bytes, BUCKET_VIDEO
            # Parse the R2 key from the URL
            # r2_url is like https://assets.example.com/{property_id}/video/...
            from urllib.parse import urlparse
            parsed_url = urlparse(r2_url)
            r2_key = parsed_url.path.lstrip("/")
            master_bytes = download_bytes(BUCKET_VIDEO, r2_key)
            if master_bytes is None:
                raise RuntimeError(f"Failed to download master from R2: {r2_url}")
        except Exception as exc:
            raise RuntimeError(f"Failed to download master video: {exc}") from exc

    # ── Generate captions (one Claude call for all 4 platforms) ────────
    if not dry_run:
        captions = _generate_captions(title, premise)
    else:
        captions = {p: f"[dry_run] {title} — {p}" for p in PLATFORMS}

    # ── Build variants ─────────────────────────────────────────────────
    variants: list[dict] = []
    seen_utm: set[str] = set()

    for platform in PLATFORMS:
        # Check existing variant (unless force)
        if not force:
            existing = _load_existing_variant(assembly_id, platform)
            if existing:
                logger.info(
                    "[Agent 8 Stage 9] Existing variant for %s/%s — skipping",
                    assembly_id, platform,
                )
                variants.append(existing)
                continue

        spec = PLATFORM_SPECS[platform]
        caption = captions.get(platform, "")

        # ── Governance checks on caption ───────────────────────────────
        caption_violations = _check_caption_governance(caption, platform, kb)
        if caption_violations:
            logger.warning(
                "[Agent 8 Stage 9] Caption governance violations for %s: %s",
                platform, caption_violations,
            )
            # Mark as held — cannot publish with violations
            status = "held"
        else:
            status = "draft"  # Cannot be "ready" without passing compliance check

        # ── Build utm_content ──────────────────────────────────────────
        utm_content = _build_variant_utm_content(concept_slug, platform)

        # Verify uniqueness
        if utm_content in seen_utm:
            logger.error(
                "[Agent 8 Stage 9] utm_content collision: %s", utm_content,
            )
        seen_utm.add(utm_content)

        # ── Adapt video ───────────────────────────────────────────────
        adapted_r2_url: str | None = None
        if not dry_run and master_bytes is not None:
            if spec["crop"]:
                # Pinterest: crop bottom 300px
                adapted_bytes = _crop_for_pinterest(master_bytes)
            else:
                # TikTok/Instagram/Facebook: copy master unchanged
                adapted_bytes = master_bytes

            # Upload to R2
            try:
                from core.r2_storage import upload_video
                adapted_r2_url = upload_video(
                    property_id=property_id,
                    video_bytes=adapted_bytes,
                    video_type=f"variant_{platform}",
                    format_label=spec["aspect"].replace(":", "x"),
                    extension="mp4",
                )
            except Exception as exc:
                logger.error(
                    "[Agent 8 Stage 9] R2 upload failed for %s: %s",
                    platform, exc,
                )
                adapted_r2_url = None
                status = "failed"

        # ── Build variant record ──────────────────────────────────────
        variant: dict = {
            "variant_id": str(uuid.uuid4()),
            "assembly_id": assembly_id,
            "property_id": property_id,
            "concept_id": concept_id,
            "platform": platform,
            "aspect_ratio": spec["aspect"],
            "width": spec["width"],
            "height": spec["height"],
            "caption": caption,
            "caption_violations": caption_violations,
            "utm_content": utm_content,
            "link_url": link_url,
            "r2_url": adapted_r2_url,
            "compliance_check_id": None,  # Stage 8 checks after
            "status": status,
            "safe_zone_warnings": safe_zone_warnings if platform == "pinterest" else [],
            "created_by_agent": "agent8_stage9",
        }

        if not dry_run:
            _persist_variant(variant)

        variants.append(variant)
        logger.info(
            "[Agent 8 Stage 9] Variant %s: aspect=%s status=%s utm=%s",
            platform, spec["aspect"], status, utm_content,
        )

    logger.info(
        "[Agent 8 Stage 9] Adapted %d variants for assembly %s",
        len(variants), assembly_id,
    )

    return variants
