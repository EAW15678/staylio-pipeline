"""
Skill: render_page — build a property landing page from substrate data.

Reads photographs, renditions, observations, guest_evidence, copy_versions,
local_guides, and properties from the substrate. Assembles the dicts that
page_builder.build_landing_page_html expects, then calls it unchanged.

No Redis. No media_assets. No property_image_curations.
The substrate IS the data layer.

Usage:
    from skills.render_page import render_page
    result = render_page("a1b2c3d4-...")
    if result.is_ok:
        html = result.data["html"]
"""

import logging
from typing import Optional

from skills.contract import SkillResult, get_substrate

logger = logging.getLogger(__name__)


def render_page(property_id: str) -> SkillResult:
    """Build a landing page HTML from substrate data.

    Returns SkillResult.ok({"html": str, "sections_built": [...], "gallery_count": int})
    or SkillResult.failed(reason).
    """
    try:
        sb = get_substrate()
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    # ── Load property ────────────────────────────────────────────────────
    prop_resp = sb.table("properties").select("*").eq("id", property_id).limit(1).execute()
    if not prop_resp.data:
        return SkillResult.failed(f"Property {property_id} not found in substrate")
    prop = prop_resp.data[0]
    slug = prop.get("slug") or "unknown"

    # ── Build KB dict (what page_builder expects) ────────────────────────
    kb = {
        "property_id": property_id,
        "name": {"value": prop.get("name")},
        "city": {"value": prop.get("city")},
        "state": {"value": prop.get("state_region")},
        "property_type": {"value": prop.get("property_type")},
        "vibe_profile": prop.get("vibe_profile") or "",
        "booking_url": prop.get("booking_url") or "#",
        "bedrooms": {"value": prop.get("bedrooms")},
        "bathrooms": {"value": prop.get("bathrooms")},
        "max_occupancy": {"value": prop.get("max_occupancy")},
        "slug": slug,
        "latitude": {"value": prop.get("latitude")},
        "longitude": {"value": prop.get("longitude")},
        "ical_url": prop.get("ical_url"),
        "neighborhood_description": {"value": None},
        "owner_story": None,
        "wow_factor": None,
        "hidden_gems": None,
        "seasonal_notes": None,
    }

    # ── Load owner_context ───────────────────────────────────────────────
    ctx_resp = sb.table("owner_context").select("*").eq(
        "property_id", property_id
    ).is_("superseded_at", "null").order("version", desc=True).limit(1).execute()
    if ctx_resp.data:
        ctx = ctx_resp.data[0]
        kb["owner_story"] = ctx.get("owner_story")
        kb["wow_factor"] = ctx.get("wow_factor")
        kb["hidden_gems"] = ctx.get("hidden_gems")
        kb["seasonal_notes"] = ctx.get("seasonal_notes")
        kb["arrival_info"] = ctx.get("arrival_info")
        kb["extra_notes"] = ctx.get("extra_notes")
        kb["dont_miss"] = ctx.get("dont_miss")
        kb["surround_areas"] = ctx.get("surround_areas")
        if ctx.get("area_vibe"):
            kb["neighborhood_description"] = {"value": ctx["area_vibe"]}

    # ── Load guest evidence → kb.guest_reviews ───────────────────────────
    ev_resp = sb.table("guest_evidence").select("*").eq("property_id", property_id).execute()
    guest_reviews = []
    for ev in (ev_resp.data or []):
        text = ev.get("written_text") or ""
        verbal = ev.get("verbal_text") or ""
        combined = f"{text}\n\n{verbal}".strip() if text and verbal else (text or verbal)
        guest_reviews.append({
            "text": combined,
            "reviewer_name": ev.get("reviewer_name"),
            "stay_date": ev.get("stay_date"),
            "is_guest_book": ev.get("is_guest_book", True),
            "star_rating": ev.get("star_rating"),
            "host_response": ev.get("host_response"),
            "source": ev.get("source"),
        })
    kb["guest_reviews"] = guest_reviews

    # ── Load copy version ────────────────────────────────────────────────
    copy_resp = sb.table("copy_versions").select("content, version").eq(
        "property_id", property_id
    ).order("version", desc=True).limit(1).execute()

    content_package = {}
    if copy_resp.data:
        content_package = copy_resp.data[0].get("content") or {}

    # ── Load photographs + renditions + observations → visual_media ──────
    # Only render canonical photographs (post-dedupe).
    photos_resp = sb.table("photographs").select(
        "photo_id, content_hash, phash, is_canonical, image_width, image_height"
    ).eq("property_id", property_id).eq("is_canonical", True).execute()
    photos = photos_resp.data or []

    # Get enhanced renditions for each photo
    photo_ids = [p["photo_id"] for p in photos]
    renditions_by_photo = {}
    for pid in photo_ids:
        rend_resp = sb.table("renditions").select("kind, storage_url").eq("photo_id", pid).execute()
        for r in (rend_resp.data or []):
            renditions_by_photo.setdefault(pid, {})[r["kind"]] = r["storage_url"]

    # Get active observations
    obs_resp = sb.table("observations").select("*").eq(
        "property_id", property_id
    ).is_("superseded_at", "null").execute()
    obs_by_photo = {o["photo_id"]: o for o in (obs_resp.data or [])}

    # Build media_assets-shaped list for page_builder
    media_assets = []
    hero_photo_url = None
    unresolvable = 0

    for photo in photos:
        pid = photo["photo_id"]
        rends = renditions_by_photo.get(pid, {})
        obs = obs_by_photo.get(pid, {})

        original_url = rends.get("original", "")
        enhanced_url = rends.get("enhanced", "")
        display_url = enhanced_url or original_url

        if not display_url:
            unresolvable += 1
            continue

        cat = obs.get("curated_section") or "uncategorised"
        # Map section names to GCV categories for page_builder compatibility
        section_to_cat = {
            "Exterior": "exterior", "Pool": "pool_hot_tub",
            "Living Areas": "living_room", "Kitchen": "kitchen",
            "Bedrooms": "standard_bedroom", "Bathrooms": "bathroom",
            "Extras": "uncategorised",
        }
        subject_category = section_to_cat.get(cat, cat.lower().replace(" ", "_"))

        asset = {
            "asset_url_original": original_url,
            "asset_url_enhanced": enhanced_url,
            "subject_category": subject_category,
            "category_rank": obs.get("rank_within_section") or 999,
            "composition_score": obs.get("quality_score") or 0.0,
            "labels_enhanced": [],
            "brightness": None,
            "dominant_colors": None,
            "hero_rank": None,
            "is_hero": (obs.get("role") == "hero"),
            "source": "substrate",
        }
        media_assets.append(asset)

        if obs.get("role") == "hero" and not hero_photo_url:
            hero_photo_url = display_url

    if not hero_photo_url and media_assets:
        hero_photo_url = (media_assets[0].get("asset_url_enhanced")
                         or media_assets[0].get("asset_url_original"))

    # ── Load hero video from video_artifacts ───────────────────────────
    hero_video_url = None
    hero_video_resp = sb.table("video_artifacts").select("storage_url").eq(
        "property_id", property_id
    ).eq("kind", "master").eq("status", "ready").is_(
        "superseded_at", "null"
    ).order("created_at", desc=True).limit(1).execute()
    if hero_video_resp.data:
        hero_video_url = hero_video_resp.data[0].get("storage_url")
        logger.info("[render_page] Hero video found: %s", hero_video_url[:60] if hero_video_url else "none")

    # ── Load guest review narration audio from video_artifacts ────────
    review_audio_urls = {}
    narr_resp = sb.table("video_artifacts").select(
        "storage_url, source_reference"
    ).eq("property_id", property_id).eq("kind", "narration").eq(
        "status", "ready"
    ).eq("provenance", "guest_book").is_("superseded_at", "null").execute()
    for narr in (narr_resp.data or []):
        url = narr.get("storage_url")
        ref = narr.get("source_reference") or {}
        # Key by reviewer name or index for guest card matching
        reviewer = ref.get("reviewer_name") if isinstance(ref, dict) else None
        if url and reviewer:
            review_audio_urls[reviewer] = url

    visual_media = {
        "hero_photo_url": hero_photo_url,
        "hero_video_url": hero_video_url,
        "review_audio_urls": review_audio_urls,
        "media_assets": media_assets,
        "category_winners": {},
    }

    # Build image_curation dict for the LLM curation path
    image_curation = _build_image_curation(obs_resp.data or [], renditions_by_photo, photos)
    if image_curation:
        visual_media["image_curation"] = image_curation

    # ── Load local guide ─────────────────────────────────────────────────
    guide_resp = sb.table("local_guides").select("*").eq("property_id", property_id).limit(1).execute()
    local_guide = {}
    if guide_resp.data:
        g = guide_resp.data[0]
        local_guide = {
            "area_introduction": g.get("area_introduction") or "",
            "primary_recommendations": g.get("primary_recommendations") or [],
            "dont_miss_picks": g.get("dont_miss_picks") or [],
        }

    # Wire owner's dont_miss into the guide's picks
    owner_dont_miss = kb.get("dont_miss") or ""
    if owner_dont_miss:
        # Parse free-text dont_miss into structured picks
        # Owner writes newline-separated or dash-separated items
        import re
        items = [line.strip().lstrip("-•").strip() for line in owner_dont_miss.split("\n") if line.strip()]
        for item in items:
            if item and len(item) > 3:
                # Split on " — " or " -- " to get name and description
                parts = re.split(r"\s*[—–-]{2,}\s*|\s*—\s*", item, maxsplit=1)
                pick = {"name": parts[0].strip()}
                if len(parts) > 1:
                    pick["description"] = parts[1].strip()
                local_guide.setdefault("dont_miss_picks", []).append(pick)

    # ── Build the page ───────────────────────────────────────────────────
    page_url = f"https://{slug}.upliftstays.com"

    from agents.agent5.page_builder import build_landing_page_html
    html = build_landing_page_html(
        kb=kb,
        content_package=content_package,
        visual_media=visual_media,
        local_guide=local_guide,
        page_url=page_url,
        slug=slug,
    )

    # Count sections
    sections_built = []
    for section_name in ["hero", "description", "spotlights", "photo-tour", "gallery",
                         "guest-book", "amenities", "availability", "local-guide",
                         "owner-story", "faqs", "footer-cta"]:
        if f'id="{section_name}"' in html or f'class="{section_name}"' in html:
            sections_built.append(section_name)

    # Count gallery images
    import re
    gallery_count = len(re.findall(r'class="gallery-thumb"', html))
    lightbox_present = "window.openLightbox" in html

    logger.info(
        "[render_page] Property %s: %d photos, %d sections, gallery=%d, lightbox=%s, unresolvable=%d",
        property_id[:12], len(media_assets), len(sections_built), gallery_count,
        lightbox_present, unresolvable,
    )

    return SkillResult.ok({
        "html": html,
        "sections_built": sections_built,
        "gallery_count": gallery_count,
        "photo_count": len(media_assets),
        "lightbox_present": lightbox_present,
        "unresolvable": unresolvable,
    })


def _build_image_curation(observations: list, renditions_by_photo: dict, photos: list) -> dict:
    """Build an image_curation dict from observations for page_builder's LLM curation path."""
    if not observations:
        return {}

    images = []
    sections = {}

    for obs in observations:
        pid = obs["photo_id"]
        rends = renditions_by_photo.get(pid, {})
        original_url = rends.get("original", "")
        enhanced_url = rends.get("enhanced", "")

        img = {
            "asset_id": original_url,
            "role": obs.get("role") or "gallery_only",
            "curated_section": obs.get("curated_section"),
            "quality_score": obs.get("quality_score"),
            "alt": obs.get("alt_text") or "",
            "llm_category": obs.get("curated_section"),
            "rank_within_category": obs.get("rank_within_section") or 999,
            "gallery_visible": obs.get("gallery_visible", True),
            "tour_eligible": obs.get("tour_eligible", True),
            "duplicate_group": obs.get("duplicate_group"),
            "is_best_in_duplicate_group": obs.get("is_best_in_group", True),
        }
        images.append(img)

        # Build tour sections
        section = obs.get("curated_section")
        role = obs.get("role")
        if section and role in ("hero", "supporting"):
            if section not in sections:
                sections[section] = {"section": section, "hero": None, "supporting": []}
            if role == "hero" and not sections[section]["hero"]:
                sections[section]["hero"] = original_url
            elif role == "supporting":
                sections[section]["supporting"].append(original_url)

    return {
        "images": images,
        "property": {
            "photo_tour_sections": [
                {"section": s["section"], "hero": s["hero"], "supporting": s["supporting"]}
                for s in sections.values() if s["hero"]
            ],
        },
        "_source": "substrate",
    }
