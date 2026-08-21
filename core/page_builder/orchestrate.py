"""
Substrate-native page orchestrator — the single data path.

Created 2026-08-19 (PAGE-3). Zero imports from agents/.

This module exists because PAGE-2 proved that the old orchestrator
(agents/agent5/page_builder.build_landing_page_html) never actually used
substrate curation data. The _curation_active check at page_builder.py:274
tests image_curation.get("status") == "complete", but render_page.py's
_build_image_curation never sets a "status" key. Every substrate property
silently rendered through the old Google Vision heuristic path instead.

This orchestrator has exactly one data path — by design, not by omission.
Observations either exist and are used directly, or they don't and the page
holds. No _curation_active check, no heuristic fallback, no branching
between "real data" and "degraded data."

Hold conditions (ruled by Erick, 2026-08-19):
  - No canonical photographs → hold, do not publish
  - Photographs exist but none have an active observation → hold, do not publish
  - No hero video → publish anyway, section absent
  - No guest narration audio → publish anyway, no "hear it" button

If a page is already live for this property, a hold must never take it down
or overwrite it. The existing landing_pages row and its published R2 file
are left exactly as they are. The hold only prevents this run from replacing
something that currently works.
"""

import logging
import re
import uuid

from core.page_builder.vocabulary import normalise_section, SECTIONS
from core.page_builder.gallery import _prepare_gallery_items, _build_category_modules
from core.page_builder.sections import (
    _build_spotlights_section,
    _build_gallery_section,
    _build_guest_book_section,
    _build_ota_reviews_section,
    _build_amenities_section,
    _build_good_to_know_section,
    _build_local_guide_section,
    _build_owner_story_section,
    _build_faq_section,
    _build_category_modules_section,
)
from core.page_builder.lightbox import _build_lightbox_gallery_js
from core.page_builder.css import _page_css
from core.page_builder.helpers import _esc, _format_description, _calendar_widget_js, _get_headline_variants
from core.page_builder.schema_markup import build_schema_from_inputs
from core.page_builder.ab_testing import generate_growthbook_snippet

logger = logging.getLogger(__name__)

# UTM parameters applied to every Book Now link
UTM_TEMPLATE = "?utm_source=booked&utm_medium=landing_page&utm_campaign={slug}&utm_content=book_now"

# ── Amenity photo-proof matching ──────────────────────────────────────────────
# Replaced the old 16-entry _AMENITY_SYNONYMS dict (AMENITY-2, 2026-08-20).
# Now uses the canonical taxonomy from amenity_taxonomy.py.
#
# For the 7 object-based categories (water_wellness, recreation, etc.), match
# against located_amenities WITHIN THE SAME CATEGORY using substring matching
# on the name — category narrows the candidate pool, name decides precision
# (a "Fire pit" claim must not match a "Fireplace" photo just because both
# are in "gathering").
#
# For the `setting` category (Ocean views, Mountain views, etc.), match against
# observations.is_setting = true and setting_subject instead — these describe
# fixed geography, not a photographable object. located_amenities does not
# carry setting data.

from core.page_builder.amenity_taxonomy import (
    _AMENITY_PHOTO_SYNONYMS,
    _SETTING_PHOTO_SYNONYMS,
    SETTING_AMENITIES,
    extract_amenity_names,
)


def _find_amenity_photo(amenity_name: str, amenity_category: str,
                        photos_with_obs: list) -> str:
    """Find a photo whose observations match an amenity claim.

    For setting-category amenities: matches against is_setting/setting_subject.
    For all other categories: matches against located_amenities within the
    same category using substring matching on the name.

    Returns the display URL of the best matching photo, or empty string.
    """
    if amenity_category == "setting":
        synonyms = _SETTING_PHOTO_SYNONYMS.get(amenity_name, set())
        if not synonyms:
            return ""
        for photo in photos_with_obs:
            if photo.get("is_setting") and photo.get("setting_subject"):
                subject = photo["setting_subject"].lower()
                for syn in synonyms:
                    if syn in subject:
                        return photo.get("display_url") or ""
        return ""

    # Non-setting: match against located_amenities
    synonyms = _AMENITY_PHOTO_SYNONYMS.get(amenity_name, set())
    if not synonyms:
        return ""

    for photo in photos_with_obs:
        located = photo.get("located_amenities") or []
        for loc in located:
            loc_name = (loc.get("name") or "").lower()
            loc_cat = (loc.get("category") or "").lower()
            # Category narrows the candidate pool when available
            if amenity_category and loc_cat and loc_cat != amenity_category:
                continue
            for syn in synonyms:
                if syn in loc_name:
                    return photo.get("display_url") or ""
    return ""


def build_page(sb, property_id: str) -> dict:
    """Build a landing page from substrate data. Single data path.

    Args:
        sb: Supabase client from get_substrate()
        property_id: UUID string

    Returns dict with either:
        {"ok": True, "html": str, "photo_count": int, "gallery_count": int,
         "sections_built": list, "unresolvable": int, "lightbox_present": bool}
    or:
        {"ok": False, "hold": True, "reason": str}
    """
    # ── Load property ────────────────────────────────────────────────────
    prop_resp = sb.table("properties").select("*").eq("id", property_id).limit(1).execute()
    if not prop_resp.data:
        return {"ok": False, "hold": False, "reason": f"Property {property_id} not found"}
    prop = prop_resp.data[0]
    slug = prop.get("slug") or "unknown"
    name = prop.get("name") or "Property"

    # ── Load canonical photographs ───────────────────────────────────────
    photos_resp = sb.table("photographs").select(
        "photo_id, content_hash, phash, is_canonical, image_width, image_height"
    ).eq("property_id", property_id).eq("is_canonical", True).execute()
    photos = photos_resp.data or []

    # ── HOLD: no canonical photographs ───────────────────────────────────
    # Ruled by Erick, 2026-08-19: no canonical photographs → hold, do not publish.
    # If a page is already live, the hold must never take it down — the existing
    # landing_pages row and R2 file are left exactly as they are.
    if not photos:
        reason = f"No canonical photographs for property {property_id[:12]}"
        hold_code = "no_canonical_photos"
        raise_hold(sb, property_id, reason, hold_code)
        return {
            "ok": False, "hold": True,
            "reason": reason,
            "hold_code": hold_code,
        }

    # ── Load renditions ──────────────────────────────────────────────────
    photo_ids = [p["photo_id"] for p in photos]
    renditions_by_photo = {}
    for pid in photo_ids:
        rend_resp = sb.table("renditions").select("kind, storage_url").eq("photo_id", pid).execute()
        for r in (rend_resp.data or []):
            renditions_by_photo.setdefault(pid, {})[r["kind"]] = r["storage_url"]

    # ── Load observations ────────────────────────────────────────────────
    obs_resp = sb.table("observations").select("*").eq(
        "property_id", property_id
    ).is_("superseded_at", "null").execute()
    obs_by_photo = {o["photo_id"]: o for o in (obs_resp.data or [])}

    # ── HOLD: photographs exist but none have observations ───────────────
    # Ruled by Erick, 2026-08-19: same hold behaviour as zero photos.
    observed_count = sum(1 for p in photos if p["photo_id"] in obs_by_photo)
    if observed_count == 0:
        reason = f"Property {property_id[:12]} has {len(photos)} photos but 0 observations"
        hold_code = "no_observations"
        raise_hold(sb, property_id, reason, hold_code)
        return {
            "ok": False, "hold": True,
            "reason": reason,
            "hold_code": hold_code,
        }

    # ── Build media assets with curated_section directly ─────────────────
    # No translation dict. No GCV names. curated_section passes straight through.
    media_assets = []
    hero_photo_url = None
    unresolvable = 0
    photos_with_obs = []  # for amenity photo-proof

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

        section = normalise_section(obs.get("curated_section") or "Extras")

        asset = {
            "asset_url_original": original_url,
            "asset_url_enhanced": enhanced_url,
            "curated_section": section,
            "category_rank": obs.get("rank_within_section") or 999,
            "composition_score": obs.get("quality_score") or 0.0,
            "labels_enhanced": [],
            "source": "substrate",
        }
        media_assets.append(asset)

        # Track for amenity photo-proof — both located_amenities and setting data
        if obs.get("located_amenities") or obs.get("is_setting"):
            photos_with_obs.append({
                "display_url": display_url,
                "located_amenities": obs.get("located_amenities") or [],
                "is_setting": obs.get("is_setting", False),
                "setting_subject": obs.get("setting_subject") or "",
            })

        if obs.get("role") == "hero" and not hero_photo_url:
            hero_photo_url = display_url

    if not hero_photo_url and media_assets:
        hero_photo_url = (media_assets[0].get("asset_url_enhanced")
                         or media_assets[0].get("asset_url_original"))

    # ── Gallery ──────────────────────────────────────────────────────────
    gallery_items = _prepare_gallery_items(
        media_assets=media_assets,
        hero_photo=hero_photo_url or "",
        kb_photos=[],
        property_name=name,
    )
    category_modules = _build_category_modules(gallery_items) if media_assets else {}

    # ── Hero video — no fallback, single query ───────────────────────────
    hero_video_url = None
    hero_video_resp = sb.table("video_artifacts").select("storage_url").eq(
        "property_id", property_id
    ).eq("kind", "master").eq("status", "ready").is_(
        "superseded_at", "null"
    ).order("created_at", desc=True).limit(1).execute()
    if hero_video_resp.data:
        hero_video_url = hero_video_resp.data[0].get("storage_url")

    # ── Guest narration audio — no fallback ──────────────────────────────
    review_audio_urls = {}
    narr_resp = sb.table("video_artifacts").select(
        "storage_url, source_reference"
    ).eq("property_id", property_id).eq("kind", "narration").eq(
        "status", "ready"
    ).eq("provenance", "guest_book").is_("superseded_at", "null").execute()
    for narr in (narr_resp.data or []):
        url = narr.get("storage_url")
        ref = narr.get("source_reference") or {}
        reviewer = ref.get("reviewer_name") if isinstance(ref, dict) else None
        if url and reviewer:
            review_audio_urls[reviewer] = url

    # ── Owner context ────────────────────────────────────────────────────
    ctx_resp = sb.table("owner_context").select("*").eq(
        "property_id", property_id
    ).is_("superseded_at", "null").order("version", desc=True).limit(1).execute()
    ctx = ctx_resp.data[0] if ctx_resp.data else {}

    # ── Guest evidence ───────────────────────────────────────────────────
    ev_resp = sb.table("guest_evidence").select("*").eq("property_id", property_id).execute()
    guest_reviews = []
    ota_reviews = []
    for ev in (ev_resp.data or []):
        text = ev.get("written_text") or ""
        verbal = ev.get("verbal_text") or ""
        combined = f"{text}\n\n{verbal}".strip() if text and verbal else (text or verbal)
        review = {
            "text": combined,
            "reviewer_name": ev.get("reviewer_name"),
            "stay_date": ev.get("stay_date"),
            "star_rating": ev.get("star_rating"),
        }
        if ev.get("is_guest_book"):
            guest_reviews.append(review)
        else:
            ota_reviews.append(review)

    # ── Copy version ─────────────────────────────────────────────────────
    copy_resp = sb.table("copy_versions").select("content, version").eq(
        "property_id", property_id
    ).order("version", desc=True).limit(1).execute()
    content_package = {}
    if copy_resp.data:
        content_package = copy_resp.data[0].get("content") or {}

    # ── Local guide ──────────────────────────────────────────────────────
    guide_resp = sb.table("local_guides").select("*").eq("property_id", property_id).limit(1).execute()
    guide = guide_resp.data[0] if guide_resp.data else {}
    area_intro = guide.get("area_introduction") or ""
    primary_recs = guide.get("primary_recommendations") or []
    dont_miss_picks = list(guide.get("dont_miss_picks") or [])

    # Wire owner's dont_miss into picks
    owner_dont_miss = ctx.get("dont_miss") or ""
    if owner_dont_miss:
        items = [line.strip().lstrip("-\u2022").strip() for line in owner_dont_miss.split("\n") if line.strip()]
        for item in items:
            if item and len(item) > 3:
                parts = re.split(r"\s*[\u2014\u2013-]{2,}\s*|\s*\u2014\s*", item, maxsplit=1)
                pick = {"name": parts[0].strip()}
                if len(parts) > 1:
                    pick["description"] = parts[1].strip()
                dont_miss_picks.append(pick)

    # ── Amenity photo-proof (ruled by Erick, 2026-08-19; updated AMENITY-2) ──
    # For each amenity claim, find a matching photo. Setting-category items match
    # via is_setting/setting_subject; all others via located_amenities within
    # the same category. If none exists, the claim stays text-only.
    amenity_highlights = content_package.get("amenity_highlights") or {}
    amenities_list = prop.get("amenities") or []
    amenity_photos = {}
    for amenity_entry in amenities_list:
        # Handle both structured {name, category} and legacy flat strings
        if isinstance(amenity_entry, dict):
            a_name = amenity_entry.get("name") or ""
            a_cat = amenity_entry.get("category") or ""
        elif isinstance(amenity_entry, str):
            a_name = amenity_entry
            a_cat = ""
        else:
            continue
        if a_name:
            photo_url = _find_amenity_photo(a_name, a_cat, photos_with_obs)
            if photo_url:
                amenity_photos[a_name] = photo_url

    # ── Build page URL and booking URL ───────────────────────────────────
    page_url = f"https://{slug}.upliftstays.com"
    booking_url = prop.get("booking_url") or "#"
    utm_params = UTM_TEMPLATE.format(slug=slug)
    book_now_url = booking_url + utm_params if booking_url != "#" else "#"

    # ── Schema.org markup ────────────────────────────────────────────────
    kb_for_schema = {
        "name": prop.get("name"),
        "description": content_package.get("property_description"),
        "booking_url": booking_url,
        "city": {"value": prop.get("city")},
        "state": {"value": prop.get("state_region")},
        "zip_code": {"value": prop.get("zip_code")},
        "address_line1": {"value": prop.get("address_line1")},
        "latitude": {"value": prop.get("latitude")},
        "longitude": {"value": prop.get("longitude")},
        "bedrooms": {"value": prop.get("bedrooms")},
        "bathrooms": {"value": prop.get("bathrooms")},
        "max_occupancy": {"value": prop.get("max_occupancy")},
        "avg_nightly_rate": {"value": prop.get("avg_nightly_rate")},
        "airbnb_rating": {"value": prop.get("airbnb_rating")},
        "airbnb_review_count": {"value": prop.get("airbnb_review_count")},
        "amenities": [{"value": n} for n in extract_amenity_names(amenities_list)],
    }
    schema_html = build_schema_from_inputs(
        kb=kb_for_schema,
        content_package=content_package,
        visual_media={"hero_photo_url": hero_photo_url},
        page_url=page_url,
        slug=slug,
    )

    # ── GrowthBook ───────────────────────────────────────────────────────
    headline_variants = _get_headline_variants(content_package)
    growthbook_html = generate_growthbook_snippet(property_id, slug, headline_variants)

    # ── Assemble HTML ────────────────────────────────────────────────────
    city = prop.get("city") or ""
    state = prop.get("state_region") or ""
    location_str = f"{city}, {state}" if city else state
    description = content_package.get("property_description") or ""
    tagline = content_package.get("tagline") or ""
    spotlights = content_package.get("spotlights") or []
    owner_story = ctx.get("owner_story") or ""
    arrival_info = ctx.get("arrival_info") or ""
    extra_notes = ctx.get("extra_notes") or ""
    faqs = content_package.get("faqs") or []
    ical_url = prop.get("ical_url") or ""

    bedrooms = prop.get("bedrooms")
    bathrooms = prop.get("bathrooms")
    max_occ = prop.get("max_occupancy")
    specs_parts = []
    if bedrooms:
        specs_parts.append(f"{bedrooms} Bedrooms")
    if bathrooms:
        specs_parts.append(f"{bathrooms} Bathrooms")
    if max_occ:
        specs_parts.append(f"Sleeps {max_occ}")

    # Hero section
    hero_media_html = ""
    if hero_video_url:
        hero_media_html = (
            f'<video autoplay muted loop playsinline id="hero-video">'
            f'<source src="{_esc(hero_video_url)}" type="video/mp4"></video>'
        )
    elif hero_photo_url:
        hero_media_html = f'<img src="{_esc(hero_photo_url)}" alt="{_esc(name)}">'

    specs_html = "".join(f"<span>{_esc(s)}</span>" for s in specs_parts)

    hero_html = f"""
  <section class="hero">
    <div class="hero-media">{hero_media_html}</div>
    <div class="hero-overlay"></div>
    <div id="hero-cta-overlay">
      <a id="hero-cta-btn" href="{_esc(book_now_url)}">
        <span class="hero-cta-icon">&#9758;</span>
        <span class="hero-cta-text">Check Availability</span>
      </a>
    </div>
    <div class="hero-content">
      <p class="location-tag">{_esc(location_str)}</p>
      <h1 class="hero-headline" id="hero-headline">{_esc(name)}</h1>
      <p class="hero-tagline">{_esc(tagline)}</p>
      <div class="hero-specs">{specs_html}</div>
    </div>
  </section>"""

    # Description
    desc_html = ""
    if description:
        desc_html = f"""
  <section class="description" id="description">
    <div class="container">
      <div class="description-text">{_format_description(description)}</div>
    </div>
  </section>"""

    # Sections
    spotlights_html = _build_spotlights_section(spotlights) if spotlights else ""
    modules_html = _build_category_modules_section(category_modules, gallery_items)
    gallery_html = _build_gallery_section(gallery_items, name)
    lightbox_html = _build_lightbox_gallery_js(gallery_items)
    guest_book_html = _build_guest_book_section(guest_reviews, review_audio_urls) if guest_reviews else ""
    ota_html = _build_ota_reviews_section(ota_reviews) if ota_reviews else ""
    amenities_html = _build_amenities_section(amenity_highlights, amenity_photos) if amenity_highlights else ""
    good_to_know_html = _build_good_to_know_section(arrival_info, extra_notes)
    local_html = _build_local_guide_section(area_intro, dont_miss_picks, primary_recs, location_str) if (area_intro or primary_recs or dont_miss_picks) else ""
    owner_html = _build_owner_story_section(owner_story) if owner_story else ""
    faq_html = _build_faq_section(faqs) if faqs else ""

    # Calendar
    calendar_html = ""
    if ical_url:
        calendar_html = f"""
  <section class="availability" id="availability">
    <div class="container">
      <h2>Availability</h2>
      <p class="calendar-subtext">Select your dates to check availability</p>
      <div id="calendar-widget" data-cache-url="{_esc(ical_url)}">
        <div class="cal-nav">
          <button class="cal-nav-btn" id="cal-prev-btn" type="button">&larr; Previous</button>
          <button class="cal-nav-btn" id="cal-next-btn" type="button">Next &rarr;</button>
        </div>
        <div class="cal-month-grid" id="cal-month-grid"></div>
        <p class="calendar-legend">
          <span class="legend-available"></span> Available
          <span class="legend-blocked"></span> Booked
        </p>
        <p class="calendar-helper">Dates shown may not reflect real-time availability. Confirm with the host.</p>
      </div>
      <div class="calendar-cta">
        <a href="{_esc(book_now_url)}" class="staylio-cta-btn staylio-cta-full">Check Availability</a>
      </div>
    </div>
  </section>
  <script>{_calendar_widget_js()}</script>"""

    # Footer
    footer_html = f"""
  <section class="footer-cta">
    <div class="container">
      <h2>Ready to Book Your Stay?</h2>
      <p>Experience {_esc(name)} for yourself</p>
      <a href="{_esc(book_now_url)}" class="staylio-cta-btn cta-large">Check Availability</a>
    </div>
  </section>
  <footer class="site-footer">
    <div class="container">
      <p>&copy; {_esc(name)} &middot; <span class="powered-by">Powered by <a href="https://staylio.ai">Staylio</a></span></p>
    </div>
  </footer>"""

    # Hero video "Hear me" unmute control — restored from the pre-PAGE-2
    # implementation. Dropped during the PAGE-2/3/4 rebuild, not intentionally
    # removed. On click: unmutes and restarts from the beginning. On loop
    # restart: re-mutes automatically so unmuted playback is a deliberate,
    # user-requested state, not permanent. The video loops muted by default
    # (autoplay muted loop playsinline) for browser compliance.
    hero_video_js = ""
    if hero_video_url:
        hero_video_js = """
  <script>
  (function(){
    var v=document.getElementById('hero-video');
    if(!v)return;
    var hb=document.createElement('button');
    hb.id='hero-hear-btn';
    hb.textContent='\\uD83D\\uDD0A Hear me';
    hb.setAttribute('aria-label','Unmute and play video with sound');
    v.parentElement.parentElement.appendChild(hb);
    hb.addEventListener('click',function(){
      v.muted=false;
      v.currentTime=0;
      v.play();
      hb.textContent='\\uD83D\\uDD07 Mute';
    });
    v.addEventListener('timeupdate',function(){
      if(!v.muted && v.currentTime<0.1 && v.duration>0){
        v.muted=true;
        hb.textContent='\\uD83D\\uDD0A Hear me';
      }
    });
  })();
  </script>"""

    # Guest audio toggle — clicking a playing track stops it; clicking a
    # different track stops the old one and starts the new one. The previous
    # implementation unconditionally created a new Audio object on every click,
    # including re-clicking the same button — no way to stop playback.
    # Dropped feature restored from pre-PAGE-2, found 2026-08-20.
    audio_js = ""
    if guest_reviews and review_audio_urls:
        audio_js = """
  <script>
  (function(){
    var current=null,currentBtn=null;
    document.querySelectorAll('.audio-play-btn').forEach(function(btn){
      btn.addEventListener('click',function(){
        var src=this.getAttribute('data-audio-src');
        if(!src)return;
        if(current && currentBtn===this){
          current.pause();current=null;currentBtn=null;
          return;
        }
        if(current){current.pause();current=null;}
        current=new Audio(src);
        currentBtn=this;
        current.addEventListener('ended',function(){current=null;currentBtn=null;});
        current.play().catch(function(){});
      });
    });
  })();
  </script>"""

    css = _page_css()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_esc(name)} | Staylio</title>
  <meta name="description" content="{_esc(tagline or description[:160])}">
  <link rel="canonical" href="{_esc(page_url)}">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
  {schema_html}
  <style>{css}</style>
</head>
<body>
{hero_html}
{desc_html}
{spotlights_html}
{modules_html}
{gallery_html}
{guest_book_html}
{ota_html}
{amenities_html}
{good_to_know_html}
{calendar_html}
{local_html}
{owner_html}
{faq_html}
{footer_html}
{lightbox_html}
{hero_video_js}
{audio_js}
{growthbook_html}
</body>
</html>"""

    # ── Count sections ───────────────────────────────────────────────────
    sections_built = []
    for section_name in ["hero", "description", "spotlights", "photo-tour", "gallery",
                         "reviews", "guest-reviews", "amenities", "availability",
                         "local-guide", "owner", "faq", "footer-cta"]:
        if f'id="{section_name}"' in html or f'class="{section_name}"' in html:
            sections_built.append(section_name)

    gallery_count = len(re.findall(r'class="gallery-thumb"', html))
    lightbox_present = "window.openLightbox" in html

    logger.info(
        "[orchestrate] Property %s: %d photos, %d sections, gallery=%d, unresolvable=%d",
        property_id[:12], len(media_assets), len(sections_built), gallery_count, unresolvable,
    )

    return {
        "ok": True,
        "html": html,
        "photo_count": len(media_assets),
        "gallery_count": gallery_count,
        "sections_built": sections_built,
        "lightbox_present": lightbox_present,
        "unresolvable": unresolvable,
    }


def raise_hold(sb, property_id: str, reason: str, hold_code: str):
    """Raise a hold: create hitl_queue_items row + alert.

    Reuses the exact pattern from VIDEO-WORKFLOW-1 in workflows/onboard.py.
    Does not fork send_halt_alert — calls it directly.
    """
    # Get property name and account_id for the alert.
    # account_id was hardcoded to None — found by reviewing every
    # hitl_queue_items insert site after escalate_halt's missing
    # created_by_type crashed Vista Azule on 2026-08-20.
    prop_name = property_id[:12]
    account_id = None
    try:
        prop_resp = sb.table("properties").select("name, account_id").eq("id", property_id).limit(1).execute()
        if prop_resp.data:
            prop_name = prop_resp.data[0].get("name") or property_id[:12]
            account_id = prop_resp.data[0].get("account_id")
    except Exception:
        pass

    hitl_id = str(uuid.uuid4())
    sb.table("hitl_queue_items").insert({
        "id": hitl_id,
        "property_id": property_id,
        "account_id": account_id,
        "queue_type": "pipeline_failure",
        "priority": "p0",
        "status": "open",
        "created_by_type": "system",
        "reason_code": hold_code,
        "title": f"HOLD: {prop_name} — cannot publish ({hold_code})",
        "description": reason,
        "payload": {"hold_code": hold_code, "property_id": property_id},
    }).execute()

    from skills.notify import send_halt_alert
    send_halt_alert(
        sb, hitl_id, property_id, prop_name,
        error_class=hold_code,
        detail=reason,
        step_name="orchestrate",
    )

    return hitl_id
