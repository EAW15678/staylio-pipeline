"""
HTML section builders for the substrate-native page builder.

Created 2026-08-19 (PAGE-2). Logic-preserving move from
agents/agent5/page_builder.py. No database access, no vocabulary
assumptions — sections receive plain data and emit HTML.
"""

from core.page_builder.helpers import _esc
from core.page_builder.vocabulary import MAX_VISIBLE_SUPPORTING


def _build_spotlights_section(spotlights: list) -> str:
    cards = ""
    for s in spotlights[:5]:
        if isinstance(s, dict):
            cards += f"""
        <div class="spotlight-card">
          <h3>{_esc(s.get("headline", ""))}</h3>
          <p class="spotlight-feature">{_esc(s.get("feature_name", ""))}</p>
          <p>{_esc(s.get("description", ""))}</p>
        </div>"""
    return f"""
  <section class="spotlights" id="features">
    <div class="container">
      <h2>What Makes This Place Special</h2>
      <div class="spotlight-grid">{cards}
      </div>
    </div>
  </section>"""


def _build_gallery_section(items: list, property_name: str) -> str:
    """Render the All Photos section as a flat grid. No section headers."""
    if not items:
        return ""

    grid_html = ""
    for photo_index, item in enumerate(items):
        grid_html += (
            f'<img src="{_esc(item["url"])}" alt="{_esc(item["alt"])}" loading="lazy" '
            f'class="gallery-thumb" onclick="openLightbox({photo_index})">\n'
        )

    return f"""
  <section class="gallery" id="gallery">
    <div class="container">
      <h2>All Photos</h2>
      <div class="gallery-grid">{grid_html}</div>
    </div>
  </section>"""


def _build_guest_book_section(reviews: list, audio_urls: dict = None) -> str:
    if audio_urls is None:
        audio_urls = {}
    _AUDIO_KEYS = ["audio_guest_review_1", "audio_guest_review_2", "audio_guest_review_3"]
    cards = ""
    for i, r in enumerate(reviews[:6]):
        if not isinstance(r, dict):
            continue
        name_str = r.get("reviewer_name") or "Guest"
        date_str = r.get("stay_date") or ""
        audio_url = (
            audio_urls.get(name_str.strip())
            or (audio_urls.get(_AUDIO_KEYS[i]) if i < len(_AUDIO_KEYS) else None)
        )
        audio_btn = (
            f'<button class="audio-play-btn" data-audio-src="{_esc(audio_url)}" aria-label="Play review audio">&#9654; Play</button>'
            if audio_url else ""
        )
        cards += f"""
      <div class="review-card guest-book-card">
        <span class="badge">Guest Book</span>
        <blockquote>{_esc(r.get("text", ""))}</blockquote>
        <cite>— {_esc(name_str)}{f", {_esc(date_str)}" if date_str else ""}</cite>
        {audio_btn}
      </div>"""
    return f"""
  <section class="reviews guest-book" id="reviews">
    <div class="container">
      <h2>From the Guest Book</h2>
      <p class="guest-book-helper">Click the audio play button to hear what guests are saying.</p>
      <div class="reviews-grid">{cards}
      </div>
    </div>
  </section>"""


def _build_ota_reviews_section(reviews: list) -> str:
    if not reviews:
        return ""
    cards = ""
    for r in reviews[:8]:
        if not isinstance(r, dict):
            continue
        rating = r.get("star_rating")
        stars = "\u2605" * int(rating) if rating else ""
        stars_html = '<p class="stars">' + stars + '</p>' if stars else ""
        review_text = _esc((r.get("text") or "")[:300])
        reviewer = _esc(r.get("reviewer_name") or "Verified Guest")
        cards += f"""
      <div class="review-card ota-review-card">
        {stars_html}
        <blockquote>{review_text}</blockquote>
        <cite>\u2014 {reviewer}</cite>
      </div>"""
    return f"""
  <section class="reviews ota-reviews" id="guest-reviews">
    <div class="container">
      <h2>What Guests Are Saying</h2>
      <div class="reviews-grid">{cards}
      </div>
    </div>
  </section>"""


def _build_amenities_section(amenity_highlights: dict, amenity_photos: dict = None) -> str:
    """Render the Amenities section with optional photo proof.

    amenity_highlights: {amenity_name: copy_text} from content package
    amenity_photos: {amenity_name: photo_url} — photo evidence for claims.
        Ruled by Erick, 2026-08-19: if a photograph's located_amenities names
        the amenity, show that photo beside the claim. If none exists, the
        claim renders as text only — no invented evidence.
    """
    if amenity_photos is None:
        amenity_photos = {}
    items = ""
    for amenity, copy in list(amenity_highlights.items())[:8]:
        photo_url = amenity_photos.get(amenity)
        photo_html = ""
        if photo_url:
            photo_html = (
                '<img src="' + _esc(photo_url) + '" alt="' + _esc(amenity)
                + '" loading="lazy" class="amenity-photo">'
            )
        items += f"""
      <div class="amenity-item">
        {photo_html}
        <h3>{_esc(amenity)}</h3>
        <p>{_esc(copy)}</p>
      </div>"""
    return f"""
  <section class="amenities" id="amenities">
    <div class="container">
      <h2>Amenities</h2>
      <div class="amenities-grid">{items}
      </div>
    </div>
  </section>"""


def _build_good_to_know_section(arrival_info: str, extra_notes: str = None) -> str:
    """Render the Good to Know section."""
    if not arrival_info:
        return ""

    items_html = f'<p class="story-text">{_esc(arrival_info)}</p>'
    if extra_notes:
        items_html += f'\n      <p class="story-text" style="margin-top: 1rem;">{_esc(extra_notes)}</p>'

    return f"""
  <section class="good-to-know" id="good-to-know">
    <div class="container">
      <h2>Good to Know</h2>
      {items_html}
    </div>
  </section>"""


def _build_local_guide_section(
    area_intro: str,
    dont_miss: list,
    primary_recs: list,
    location_str: str,
) -> str:
    intro_html = f"<p class='area-intro'>{_esc(area_intro)}</p>" if area_intro else ""

    dont_miss_html = ""
    if dont_miss:
        picks = ""
        for p in dont_miss[:5]:
            if not isinstance(p, dict):
                continue
            picks += f"""
        <div class="dont-miss-item">
          <h3>{_esc(p.get("name", ""))}</h3>
          <p>{_esc(p.get("description", ""))}</p>
        </div>"""
        dont_miss_html = f"""
      <div class="dont-miss">
        <h3>Owner's Don't Miss Picks</h3>
        <div class="dont-miss-grid">{picks}
        </div>
      </div>"""

    rec_cards = ""
    for rec in primary_recs[:10]:
        if not isinstance(rec, dict):
            continue
        rating = rec.get("composite_rating") or rec.get("google_rating") or ""
        price  = rec.get("price_level") or ""
        dist   = rec.get("distance_miles")
        dist_str = f"{dist} mi" if dist else ""
        star = "\u2605"
        rating_str = str(rating) + star if rating else ""
        photo_url = rec.get("photo_url", "")
        img_tag = f"<img src='{_esc(photo_url)}' loading='lazy' alt='{_esc(rec.get(chr(110) + chr(97) + chr(109) + chr(101), chr(0)))}'>" if photo_url else ""
        rec_name = _esc(rec.get("name", ""))
        img_tag = ""
        if rec.get("photo_url"):
            img_tag = "<img src='" + _esc(rec.get("photo_url","")) + "' loading='lazy' alt='" + _esc(rec.get("name","")) + "'>"
        meta_parts = [rating_str, price, dist_str]
        meta_str = " \u00b7 ".join(filter(None, meta_parts))
        desc_html = "<p>" + _esc(rec.get("description","")) + "</p>" if rec.get("description") else ""
        rec_cards += f"""
      <div class="rec-card">
        {img_tag}
        <div class="rec-info">
          <h4>{rec_name}</h4>
          <p class="rec-meta">{meta_str}</p>
          {desc_html}
        </div>
      </div>"""

    return f"""
  <section class="local-guide" id="local-guide">
    <div class="container">
      <h2>Explore {_esc(location_str)}</h2>
      {intro_html}
      {dont_miss_html}
      <div class="recommendations-grid">{rec_cards}
      </div>
    </div>
  </section>"""


def _build_owner_story_section(story: str) -> str:
    return f"""
  <section class="owner-story" id="owner">
    <div class="container">
      <h2>About This Home</h2>
      <div class="story-text">
        <p>{_esc(story)}</p>
      </div>
    </div>
  </section>"""


def _build_faq_section(faqs: list) -> str:
    items = ""
    for faq in faqs[:7]:
        if not isinstance(faq, dict):
            continue
        items += f"""
      <details class="faq-item">
        <summary>{_esc(faq.get("question", ""))}</summary>
        <p>{_esc(faq.get("answer", ""))}</p>
      </details>"""
    return f"""
  <section class="faqs" id="faq">
    <div class="container">
      <h2>Frequently Asked Questions</h2>
      <div class="faq-list">{items}
      </div>
    </div>
  </section>"""


def _build_category_modules_section(modules: dict, gallery_items: list) -> str:
    """Render the Photo Tour section with section-name headers.

    In the new vocabulary, the section name IS the display header and the
    gallery filter key — no GCV-to-display translation needed.
    """
    if not modules:
        return ""

    from core.page_builder.helpers import _esc_js

    lightbox_idx = {item["url"]: i for i, item in enumerate(gallery_items)}

    modules_html = ""
    for label, module in modules.items():
        hero = module["hero"]
        all_supporting = module["supporting"]
        all_items = module.get("all") or ([hero] + all_supporting)

        visible_supporting = all_supporting[:MAX_VISIBLE_SUPPORTING]

        h_idx = lightbox_idx.get(hero["url"], 0)
        hero_html = (
            f'<img src="{_esc(hero["url"])}" alt="{_esc(hero["alt"])}" loading="lazy" '
            f'class="cat-module-hero" onclick="openLightbox({h_idx})">'
        )

        sup_html = ""
        for img in visible_supporting:
            i_idx = lightbox_idx.get(img["url"], 0)
            sup_html += (
                f'<img src="{_esc(img["url"])}" alt="{_esc(img["alt"])}" loading="lazy" '
                f'class="cat-module-thumb" onclick="openLightbox({i_idx})">'
            )

        if visible_supporting:
            grid_class = "cat-module-grid"
            grid_inner = hero_html + f'<div class="cat-module-supporting">{sup_html}</div>'
        else:
            grid_class = "cat-module-grid cat-module-grid--solo"
            grid_inner = hero_html

        # "View all N photos" — filter by section name directly
        section_btn = ""
        gallery_count = sum(
            1 for gi in gallery_items
            if (gi.get("section") or gi.get("category")) == label
        )
        if gallery_count > MAX_VISIBLE_SUPPORTING + 1:
            cat_js = _esc_js(label)
            section_btn = (
                f'<div class="cat-module-more">'
                f'<a href="#gallery" class="cat-module-more-btn" '
                f'onclick="if(window.openGalleryFiltered){{event.preventDefault();'
                f"openGalleryFiltered(['{cat_js}']);"
                f'}}">View all {gallery_count} photos</a></div>'
            )

        modules_html += f"""
    <div class="cat-module">
      <h3 class="cat-module-label">{_esc(label)}</h3>
      <div class="{grid_class}">{grid_inner}</div>{section_btn}
    </div>"""

    return f"""
  <section class="cat-modules" id="photo-tour">
    <div class="container">
      <h2>Photo Tour</h2>
      <div class="cat-modules-list">{modules_html}
      </div>
      <div class="view-all-wrap">
        <a href="#gallery" class="staylio-cta-btn" onclick="if(window.openGallery){{event.preventDefault();openGallery();}}">View all photos</a>
      </div>
    </div>
  </section>"""
