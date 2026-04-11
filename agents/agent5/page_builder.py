"""
Landing Page Builder

Assembles the complete property landing page HTML from the outputs of
Agents 1, 2, 3, and 4. Embeds Schema.org markup, GrowthBook SDK,
and the iCal calendar widget configuration.

Page sections (in order):
  1. <head> — meta, schema JSON-LD, GA4, fonts
  2. Hero — full-bleed photo, headline, tagline, Book Now CTA
  3. Vibe intro — property description (first 2 paragraphs)
  4. Feature spotlights — 3-5 visual cards
  5. Photo gallery — lightbox with all enhanced photos
  6. Guest book — physical reviews with "Guest Book" badge
  7. OTA reviews — Airbnb/VRBO reviews (visually distinct)
  8. Amenities — highlighted amenities with icons
  9. Availability calendar — iCal widget + Book Now
  10. Local area guide — Don't Miss picks + vibe-filtered recommendations
  11. Owner story — owner's personal narrative
  12. FAQs — collapsible accordion
  13. Footer — Book Now CTA, social links, legal

Design principles:
  - Mobile-first responsive CSS
  - All Book Now buttons use booking_url + UTM parameters
  - Hero image loads from R2 with progressive enhancement
  - Video embed for hero video (Video 1 from Agent 3)
  - GrowthBook snippet before </body>
"""

import logging
from typing import Optional

from agents.agent5.ab_testing import generate_growthbook_snippet
from agents.agent5.schema_markup import build_schema_from_inputs

logger = logging.getLogger(__name__)

# UTM parameters applied to every Book Now link
UTM_TEMPLATE = "?utm_source=booked&utm_medium=landing_page&utm_campaign={slug}&utm_content=book_now"


def build_landing_page_html(
    kb: dict,
    content_package: dict,
    visual_media: dict,
    local_guide: dict,
    page_url: str,
    slug: str,
    calendar_cache_endpoint: Optional[str] = None,
    api_base_url: str = "",
) -> str:
    """
    Assemble the complete landing page HTML string.

    Args:
        kb:                      Property knowledge base (Agent 1)
        content_package:         Generated content (Agent 2)
        visual_media:            Photo/video metadata (Agent 3)
        local_guide:             Local recommendations (Agent 4)
        page_url:                Full public URL of this page
        slug:                    URL slug for UTM and IDs
        calendar_cache_endpoint: Cloudflare Worker URL for iCal data

    Returns:
        Complete HTML string ready for Cloudflare Pages deployment
    """
    # ── Extract key values ─────────────────────────────────────────────
    def _val(obj, key):
        f = obj.get(key)
        return f.get("value") if isinstance(f, dict) else f

    name           = _val(kb, "name") or "Vacation Rental"
    booking_url    = kb.get("booking_url") or "#"
    vibe_profile   = kb.get("vibe_profile") or ""
    hero_photo     = visual_media.get("hero_photo_url") or ""
    hero_headline  = content_package.get("hero_headline") or name
    vibe_tagline   = content_package.get("vibe_tagline") or ""
    description    = content_package.get("property_description") or ""
    city           = _val(kb, "city") or ""
    state_abbr     = _val(kb, "state") or ""
    location_str   = f"{city}, {state_abbr}".strip(", ")
    bedrooms       = _val(kb, "bedrooms")
    bathrooms      = _val(kb, "bathrooms")
    max_occupancy  = _val(kb, "max_occupancy")
    owner_story    = content_package.get("owner_story_refined") or kb.get("owner_story") or ""
    seo_title      = content_package.get("seo_page_title") or f"{name} | {location_str}"
    seo_meta_desc  = content_package.get("seo_meta_description") or ""
    book_url_utm   = booking_url + UTM_TEMPLATE.format(slug=slug)

    # Feature spotlights, amenities, FAQs, social captions
    spotlights     = content_package.get("feature_spotlights") or []
    amenity_highlights = content_package.get("amenity_highlights") or {}
    faqs           = content_package.get("faqs") or []

    # Reviews
    all_reviews    = kb.get("guest_reviews") or []
    guest_book_reviews = [r for r in all_reviews if r.get("is_guest_book")]
    ota_reviews        = [r for r in all_reviews if not r.get("is_guest_book")]

    # Photos — prefer enhanced R2 URLs from Agent 3, fall back to KB photos
    media_assets = visual_media.get("media_assets", [])
    if media_assets:
        gallery_photos = [
            a.get("asset_url_enhanced") or a.get("asset_url_original")
            for a in media_assets
            if (a.get("asset_url_enhanced") or a.get("asset_url_original"))
            and (a.get("asset_url_enhanced") or a.get("asset_url_original")) != hero_photo
        ][:24]
    else:
        gallery_photos = [
            p.get("url") for p in (kb.get("photos") or [])
            if p.get("url") and p.get("url") != hero_photo
        ][:24]

    # Hero video (Video 1 from Agent 3, 16:9 format for landing page)
    hero_video_url = _get_hero_video_url(kb.get("property_id", ""))

    # Review audio URLs from Agent 3 (mp3s per guest review)
    review_audio_urls = _get_review_audio_urls(kb.get("property_id", ""))

    # Local guide
    dont_miss_picks       = local_guide.get("dont_miss_picks") or []
    primary_recs          = local_guide.get("primary_recommendations") or []
    area_introduction     = local_guide.get("area_introduction") or ""

    # Schema.org
    schema_script = build_schema_from_inputs(kb, content_package, visual_media, page_url, slug)

    # GrowthBook — pass headline variants for Experiment 1
    headline_variants = _get_headline_variants(content_package)
    growthbook_script = generate_growthbook_snippet(
        property_id=kb.get("property_id", ""),
        slug=slug,
        hero_headline_variants=headline_variants,
    )

    # ── Assemble HTML ──────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{_esc(seo_title)}</title>
  <meta name="description" content="{_esc(seo_meta_desc)}">
  <meta property="og:title" content="{_esc(hero_headline)}">
  <meta property="og:description" content="{_esc(vibe_tagline)}">
  <meta property="og:image" content="{_esc(hero_photo)}">
  <meta property="og:url" content="{_esc(page_url)}">
  <meta property="og:type" content="website">
  <meta name="twitter:card" content="summary_large_image">
  <link rel="canonical" href="{_esc(page_url)}">

  <!-- Schema.org Structured Data (TS-21) -->
  {schema_script}

  <!-- Google Analytics 4 (TS-16) -->
  <script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
  <script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){{dataLayer.push(arguments);}}
    gtag('js', new Date());
    gtag('config', 'GA_MEASUREMENT_ID', {{
      'custom_map': {{'dimension1': 'property_id'}}
    }});
    gtag('set', {{'property_id': '{_esc(kb.get("property_id",""))}'}});
  </script>

  <!-- Staylio Tracking Config -->
  <script>
    window.STAYLIO_CONFIG = {{
      apiBaseUrl: "{_esc(api_base_url)}",
      subdomain: "{_esc(slug)}",
      propertyId: "{_esc(kb.get('property_id', ''))}"
    }};
  </script>

  <!-- Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">

  <style>
    {_page_css()}
  </style>
</head>
<body class="vibe-{vibe_profile.replace('_','-')}">

  <!-- ── HERO ─────────────────────────────────────────────────────── -->
  <section class="hero" id="hero">
    <div class="hero-media">
      {(f"""
    <video id="hero-video" muted playsinline poster="{_esc(hero_photo)}" preload="auto">
      <source src="{_esc(hero_video_url)}" type="video/mp4">
    </video>
    <div id="hero-cta-overlay">
      <button id="hero-cta-btn" aria-label="Play hero story">
        <span class="hero-cta-icon">&#9654;</span>
        <span class="hero-cta-text">Hear the story behind this home</span>
      </button>
    </div>
    <button id="hero-replay-btn" style="display:none" aria-label="Replay hero story">&#8635; Replay</button>
    """) if hero_video_url else f'<img src="{_esc(hero_photo)}" alt="{_esc(name)}" loading="eager">'}
    </div>
    <div class="hero-overlay"></div>
    <div class="hero-content">
      <p class="location-tag">{_esc(location_str)}</p>
      <h1 id="hero-headline" class="hero-headline">{_esc(hero_headline)}</h1>
      <p class="hero-tagline">{_esc(vibe_tagline)}</p>
      <div class="hero-specs">
        {"<span>" + str(bedrooms) + " Beds</span>" if bedrooms else ""}
        {"<span>" + str(bathrooms) + " Baths</span>" if bathrooms else ""}
        {"<span>Sleeps " + str(max_occupancy) + "</span>" if max_occupancy else ""}
      </div>
      <a href="{_esc(book_url_utm)}" class="staylio-cta-btn cta-primary" target="_blank" rel="noopener" data-cta-type="check_availability" data-cta-location="hero">
        Check Availability
      </a>
    </div>
  </section>

  <!-- ── PROPERTY DESCRIPTION ─────────────────────────────────────── -->
  <section class="description" id="about">
    <div class="container">
      <div class="description-text">
        {_format_description(description)}
      </div>
    </div>
  </section>

  <!-- ── FEATURE SPOTLIGHTS ───────────────────────────────────────── -->
  {_build_spotlights_section(spotlights) if spotlights else ""}

  <!-- ── PHOTO GALLERY ────────────────────────────────────────────── -->
  {_build_gallery_section(gallery_photos, name) if gallery_photos else ""}

  <!-- ── AVAILABILITY CALENDAR ────────────────────────────────────── -->
  <section class="availability" id="availability">
    <div class="container">
      <h2>Check Availability</h2>
      <div id="calendar-widget" data-cache-url="{_esc(calendar_cache_endpoint or '')}"></div>
      <div class="calendar-cta">
        <a href="{_esc(book_url_utm)}" class="staylio-cta-btn" target="_blank" rel="noopener" data-cta-type="book_now" data-cta-location="calendar">
          Book Direct — No OTA Fees
        </a>
      </div>
    </div>
  </section>

  <!-- ── GUEST BOOK REVIEWS ───────────────────────────────────────── -->
  {_build_guest_book_section(guest_book_reviews, review_audio_urls) if guest_book_reviews else ""}

  <!-- ── OTA REVIEWS ──────────────────────────────────────────────── -->
  {_build_ota_reviews_section(ota_reviews) if ota_reviews else ""}

  <!-- ── AMENITIES ─────────────────────────────────────────────────── -->
  {_build_amenities_section(amenity_highlights) if amenity_highlights else ""}

  <!-- ── LOCAL AREA GUIDE ─────────────────────────────────────────── -->
  {_build_local_guide_section(area_introduction, dont_miss_picks, primary_recs, location_str) if (area_introduction or primary_recs) else ""}

  <!-- ── OWNER STORY ───────────────────────────────────────────────── -->
  {_build_owner_story_section(owner_story) if owner_story else ""}

  <!-- ── FAQs ─────────────────────────────────────────────────────── -->
  {_build_faq_section(faqs) if faqs else ""}

  <!-- ── FOOTER CTA ────────────────────────────────────────────────── -->
  <section class="footer-cta">
    <div class="container">
      <h2>Ready to Book?</h2>
      <p>Book direct and skip the OTA fees.</p>
      <a href="{_esc(book_url_utm)}" class="staylio-cta-btn cta-large" target="_blank" rel="noopener" data-cta-type="book_now" data-cta-location="footer">
        Book Direct Now
      </a>
    </div>
  </section>

  <footer class="site-footer">
    <div class="container">
      <p class="powered-by">Marketing by <a href="https://staylio.ai">Staylio</a></p>
    </div>
  </footer>

  <script>
    (function() {{
      var cfg = window.STAYLIO_CONFIG || {{}};
      var apiBase = cfg.apiBaseUrl || "";
      var subdomain = cfg.subdomain || "";

      if (!apiBase || !subdomain) return;

      var SESSION_KEY = "staylio_sk_" + subdomain;
      var sessionKey = sessionStorage.getItem(SESSION_KEY);
      if (!sessionKey) {{
        sessionKey = "sk_" + Math.random().toString(36).slice(2) + Date.now();
        sessionStorage.setItem(SESSION_KEY, sessionKey);
      }}

      var visitorKey = localStorage.getItem("staylio_vk");
      if (!visitorKey) {{
        visitorKey = "vk_" + Math.random().toString(36).slice(2) + Date.now();
        localStorage.setItem("staylio_vk", visitorKey);
      }}

      var sessionId = null;
      var params = new URLSearchParams(window.location.search);

      fetch(apiBase + "/public/sessions/start", {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify({{
          subdomain: subdomain,
          session_key: sessionKey,
          visitor_key: visitorKey,
          landing_url: window.location.href,
          referrer_url: document.referrer || null,
          device_type: /Mobi|Android/i.test(navigator.userAgent) ? "mobile" : "desktop",
          utm_source: params.get("utm_source"),
          utm_medium: params.get("utm_medium"),
          utm_campaign: params.get("utm_campaign"),
          utm_term: params.get("utm_term"),
          utm_content: params.get("utm_content")
        }})
      }})
      .then(function(r) {{
        if (!r.ok) return null;
        return r.json().catch(function() {{ return null; }});
      }})
      .then(function(data) {{
        if (!data || !data.visitor_session_id) return;

        sessionId = data.visitor_session_id;

        fetch(apiBase + "/public/events", {{
          method: "POST",
          headers: {{"Content-Type": "application/json"}},
          body: JSON.stringify({{
            visitor_session_id: sessionId,
            event_name: "page_viewed",
            event_payload: {{
              url: window.location.href,
              referrer: document.referrer || null
            }},
            occurred_at: new Date().toISOString()
          }})
        }}).catch(function() {{}});
      }})
      .catch(function() {{}});

      document.addEventListener("click", function(e) {{
        var link = e.target.closest("a[data-cta-type]");
        if (!link || !sessionId) return;

        var ctaType = link.dataset.ctaType;
        var destination = link.href;
        if (!ctaType || !destination) return;

        e.preventDefault();

        var redirected = false;
        function go() {{
          if (redirected) return;
          redirected = true;
          window.location.assign(destination);
        }}

        var timeout = setTimeout(go, 400);

        fetch(apiBase + "/public/cta-clicks", {{
          method: "POST",
          headers: {{"Content-Type": "application/json"}},
          body: JSON.stringify({{
            visitor_session_id: sessionId,
            cta_type: ctaType,
            cta_location: link.dataset.ctaLocation || null,
            destination_url: destination,
            clicked_at: new Date().toISOString()
          }})
        }})
        .catch(function() {{}})
        .finally(function() {{
          clearTimeout(timeout);
          go();
        }});
      }});

    }})();
  </script>

  <!-- Calendar widget JS -->
  <script>
    {_calendar_widget_js()}
  </script>

  <!-- GrowthBook A/B Testing (TS-24) -->
  {growthbook_script}

  <!-- Staylio Audio Player -->
  <script>
  var StaylioAudio = (function() {{
    var current = null;

    function stopCurrent() {{
      if (!current) return;
      if (current.media) {{
        current.media.pause();
        current.media.currentTime = 0;
      }}
      if (current.onStop) current.onStop();
      current = null;
      StaylioAudio.current = null;
    }}

    return {{
      current: null,

      _clearCurrent: function() {{
        current = null;
        StaylioAudio.current = null;
      }},

      play: function(media, onStop) {{
        stopCurrent();
        current = {{ media: media, onStop: onStop || null }};
        StaylioAudio.current = current;
        if (!media) return;
        var playPromise = media.play();
        if (playPromise !== undefined) {{
          playPromise.catch(function() {{}});
        }}
      }},

      stop: function() {{
        stopCurrent();
      }}
    }};
  }})();

  document.addEventListener('DOMContentLoaded', function() {{

    // HERO
    var heroVideo  = document.getElementById('hero-video');
    var heroCta    = document.getElementById('hero-cta-overlay');
    var heroCtaBtn = document.getElementById('hero-cta-btn');
    var heroReplay = document.getElementById('hero-replay-btn');

    function initHeroPreview() {{
      if (!heroVideo) return;
      heroVideo.muted = true;
      heroVideo.loop = true;
      heroVideo.play().catch(function() {{}});
    }}

    function startHero() {{
      if (!heroVideo) return;
      heroVideo.loop = false;
      heroVideo.muted = false;
      heroVideo.currentTime = 0;
      if (heroCta) heroCta.style.display = 'none';
      if (heroReplay) heroReplay.style.display = 'none';
      StaylioAudio.play(heroVideo, function() {{
        heroVideo.pause();
        heroVideo.currentTime = 0;
        heroVideo.muted = true;
        heroVideo.loop = true;
        heroVideo.play().catch(function() {{}});
        if (heroCta) heroCta.style.display = 'flex';
        if (heroReplay) heroReplay.style.display = 'none';
      }});
      heroVideo.onended = function() {{
        StaylioAudio._clearCurrent();
        if (heroReplay) heroReplay.style.display = 'inline-block';
      }};
    }}

    initHeroPreview();
    if (heroCtaBtn) heroCtaBtn.addEventListener('click', startHero);
    if (heroReplay) heroReplay.addEventListener('click', startHero);

    // GUEST REVIEWS
    document.querySelectorAll('.audio-play-btn').forEach(function(btn) {{
      btn.addEventListener('click', function() {{
        var src = btn.getAttribute('data-audio-src');
        if (StaylioAudio.current && StaylioAudio.current.media === btn._audio) {{
          StaylioAudio.stop();
          btn.textContent = '\u25b6 Play';
        }} else {{
          var audio = new Audio(src);
          btn._audio = audio;
          btn.textContent = '\u23f9 Stop';
          StaylioAudio.play(audio, function() {{
            btn.textContent = '\u25b6 Play';
          }});
          audio.onended = function() {{
            btn.textContent = '\u25b6 Play';
            if (StaylioAudio.current && StaylioAudio.current.media === audio) {{
              StaylioAudio._clearCurrent();
            }}
          }};
        }}
      }});
    }});

  }});
  </script>

</body>
</html>"""

    return html


# ── Section builders ──────────────────────────────────────────────────────

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


def _build_gallery_section(photos: list, property_name: str) -> str:
    thumbs = ""
    for i, photo in enumerate(photos[:24]):
        url = photo.get("url", "") if isinstance(photo, dict) else photo
        thumbs += f'<img src="{_esc(url)}" alt="{_esc(property_name)} photo {i+1}" loading="lazy" class="gallery-thumb" onclick="openLightbox({i})">\n'
    return f"""
  <section class="gallery" id="gallery">
    <div class="container">
      <h2>Gallery</h2>
      <div class="gallery-grid">{thumbs}</div>
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
        audio_url = audio_urls.get(_AUDIO_KEYS[i]) if i < len(_AUDIO_KEYS) else None
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
        stars = "★" * int(rating) if rating else ""
        cards += f"""
      <div class="review-card ota-review-card">
        {f'<p class="stars">{stars}</p>' if stars else ""}
        <blockquote>{_esc((r.get("text") or "")[:300])}</blockquote>
        <cite>— {_esc(r.get("reviewer_name") or "Verified Guest")}</cite>
      </div>"""
    return f"""
  <section class="reviews ota-reviews" id="guest-reviews">
    <div class="container">
      <h2>What Guests Are Saying</h2>
      <div class="reviews-grid">{cards}
      </div>
    </div>
  </section>"""


def _build_amenities_section(amenity_highlights: dict) -> str:
    items = ""
    for amenity, copy in list(amenity_highlights.items())[:8]:
        items += f"""
      <div class="amenity-item">
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
        rec_cards += f"""
      <div class="rec-card">
        {"<img src='" + _esc(rec.get("photo_url","")) + "' loading='lazy' alt='" + _esc(rec.get("name","")) + "'>" if rec.get("photo_url") else ""}
        <div class="rec-info">
          <h4>{_esc(rec.get("name", ""))}</h4>
          <p class="rec-meta">{" · ".join(filter(None, [str(rating) + "★" if rating else "", price, dist_str]))}</p>
          {f'<p>{_esc(rec.get("description",""))}</p>' if rec.get("description") else ""}
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


# ── Utilities ─────────────────────────────────────────────────────────────

def _esc(text) -> str:
    """HTML-escape a value for safe embedding."""
    if text is None:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _format_description(description: str) -> str:
    """Split description into paragraphs."""
    if not description:
        return ""
    paragraphs = [p.strip() for p in description.split("\n\n") if p.strip()]
    return "\n".join(f"<p>{_esc(p)}</p>" for p in paragraphs[:5])


def _get_hero_video_url(property_id: str) -> Optional[str]:
    """Query Supabase for the 16:9 vibe_match video URL for this property."""
    if not property_id:
        return None
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("video_assets")
            .select("r2_url")
            .eq("property_id", property_id)
            .eq("video_type", "vibe_match")
            .eq("format", "16_9")
            .not_.is_("r2_url", "null")
            .limit(1)
            .execute()
        )
        rows = result.data or []
        return rows[0]["r2_url"] if rows else None
    except Exception as exc:
        logger.warning(f"[Agent 5] Could not fetch hero video URL for {property_id}: {exc}")
        return None


def _get_review_audio_urls(property_id: str) -> dict:
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("video_assets")
            .select("video_type,r2_url")
            .eq("property_id", property_id)
            .eq("format", "mp3")
            .like("video_type", "audio_guest_review%")
            .not_.is_("r2_url", "null")
            .order("video_type")
            .execute()
        )
        rows = result.data or []
        import time
        bust = int(time.time())
        return {row["video_type"]: f"{row['r2_url']}?v={bust}" for row in rows}
    except Exception as exc:
        logger.warning(f"[TS-12] Could not fetch review audio URLs: {exc}")
        return {}


def _get_headline_variants(content_package: dict) -> list[str]:
    """
    Extract alternative headline variants for GrowthBook Experiment 1.
    In Phase 1 we pass an empty list — variants added when experiments are configured.
    """
    return []


def _calendar_widget_js() -> str:
    """Inline JavaScript for the availability calendar widget."""
    return """
(function() {
  const widget = document.getElementById('calendar-widget');
  if (!widget) return;

  const cacheUrl = widget.dataset.cacheUrl;
  if (!cacheUrl) {
    widget.innerHTML = '<p class="calendar-unavailable">Calendar loading...</p>';
    return;
  }

  fetch(cacheUrl)
    .then(r => r.json())
    .then(data => {
      const blocked = (data.blocked_dates || []).map(b => ({
        start: new Date(b.start),
        end: new Date(b.end),
      }));

      // Render a simple month-view calendar
      widget.innerHTML = buildCalendarHTML(blocked, new Date());
    })
    .catch(() => {
      widget.innerHTML = '<p class="calendar-unavailable">Calendar temporarily unavailable.</p>';
    });

  function buildCalendarHTML(blockedRanges, currentDate) {
    const today = new Date();
    const year = today.getFullYear();
    const month = today.getMonth();
    const daysInMonth = new Date(year, month + 1, 0).getDate();
    const firstDay = new Date(year, month, 1).getDay();

    const monthNames = ['January','February','March','April','May','June',
                        'July','August','September','October','November','December'];

    let html = `<div class="calendar-month">`;
    html += `<h3>${monthNames[month]} ${year}</h3>`;
    html += `<div class="calendar-grid">`;
    html += ['Su','Mo','Tu','We','Th','Fr','Sa']
              .map(d => `<div class="cal-header">${d}</div>`).join('');

    for (let i = 0; i < firstDay; i++) html += '<div class="cal-day empty"></div>';

    for (let day = 1; day <= daysInMonth; day++) {
      const date = new Date(year, month, day);
      const isBlocked = blockedRanges.some(r => date >= r.start && date < r.end);
      const isPast = date < today;
      const cls = isPast ? 'cal-day past' : isBlocked ? 'cal-day blocked' : 'cal-day available';
      html += `<div class="${cls}">${day}</div>`;
    }

    html += '</div></div>';
    html += '<p class="calendar-legend"><span class="legend-available"></span> Available '
          + '<span class="legend-blocked"></span> Booked</p>';
    return html;
  }
})();
"""


def _page_css() -> str:
    """Core CSS for the landing page. Mobile-first, minimal, professional."""
    return """
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --font-serif: 'Cormorant Garamond', Georgia, serif;
      --font-sans: 'Inter', system-ui, sans-serif;
      --color-text: #1a1a1a;
      --color-muted: #666;
      --color-bg: #fff;
      --color-accent: #2c3e50;
      --color-cta: #1a1a1a;
      --max-width: 1200px;
      --spacing: clamp(2rem, 5vw, 4rem);
    }
    body { font-family: var(--font-sans); color: var(--color-text); background: var(--color-bg); }
    .container { max-width: var(--max-width); margin: 0 auto; padding: 0 1.5rem; }
    h1, h2 { font-family: var(--font-serif); font-weight: 400; }
    h2 { font-size: clamp(1.8rem, 3vw, 2.8rem); margin-bottom: 1.5rem; }

    /* Hero */
    .hero { position: relative; height: 100svh; min-height: 600px; display: flex;
            align-items: flex-end; overflow: hidden; }
    .hero-media { position: absolute; inset: 0; }
    .hero-media img, .hero-media video { width: 100%; height: 100%; object-fit: cover; }
    .hero-overlay { position: absolute; inset: 0;
                    background: linear-gradient(to top, rgba(0,0,0,.7) 0%, rgba(0,0,0,.1) 60%); }
    #hero-cta-overlay {
      position: absolute;
      inset: 0;
      z-index: 3;
      display: flex;
      align-items: center;
      justify-content: center;
      pointer-events: none;
    }
    #hero-cta-btn {
      pointer-events: all;
      display: flex;
      align-items: center;
      gap: 12px;
      background: rgba(255,255,255,0.12);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(255,255,255,0.35);
      border-radius: 50px;
      color: #fff;
      padding: 18px 36px;
      font-size: 1rem;
      font-family: inherit;
      cursor: pointer;
      letter-spacing: 0.02em;
      transition: background 0.2s;
    }
    #hero-cta-btn:hover {
      background: rgba(255,255,255,0.22);
    }
    .hero-cta-icon {
      font-size: 1.1rem;
      line-height: 1;
    }
    .hero-cta-text {
      line-height: 1;
    }
    #hero-replay-btn {
      position: absolute;
      bottom: 24px;
      right: 24px;
      z-index: 3;
      background: rgba(0,0,0,0.45);
      border: 1px solid rgba(255,255,255,0.3);
      border-radius: 20px;
      color: #fff;
      padding: 8px 18px;
      font-size: 0.85rem;
      font-family: inherit;
      cursor: pointer;
      letter-spacing: 0.02em;
    }
    .hero-content { position: relative; z-index: 1; padding: 2rem 1.5rem 3rem;
                    max-width: var(--max-width); margin: 0 auto; width: 100%; color: #fff; }
    .location-tag { font-size: .85rem; letter-spacing: .15em; text-transform: uppercase;
                    opacity: .8; margin-bottom: .5rem; }
    .hero-headline { font-size: clamp(2.2rem, 5vw, 4rem); line-height: 1.1;
                     margin-bottom: .75rem; }
    .hero-tagline { font-size: clamp(1rem, 2vw, 1.25rem); opacity: .85; margin-bottom: 1.5rem; }
    .hero-specs { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem;
                  font-size: .9rem; opacity: .9; }
    .hero-specs span::before { content: "·"; margin-right: 1rem; }
    .hero-specs span:first-child::before { content: ""; margin: 0; }

    /* CTAs */
    .staylio-cta-btn {
      display: inline-block; padding: .9rem 2rem; font-size: 1rem; font-weight: 500;
      background: #fff; color: #1a1a1a; border: 2px solid #fff; cursor: pointer;
      text-decoration: none; transition: all .2s ease; letter-spacing: .05em;
    }
    .staylio-cta-btn:hover { background: transparent; color: #fff; }
    .cta-primary { font-size: 1.1rem; padding: 1rem 2.5rem; }
    .cta-large { font-size: 1.2rem; padding: 1.1rem 3rem; }
    section:not(.hero) .staylio-cta-btn {
      background: var(--color-cta); color: #fff; border-color: var(--color-cta);
    }
    section:not(.hero) .staylio-cta-btn:hover {
      background: transparent; color: var(--color-cta);
    }

    /* Sections */
    section { padding: var(--spacing) 0; }
    section:nth-child(even) { background: #f8f8f6; }

    /* Description */
    .description-text { max-width: 720px; }
    .description-text p { font-size: 1.1rem; line-height: 1.8; margin-bottom: 1.25rem;
                          color: #333; }

    /* Spotlights */
    .spotlight-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                      gap: 1.5rem; }
    .spotlight-card { padding: 1.5rem; border: 1px solid #e8e8e4; }
    .spotlight-card h3 { font-family: var(--font-serif); font-size: 1.4rem;
                         margin-bottom: .25rem; }
    .spotlight-feature { font-size: .85rem; color: var(--color-muted); text-transform: uppercase;
                         letter-spacing: .1em; margin-bottom: .75rem; }

    /* Gallery */
    .gallery-grid { display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: .5rem; }
    .gallery-thumb { width: 100%; height: 200px; object-fit: cover; cursor: pointer;
                     transition: opacity .2s; }
    .gallery-thumb:hover { opacity: .85; }

    /* Reviews */
    .reviews-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                    gap: 1.5rem; }
    .review-card { padding: 1.5rem; border-left: 3px solid #e8e8e4; }
    .guest-book-card { border-left-color: var(--color-accent); }
    .badge { display: inline-block; padding: .2rem .6rem; font-size: .75rem;
             background: var(--color-accent); color: #fff; margin-bottom: .75rem;
             letter-spacing: .05em; text-transform: uppercase; }
    .review-card blockquote { font-style: italic; line-height: 1.7;
                              margin-bottom: .75rem; color: #444; }
    .review-card cite { font-size: .85rem; color: var(--color-muted); }

    /* Amenities */
    .amenities-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
                      gap: 1.5rem; }
    .amenity-item h3 { font-family: var(--font-serif); font-size: 1.2rem;
                       margin-bottom: .4rem; }
    .amenity-item p { font-size: .95rem; color: #555; line-height: 1.6; }

    /* Calendar */
    .availability { text-align: center; }
    #calendar-widget { margin: 1.5rem auto; max-width: 640px; }
    .calendar-month h3 { font-family: var(--font-serif); font-size: 1.4rem;
                         margin-bottom: 1rem; }
    .calendar-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 2px; }
    .cal-header { text-align: center; font-size: .8rem; color: var(--color-muted);
                  padding: .4rem 0; font-weight: 500; }
    .cal-day { text-align: center; padding: .6rem .2rem; font-size: .9rem; }
    .cal-day.available { background: #e8f5e9; color: #2e7d32; }
    .cal-day.blocked { background: #fce4ec; color: #c62828; }
    .cal-day.past { color: #ccc; }
    .calendar-legend { font-size: .8rem; color: var(--color-muted); margin-top: .75rem; }
    .legend-available, .legend-blocked { display: inline-block; width: 12px; height: 12px;
                                          margin-right: 4px; vertical-align: middle; }
    .legend-available { background: #e8f5e9; }
    .legend-blocked { background: #fce4ec; }
    .calendar-cta { margin-top: 2rem; }

    /* Local Guide */
    .area-intro { font-size: 1.05rem; line-height: 1.8; max-width: 680px;
                  margin-bottom: 2rem; color: #444; }
    .dont-miss h3 { font-family: var(--font-serif); font-size: 1.4rem;
                    margin-bottom: 1rem; }
    .dont-miss-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
                      gap: 1rem; margin-bottom: 2.5rem; }
    .dont-miss-item { padding: 1.25rem; background: #fff; border: 1px solid #e8e8e4; }
    .dont-miss-item h3 { font-size: 1rem; font-weight: 600; margin-bottom: .4rem; }
    .recommendations-grid { display: grid;
                             grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
                             gap: 1.25rem; }
    .rec-card { border: 1px solid #e8e8e4; overflow: hidden; }
    .rec-card img { width: 100%; height: 160px; object-fit: cover; }
    .rec-info { padding: 1rem; }
    .rec-info h4 { font-size: 1rem; font-weight: 500; margin-bottom: .25rem; }
    .rec-meta { font-size: .8rem; color: var(--color-muted); margin-bottom: .4rem; }

    /* Owner Story */
    .story-text { max-width: 680px; }
    .story-text p { font-size: 1.05rem; line-height: 1.85; color: #444; }

    /* FAQs */
    .faq-list { max-width: 720px; }
    .faq-item { border-bottom: 1px solid #e8e8e4; }
    .faq-item summary { padding: 1.1rem 0; font-size: 1rem; font-weight: 500;
                         cursor: pointer; list-style: none; display: flex;
                         justify-content: space-between; align-items: center; }
    .faq-item summary::after { content: "+"; font-size: 1.2rem; color: var(--color-muted); }
    .faq-item[open] summary::after { content: "−"; }
    .faq-item p { padding: 0 0 1.1rem; color: #555; line-height: 1.7; }

    /* Footer */
    .footer-cta { text-align: center; background: var(--color-accent) !important;
                  color: #fff; }
    .footer-cta h2, .footer-cta p { color: #fff; }
    .footer-cta h2 { color: #fff; margin-bottom: .5rem; }
    .footer-cta p { margin-bottom: 1.5rem; opacity: .85; }
    .footer-cta .staylio-cta-btn { background: #fff !important; color: var(--color-accent) !important;
                                   border-color: #fff !important; }
    .site-footer { padding: 1.5rem 0; background: #111; color: #888; text-align: center;
                   font-size: .85rem; }
    .site-footer a { color: #888; }
    .powered-by { opacity: .6; }

    @media (max-width: 600px) {
      .hero-headline { font-size: 2rem; }
      .gallery-grid { grid-template-columns: repeat(2, 1fr); }
    }
  """
