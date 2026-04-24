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

# ── Gallery ordering & grouping ───────────────────────────────────────────

MAX_GALLERY_IMAGES = 50          # maximum photos shown in the gallery
MAX_IMAGES_PER_GALLERY_CATEGORY = 8  # per-category cap (first pass balancing)

# Category sort order for the gallery (lower index = appears first).
# Matches the SubjectCategory enum values from agent3/models.py.
_GALLERY_CATEGORY_ORDER: list[str] = [
    "exterior",
    "view",
    "pool_hot_tub",
    "outdoor_entertaining",
    "living_room",
    "kitchen",
    "master_bedroom",
    "standard_bedroom",
    "bathroom",
    "game_entertainment",
    "local_area",
    "uncategorised",
]

# Section headers: label → set of categories that fall under it.
# Ordering here determines header insertion order in the gallery.
_GALLERY_SECTIONS: list[tuple[str, frozenset]] = [
    ("Exterior & Views",  frozenset({"exterior", "view"})),
    ("Outdoor & Pool",    frozenset({"pool_hot_tub", "outdoor_entertaining"})),
    ("Living & Kitchen",  frozenset({"living_room", "kitchen", "game_entertainment"})),
    ("Bedrooms",          frozenset({"master_bedroom", "standard_bedroom"})),
    ("Bathrooms",         frozenset({"bathroom"})),
    # local_area and uncategorised: no header — appended silently at end
]

# ── Category modules (Photo Tour section) ─────────────────────────────────
#
# Ordered list of (display_label, frozenset of subject_categories).
# Each entry becomes one 1-hero + 2-supporting module in the Photo Tour section.
# Ordering determines vertical render order on the page.
_CATEGORY_MODULES: list[tuple[str, frozenset]] = [
    ("Exterior & Views",    frozenset({"exterior", "view"})),
    ("Outdoor & Pool",      frozenset({"pool_hot_tub", "outdoor_entertaining"})),
    ("Living Room",         frozenset({"living_room", "game_entertainment"})),
    ("Kitchen",             frozenset({"kitchen"})),
    ("Bedrooms",            frozenset({"master_bedroom", "standard_bedroom"})),
    ("Bathrooms",           frozenset({"bathroom"})),
    ("Amenities & Extras",  frozenset({"local_area", "uncategorised"})),
]

# Modules whose absence is silently skipped (no log warning).
# All others log a warning when no images are available.
_OPTIONAL_MODULE_LABELS: frozenset = frozenset({"Amenities & Extras"})

# ── Near-duplicate suppression ─────────────────────────────────────────────

# Jaccard similarity of Vision label sets at or above this threshold
# treats two images as near-duplicates (same scene, different angle/crop).
NEAR_DUPE_LABEL_THRESHOLD = 0.5

# Maximum images kept per near-duplicate cluster (default for most categories).
MAX_PER_DUPE_CLUSTER = 2

# These room categories are highly repetitive; use a stricter per-cluster cap.
_STRICT_DUPE_CATEGORIES: frozenset = frozenset({
    "bathroom",
    "master_bedroom",
    "standard_bedroom",
})

# Vision labels that indicate low-value shots (e.g. utility, circulation).
# Images whose enhanced labels overlap with these receive a score penalty.
_DEPRIORITIZED_LABELS: frozenset = frozenset({
    "laundry",
    "washing machine",
    "dryer",
    "garage",
    "garage door",
    "parking lot",
    "hallway",
    "corridor",
    "closet",
    "storage room",
})

# Weight applied to max similarity when computing diversity-adjusted score.
# adjusted_score = raw_score - max_similarity_to_selected * SIMILARITY_PENALTY_WEIGHT
SIMILARITY_PENALTY_WEIGHT = 0.3


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

    # Photos — sorted by category priority + quality rank using Agent 3 metadata
    media_assets = visual_media.get("media_assets", [])
    gallery_items = _prepare_gallery_items(
        media_assets=media_assets,
        hero_photo=hero_photo,
        kb_photos=kb.get("photos") or [],
        property_name=name,
    )

    # Category modules — curated 1-hero + 2-supporting per section (Agent 3 path only)
    category_modules = _build_category_modules(gallery_items) if media_assets else {}

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

  <!-- ── PHOTO TOUR (category modules) ──────────────────────────── -->
  {_build_category_modules_section(category_modules, gallery_items) if category_modules else ""}

  <!-- ── ALL PHOTOS (full gallery) ───────────────────────────────── -->
  {_build_gallery_section(gallery_items, name) if gallery_items else ""}

  <!-- ── AVAILABILITY CALENDAR ────────────────────────────────────── -->
  <section class="availability" id="availability">
    <div class="container">
      <h2>Check Availability</h2>
      <p class="calendar-subtext">Real-time availability based on the booking system</p>
      <div id="calendar-widget" data-cache-url="{_esc(calendar_cache_endpoint or '')}">
        <div class="cal-nav">
          <button id="cal-prev-btn" class="cal-nav-btn" type="button" disabled>&#8592; Prev</button>
          <button id="cal-next-btn" class="cal-nav-btn" type="button">Next &#8594;</button>
        </div>
        <div id="cal-month-grid" class="cal-month-grid"></div>
        <p class="calendar-legend"><span class="legend-available"></span> Available &nbsp;<span class="legend-blocked"></span> Unavailable</p>
      </div>
      <p class="calendar-helper">Unavailable dates cannot be booked.</p>
      <div class="calendar-cta">
        <a href="{_esc(book_url_utm)}" class="staylio-cta-btn staylio-cta-full" target="_blank" rel="noopener" data-cta-type="book_now" data-cta-location="calendar">
          Check exact pricing &amp; book
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


def _jaccard(a, b):
    """
    Jaccard similarity of two frozensets (0.0–1.0).
    Returns 0.0 when either set is empty (incomparable, not identical).
    """
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _filename_seq_num(url):
    """
    Extract the sequential upload number from an R2 photo filename.
    Filename pattern: photo_NNN_<hash>.jpg  → returns NNN as int.
    Returns None when the pattern is not found.
    """
    import os
    import re
    try:
        stem = os.path.splitext(os.path.basename(url))[0]
        m = re.match(r"^photo_(\d+)", stem)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


def _caption_word_overlap(labels_a, labels_b):
    """
    Word-level Jaccard similarity between two label lists.
    Splits each label string into individual words (>2 chars) to catch
    partial matches that label-level Jaccard misses
    (e.g. "swimming pool" vs "pool deck" share the word "pool").

    Returns 0.0 when either tokenised set is empty.
    """
    def _tokenize(labels):
        words = set()
        for label in labels:
            for word in label.lower().split():
                if len(word) > 2:
                    words.add(word)
        return words

    ta = _tokenize(labels_a)
    tb = _tokenize(labels_b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _combined_similarity(label_set_a, label_set_b, asset_a, asset_b):
    """
    Multi-signal similarity score between two assets (0.0–1.0).

    Weights:
      0.60  Vision label Jaccard (frozenset)
      0.25  Word-level caption overlap (catches partial label matches)
      0.15  Filename sequential proximity (same source, ≤3 photos apart)

    Filename proximity is only applied when both images came from the same
    source (same scrape batch → likely same burst of shots on the property).
    """
    label_sim = _jaccard(label_set_a, label_set_b)
    word_sim = _caption_word_overlap(
        asset_a.get("labels_enhanced") or asset_a.get("labels_original") or [],
        asset_b.get("labels_enhanced") or asset_b.get("labels_original") or [],
    )
    fname_sim = 0.0
    src_a = (asset_a.get("source") or "").lower()
    src_b = (asset_b.get("source") or "").lower()
    if src_a and src_a == src_b:
        url_a = asset_a.get("url") or asset_a.get("asset_url_enhanced") or ""
        url_b = asset_b.get("url") or asset_b.get("asset_url_enhanced") or ""
        n_a = _filename_seq_num(url_a)
        n_b = _filename_seq_num(url_b)
        if n_a is not None and n_b is not None and abs(n_a - n_b) <= 3:
            fname_sim = 1.0
    return min(1.0, label_sim * 0.60 + word_sim * 0.25 + fname_sim * 0.15)


def _asset_score(asset):
    """
    Quality score for a single asset dict (higher = better).

    Weights:
      0.50  category_rank (inverted: rank 1 → 1.0, rank 10 → 0.1, unranked → ~0.001)
      0.30  composition_score from Vision API
      0.15  enhanced URL available (post-Claid quality)
      0.05  source priority (intake > PMC > VRBO > Airbnb)
      ×0.7  penalty when enhanced labels overlap with _DEPRIORITIZED_LABELS
    """
    rank = asset.get("category_rank") or 999
    rank_score = 1.0 / rank
    comp = float(asset.get("composition_score") or 0.0)
    has_enhanced = 1.0 if asset.get("has_enhanced") else 0.0
    _SRC = {
        "intake_upload": 0.3,
        "pmc_website": 0.2,
        "vrbo_scraped": 0.1,
        "airbnb_scraped": 0.0,
        "unknown": 0.0,
    }
    src = _SRC.get((asset.get("source") or "unknown").lower(), 0.0)
    score = rank_score * 0.5 + comp * 0.3 + has_enhanced * 0.15 + src * 0.05

    # Penalise low-value subject matter
    labels = set(lbl.lower() for lbl in (asset.get("labels_enhanced") or []))
    if labels & _DEPRIORITIZED_LABELS:
        score *= 0.7

    return score


def _suppress_near_dupes(assets):
    """
    Within a single subject_category, cluster images by multi-signal similarity
    and select diverse representatives from each cluster.

    Two-phase algorithm (O(n²) within category — acceptable for n ≤ ~50):

    Phase 1 — Cluster assignment (greedy):
      Sort by _asset_score descending so the best image founds each cluster.
      For each candidate compute _combined_similarity against every cluster
      founder. If similarity ≥ NEAR_DUPE_LABEL_THRESHOLD, assign to that
      cluster; otherwise start a new one.

    Phase 2 — Diversity-aware selection within each cluster:
      For each cluster, iteratively pick the image with the highest
      adjusted_score = raw_score - max_similarity_to_already_selected * SIMILARITY_PENALTY_WEIGHT
      until the per-cluster cap is reached.

    Per-category cap:
      _STRICT_DUPE_CATEGORIES (bathroom/bedrooms): 1 per cluster
      all others: MAX_PER_DUPE_CLUSTER (default 2)

    Returns (kept_assets, n_dupes_removed, n_clusters).
    Images with no labels are never clustered together
    (empty label sets always return similarity = 0.0).
    """
    if not assets:
        return assets, 0, 0

    category = (assets[0].get("subject_category") or "uncategorised").lower()
    max_per_cluster = 1 if category in _STRICT_DUPE_CATEGORIES else MAX_PER_DUPE_CLUSTER

    # ── Phase 1: cluster assignment ──────────────────────────────────────
    sorted_assets = sorted(assets, key=_asset_score, reverse=True)

    # Each entry: list of asset dicts
    clusters = []            # clusters[cluster_id] = [asset, ...]
    founder_label_sets = []  # frozenset of labels for each cluster founder
    founder_assets = []      # first asset in each cluster (for _combined_similarity)

    for asset in sorted_assets:
        raw_labels = asset.get("labels_enhanced") or asset.get("labels_original") or []
        label_set = frozenset(lbl.lower() for lbl in raw_labels)

        matched_cluster = None
        best_sim = 0.0
        for i, founder_set in enumerate(founder_label_sets):
            sim = _combined_similarity(label_set, founder_set, asset, founder_assets[i])
            if sim >= NEAR_DUPE_LABEL_THRESHOLD and sim > best_sim:
                best_sim = sim
                matched_cluster = i

        if matched_cluster is not None:
            clusters[matched_cluster].append(asset)
        else:
            matched_cluster = len(clusters)
            clusters.append([asset])
            founder_label_sets.append(label_set)
            founder_assets.append(asset)

    # ── Phase 2: diversity-aware selection within each cluster ───────────
    kept = []
    n_dupes = 0

    for cluster in clusters:
        if len(cluster) == 1:
            kept.extend(cluster)
            continue

        # Precompute label sets and raw scores for every asset in the cluster
        label_sets = []
        raw_scores = []
        for a in cluster:
            raw_lbl = a.get("labels_enhanced") or a.get("labels_original") or []
            label_sets.append(frozenset(lbl.lower() for lbl in raw_lbl))
            raw_scores.append(_asset_score(a))

        selected_indices = []
        remaining = list(range(len(cluster)))

        for _ in range(max_per_cluster):
            if not remaining:
                break
            best_idx = None
            best_adj = -1.0
            for ri in remaining:
                if not selected_indices:
                    adj = raw_scores[ri]
                else:
                    # Penalty = max similarity to any already-selected asset
                    max_sim = max(
                        _combined_similarity(
                            label_sets[ri], label_sets[si], cluster[ri], cluster[si]
                        )
                        for si in selected_indices
                    )
                    adj = raw_scores[ri] - max_sim * SIMILARITY_PENALTY_WEIGHT
                if adj > best_adj:
                    best_adj = adj
                    best_idx = ri
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        for i, a in enumerate(cluster):
            if i in selected_indices:
                kept.append(a)
            else:
                n_dupes += 1

    n_clusters = len(clusters)
    return kept, n_dupes, n_clusters


def _prepare_gallery_items(
    media_assets,
    hero_photo,
    kb_photos,
    property_name,
):
    """
    Build a sorted list of gallery item dicts from Agent 3 media_assets.

    Each item: {url, alt, category, rank}

    Pipeline:
      1. Extract all candidate assets (hero excluded).
      2. Near-duplicate suppression within each subject_category
         using Vision label Jaccard similarity (Agent 3 data only, no paid calls).
      3. Sort by (category_priority, category_rank).
      4. Two-pass category balancing:
           pass 1 — up to MAX_IMAGES_PER_GALLERY_CATEGORY per category
           pass 2 — fill remaining slots from overflow
      5. Cap total at MAX_GALLERY_IMAGES.

    Falls back to raw KB photos (no near-dupe suppression) when Agent 3 absent.
    """
    raw_assets = []

    if media_assets:
        for asset in media_assets:
            url = asset.get("asset_url_enhanced") or asset.get("asset_url_original") or ""
            if not url or url == hero_photo:
                continue

            labels = asset.get("labels_enhanced") or []
            if labels:
                alt = f"{property_name} \u2013 " + ", ".join(labels[:3])
            else:
                alt = f"{property_name} photo"

            raw_assets.append({
                # display fields
                "url": url,
                "alt": alt,
                "category": (asset.get("subject_category") or "uncategorised").lower(),
                "rank": asset.get("category_rank") or 999,
                # scoring / clustering fields (stripped before return)
                "labels_enhanced": asset.get("labels_enhanced") or [],
                "labels_original": asset.get("labels_original") or [],
                "composition_score": asset.get("composition_score") or 0.0,
                "has_enhanced": bool(asset.get("asset_url_enhanced")),
                "source": asset.get("source") or "unknown",
                "category_rank": asset.get("category_rank") or 999,
                # keep subject_category as string for _suppress_near_dupes
                "subject_category": (asset.get("subject_category") or "uncategorised").lower(),
            })
    else:
        # Fallback: KB photos have no metadata — no near-dupe suppression
        for p in kb_photos:
            url = p.get("url") or ""
            if not url or url == hero_photo:
                continue
            raw_assets.append({
                "url": url,
                "alt": p.get("caption") or (property_name + " photo"),
                "category": "uncategorised",
                "rank": 999,
                "labels_enhanced": [],
                "labels_original": [],
                "composition_score": 0.0,
                "has_enhanced": False,
                "source": "unknown",
                "category_rank": 999,
                "subject_category": "uncategorised",
            })

    total_candidates = len(raw_assets)
    total_dupes_removed = 0
    total_clusters = 0

    # ── Near-duplicate suppression (Agent 3 path only) ──────────────────
    if media_assets:
        by_cat = {}
        for a in raw_assets:
            cat = a["category"]
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append(a)

        suppressed = []
        cat_cluster_info = {}  # cat -> (n_clusters, n_dupes) for logging
        for cat, cat_assets in by_cat.items():
            kept, n_dupes, n_cl = _suppress_near_dupes(cat_assets)
            suppressed.extend(kept)
            total_dupes_removed += n_dupes
            total_clusters += n_cl
            cat_cluster_info[cat] = (n_cl, n_dupes)
        raw_assets = suppressed

    # ── Sort: (category_priority_index, rank) ───────────────────────────
    def _sort_key(item):
        cat = item["category"]
        try:
            priority = _GALLERY_CATEGORY_ORDER.index(cat)
        except ValueError:
            priority = len(_GALLERY_CATEGORY_ORDER)
        return (priority, item["rank"])

    raw_assets.sort(key=_sort_key)

    # ── Two-pass category balancing ──────────────────────────────────────
    selected = []
    per_cat_count = {}
    remainder = []
    for item in raw_assets:
        cat = item["category"]
        if (per_cat_count.get(cat, 0) < MAX_IMAGES_PER_GALLERY_CATEGORY
                and len(selected) < MAX_GALLERY_IMAGES):
            selected.append(item)
            per_cat_count[cat] = per_cat_count.get(cat, 0) + 1
        else:
            remainder.append(item)

    if len(selected) < MAX_GALLERY_IMAGES:
        slots = MAX_GALLERY_IMAGES - len(selected)
        selected.extend(remainder[:slots])

    # ── Logging ──────────────────────────────────────────────────────────
    cat_counts = {}
    for item in selected:
        cat_counts[item["category"]] = cat_counts.get(item["category"], 0) + 1
    src_label = "Agent 3 media_assets" if media_assets else "KB photos (fallback)"
    sorted_cats = sorted(
        cat_counts.items(),
        key=lambda kv: _sort_key({"category": kv[0], "rank": 0}),
    )
    cat_summary = ", ".join((c + "=" + str(n)) for c, n in sorted_cats)
    logger.info(
        "[Agent 5] Gallery selection (%s): candidates=%d, "
        "near-dupes suppressed=%d, clusters=%d, after-dedup=%d, final=%d. "
        "Categories: %s",
        src_label,
        total_candidates,
        total_dupes_removed,
        total_clusters,
        total_candidates - total_dupes_removed,
        len(selected),
        cat_summary,
    )
    if media_assets and cat_cluster_info:
        cluster_detail = ", ".join(
            "{cat}({cl}cl/{dp}dp)".format(cat=cat, cl=cl, dp=dp)
            for cat, (cl, dp) in sorted(
                cat_cluster_info.items(),
                key=lambda kv: _sort_key({"category": kv[0], "rank": 0}),
            )
            if dp > 0 or cl > 1
        )
        if cluster_detail:
            logger.info("[Agent 5] Cluster detail: %s", cluster_detail)

    # Strip scoring/clustering fields — return only display fields
    return [
        {"url": i["url"], "alt": i["alt"], "category": i["category"], "rank": i["rank"]}
        for i in selected
    ]


def _build_category_modules(gallery_items: list) -> dict:
    """
    Build curated category modules from the flat gallery items list.

    Groups the already-prepared gallery items (from _prepare_gallery_items) by
    _CATEGORY_MODULES section definitions and selects up to 3 featured images
    per module: 1 hero + 2 supporting.

    Selection:
      - hero      = lowest category_rank in the section (best quality)
      - supporting = next 2 by rank; URLs guaranteed distinct from hero
      - all        = all section images (for full gallery cross-referencing)

    Returns an ordered dict: {label: {hero, supporting, all}}.
    Modules with no images are omitted; missing required modules are logged.
    """
    by_category: dict = {}
    for item in gallery_items:
        by_category.setdefault(item["category"], []).append(item)

    result: dict = {}
    missing_required: list = []
    missing_optional: list = []

    for label, cats in _CATEGORY_MODULES:
        section_items: list = []
        for cat in cats:
            section_items.extend(by_category.get(cat, []))

        # Sort ascending by rank (rank 1 = best); break ties by URL for stability
        section_items.sort(key=lambda x: (x["rank"], x["url"]))

        if not section_items:
            if label in _OPTIONAL_MODULE_LABELS:
                missing_optional.append(label)
            else:
                missing_required.append(label)
            continue

        hero = section_items[0]
        supporting = [x for x in section_items[1:] if x["url"] != hero["url"]][:2]

        result[label] = {
            "hero": hero,
            "supporting": supporting,
            "all": section_items,
        }

    for lbl in missing_required:
        logger.info("[Agent 5] Photo tour module '%s' skipped — no images available", lbl)

    featured_summary = ", ".join(
        "{lbl}={n}img".format(lbl=lbl, n=1 + len(mod["supporting"]))
        for lbl, mod in result.items()
    )
    skipped = missing_required + missing_optional
    logger.info(
        "[Agent 5] Photo tour: %d modules built (%s); skipped: %s",
        len(result),
        featured_summary or "none",
        ", ".join(skipped) if skipped else "none",
    )

    return result


def _build_category_modules_section(modules: dict, gallery_items: list) -> str:
    """
    Render the Photo Tour section: one hero + up to two supporting images per
    category module. Clicking any image opens the full gallery lightbox at the
    correct position. A 'View all photos' anchor scrolls to the full gallery.

    Layout per module (desktop):
      ┌────────────────────────────┬──────────────┐
      │  hero (2fr, 420px tall)    │ supporting 1 │
      │                            │──────────────│
      │                            │ supporting 2 │
      └────────────────────────────┴──────────────┘

    Layout falls back to stacked single-column on mobile.
    """
    if not modules:
        return ""

    # Build url → lightbox-index lookup (gallery_items order = lightbox order)
    lightbox_idx: dict = {item["url"]: i for i, item in enumerate(gallery_items)}

    modules_html = ""
    for label, module in modules.items():
        hero = module["hero"]
        supporting = module["supporting"]

        h_idx = lightbox_idx.get(hero["url"], 0)
        hero_html = (
            f'<img src="{_esc(hero["url"])}" alt="{_esc(hero["alt"])}" loading="lazy" '
            f'class="cat-module-hero" onclick="openLightbox({h_idx})">'
        )

        sup_html = ""
        for img in supporting:
            i_idx = lightbox_idx.get(img["url"], 0)
            sup_html += (
                f'<img src="{_esc(img["url"])}" alt="{_esc(img["alt"])}" loading="lazy" '
                f'class="cat-module-thumb" onclick="openLightbox({i_idx})">'
            )

        # If no supporting images, hero spans full width (no 2-column grid)
        if supporting:
            grid_class = "cat-module-grid"
            grid_inner = hero_html + f'<div class="cat-module-supporting">{sup_html}</div>'
        else:
            grid_class = "cat-module-grid cat-module-grid--solo"
            grid_inner = hero_html

        modules_html += f"""
    <div class="cat-module">
      <h3 class="cat-module-label">{_esc(label)}</h3>
      <div class="{grid_class}">{grid_inner}</div>
    </div>"""

    return f"""
  <section class="cat-modules" id="photo-tour">
    <div class="container">
      <h2>Photo Tour</h2>
      <div class="cat-modules-list">{modules_html}
      </div>
      <div class="view-all-wrap">
        <a href="#gallery" class="staylio-cta-btn">View all photos</a>
      </div>
    </div>
  </section>"""


def _build_gallery_section(items: list, property_name: str) -> str:
    """
    Render the full gallery section with category-ordered photos and section headers.

    Section headers span the full grid width (grid-column: 1 / -1).
    Photo count and CSS structure are unchanged from the original layout.
    Lightbox indices (0-based) track photo position, not including headers.
    """
    if not items:
        return ""

    # Build set of which categories are present for header decisions
    present_categories = {item["category"] for item in items}

    # Map each item to its section label (for header insertion)
    def _section_for(cat: str) -> Optional[str]:
        for label, cats in _GALLERY_SECTIONS:
            if cat in cats:
                return label
        return None  # no header for local_area / uncategorised

    grid_html = ""
    last_section: Optional[str] = None
    photo_index = 0  # lightbox index (headers don't count)

    for item in items:
        url = item["url"]
        alt = item["alt"]
        section = _section_for(item["category"])

        # Insert section header when crossing into a new labelled section
        if section and section != last_section:
            # Only show header if this section has at least 1 photo (always true here)
            grid_html += (
                f'<p class="gallery-section-label" '
                f'style="grid-column: 1 / -1">{_esc(section)}</p>\n'
            )
            last_section = section

        grid_html += (
            f'<img src="{_esc(url)}" alt="{_esc(alt)}" loading="lazy" '
            f'class="gallery-thumb" onclick="openLightbox({photo_index})">\n'
        )
        photo_index += 1

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

  const MONTH_NAMES = ['January','February','March','April','May','June',
                       'July','August','September','October','November','December'];
  const MAX_ADVANCE = 12; // months forward from today

  const today       = new Date();
  const originYear  = today.getFullYear();
  const originMonth = today.getMonth(); // 0-indexed, never goes backward

  // Today as a comparable "YYYY-MM-DD" string (local date, no UTC drift)
  const todayYMD = originYear + '-'
    + String(originMonth + 1).padStart(2, '0') + '-'
    + String(today.getDate()).padStart(2, '0');

  // offset = how many months past today's month the LEFT panel shows (0 = current)
  let offset = 0;

  // Stored as raw "YYYY-MM-DD" strings — avoids UTC vs local-midnight drift
  let blockedRanges = [];

  function dayYMD(year, month, day) {
    return year + '-'
      + String(month + 1).padStart(2, '0') + '-'
      + String(day).padStart(2, '0');
  }

  // String comparison is safe for ISO dates: "2026-05-02" >= "2026-05-02" etc.
  function isBlocked(ymd) {
    return blockedRanges.some(r => ymd >= r.start && ymd < r.end);
  }

  function buildMonthHTML(year, month) {
    const daysInMonth = new Date(year, month + 1, 0).getDate();
    const firstDay    = new Date(year, month, 1).getDay();
    let html = `<div class="calendar-month">`;
    html += `<h3>${MONTH_NAMES[month]} ${year}</h3>`;
    html += `<div class="calendar-grid">`;
    html += ['Su','Mo','Tu','We','Th','Fr','Sa']
              .map(d => `<div class="cal-header">${d}</div>`).join('');
    for (let i = 0; i < firstDay; i++) html += '<div class="cal-day empty"></div>';
    for (let day = 1; day <= daysInMonth; day++) {
      const ymd = dayYMD(year, month, day);
      const past = ymd < todayYMD;
      const cls  = past ? 'cal-day past'
                        : isBlocked(ymd) ? 'cal-day blocked'
                                         : 'cal-day available';
      html += `<div class="${cls}">${day}</div>`;
    }
    html += '</div></div>';
    return html;
  }

  function render() {
    const leftDate  = new Date(originYear, originMonth + offset, 1);
    const rightDate = new Date(originYear, originMonth + offset + 1, 1);

    const atStart = offset === 0;
    const atLimit = (offset + 1) >= MAX_ADVANCE;

    const prevBtn = document.getElementById('cal-prev-btn');
    if (prevBtn) {
      prevBtn.disabled       = atStart;
      prevBtn.style.opacity  = atStart ? '0.35' : '1';
      prevBtn.style.cursor   = atStart ? 'not-allowed' : 'pointer';
    }
    const nextBtn = document.getElementById('cal-next-btn');
    if (nextBtn) {
      nextBtn.disabled       = atLimit;
      nextBtn.style.opacity  = atLimit ? '0.35' : '1';
      nextBtn.style.cursor   = atLimit ? 'not-allowed' : 'pointer';
    }

    const grid = document.getElementById('cal-month-grid');
    if (grid) {
      grid.innerHTML = buildMonthHTML(leftDate.getFullYear(),  leftDate.getMonth())
                     + buildMonthHTML(rightDate.getFullYear(), rightDate.getMonth());
    }
  }

  fetch(cacheUrl)
    .then(r => r.json())
    .then(data => {
      // Keep as strings — no Date construction, no UTC/local-midnight drift
      blockedRanges = (data.blocked_dates || []).map(b => ({
        start: b.start,
        end:   b.end,
      }));

      // Wire navigation ONCE after data loads
      const prevBtn = document.getElementById('cal-prev-btn');
      if (prevBtn) {
        prevBtn.addEventListener('click', function() {
          if (offset > 0) { offset -= 1; render(); }
        });
      }
      const nextBtn = document.getElementById('cal-next-btn');
      if (nextBtn) {
        nextBtn.addEventListener('click', function() {
          if (offset + 1 < MAX_ADVANCE) { offset += 1; render(); }
        });
      }

      render();
    })
    .catch(() => {
      widget.innerHTML = '<p class="calendar-unavailable">Calendar temporarily unavailable.</p>';
    });
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

    /* Photo Tour — category modules */
    .cat-modules-list { display: flex; flex-direction: column; gap: 3rem; }
    .cat-module-label { font-family: var(--font-serif); font-size: 1.5rem; font-weight: 400;
                        margin-bottom: .75rem; }
    .cat-module-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 4px; }
    .cat-module-grid--solo { grid-template-columns: 1fr; }
    .cat-module-hero { width: 100%; height: 420px; object-fit: cover; cursor: pointer;
                       display: block; transition: opacity .2s; }
    .cat-module-supporting { display: flex; flex-direction: column; gap: 4px; height: 420px; }
    .cat-module-thumb { width: 100%; flex: 1; min-height: 0; object-fit: cover;
                        cursor: pointer; display: block; transition: opacity .2s; }
    .cat-module-hero:hover, .cat-module-thumb:hover { opacity: .85; }
    .view-all-wrap { margin-top: 2.5rem; text-align: center; }

    /* Gallery (full / secondary) */
    .gallery-grid { display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: .5rem; }
    .gallery-thumb { width: 100%; height: 200px; object-fit: cover; cursor: pointer;
                     transition: opacity .2s; }
    .gallery-thumb:hover { opacity: .85; }
    .gallery-section-label { margin: 1rem 0 .25rem; font-size: .75rem; font-weight: 500;
                              text-transform: uppercase; letter-spacing: .1em;
                              color: var(--color-muted); }

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
    .calendar-subtext { font-size: .95rem; color: var(--color-muted); margin-bottom: 1rem; }
    .calendar-helper { font-size: .85rem; color: var(--color-muted); margin-top: .75rem; }
    #calendar-widget { margin: 1.5rem auto; max-width: 100%; }
    /* Navigation */
    .cal-nav { display: flex; justify-content: space-between; margin-bottom: .75rem; }
    .cal-nav-btn { background: transparent; border: 1px solid #ccc; border-radius: 4px;
                   padding: .35rem .85rem; font-size: .9rem; cursor: pointer;
                   color: var(--color-text, #222); }
    .cal-nav-btn:hover:not(:disabled) { border-color: #888; }
    /* Two-month grid */
    .cal-month-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;
                      text-align: left; }
    @media (max-width: 640px) { .cal-month-grid { grid-template-columns: 1fr; gap: 1.5rem; } }
    .calendar-month h3 { font-family: var(--font-serif); font-size: 1.2rem;
                         margin-bottom: .75rem; text-align: center; }
    .calendar-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 2px; }
    .cal-header { text-align: center; font-size: .75rem; color: var(--color-muted);
                  padding: .35rem 0; font-weight: 500; }
    .cal-day { text-align: center; padding: .5rem .15rem; font-size: .85rem; border-radius: 3px; }
    .cal-day.available { background: #fff; color: #222;
                         border: 1px solid #e0e0e0; cursor: default; }
    .cal-day.available:hover { border-color: #999; }
    .cal-day.blocked { background: #D9534F; color: #fff;
                       border: 1px solid #D9534F; cursor: not-allowed; pointer-events: none; }
    .cal-day.past { background: transparent; color: #ccc;
                    border: 1px solid transparent; }
    .calendar-legend { font-size: .8rem; color: var(--color-muted); margin-top: .75rem;
                       text-align: center; }
    .legend-available, .legend-blocked { display: inline-block; width: 12px; height: 12px;
                                          margin-right: 4px; vertical-align: middle;
                                          border-radius: 2px; }
    .legend-available { background: #fff; border: 1px solid #e0e0e0; }
    .legend-blocked { background: #D9534F; }
    .calendar-cta { margin-top: 2rem; }
    .staylio-cta-full { display: block; width: 100%; text-align: center; box-sizing: border-box; }

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
      .cat-module-grid, .cat-module-grid--solo { grid-template-columns: 1fr; }
      .cat-module-hero { height: 260px; }
      .cat-module-supporting { flex-direction: row; height: auto; }
      .cat-module-thumb { height: 160px; flex: 1; }
    }
  """
