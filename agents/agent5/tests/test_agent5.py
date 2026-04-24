"""
Agent 5 Test Suite
Run with: pytest booked/agents/agent5/tests/ -v

Tests cover:
  - Schema.org JSON-LD output correctness (all required fields)
  - Schema.org field mapping from KB + content package
  - HTML escaping (XSS prevention)
  - Page builder produces valid HTML with all sections
  - Calendar worker naming and URL conventions
  - iCal mode vs PMS API mode detection
  - Cloudflare Pages URL construction (both modes)
  - Slug fallback generation
  - Agent node output contract (success and failure paths)
  - GrowthBook snippet structure
"""

import json
import re
import pytest
from unittest.mock import MagicMock, patch

from agents.agent5.models import (
    CalendarConfig,
    DeployMode,
    LandingPage,
    PageStatus,
)
from agents.agent5.schema_markup import (
    build_schema_from_inputs,
    generate_schema_jsonld,
)
from agents.agent5.calendar_sync import (
    _worker_name,
    _worker_url,
    provision_calendar_sync,
)
from agents.agent5.cloudflare_deployer import (
    _build_page_url,
    _build_deployment_bundle,
)
from agents.agent5.page_builder import (
    _esc,
    _format_description,
    build_landing_page_html,
    _prepare_gallery_items,
    _jaccard,
    _filename_seq_num,
    _caption_word_overlap,
    _combined_similarity,
    _asset_score,
    _suppress_near_dupes,
    MAX_GALLERY_IMAGES,
    MAX_IMAGES_PER_GALLERY_CATEGORY,
    NEAR_DUPE_LABEL_THRESHOLD,
    MAX_PER_DUPE_CLUSTER,
    SIMILARITY_PENALTY_WEIGHT,
    _STRICT_DUPE_CATEGORIES,
    _DEPRIORITIZED_LABELS,
    _GALLERY_CATEGORY_ORDER,
)
from agents.agent5.ab_testing import generate_growthbook_snippet


# ── Fixtures ──────────────────────────────────────────────────────────────

def make_kb(with_coords: bool = True) -> dict:
    kb = {
        "property_id": "prop-test-001",
        "client_id": "client-001",
        "vibe_profile": "romantic_escape",
        "slug": "vista-azule",
        "name": {"value": "Vista Azule", "source": "intake_portal"},
        "description": {"value": "A stunning beachfront property.", "source": "intake_portal"},
        "bedrooms": {"value": 4, "source": "intake_portal"},
        "bathrooms": {"value": 3.5, "source": "intake_portal"},
        "max_occupancy": {"value": 10, "source": "intake_portal"},
        "property_type": {"value": "beach house", "source": "intake_portal"},
        "city": {"value": "Carolina Beach", "source": "intake_portal"},
        "state": {"value": "NC", "source": "intake_portal"},
        "zip_code": {"value": "28428", "source": "intake_portal"},
        "booking_url": "https://pmc.example.com/book/vista-azule",
        "ical_url": "https://airbnb.com/calendar/ical/12345.ics",
        "amenities": [
            {"value": "Heated Pool", "source": "intake_portal"},
            {"value": "Hot Tub", "source": "intake_portal"},
            {"value": "WiFi", "source": "intake_portal"},
        ],
        "airbnb_rating": {"value": 4.87, "source": "airbnb"},
        "airbnb_review_count": {"value": 142, "source": "airbnb"},
        "avg_nightly_rate": {"value": 450.0, "source": "airbnb"},
        "owner_story": "We found this home and fell in love immediately.",
        "guest_reviews": [
            {"text": "Absolutely magical.", "reviewer_name": "Sarah", "is_guest_book": True},
            {"text": "Best vacation ever.", "reviewer_name": "Mike", "is_guest_book": False,
             "star_rating": 5},
        ],
        "photos": [
            {"url": "https://r2.example.com/enhanced/photo_001.jpg", "source": "airbnb"},
            {"url": "https://r2.example.com/enhanced/photo_002.jpg", "source": "airbnb"},
        ],
    }
    if with_coords:
        kb["latitude"] = {"value": 34.0354, "source": "airbnb"}
        kb["longitude"] = {"value": -77.8970, "source": "airbnb"}
    return kb


def make_content_package() -> dict:
    return {
        "hero_headline": "Where the ocean meets the two of you",
        "vibe_tagline": "A private beachfront retreat",
        "property_description": "Vista Azule is the kind of place that makes time slow down.\n\nThe master suite rises directly above the water.",
        "feature_spotlights": [
            {"feature_name": "Rooftop Deck", "headline": "The sky is yours",
             "description": "Private rooftop with outdoor kitchen overlooking the Atlantic."},
        ],
        "amenity_highlights": {
            "Heated Pool": "Stay in long past sunset.",
            "Hot Tub": "For mornings too good to leave.",
        },
        "neighborhood_intro": "Carolina Beach moves at its own pace.",
        "faqs": [
            {"question": "Is the pool heated?", "answer": "Yes, year-round at 82°F."},
            {"question": "Distance to beach?", "answer": "3-minute walk."},
        ],
        "owner_story_refined": "Twelve sunsets in and we never wanted to leave.",
        "seo_page_title": "Vista Azule | Carolina Beach | Romantic Retreat",
        "seo_meta_description": "Romantic 4BR beachfront in Carolina Beach NC. Private pool, rooftop deck. Book direct.",
    }


def make_visual_media() -> dict:
    return {
        "hero_photo_url": "https://r2.example.com/enhanced/hero.jpg",
        "category_winners": {"view": "https://r2.example.com/enhanced/view.jpg"},
        "videos_queued": True,
        "review_videos_pending": False,
    }


def make_local_guide() -> dict:
    return {
        "area_introduction": "Carolina Beach stretches in both directions.",
        "dont_miss_picks": [
            {"name": "Britt's Donuts", "description": "A Carolina Beach institution for 80 years."},
        ],
        "primary_recommendations": [
            {
                "name": "Oceanic Restaurant",
                "category": "eat_and_drink",
                "composite_rating": 4.7,
                "price_level": "$$$",
                "distance_miles": 0.4,
                "photo_url": "https://maps.google.com/photo/ocean.jpg",
                "description": None,
            },
        ],
        "location_name": "Carolina Beach, NC",
    }


# ── Schema.org Tests ──────────────────────────────────────────────────────

class TestSchemaMarkup:
    def test_output_is_valid_script_tag(self):
        script = generate_schema_jsonld(
            name="Vista Azule",
            description="A stunning property",
            page_url="https://vista-azule.staylio.ai",
            booking_url="https://pmc.example.com/book",
            address_line1=None,
            city="Carolina Beach",
            state="NC",
            zip_code="28428",
            latitude=34.035,
            longitude=-77.897,
            bedrooms=4,
            bathrooms=3.5,
            max_occupancy=10,
            amenities=["Pool", "Hot Tub", "WiFi"],
            hero_photo_url="https://r2.example.com/hero.jpg",
            google_rating=4.87,
            google_review_count=142,
            avg_nightly_rate=450.0,
            slug="vista-azule",
        )
        assert script.startswith('<script type="application/ld+json">')
        assert script.endswith("</script>")

    def test_schema_type_is_vacation_rental(self):
        script = generate_schema_jsonld(
            name="Test Property", description=None, page_url="https://test.staylio.ai",
            booking_url=None, address_line1=None, city="Test City", state="NC",
            zip_code=None, latitude=None, longitude=None, bedrooms=None,
            bathrooms=None, max_occupancy=None, amenities=[], hero_photo_url=None,
            google_rating=None, google_review_count=None, avg_nightly_rate=None, slug="test"
        )
        # Extract JSON from script tag
        json_str = script.replace('<script type="application/ld+json">', "").replace("</script>", "").strip()
        data = json.loads(json_str)
        assert data["@type"] == "VacationRental"
        assert data["@context"] == "https://schema.org"

    def test_all_numeric_fields_populate(self):
        script = generate_schema_jsonld(
            name="Test", description="Desc", page_url="https://test.staylio.ai",
            booking_url="https://book.example.com", address_line1="123 Beach Rd",
            city="Wilmington", state="NC", zip_code="28401",
            latitude=34.2, longitude=-77.9, bedrooms=3, bathrooms=2.0,
            max_occupancy=8, amenities=["WiFi"], hero_photo_url="https://r2.example.com/photo.jpg",
            google_rating=4.5, google_review_count=100, avg_nightly_rate=300.0, slug="test"
        )
        json_str = script.split(">", 1)[1].rsplit("<", 1)[0].strip()
        data = json.loads(json_str)
        assert data["numberOfRooms"] == 3
        assert data["numberOfBathroomsTotal"] == 2.0
        assert data["occupancy"]["value"] == 8
        assert data["aggregateRating"]["ratingValue"] == 4.5
        assert data["geo"]["latitude"] == 34.2

    def test_build_schema_from_inputs_uses_kb_fields(self):
        kb = make_kb()
        cp = make_content_package()
        vm = make_visual_media()
        script = build_schema_from_inputs(kb, cp, vm, "https://vista-azule.staylio.ai", "vista-azule")
        assert "Vista Azule" in script
        assert "Carolina Beach" in script

    def test_amenity_feature_list_generated(self):
        script = generate_schema_jsonld(
            name="T", description=None, page_url="https://t.staylio.ai",
            booking_url=None, address_line1=None, city=None, state=None,
            zip_code=None, latitude=None, longitude=None, bedrooms=None,
            bathrooms=None, max_occupancy=None,
            amenities=["Pool", "Hot Tub", "WiFi", "BBQ"],
            hero_photo_url=None, google_rating=None, google_review_count=None,
            avg_nightly_rate=None, slug="t"
        )
        json_str = script.split(">", 1)[1].rsplit("<", 1)[0].strip()
        data = json.loads(json_str)
        assert "amenityFeature" in data
        assert len(data["amenityFeature"]) == 4

    def test_booking_action_uses_booking_url(self):
        script = generate_schema_jsonld(
            name="T", description=None, page_url="https://t.staylio.ai",
            booking_url="https://pmc.example.com/book/vista-azule",
            address_line1=None, city=None, state=None, zip_code=None,
            latitude=None, longitude=None, bedrooms=None, bathrooms=None,
            max_occupancy=None, amenities=[], hero_photo_url=None,
            google_rating=None, google_review_count=None, avg_nightly_rate=None, slug="t"
        )
        assert "https://pmc.example.com/book/vista-azule" in script


# ── HTML Escaping Tests ───────────────────────────────────────────────────

class TestHTMLEscaping:
    """
    XSS prevention tests. Any user-supplied content passing through _esc()
    must be safe for HTML embedding.
    """
    def test_ampersand_escaped(self):
        assert _esc("Fish & Chips") == "Fish &amp; Chips"

    def test_less_than_escaped(self):
        assert _esc("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"

    def test_double_quote_escaped(self):
        assert _esc('say "hello"') == "say &quot;hello&quot;"

    def test_single_quote_escaped(self):
        assert _esc("it's fine") == "it&#39;s fine"

    def test_none_returns_empty_string(self):
        assert _esc(None) == ""

    def test_integer_converted(self):
        assert _esc(42) == "42"


# ── Page Builder Tests ────────────────────────────────────────────────────

class TestPageBuilder:
    def test_page_builds_without_error(self):
        """build_landing_page_html should return non-empty HTML."""
        html = build_landing_page_html(
            kb=make_kb(),
            content_package=make_content_package(),
            visual_media=make_visual_media(),
            local_guide=make_local_guide(),
            page_url="https://vista-azule.staylio.ai",
            slug="vista-azule",
        )
        assert html
        assert len(html) > 5000

    def test_page_is_valid_html_structure(self):
        html = build_landing_page_html(
            kb=make_kb(),
            content_package=make_content_package(),
            visual_media=make_visual_media(),
            local_guide=make_local_guide(),
            page_url="https://vista-azule.staylio.ai",
            slug="vista-azule",
        )
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "<head>" in html
        assert "<body" in html
        assert "</body>" in html

    def test_page_contains_schema_script(self):
        html = build_landing_page_html(
            kb=make_kb(),
            content_package=make_content_package(),
            visual_media=make_visual_media(),
            local_guide=make_local_guide(),
            page_url="https://vista-azule.staylio.ai",
            slug="vista-azule",
        )
        assert 'application/ld+json' in html
        assert 'VacationRental' in html

    def test_page_contains_hero_headline(self):
        html = build_landing_page_html(
            kb=make_kb(),
            content_package=make_content_package(),
            visual_media=make_visual_media(),
            local_guide=make_local_guide(),
            page_url="https://vista-azule.staylio.ai",
            slug="vista-azule",
        )
        assert "Where the ocean meets the two of you" in html

    def test_all_cta_buttons_use_booking_url(self):
        html = build_landing_page_html(
            kb=make_kb(),
            content_package=make_content_package(),
            visual_media=make_visual_media(),
            local_guide=make_local_guide(),
            page_url="https://vista-azule.staylio.ai",
            slug="vista-azule",
        )
        # All CTAs must point to the booking URL
        assert "pmc.example.com/book" in html
        # All must have UTM parameters
        assert "utm_source=booked" in html

    def test_page_contains_utm_parameters(self):
        html = build_landing_page_html(
            kb=make_kb(),
            content_package=make_content_package(),
            visual_media=make_visual_media(),
            local_guide=make_local_guide(),
            page_url="https://vista-azule.staylio.ai",
            slug="vista-azule",
        )
        assert "utm_medium=landing_page" in html
        assert "utm_campaign=vista-azule" in html

    def test_guest_book_section_appears_when_reviews_present(self):
        html = build_landing_page_html(
            kb=make_kb(),
            content_package=make_content_package(),
            visual_media=make_visual_media(),
            local_guide=make_local_guide(),
            page_url="https://vista-azule.staylio.ai",
            slug="vista-azule",
        )
        assert "Guest Book" in html
        assert "Absolutely magical." in html

    def test_page_builds_with_empty_agent_outputs(self):
        """Page should degrade gracefully when agent outputs are empty."""
        html = build_landing_page_html(
            kb=make_kb(),
            content_package={},
            visual_media={},
            local_guide={},
            page_url="https://test.staylio.ai",
            slug="test",
        )
        assert html
        assert "<!DOCTYPE html>" in html

    def test_description_split_into_paragraphs(self):
        result = _format_description("First paragraph.\n\nSecond paragraph.")
        assert "<p>First paragraph.</p>" in result
        assert "<p>Second paragraph.</p>" in result

    def test_fa_section_appears(self):
        html = build_landing_page_html(
            kb=make_kb(),
            content_package=make_content_package(),
            visual_media=make_visual_media(),
            local_guide=make_local_guide(),
            page_url="https://vista-azule.staylio.ai",
            slug="vista-azule",
        )
        assert "Is the pool heated?" in html
        assert "82°F" in html


# ── Calendar Sync Tests ───────────────────────────────────────────────────

class TestCalendarSync:
    def test_worker_name_is_stable(self):
        """Same property_id always produces same worker name."""
        name1 = _worker_name("prop-001")
        name2 = _worker_name("prop-001")
        assert name1 == name2
        assert name1.startswith("staylio-cal-")

    def test_worker_names_differ_per_property(self):
        assert _worker_name("prop-001") != _worker_name("prop-002")

    def test_worker_url_format(self):
        url = _worker_url("prop-001")
        assert url.startswith("https://staylio-cal-")
        assert "workers.dev" in url

    def test_ical_mode_returns_config(self):
        with patch("booked.agents.agent5.calendar_sync._deploy_ical_worker",
                   return_value="https://worker.example.com/calendar"):
            config = provision_calendar_sync(
                property_id="prop-001",
                ical_url="https://airbnb.com/ical/12345.ics",
            )
        assert config.ical_url == "https://airbnb.com/ical/12345.ics"
        assert config.cache_endpoint == "https://worker.example.com/calendar"

    def test_pms_mode_records_type(self):
        config = provision_calendar_sync(
            property_id="prop-001",
            ical_url=None,
            pms_type="guesty",
            pms_api_connected=True,
        )
        assert config.pms_type == "guesty"
        assert config.pms_api_connected is True

    def test_unsupported_pms_type_excluded(self):
        config = provision_calendar_sync(
            property_id="prop-001",
            ical_url=None,
            pms_type="unknown_pms",
        )
        assert config.pms_type is None


# ── Cloudflare Deployer Tests ─────────────────────────────────────────────

class TestCloudflareDeployer:
    def test_subdomain_url_format(self):
        url = _build_page_url("vista-azule", DeployMode.STAYLIO_SUBDOMAIN, None)
        assert url == "https://vista-azule.staylio.ai"

    def test_cname_url_uses_custom_domain(self):
        url = _build_page_url("vista-azule", DeployMode.CNAME_CUSTOM, "stays.pmc.com")
        assert url == "https://stays.pmc.com"

    def test_deployment_bundle_contains_html(self):
        import tarfile, io
        bundle = _build_deployment_bundle("test-slug", "<html><body>Test</body></html>")
        tar = tarfile.open(fileobj=io.BytesIO(bundle))
        names = tar.getnames()
        assert "index.html" in names
        # Verify HTML content
        member = tar.extractfile("index.html")
        content = member.read().decode()
        assert "Test" in content

    def test_simulation_mode_returns_success(self):
        """Without Cloudflare credentials, should simulate and return success."""
        from agents.agent5.cloudflare_deployer import deploy_property_page
        with patch.dict("os.environ", {"CLOUDFLARE_API_KEY": "", "CLOUDFLARE_ACCOUNT_ID": ""}):
            success, url, dep_id = deploy_property_page(
                property_id="p1",
                slug="test-slug",
                html_content="<html></html>",
            )
        assert success is True
        assert "test-slug.staylio.ai" in url


# ── Agent Node Contract Tests ─────────────────────────────────────────────

class TestAgent5NodeContract:
    def test_successful_build_sets_flags(self):
        from agents.agent5.agent import agent5_node

        kb = make_kb()
        state = {
            "property_id": "prop-test-001",
            "knowledge_base": kb,
            "content_package": make_content_package(),
            "visual_media_package": make_visual_media(),
            "local_guide": make_local_guide(),
            "errors": [],
        }

        with patch("booked.agents.agent5.agent.get_cached_knowledge_base", return_value=kb), \
             patch("booked.agents.agent5.agent._load_from_cache_or_state",
                   side_effect=lambda state, pid, key, label: state.get(key, {})), \
             patch("booked.agents.agent5.agent.provision_calendar_sync",
                   return_value=CalendarConfig(cache_endpoint="https://worker.example.com/cal")), \
             patch("booked.agents.agent5.agent.deploy_property_page",
                   return_value=(True, "https://vista-azule.staylio.ai", "dep-001")), \
             patch("booked.agents.agent5.agent._save_landing_page"), \
             patch("booked.agents.agent5.agent._get_client_tier", return_value="base"), \
             patch("booked.agents.agent5.agent.cache_knowledge_base"), \
             patch("booked.agents.agent5.agent.update_pipeline_status"):

            result = agent5_node(state)

        assert result["agent5_complete"] is True
        assert result["agent6_ready"] is True
        assert result["page_url"] == "https://vista-azule.staylio.ai"

    def test_deployment_failure_returns_error(self):
        from agents.agent5.agent import agent5_node

        kb = make_kb()
        state = {
            "property_id": "prop-test-001",
            "knowledge_base": kb,
            "content_package": make_content_package(),
            "visual_media_package": make_visual_media(),
            "local_guide": make_local_guide(),
            "errors": [],
        }

        with patch("booked.agents.agent5.agent.get_cached_knowledge_base", return_value=kb), \
             patch("booked.agents.agent5.agent._load_from_cache_or_state",
                   side_effect=lambda state, pid, key, label: state.get(key, {})), \
             patch("booked.agents.agent5.agent.provision_calendar_sync",
                   return_value=CalendarConfig()), \
             patch("booked.agents.agent5.agent.deploy_property_page",
                   return_value=(False, "https://vista-azule.staylio.ai", None)), \
             patch("booked.agents.agent5.agent._get_client_tier", return_value="base"), \
             patch("booked.agents.agent5.agent.update_pipeline_status"):

            result = agent5_node(state)

        assert result["agent5_complete"] is False
        assert len(result["errors"]) > 0

    def test_missing_kb_returns_error(self):
        from agents.agent5.agent import agent5_node

        state = {
            "property_id": "ghost-prop",
            "knowledge_base": None,
            "errors": [],
        }

        with patch("booked.agents.agent5.agent.get_cached_knowledge_base", return_value=None), \
             patch("booked.agents.agent5.agent.update_pipeline_status"):

            result = agent5_node(state)

        assert result["agent5_complete"] is False


# ── Gallery Preparation Tests ─────────────────────────────────────────────

def _make_asset(category: str, rank: int, labels=None, url_suffix=""):
    """Helper: minimal media_asset dict."""
    return {
        "asset_url_enhanced": f"https://r2.example.com/{category}_{rank}{url_suffix}.jpg",
        "subject_category": category,
        "category_rank": rank,
        "labels_enhanced": labels or [],
    }


class TestGalleryPreparation:

    def test_max_gallery_constant_is_50(self):
        assert MAX_GALLERY_IMAGES == 50

    def test_max_per_category_constant_is_8(self):
        assert MAX_IMAGES_PER_GALLERY_CATEGORY == 8

    def test_never_exceeds_50_items(self):
        # 6 categories x 20 photos each = 120 candidates
        assets = []
        for cat in ["exterior", "view", "pool_hot_tub", "outdoor_entertaining", "living_room", "kitchen"]:
            for rank in range(1, 21):
                assets.append(_make_asset(cat, rank))
        items = _prepare_gallery_items(assets, "", [], "TestProp")
        assert len(items) <= 50

    def test_first_pass_cap_per_category(self):
        # 3 categories x 12 photos = 36 total, well under 50 — all 36 appear
        assets = []
        for cat in ["exterior", "view", "pool_hot_tub"]:
            for rank in range(1, 13):
                assets.append(_make_asset(cat, rank))
        items = _prepare_gallery_items(assets, "", [], "TestProp")
        assert len(items) == 36

    def test_category_balancing_with_overflow(self):
        # 2 categories x 30 = 60 candidates; first pass 8+8=16, second pass fills 34 -> 50
        assets = []
        for rank in range(1, 31):
            assets.append(_make_asset("exterior", rank))
            assets.append(_make_asset("view", rank))
        items = _prepare_gallery_items(assets, "", [], "TestProp")
        assert len(items) == 50
        cat_counts = {}
        for item in items:
            cat_counts[item["category"]] = cat_counts.get(item["category"], 0) + 1
        assert cat_counts.get("exterior", 0) + cat_counts.get("view", 0) == 50

    def test_priority_order_exterior_before_kitchen(self):
        assets = [
            _make_asset("kitchen", 1),
            _make_asset("exterior", 1),
        ]
        items = _prepare_gallery_items(assets, "", [], "TestProp")
        cats = [i["category"] for i in items]
        assert cats.index("exterior") < cats.index("kitchen")

    def test_priority_order_view_before_bathroom(self):
        assets = [
            _make_asset("bathroom", 1),
            _make_asset("view", 1),
            _make_asset("exterior", 1),
        ]
        items = _prepare_gallery_items(assets, "", [], "TestProp")
        cats = [i["category"] for i in items]
        assert cats.index("view") < cats.index("bathroom")

    def test_pool_before_bedroom(self):
        assets = [
            _make_asset("master_bedroom", 1),
            _make_asset("pool_hot_tub", 1),
        ]
        items = _prepare_gallery_items(assets, "", [], "TestProp")
        cats = [i["category"] for i in items]
        assert cats.index("pool_hot_tub") < cats.index("master_bedroom")

    def test_alt_text_includes_property_name_when_labels_present(self):
        assets = [_make_asset("exterior", 1, labels=["facade", "villa", "balcony"])]
        items = _prepare_gallery_items(assets, "", [], "Vista Azule")
        assert items[0]["alt"].startswith("Vista Azule")
        assert "facade" in items[0]["alt"]

    def test_alt_text_format_with_labels(self):
        assets = [_make_asset("view", 1, labels=["ocean", "horizon", "sunset", "waves"])]
        items = _prepare_gallery_items(assets, "", [], "Sea Breeze")
        assert items[0]["alt"] == "Sea Breeze \u2013 ocean, horizon, sunset"

    def test_alt_text_fallback_no_labels(self):
        assets = [_make_asset("kitchen", 1, labels=[])]
        items = _prepare_gallery_items(assets, "", [], "The Cottage")
        assert items[0]["alt"] == "The Cottage photo"

    def test_hero_photo_excluded(self):
        hero = "https://r2.example.com/hero.jpg"
        assets = [
            {"asset_url_enhanced": hero, "subject_category": "exterior", "category_rank": 1, "labels_enhanced": []},
            _make_asset("view", 1),
        ]
        items = _prepare_gallery_items(assets, hero, [], "Prop")
        urls = [i["url"] for i in items]
        assert hero not in urls

    def test_fallback_to_kb_photos_when_no_media_assets(self):
        kb_photos = [
            {"url": "https://r2.example.com/kb1.jpg", "caption": "Front entrance"},
            {"url": "https://r2.example.com/kb2.jpg", "caption": ""},
        ]
        items = _prepare_gallery_items([], "", kb_photos, "Mountain Haus")
        assert len(items) == 2
        assert items[0]["alt"] == "Front entrance"
        assert items[1]["alt"] == "Mountain Haus photo"


# ── Near-Duplicate Suppression Tests ─────────────────────────────────────

def _make_rich_asset(category, rank, labels, comp=0.5, source="airbnb_scraped", enhanced=True, url_suffix=""):
    """Minimal asset dict with all fields _suppress_near_dupes needs."""
    url = f"https://r2.example.com/{category}_{rank}{url_suffix}.jpg"
    return {
        "asset_url_enhanced": url if enhanced else None,
        "asset_url_original": url,
        "subject_category": category,
        "category_rank": rank,
        "labels_enhanced": labels or [],
        "labels_original": labels or [],
        "composition_score": comp,
        "has_enhanced": enhanced,
        "source": source,
    }


class TestJaccard:
    def test_empty_sets_return_zero(self):
        assert _jaccard(frozenset(), frozenset()) == 0.0

    def test_identical_sets_return_one(self):
        assert _jaccard(frozenset(["a", "b"]), frozenset(["a", "b"])) == 1.0

    def test_one_empty_returns_zero(self):
        assert _jaccard(frozenset(["a"]), frozenset()) == 0.0

    def test_partial_overlap(self):
        # {a,b} ∩ {b,c} = {b}, union = {a,b,c} → 1/3
        sim = _jaccard(frozenset(["a", "b"]), frozenset(["b", "c"]))
        assert abs(sim - 1/3) < 0.01

    def test_no_overlap_returns_zero(self):
        assert _jaccard(frozenset(["a"]), frozenset(["b"])) == 0.0


class TestAssetScore:
    def test_rank1_beats_rank2_all_else_equal(self):
        a1 = _make_rich_asset("exterior", 1, ["house"])
        a2 = _make_rich_asset("exterior", 2, ["house"])
        assert _asset_score(a1) > _asset_score(a2)

    def test_intake_upload_beats_airbnb_same_rank(self):
        intake = _make_rich_asset("exterior", 2, ["house"], source="intake_upload")
        airbnb = _make_rich_asset("exterior", 2, ["house"], source="airbnb_scraped")
        assert _asset_score(intake) > _asset_score(airbnb)

    def test_deprioritized_label_reduces_score(self):
        # Same rank and comp — only labels differ
        laundry = _make_rich_asset("uncategorised", 2, ["laundry", "washing machine"], comp=0.5)
        pool = _make_rich_asset("uncategorised", 2, ["pool", "water", "deck"], comp=0.5)
        assert _asset_score(laundry) < _asset_score(pool)

    def test_deprioritized_penalty_is_0_7_factor(self):
        base = _make_rich_asset("uncategorised", 2, ["pool"], comp=0.5)
        penalised = _make_rich_asset("uncategorised", 2, ["hallway", "corridor"], comp=0.5)
        ratio = _asset_score(penalised) / _asset_score(base)
        assert abs(ratio - 0.7) < 0.01

    def test_composition_score_increases_score(self):
        low = _make_rich_asset("exterior", 1, ["house"], comp=0.2)
        high = _make_rich_asset("exterior", 1, ["house"], comp=0.8)
        assert _asset_score(high) > _asset_score(low)


class TestSuppressNearDupes:
    def test_overlapping_labels_cluster_together(self):
        assets = [
            _make_rich_asset("standard_bedroom", 1, ["bed", "pillow", "bedroom", "nightstand"]),
            _make_rich_asset("standard_bedroom", 2, ["bed", "pillow", "bedroom", "lamp"]),
        ]
        # Jaccard("bed","pillow","bedroom","nightstand" vs "bed","pillow","bedroom","lamp") = 3/5 = 0.6
        kept, n_dupes, _ = _suppress_near_dupes(assets)
        # strict category: max 1 per cluster → 1 removed
        assert n_dupes == 1
        assert len(kept) == 1

    def test_best_image_kept_from_cluster(self):
        # rank 1 = best; both in same cluster
        assets = [
            _make_rich_asset("pool_hot_tub", 3, ["pool", "water", "hot tub", "tile"], comp=0.3),
            _make_rich_asset("pool_hot_tub", 1, ["pool", "water", "hot tub", "deck"], comp=0.8),
        ]
        kept, _, _ = _suppress_near_dupes(assets)
        # rank-1/high-comp should survive
        kept_ranks = [a["category_rank"] for a in kept]
        assert 1 in kept_ranks

    def test_pool_duplicates_reduced_to_max_per_cluster(self):
        # 5 near-identical pool shots → only MAX_PER_DUPE_CLUSTER=2 kept
        assets = [
            _make_rich_asset("pool_hot_tub", i+1, ["pool", "water", "hot tub", "tile", str(i)])
            for i in range(5)
        ]
        kept, n_dupes, _ = _suppress_near_dupes(assets)
        assert len(kept) <= MAX_PER_DUPE_CLUSTER

    def test_bedroom_duplicates_reduced_to_one_per_cluster(self):
        # master_bedroom in _STRICT_DUPE_CATEGORIES → max 1 per cluster
        assets = [
            _make_rich_asset("master_bedroom", i+1, ["bed", "pillow", "bedroom", "luxury"])
            for i in range(5)
        ]
        kept, _, _ = _suppress_near_dupes(assets)
        assert len(kept) == 1

    def test_bathroom_strict_cap_one_per_cluster(self):
        assets = [
            _make_rich_asset("bathroom", i+1, ["shower", "tile", "glass", "mirror"])
            for i in range(4)
        ]
        kept, _, _ = _suppress_near_dupes(assets)
        assert len(kept) == 1

    def test_distinct_scenes_not_deduplicated(self):
        assets = [
            _make_rich_asset("bathroom", 1, ["shower", "tile", "glass door", "steam"]),
            _make_rich_asset("bathroom", 2, ["bathtub", "faucet", "soaking tub", "marble"]),
            _make_rich_asset("bathroom", 3, ["sink", "vanity", "mirror", "countertop"]),
        ]
        kept, n_dupes, _ = _suppress_near_dupes(assets)
        assert len(kept) == 3
        assert n_dupes == 0

    def test_no_label_images_not_clustered_together(self):
        assets = [
            _make_rich_asset("exterior", 1, []),
            _make_rich_asset("exterior", 2, []),
            _make_rich_asset("exterior", 3, []),
        ]
        kept, n_dupes, _ = _suppress_near_dupes(assets)
        assert len(kept) == 3
        assert n_dupes == 0

    def test_empty_input_returns_empty(self):
        kept, n, nc = _suppress_near_dupes([])
        assert kept == []
        assert n == 0
        assert nc == 0

    def test_returns_three_tuple(self):
        assets = [_make_rich_asset("exterior", 1, ["house", "facade"])]
        result = _suppress_near_dupes(assets)
        assert len(result) == 3

    def test_n_clusters_counted(self):
        # 3 distinct scenes → 3 clusters
        assets = [
            _make_rich_asset("pool_hot_tub", 1, ["pool", "water", "deck"]),
            _make_rich_asset("pool_hot_tub", 2, ["hot tub", "spa", "jets"]),
            _make_rich_asset("pool_hot_tub", 3, ["fountain", "garden", "stone"]),
        ]
        kept, _, n_clusters = _suppress_near_dupes(assets)
        assert n_clusters == 3
        assert len(kept) == 3

    def test_similarity_penalty_favors_diverse_selection(self):
        # Demonstrate that the similarity penalty causes a near-clone of A to be
        # dropped in favour of a more diverse image, even when the clone has a
        # slightly higher raw score.
        #
        # Setup (all 3 land in the same cluster — verified by hand below):
        #   A: rank=1, labels=["pool","water","deck"]  raw≈0.89
        #   B: rank=2, labels=["pool","water","deck"]  raw≈0.64  ← identical to A
        #   C: rank=3, labels=["pool","water","deck","sunset","sky"]  raw≈0.56  ← diverse
        #
        # combined_sim(B, A) ≈ 0.85  →  adj_B = 0.64 - 0.85*0.3 = 0.385
        # combined_sim(C, A) ≈ 0.51  →  adj_C = 0.56 - 0.51*0.3 = 0.407
        #
        # adj_C > adj_B → C is selected instead of B, proving the penalty works.
        assets = [
            _make_rich_asset("pool_hot_tub", 1, ["pool", "water", "deck"], comp=0.8),
            _make_rich_asset("pool_hot_tub", 2, ["pool", "water", "deck"], comp=0.8),  # clone of A
            _make_rich_asset("pool_hot_tub", 3, ["pool", "water", "deck", "sunset", "sky"], comp=0.8),  # diverse
        ]
        kept, _, _ = _suppress_near_dupes(assets)
        assert len(kept) == 2
        kept_ranks = {a["category_rank"] for a in kept}
        assert 1 in kept_ranks   # best quality always kept
        assert 2 not in kept_ranks  # near-clone of A was penalised and dropped
        assert 3 in kept_ranks   # diverse pick kept over near-copy


class TestNearDupeGallery:
    """End-to-end _prepare_gallery_items tests with near-dupe suppression."""

    def _to_media_asset(self, a):
        return {
            "asset_url_enhanced": a["asset_url_enhanced"],
            "asset_url_original": a["asset_url_original"],
            "subject_category": a["subject_category"],
            "category_rank": a["category_rank"],
            "labels_enhanced": a["labels_enhanced"],
            "labels_original": a["labels_original"],
            "composition_score": a["composition_score"],
            "source": a["source"],
        }

    def test_final_gallery_includes_multiple_categories(self):
        assets = []
        for cat, labels in [
            ("exterior", ["house", "facade", "driveway"]),
            ("view", ["ocean", "horizon", "sunset"]),
            ("pool_hot_tub", ["pool", "water", "deck"]),
            ("living_room", ["sofa", "couch", "furniture"]),
            ("kitchen", ["kitchen", "countertop", "stove"]),
            ("master_bedroom", ["bed", "pillow", "bedroom"]),
            ("bathroom", ["shower", "tile", "glass"]),
        ]:
            for i in range(3):
                assets.append(_make_rich_asset(cat, i+1, labels + [str(i)]))
        items = _prepare_gallery_items(
            [self._to_media_asset(a) for a in assets], "", [], "TestProp"
        )
        cats = {i["category"] for i in items}
        assert len(cats) >= 5

    def test_max_gallery_count_50(self):
        assets = []
        for cat in _GALLERY_CATEGORY_ORDER[:6]:
            for i in range(20):
                assets.append(_make_rich_asset(cat, i+1, ["label_a", "label_b", str(i)]))
        items = _prepare_gallery_items(
            [self._to_media_asset(a) for a in assets], "", [], "BigProp"
        )
        assert len(items) <= MAX_GALLERY_IMAGES

    def test_near_dupe_pool_shots_reduced(self):
        # 10 near-identical pool shots + varied categories — pool should be trimmed
        assets = []
        for i in range(10):
            assets.append(_make_rich_asset("pool_hot_tub", i+1, ["pool", "water", "hot tub", "tile"]))
        for i in range(3):
            assets.append(_make_rich_asset("exterior", i+1, ["house", "facade", str(i)]))
        items = _prepare_gallery_items(
            [self._to_media_asset(a) for a in assets], "", [], "Prop"
        )
        pool_count = sum(1 for i in items if i["category"] == "pool_hot_tub")
        # 10 near-identical pool shots → max MAX_PER_DUPE_CLUSTER=2 from same cluster
        assert pool_count <= MAX_PER_DUPE_CLUSTER

    def test_near_dupe_bedroom_shots_reduced(self):
        # 8 near-identical bedroom shots → max 1 per cluster (strict category)
        assets = []
        for i in range(8):
            assets.append(_make_rich_asset("master_bedroom", i+1, ["bed", "pillow", "bedroom", "luxury"]))
        for i in range(3):
            assets.append(_make_rich_asset("exterior", i+1, ["house", str(i)]))
        items = _prepare_gallery_items(
            [self._to_media_asset(a) for a in assets], "", [], "Prop"
        )
        bed_count = sum(1 for i in items if i["category"] == "master_bedroom")
        assert bed_count <= 1

    def test_constants(self):
        assert MAX_GALLERY_IMAGES == 50
        assert MAX_PER_DUPE_CLUSTER == 2
        assert NEAR_DUPE_LABEL_THRESHOLD == 0.5
        assert "bathroom" in _STRICT_DUPE_CATEGORIES
        assert "master_bedroom" in _STRICT_DUPE_CATEGORIES
        assert "standard_bedroom" in _STRICT_DUPE_CATEGORIES
        assert "laundry" in _DEPRIORITIZED_LABELS

    def test_diverse_pool_shots_retained(self):
        # 5 pool shots, but 3 are visually distinct — all 3 should survive
        assets = []
        # Shot A: pool deck angle (high comp)
        assets.append(_make_rich_asset("pool_hot_tub", 1, ["pool", "water", "deck", "lounge"], comp=0.9))
        # Shot B: near-duplicate of A
        assets.append(_make_rich_asset("pool_hot_tub", 2, ["pool", "water", "deck", "umbrella"], comp=0.85))
        # Shot C: night shot of pool — distinct
        assets.append(_make_rich_asset("pool_hot_tub", 3, ["pool", "night", "lighting", "reflection"], comp=0.8))
        # Shot D: close-up of hot tub — distinct
        assets.append(_make_rich_asset("pool_hot_tub", 4, ["hot tub", "spa", "jets", "bubbly"], comp=0.75))
        items = _prepare_gallery_items(
            [self._to_media_asset(a) for a in assets], "", [], "Prop"
        )
        pool_count = sum(1 for i in items if i["category"] == "pool_hot_tub")
        # A+B cluster → 2 kept (max_per_cluster=2); C and D are distinct → 1 each
        # Total: up to 4, min distinct clusters = 3 images
        assert pool_count >= 3


class TestFilenameSeqNum:
    def test_standard_r2_url(self):
        url = "https://r2.staylio.ai/prop-001/photo_042_abc123.jpg"
        assert _filename_seq_num(url) == 42

    def test_leading_zeros(self):
        url = "https://r2.staylio.ai/prop/photo_007_def.jpg"
        assert _filename_seq_num(url) == 7

    def test_no_photo_prefix(self):
        url = "https://r2.staylio.ai/prop/hero_image.jpg"
        assert _filename_seq_num(url) is None

    def test_empty_url(self):
        assert _filename_seq_num("") is None

    def test_url_without_extension(self):
        url = "https://r2.staylio.ai/prop/photo_010_hash"
        assert _filename_seq_num(url) == 10


class TestCaptionWordOverlap:
    def test_identical_labels_return_one(self):
        labels = ["swimming pool", "deck chairs", "water feature"]
        assert _caption_word_overlap(labels, labels) == 1.0

    def test_empty_labels_return_zero(self):
        assert _caption_word_overlap([], ["pool", "water"]) == 0.0
        assert _caption_word_overlap(["pool"], []) == 0.0

    def test_partial_word_overlap(self):
        # "swimming pool" vs "pool deck" → shared word: "pool"
        a = ["swimming pool"]
        b = ["pool deck"]
        sim = _caption_word_overlap(a, b)
        assert sim > 0.0

    def test_no_overlap(self):
        a = ["bedroom", "pillow", "nightstand"]
        b = ["kitchen", "countertop", "stove"]
        assert _caption_word_overlap(a, b) == 0.0

    def test_short_words_ignored(self):
        # Words ≤2 chars ("a", "of", "to") should not contribute
        a = ["a pool"]
        b = ["a spa"]
        sim = _caption_word_overlap(a, b)
        # Only "pool" and "spa" qualify (>2 chars) — no overlap
        assert sim == 0.0


class TestCombinedSimilarity:
    def test_identical_labels_same_source_adjacent_files_max(self):
        # Identical labels + same source + adjacent filenames → very high similarity
        a = _make_rich_asset("pool_hot_tub", 1, ["pool", "water", "deck"], source="vrbo_scraped")
        b = _make_rich_asset("pool_hot_tub", 2, ["pool", "water", "deck"], source="vrbo_scraped")
        a["url"] = "https://r2.staylio.ai/prop/photo_010_abc.jpg"
        b["url"] = "https://r2.staylio.ai/prop/photo_012_def.jpg"
        ls_a = frozenset(lbl.lower() for lbl in a["labels_enhanced"])
        ls_b = frozenset(lbl.lower() for lbl in b["labels_enhanced"])
        sim = _combined_similarity(ls_a, ls_b, a, b)
        assert sim > 0.9

    def test_different_labels_different_source_far_files_low(self):
        a = _make_rich_asset("exterior", 1, ["house", "facade", "driveway"], source="airbnb_scraped")
        b = _make_rich_asset("exterior", 2, ["ocean", "horizon", "sunset"], source="vrbo_scraped")
        a["url"] = "https://r2.staylio.ai/prop/photo_001_abc.jpg"
        b["url"] = "https://r2.staylio.ai/prop/photo_099_def.jpg"
        ls_a = frozenset(lbl.lower() for lbl in a["labels_enhanced"])
        ls_b = frozenset(lbl.lower() for lbl in b["labels_enhanced"])
        sim = _combined_similarity(ls_a, ls_b, a, b)
        assert sim < 0.3

    def test_filename_proximity_only_applies_same_source(self):
        a = _make_rich_asset("exterior", 1, ["house"], source="vrbo_scraped")
        b = _make_rich_asset("exterior", 2, ["house"], source="airbnb_scraped")
        a["url"] = "https://r2.staylio.ai/prop/photo_010_abc.jpg"
        b["url"] = "https://r2.staylio.ai/prop/photo_011_def.jpg"
        ls_a = frozenset(["house"])
        ls_b = frozenset(["house"])
        sim = _combined_similarity(ls_a, ls_b, a, b)
        # filename proximity not applied (different source) → only label + word similarity
        sim_no_fname = 1.0 * 0.60 + 1.0 * 0.25  # = 0.85 (identical labels/words, no fname bonus)
        assert abs(sim - sim_no_fname) < 0.01

    def test_result_capped_at_one(self):
        a = _make_rich_asset("exterior", 1, ["house", "facade"], source="vrbo_scraped")
        b = _make_rich_asset("exterior", 2, ["house", "facade"], source="vrbo_scraped")
        a["url"] = "https://r2.staylio.ai/prop/photo_001_abc.jpg"
        b["url"] = "https://r2.staylio.ai/prop/photo_002_def.jpg"
        ls_a = frozenset(["house", "facade"])
        ls_b = frozenset(["house", "facade"])
        sim = _combined_similarity(ls_a, ls_b, a, b)
        assert sim <= 1.0


class TestSimilarityPenaltyConstant:
    def test_constant_value(self):
        assert SIMILARITY_PENALTY_WEIGHT == 0.3
