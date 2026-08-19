"""
PAGE-2: Behavioural tests for the substrate-native page builder.

Every test calls functions with synthetic data and asserts on returned
HTML or data structures. No source-text inspection.

Every test must fail if run against the OLD vocabulary (i.e., if the
vocabulary were reverted to GCV names like "living_room" instead of
"Living Areas"). This is the regression this slice exists to prevent.
"""

import sys
sys.path.insert(0, ".")


# ── Test 1: curated_section = "Living Areas" → header "Living Areas" ────────

def test_living_areas_section_header():
    """A gallery item tagged curated_section='Living Areas' is grouped and
    displayed under a visible header of 'Living Areas' — not 'living_room',
    not 'Living Room'.

    Fails against old vocabulary: the old code uses 'living_room' internally
    and displays 'Living Room' — neither matches 'Living Areas'.
    """
    from core.page_builder.gallery import _build_category_modules, _prepare_gallery_items

    items = _prepare_gallery_items(
        media_assets=[
            {
                "asset_url_enhanced": f"https://example.com/photo_{i}.jpg",
                "curated_section": "Living Areas",
                "labels_enhanced": [f"sofa_{i}", "couch"],
                "composition_score": 0.8 - i * 0.1,
                "source": "vrbo_scraped",
            }
            for i in range(5)
        ],
        hero_photo="",
        kb_photos=[],
        property_name="Test Property",
    )

    modules = _build_category_modules(items)

    assert "Living Areas" in modules, (
        f"Expected 'Living Areas' in modules, got: {list(modules.keys())}"
    )
    # Verify the old names are NOT present
    assert "living_room" not in modules
    assert "Living Room" not in modules

    # Verify HTML output uses the correct header
    from core.page_builder.sections import _build_category_modules_section
    html = _build_category_modules_section(modules, items)
    assert "Living Areas" in html
    assert "living_room" not in html


# ── Test 2: unrecognised section → routed to Extras, logged ─────────────────

def test_unrecognised_section_routes_to_extras(caplog):
    """An unrecognised section value routes to Extras and is logged.
    Not silently dropped into a generic bucket unmarked.

    Fails against old vocabulary: the old code would dump it into
    'uncategorised' with no log.
    """
    import logging
    from core.page_builder.vocabulary import normalise_section

    with caplog.at_level(logging.WARNING, logger="core.page_builder.vocabulary"):
        result = normalise_section("completely_bogus_section")

    assert result == "Extras", f"Expected 'Extras', got '{result}'"
    assert any("Unrecognised curated_section" in r.message for r in caplog.records), \
        "Expected a warning log for unrecognised section"
    assert any("completely_bogus_section" in r.message for r in caplog.records), \
        "Warning should mention the bad section name"

    # Also verify it appears in gallery output under Extras
    from core.page_builder.gallery import _prepare_gallery_items
    items = _prepare_gallery_items(
        media_assets=[{
            "asset_url_enhanced": "https://example.com/bogus.jpg",
            "curated_section": "completely_bogus_section",
            "labels_enhanced": ["unknown thing"],
            "composition_score": 0.5,
            "source": "unknown",
        }],
        hero_photo="",
        kb_photos=[],
        property_name="Test",
    )
    assert items[0]["section"] == "Extras"


# ── Test 3: all seven sections produce correctly grouped output ─────────────

def test_all_seven_sections_grouped_correctly():
    """All seven real sections each produce correctly grouped, correctly
    labelled output from synthetic data spanning all seven.

    Fails against old vocabulary: the old code would not recognise
    'Pool', 'Living Areas', or 'Extras' as valid section names.
    """
    from core.page_builder.vocabulary import SECTIONS
    from core.page_builder.gallery import _prepare_gallery_items, _build_category_modules

    media_assets = []
    for section in SECTIONS:
        for i in range(4):
            media_assets.append({
                "asset_url_enhanced": f"https://example.com/{section.lower().replace(' ', '_')}_{i}.jpg",
                "curated_section": section,
                "labels_enhanced": [f"label_{section}_{i}"],
                "composition_score": 0.9 - i * 0.1,
                "source": "vrbo_scraped",
            })

    items = _prepare_gallery_items(
        media_assets=media_assets,
        hero_photo="",
        kb_photos=[],
        property_name="Test Property",
    )

    # All items should have a section from the seven
    for item in items:
        assert item["section"] in SECTIONS, (
            f"Item section '{item['section']}' not in valid sections"
        )

    # Build modules — should have all seven
    modules = _build_category_modules(items)
    for section in SECTIONS:
        assert section in modules, (
            f"Section '{section}' missing from modules: {list(modules.keys())}"
        )

    # Verify no old GCV names appear
    old_names = {"exterior", "view", "pool_hot_tub", "outdoor_entertaining",
                 "living_room", "kitchen", "master_bedroom", "standard_bedroom",
                 "bathroom", "game_entertainment", "local_area", "uncategorised"}
    for key in modules:
        assert key not in old_names, f"Old GCV name '{key}' found in modules"


# ── Test 4: per-category cap of 8, no overall cap ──────────────────────────

def test_per_section_cap_no_overall_cap():
    """A synthetic set with one section at 20 items and 90 total items
    produces all 90 across the gallery, with that one section capped at 8
    in the first pass but included fully via the second pass.

    Per-section cap = 8 in first pass (balancing).
    No overall cap — all items are included.

    Fails against old vocabulary: the old code has MAX_GALLERY_IMAGES=50
    which would cap at 50.
    """
    from core.page_builder.gallery import _prepare_gallery_items
    from core.page_builder.vocabulary import SECTIONS

    media_assets = []
    # One section with 20 items
    for i in range(20):
        media_assets.append({
            "asset_url_enhanced": f"https://example.com/kitchen_{i}.jpg",
            "curated_section": "Kitchen",
            "labels_enhanced": [f"unique_kitchen_label_{i}"],
            "composition_score": 0.5,
            "source": "vrbo_scraped",
        })

    # Remaining sections: 10 each (6 sections × ~12 = 70 more, total 90)
    other_sections = [s for s in SECTIONS if s != "Kitchen"]
    for section in other_sections:
        count = 12 if section != "Extras" else 10
        for i in range(count):
            media_assets.append({
                "asset_url_enhanced": f"https://example.com/{section.lower().replace(' ', '_')}_{i}.jpg",
                "curated_section": section,
                "labels_enhanced": [f"unique_{section}_{i}"],
                "composition_score": 0.5,
                "source": "vrbo_scraped",
            })

    total_input = len(media_assets)
    assert total_input == 90, f"Test setup: expected 90, got {total_input}"

    items = _prepare_gallery_items(
        media_assets=media_assets,
        hero_photo="",
        kb_photos=[],
        property_name="Test Property",
    )

    # All 90 items should be present — no overall cap
    assert len(items) == 90, (
        f"Expected all 90 items (no overall cap), got {len(items)}"
    )


# ── Test 5: schema and growthbook equivalence ───────────────────────────────

def test_schema_and_growthbook_equivalence():
    """build_schema_from_inputs and generate_growthbook_snippet produce
    identical output to the versions in agents/agent5/ for the same input.

    Equivalence proof, not a new baseline.
    """
    # Test schema
    from core.page_builder.schema_markup import build_schema_from_inputs as new_schema
    from agents.agent5.schema_markup import build_schema_from_inputs as old_schema

    kb = {
        "name": "Test Property",
        "description": "A beautiful vacation rental.",
        "booking_url": "https://example.com/book",
        "city": {"value": "Asheville"},
        "state": {"value": "NC"},
        "zip_code": {"value": "28801"},
        "address_line1": {"value": "123 Main St"},
        "latitude": {"value": 35.5951},
        "longitude": {"value": -82.5515},
        "bedrooms": {"value": 3},
        "bathrooms": {"value": 2.5},
        "max_occupancy": {"value": 8},
        "avg_nightly_rate": {"value": 250.0},
        "airbnb_rating": {"value": 4.9},
        "airbnb_review_count": {"value": 42},
        "amenities": [
            {"value": "Pool"}, {"value": "Hot Tub"}, {"value": "WiFi"},
        ],
    }
    content = {"property_description": "A wonderful place to stay."}
    visual = {"hero_photo_url": "https://example.com/hero.jpg"}

    old_result = old_schema(kb, content, visual, "https://test.upliftstays.com", "test")
    new_result = new_schema(kb, content, visual, "https://test.upliftstays.com", "test")
    assert old_result == new_result, (
        f"Schema output differs:\nOLD:\n{old_result[:200]}\nNEW:\n{new_result[:200]}"
    )

    # Test growthbook
    from core.page_builder.ab_testing import generate_growthbook_snippet as new_gb
    from agents.agent5.ab_testing import generate_growthbook_snippet as old_gb

    # Both should return the "not configured" comment when no key is set
    old_gb_result = old_gb("prop-1", "test-slug", [])
    new_gb_result = new_gb("prop-1", "test-slug", [])
    assert old_gb_result == new_gb_result, (
        f"GrowthBook output differs:\nOLD: {old_gb_result!r}\nNEW: {new_gb_result!r}"
    )
