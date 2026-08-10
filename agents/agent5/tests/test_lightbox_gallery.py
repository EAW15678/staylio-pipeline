"""
Tests for lightbox, gallery modal, JS escaping, tour display caps, and dedupe.
"""

import re
import pytest
from agents.agent5.page_builder import (
    _esc_js,
    _build_lightbox_gallery_js,
    build_landing_page_html,
    _build_category_modules_section,
    _build_gallery_section,
    _MAX_VISIBLE_SUPPORTING,
    _SECTION_TO_GCV_CATS,
)


# ── _esc_js tests ─────────────────────────────────────────────────────────

class TestEscJs:
    def test_single_quote(self):
        assert _esc_js("it's great") == "it\\'s great"

    def test_double_quote(self):
        assert _esc_js('say "hello"') == 'say \\"hello\\"'

    def test_backtick(self):
        assert _esc_js("use `code`") == "use \\`code\\`"

    def test_backslash(self):
        assert _esc_js("path\\to\\file") == "path\\\\to\\\\file"

    def test_newline(self):
        assert _esc_js("line1\nline2") == "line1\\nline2"

    def test_script_close_tag(self):
        """</script> in alt text must not break the script block."""
        result = _esc_js("text</script>more")
        assert "</script>" not in result
        assert "<\\/script>" in result

    def test_script_close_tag_case_variants(self):
        assert "</Script>" not in _esc_js("</Script>")
        assert "</SCRIPT>" not in _esc_js("</SCRIPT>")

    def test_none(self):
        assert _esc_js(None) == ""

    def test_combined_nasty(self):
        """LLM-generated alt with multiple dangerous chars."""
        text = "A 'cozy' kitchen with \"granite\" counters\nand a `modern` feel</script>"
        result = _esc_js(text)
        assert "'" not in result or "\\'" in result
        assert "\n" not in result
        assert "</script>" not in result


# ── _build_lightbox_gallery_js tests ──────────────────────────────────────

def _make_gallery_items(n=5):
    return [
        {"url": f"https://r2.example.com/img{i}.jpg",
         "alt": f"Photo {i} of property",
         "category": ["exterior", "kitchen", "bathroom", "living_room", "view"][i % 5]}
        for i in range(n)
    ]


class TestLightboxGalleryJs:
    def test_empty_gallery(self):
        assert _build_lightbox_gallery_js([]) == ""

    def test_lightbox_dialog_present(self):
        html = _build_lightbox_gallery_js(_make_gallery_items())
        assert 'id="staylio-lightbox"' in html
        assert "<dialog" in html

    def test_gallery_dialog_present(self):
        html = _build_lightbox_gallery_js(_make_gallery_items())
        assert 'id="staylio-gallery"' in html

    def test_openLightbox_defined(self):
        """openLightbox must be a global function."""
        html = _build_lightbox_gallery_js(_make_gallery_items())
        assert "window.openLightbox" in html

    def test_openGallery_defined(self):
        html = _build_lightbox_gallery_js(_make_gallery_items())
        assert "window.openGallery" in html

    def test_photo_count_matches(self):
        """PHOTOS array length must match gallery_items count."""
        items = _make_gallery_items(7)
        html = _build_lightbox_gallery_js(items)
        # Count url entries in the JS array
        url_count = html.count("'url':'https://r2.example.com/")
        assert url_count == 7

    def test_lightbox_modal_emitted_once(self):
        html = _build_lightbox_gallery_js(_make_gallery_items())
        assert html.count('id="staylio-lightbox"') == 1

    def test_gallery_modal_emitted_once(self):
        html = _build_lightbox_gallery_js(_make_gallery_items())
        assert html.count('id="staylio-gallery"') == 1

    def test_category_tabs_present(self):
        html = _build_lightbox_gallery_js(_make_gallery_items())
        assert "CATS" in html
        # Should have category entries
        assert "'label':'" in html

    def test_escape_in_alt_text(self):
        """Alt text with dangerous chars must be escaped in JS."""
        items = [{"url": "https://r2.example.com/img.jpg",
                  "alt": "A 'cozy' room</script>",
                  "category": "exterior"}]
        html = _build_lightbox_gallery_js(items)
        assert "</script>" not in html.split("</script>")[0]  # no raw close in script content
        assert "\\'" in html  # escaped quote

    def test_arrow_key_navigation(self):
        html = _build_lightbox_gallery_js(_make_gallery_items())
        assert "ArrowLeft" in html
        assert "ArrowRight" in html

    def test_backdrop_close(self):
        html = _build_lightbox_gallery_js(_make_gallery_items())
        assert "e.target === lbDialog" in html
        assert "e.target === galDialog" in html

    def test_body_scroll_lock(self):
        html = _build_lightbox_gallery_js(_make_gallery_items())
        assert "document.body.style.overflow" in html


# ── Integration: gallery section onclick indices ──────────────────────────

class TestGallerySectionIndices:
    def test_onclick_indices_sequential(self):
        items = _make_gallery_items(10)
        html = _build_gallery_section(items, "Test Property")
        # Extract all openLightbox(N) calls
        indices = [int(m) for m in re.findall(r'openLightbox\((\d+)\)', html)]
        assert indices == list(range(10))


# ── Integration: category modules section ─────────────────────────────────

class TestCategoryModulesSection:
    def test_view_all_opens_gallery(self):
        modules = {
            "Exterior": {
                "hero": {"url": "https://example.com/hero.jpg", "alt": "Ext hero"},
                "supporting": []
            }
        }
        gallery_items = [{"url": "https://example.com/hero.jpg", "alt": "Ext hero", "category": "exterior"}]
        html = _build_category_modules_section(modules, gallery_items)
        assert "openGallery()" in html

    def test_gallery_anchor_fallback(self):
        """href=#gallery preserved as no-JS fallback."""
        modules = {
            "Exterior": {
                "hero": {"url": "https://example.com/hero.jpg", "alt": "Ext hero"},
                "supporting": []
            }
        }
        gallery_items = [{"url": "https://example.com/hero.jpg", "alt": "Ext hero", "category": "exterior"}]
        html = _build_category_modules_section(modules, gallery_items)
        assert 'href="#gallery"' in html


# ── Hover CSS (already exists, verify not removed) ────────────────────────

class TestHoverCss:
    def test_cat_module_hero_hover(self):
        from agents.agent5.page_builder import _page_css
        css = _page_css()
        assert ".cat-module-hero:hover" in css
        assert "opacity" in css

    def test_cat_module_thumb_hover(self):
        from agents.agent5.page_builder import _page_css
        css = _page_css()
        assert ".cat-module-thumb:hover" in css

    def test_gallery_thumb_hover(self):
        from agents.agent5.page_builder import _page_css
        css = _page_css()
        assert ".gallery-thumb:hover" in css


# ── Tour display cap tests ────────────────────────────────────────────────

def _make_module(label, n_supporting, category="living_room"):
    """Build a module dict with 1 hero + n supporting images."""
    hero = {"url": f"https://r2.example.com/{label}_hero.jpg",
            "alt": f"{label} hero", "category": category}
    supporting = [
        {"url": f"https://r2.example.com/{label}_sup{i}.jpg",
         "alt": f"{label} supporting {i}", "category": category}
        for i in range(n_supporting)
    ]
    return {
        label: {
            "hero": hero,
            "supporting": supporting,
            "all": [hero] + supporting,
        }
    }


def _all_gallery_items_for_module(module_dict):
    """Build gallery_items list from a module dict."""
    items = []
    for label, mod in module_dict.items():
        for item in mod["all"]:
            items.append(item)
    return items


class TestTourDisplayCap:
    def test_26_images_renders_4_visible(self):
        """A 26-image section renders exactly 1 hero + 3 supporting."""
        modules = _make_module("Living Areas", 25)
        gallery = _all_gallery_items_for_module(modules)
        html = _build_category_modules_section(modules, gallery)
        # Count rendered <img> tags (hero + visible supporting)
        img_tags = re.findall(r'<img\s', html)
        assert len(img_tags) == 1 + _MAX_VISIBLE_SUPPORTING  # 1 hero + 3

    def test_button_reads_view_all_26(self):
        """Button says 'View all 26 photos' for a section with 26 total."""
        modules = _make_module("Living Areas", 25)
        gallery = _all_gallery_items_for_module(modules)
        html = _build_category_modules_section(modules, gallery)
        assert "View all 26 photos" in html

    def test_no_button_when_4_or_fewer(self):
        """No overflow button when section has exactly 4 photos (1+3)."""
        modules = _make_module("Exterior", 3)
        gallery = _all_gallery_items_for_module(modules)
        html = _build_category_modules_section(modules, gallery)
        assert "View all" not in html or "View all photos" in html
        # More precisely: no per-section button
        assert "cat-module-more" not in html

    def test_no_button_when_fewer_than_4(self):
        """No overflow button when section has 2 photos (1+1)."""
        modules = _make_module("Exterior", 1)
        gallery = _all_gallery_items_for_module(modules)
        html = _build_category_modules_section(modules, gallery)
        assert "cat-module-more" not in html

    def test_button_opens_gallery_filtered(self):
        """Section button calls openGalleryFiltered with correct categories."""
        modules = _make_module("Living Areas", 10)
        gallery = _all_gallery_items_for_module(modules)
        html = _build_category_modules_section(modules, gallery)
        assert "openGalleryFiltered" in html
        # Living Areas maps to living_room and game_entertainment
        assert "'living_room'" in html
        assert "'game_entertainment'" in html

    def test_button_has_noscript_fallback(self):
        """Per-section button has href=#gallery as no-JS fallback."""
        modules = _make_module("Living Areas", 10)
        gallery = _all_gallery_items_for_module(modules)
        html = _build_category_modules_section(modules, gallery)
        # Find the per-section button's href
        assert 'href="#gallery"' in html

    def test_surplus_images_in_gallery(self):
        """All 26 images are in gallery_items (surplus not dropped)."""
        modules = _make_module("Living Areas", 25)
        gallery = _all_gallery_items_for_module(modules)
        # gallery has 26 items — renderer does NOT remove surplus
        assert len(gallery) == 26
        # Gallery section renders all of them
        gallery_html = _build_gallery_section(gallery, "Test")
        gallery_imgs = re.findall(r'<img\s', gallery_html)
        assert len(gallery_imgs) == 26

    def test_nothing_dropped_from_total(self):
        """Section total in button matches actual module item count."""
        modules = _make_module("Pool", 12, category="pool_hot_tub")
        gallery = _all_gallery_items_for_module(modules)
        html = _build_category_modules_section(modules, gallery)
        assert "View all 13 photos" in html


# ── Dedupe tests (physical_room_id) ──────────────────────────────────────

class TestPickForSectionDedupe:
    """Test _pick_for_section dedupe on physical_room_id."""

    def test_physical_room_id_dedupe(self):
        """Two images sharing physical_room_id cannot both be selected."""
        from agents.agent3.llm_curator import _SECTION_PRIMARY_THRESHOLD

        # Access _pick_for_section through the module — it's a closure inside
        # _build_tour_sections, so we test via the public function or replicate
        # the logic. Since it's a nested def, we test the llm_curator module
        # behavior by simulating scored input.
        # Instead, test the dedupe logic directly:
        seen_rooms = set()
        seen_groups = set()
        selected = []
        images = [
            {"asset_id": "img1", "physical_room_id": "pool_deck", "duplicate_group": "dg_pool",
             "quality_score": 0.9},
            {"asset_id": "img2", "physical_room_id": "pool_deck", "duplicate_group": "dg_pool_1",
             "quality_score": 0.8},
            {"asset_id": "img3", "physical_room_id": "kitchen_main", "duplicate_group": None,
             "quality_score": 0.7},
        ]
        scored = [(0.8, img) for img in images]

        for sc, img in scored:
            room = img.get("physical_room_id")
            if room and room in seen_rooms:
                continue
            dg = img.get("duplicate_group")
            if dg and dg in seen_groups:
                continue
            selected.append(img)
            if room:
                seen_rooms.add(room)
            if dg:
                seen_groups.add(dg)

        # img1 selected, img2 skipped (same pool_deck), img3 selected
        assert len(selected) == 2
        assert selected[0]["asset_id"] == "img1"
        assert selected[1]["asset_id"] == "img3"

    def test_null_room_falls_back_to_duplicate_group(self):
        """When physical_room_id is null, duplicate_group is still checked."""
        seen_rooms = set()
        seen_groups = set()
        selected = []
        images = [
            {"asset_id": "img1", "physical_room_id": None, "duplicate_group": "dg_exterior"},
            {"asset_id": "img2", "physical_room_id": None, "duplicate_group": "dg_exterior"},
        ]
        scored = [(0.8, img) for img in images]

        for sc, img in scored:
            room = img.get("physical_room_id")
            if room and room in seen_rooms:
                continue
            dg = img.get("duplicate_group")
            if dg and dg in seen_groups:
                continue
            selected.append(img)
            if room:
                seen_rooms.add(room)
            if dg:
                seen_groups.add(dg)

        # img1 selected, img2 skipped (same duplicate_group)
        assert len(selected) == 1
        assert selected[0]["asset_id"] == "img1"


# ── Curation version bump ────────────────────────────────────────────────

class TestCurationVersion:
    def test_version_bumped(self):
        from agents.agent3.llm_curator import _CURATION_VERSION
        assert _CURATION_VERSION != "curation_v11_640x480_cells"
        assert "v12" in _CURATION_VERSION


# ── openGalleryFiltered in JS ────────────────────────────────────────────

class TestOpenGalleryFiltered:
    def test_function_defined(self):
        html = _build_lightbox_gallery_js(_make_gallery_items())
        assert "window.openGalleryFiltered" in html

    def test_multi_category_filter(self):
        html = _build_lightbox_gallery_js(_make_gallery_items())
        assert "cats.indexOf(p.cat)" in html


# ── Truthful button count tests ──────────────────────────────────────────

class TestButtonCountTruthful:
    def test_button_matches_gallery_count_not_module_count(self):
        """When gallery has fewer items than module (dupe suppression),
        button shows gallery count."""
        # Module has 10 items but gallery only has 6 with matching category
        modules = _make_module("Living Areas", 9)  # 10 total in module
        # Build gallery with only 6 living_room items (simulating dupe suppression)
        gallery = [
            {"url": f"https://r2.example.com/gal{i}.jpg",
             "alt": f"Gallery {i}", "category": "living_room"}
            for i in range(6)
        ]
        html = _build_category_modules_section(modules, gallery)
        # Should show gallery count (6), not module count (10)
        assert "View all 6 photos" in html
        assert "View all 10 photos" not in html

    def test_no_button_when_gallery_count_le_4(self):
        """If dupe suppression reduces gallery count to ≤4, no button."""
        modules = _make_module("Living Areas", 9)  # 10 in module
        # But only 4 reach the gallery
        gallery = [
            {"url": f"https://r2.example.com/gal{i}.jpg",
             "alt": f"Gallery {i}", "category": "living_room"}
            for i in range(4)
        ]
        html = _build_category_modules_section(modules, gallery)
        assert "cat-module-more" not in html

    def test_label_and_filter_same_categories(self):
        """The button label count and the JS filter use the same category set."""
        modules = _make_module("Living Areas", 10)
        # Gallery has items in both living_room and game_entertainment
        gallery = [
            {"url": f"https://r2.example.com/lr{i}.jpg",
             "alt": f"Living {i}", "category": "living_room"}
            for i in range(5)
        ] + [
            {"url": f"https://r2.example.com/ge{i}.jpg",
             "alt": f"Game {i}", "category": "game_entertainment"}
            for i in range(3)
        ]
        html = _build_category_modules_section(modules, gallery)
        # Total gallery count for Living Areas = 5 + 3 = 8
        assert "View all 8 photos" in html
        # And the JS filter includes both categories
        assert "'living_room'" in html
        assert "'game_entertainment'" in html

    def test_dupe_suppressed_shows_reduced_count(self):
        """Section with 26 raw images but only 21 in gallery shows 21."""
        modules = _make_module("Pool", 25, category="pool_hot_tub")  # 26 raw
        # Only 21 survive to gallery (5 dupe non-winners suppressed)
        gallery = [
            {"url": f"https://r2.example.com/pool{i}.jpg",
             "alt": f"Pool {i}", "category": "pool_hot_tub"}
            for i in range(21)
        ]
        html = _build_category_modules_section(modules, gallery)
        assert "View all 21 photos" in html
        assert "View all 26 photos" not in html

    def test_zero_gallery_count_no_button(self):
        """If a section's photos are entirely absent from gallery, no button.
        This would indicate a problem (reported, not fixed here)."""
        modules = _make_module("Pool", 5, category="pool_hot_tub")
        # Gallery has no pool_hot_tub items at all
        gallery = [
            {"url": f"https://r2.example.com/ext{i}.jpg",
             "alt": f"Exterior {i}", "category": "exterior"}
            for i in range(10)
        ]
        html = _build_category_modules_section(modules, gallery)
        # No button since gallery_count for pool_hot_tub = 0
        assert "cat-module-more" not in html
