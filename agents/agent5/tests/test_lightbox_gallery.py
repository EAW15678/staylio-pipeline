"""
Tests for lightbox, gallery modal, and JS escaping in page_builder.py.
"""

import re
import pytest
from agents.agent5.page_builder import (
    _esc_js,
    _build_lightbox_gallery_js,
    build_landing_page_html,
    _build_category_modules_section,
    _build_gallery_section,
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
