"""
B2 FIX: Room module thumbnails must open the section-scoped gallery,
not the unscoped lightbox.

Erick's reproduction: clicking a Living Areas photo showed "59/72" from
the full gallery instead of staying within the 18 Living Areas photos.
"""

import sys
import re

sys.path.insert(0, ".")

from core.page_builder.sections import _build_category_modules_section, _build_gallery_section


def _make_items(sections_with_counts):
    """Build synthetic gallery items across multiple sections."""
    items = []
    for section, count in sections_with_counts:
        for i in range(count):
            items.append({
                "url": f"https://r2.dev/{section.lower().replace(' ', '_')}_{i}.jpg",
                "alt": f"{section} photo {i}",
                "section": section,
                "category": section,
                "rank": i,
            })
    return items


def _make_modules(items):
    """Build modules dict from items, one per section."""
    by_section = {}
    for item in items:
        by_section.setdefault(item["section"], []).append(item)

    modules = {}
    for section, sec_items in by_section.items():
        modules[section] = {
            "hero": sec_items[0],
            "supporting": sec_items[1:3],
            "all": sec_items,
        }
    return modules


# ── Test 1: Room module thumbnail opens section-scoped gallery ──────────────

def test_room_thumbnail_opens_scoped_gallery():
    """Clicking a thumbnail inside a room module opens the gallery filtered
    to exactly that section's photos — not the full set."""
    items = _make_items([("Living Areas", 18), ("Bedrooms", 11), ("Bathrooms", 12)])
    modules = _make_modules(items)

    html = _build_category_modules_section(modules, items)

    # Room module thumbnails must call openGalleryFiltered, NOT openLightbox
    # Find all onclick handlers on cat-module-hero and cat-module-thumb
    hero_clicks = re.findall(r'class="cat-module-hero" onclick="([^"]+)"', html)
    thumb_clicks = re.findall(r'class="cat-module-thumb" onclick="([^"]+)"', html)

    all_module_clicks = hero_clicks + thumb_clicks
    assert len(all_module_clicks) > 0, "Expected module thumbnail onclick handlers"

    for click in all_module_clicks:
        assert "openGalleryFiltered" in click, (
            f"Room module thumbnail should use openGalleryFiltered, got: {click}"
        )
        assert "openLightbox" not in click, (
            f"Room module thumbnail must NOT use openLightbox, got: {click}"
        )


# ── Test 2: Opens positioned on the specific photo clicked ──────────────────

def test_room_thumbnail_opens_on_correct_photo():
    """The onclick passes the specific photo URL so the gallery opens on
    that photo, not always index 0."""
    items = _make_items([("Kitchen", 7)])
    modules = _make_modules(items)

    html = _build_category_modules_section(modules, items)

    # The hero photo URL should appear in the onclick handler
    hero_url = items[0]["url"]
    assert hero_url.replace("/", "\\/") in html or hero_url in html, (
        f"Hero URL {hero_url} should appear in onclick handler"
    )

    # Supporting photo URLs should also appear
    if len(items) > 1:
        sup_url = items[1]["url"]
        assert sup_url.replace("/", "\\/") in html or sup_url in html, (
            f"Supporting URL {sup_url} should appear in onclick handler"
        )


# ── Test 3: Scoped gallery only contains section's photos ───────────────────

def test_scoped_gallery_stays_within_section():
    """The onclick handler passes only the section's own label as the
    filter — scrolling prev/next stays within that section."""
    items = _make_items([("Living Areas", 18), ("Bedrooms", 11)])
    modules = _make_modules(items)

    html = _build_category_modules_section(modules, items)

    # Find Living Areas module's hero onclick
    la_hero_match = re.search(
        r'class="cat-module-hero" onclick="([^"]*Living Areas[^"]*)"',
        html,
    )
    assert la_hero_match, "Living Areas hero should have an onclick with 'Living Areas'"

    onclick = la_hero_match.group(1)
    # Must filter by 'Living Areas', not 'Bedrooms'
    assert "'Living Areas'" in onclick, f"Should filter by 'Living Areas', got: {onclick}"
    assert "'Bedrooms'" not in onclick, f"Should NOT include 'Bedrooms' in filter, got: {onclick}"


# ── Test 4: Flat "All Photos" section thumbnails unchanged ──────────────────

def test_flat_gallery_still_uses_lightbox():
    """The flat 'All Photos' section's thumbnails must still use
    openLightbox — the unscoped viewer — not openGalleryFiltered."""
    items = _make_items([("Living Areas", 5), ("Bedrooms", 5)])

    html = _build_gallery_section(items, "Test Property")

    # All flat gallery thumbnails must use openLightbox
    gallery_clicks = re.findall(r'class="gallery-thumb" onclick="([^"]+)"', html)
    assert len(gallery_clicks) > 0, "Expected gallery thumbnail onclick handlers"

    for click in gallery_clicks:
        assert "openLightbox" in click, (
            f"Flat gallery thumbnail should use openLightbox, got: {click}"
        )
        assert "openGalleryFiltered" not in click, (
            f"Flat gallery thumbnail must NOT use openGalleryFiltered, got: {click}"
        )
