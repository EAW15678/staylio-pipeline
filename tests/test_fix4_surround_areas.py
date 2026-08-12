"""
FIX-4: surround_areas parser + anchor-merge + section removal tests.

Proves:
  - parse_search_anchors splits on newlines/slashes, NOT commas
  - "Carolina Beach, NC" stays as one anchor
  - merge_venues_by_place_id deduplicates correctly
  - _build_surrounding_areas_section is removed from page_builder
"""

from skills.build_guide import parse_search_anchors, merge_venues_by_place_id


# ── Parser tests ─────────────────────────────────────────────────────────

def test_parse_newline_separated():
    """Newline-separated areas parse into individual anchors."""
    raw = "Carolina Beach, NC\nWilmington, NC\nSouthport\nKure Beach"
    result = parse_search_anchors(raw)
    assert result == ["Carolina Beach, NC", "Wilmington, NC", "Southport", "Kure Beach"]


def test_parse_preserves_commas():
    """Commas within city, state pairs are preserved (NOT split)."""
    raw = "Carolina Beach, NC\nWilmington, NC"
    result = parse_search_anchors(raw)
    assert len(result) == 2
    assert result[0] == "Carolina Beach, NC"
    assert result[1] == "Wilmington, NC"


def test_parse_slash_separated():
    """Slash-separated areas parse correctly."""
    raw = "Wilmington / Southport / Kure Beach"
    result = parse_search_anchors(raw)
    assert result == ["Wilmington", "Southport", "Kure Beach"]


def test_parse_mixed_newlines_and_slashes():
    """Mixed separators work."""
    raw = "Carolina Beach, NC\nWilmington / Southport"
    result = parse_search_anchors(raw)
    assert result == ["Carolina Beach, NC", "Wilmington", "Southport"]


def test_parse_empty_string():
    """Empty string returns empty list."""
    assert parse_search_anchors("") == []
    assert parse_search_anchors(None) == []


def test_parse_strips_whitespace():
    """Leading/trailing whitespace is stripped from each anchor."""
    raw = "  Carolina Beach  \n  Wilmington  "
    result = parse_search_anchors(raw)
    assert result == ["Carolina Beach", "Wilmington"]


def test_parse_vista_azule_stored_value():
    """Dry-run on Vista Azule Test's actual stored value.

    Stored: "Carolina Beach, NC\\nWilmington, NC\\nSoutbhport\\nKure Beach"
    Old comma-split produced: ["Carolina Beach", "NC", "Wilmington", "NC", "Soutbhport", "Kure Beach"]
    New newline-split produces: ["Carolina Beach, NC", "Wilmington, NC", "Soutbhport", "Kure Beach"]
    """
    stored = "Carolina Beach, NC\nWilmington, NC\nSoutbhport\nKure Beach"
    result = parse_search_anchors(stored)
    assert len(result) == 4
    assert "Carolina Beach, NC" in result
    assert "Wilmington, NC" in result
    assert "Soutbhport" in result  # Owner typo preserved
    assert "Kure Beach" in result
    # No standalone "NC" cards
    assert "NC" not in result


# ── Merge tests ──────────────────────────────────────────────────────────

def test_merge_deduplicates_by_place_id():
    """Venues with the same place_id are deduplicated; first occurrence wins."""
    list1 = [
        {"place_id": "A", "name": "Beach Cafe", "rating": 4.5},
        {"place_id": "B", "name": "Surf Shop", "rating": 4.0},
    ]
    list2 = [
        {"place_id": "A", "name": "Beach Cafe (dup)", "rating": 3.0},
        {"place_id": "C", "name": "Pier Bar", "rating": 4.2},
    ]
    merged = merge_venues_by_place_id([
        ("Carolina Beach", list1),
        ("Wilmington", list2),
    ])
    assert len(merged) == 3
    # First occurrence wins
    assert merged[0]["name"] == "Beach Cafe"
    assert merged[0]["anchor_area"] == "Carolina Beach"
    # Deduped
    assert not any(v["name"] == "Beach Cafe (dup)" for v in merged)
    # Third venue labeled with its source area
    assert merged[2]["anchor_area"] == "Wilmington"


def test_merge_empty_lists():
    """Empty venue lists produce empty result."""
    assert merge_venues_by_place_id([]) == []
    assert merge_venues_by_place_id([("Area", [])]) == []


# ── Section removal test ─────────────────────────────────────────────────

def test_surrounding_areas_section_removed():
    """_build_surrounding_areas_section no longer exists in page_builder."""
    from agents.agent5 import page_builder
    assert not hasattr(page_builder, '_build_surrounding_areas_section'), \
        "_build_surrounding_areas_section should be removed from page_builder"


def test_surrounding_areas_not_in_html_template():
    """The surrounding-areas section ID no longer appears in the HTML template."""
    import inspect
    from agents.agent5.page_builder import build_landing_page_html
    source = inspect.getsource(build_landing_page_html)
    assert 'id="surrounding-areas"' not in source
    assert '_build_surrounding_areas_section' not in source
