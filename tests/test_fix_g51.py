"""
FIX-G51: Tests for overlay_placement and no_guest_names false-positive fixes.
"""

from skills.direct import (
    validate_overlay_placement,
    validate_no_guest_names,
    _extract_regions,
)


# ── _extract_regions ──────────────────────────────────────────────────


def test_extract_regions_dict_format():
    """Current format: list of dicts with 'region' key."""
    neg = [
        {"region": "top_center", "size": "medium", "contrast": "high"},
        {"region": "bottom_left", "size": "small", "contrast": "low"},
    ]
    assert _extract_regions(neg) == {"top_center", "bottom_left"}


def test_extract_regions_flat_format():
    """Legacy format: list of strings."""
    neg = ["top_center", "bottom_left"]
    assert _extract_regions(neg) == {"top_center", "bottom_left"}


def test_extract_regions_empty():
    assert _extract_regions([]) == set()
    assert _extract_regions(None) == set()


# ── validate_overlay_placement ────────────────────────────────────────


def _make_direction_with_overlay(grid_region, beat_photo_id="photo1"):
    return {
        "beats": [{"ordinal": 1, "photo_id": beat_photo_id}],
        "overlay_register": [{"beat_ordinal": 1, "grid_region": grid_region, "text": "Test"}],
    }


def test_overlay_placement_pass_dict_neg_space():
    """Overlay in a region that IS in negative_space (dict format) → PASS."""
    direction = _make_direction_with_overlay("top_center")
    obs_map = {
        "photo1": {
            "negative_space": [
                {"region": "top_center", "size": "medium", "contrast": "high"},
            ]
        }
    }
    assert validate_overlay_placement(direction, obs_map) == []


def test_overlay_placement_fail_dict_neg_space():
    """Overlay in a region NOT in negative_space (dict format) → FAIL."""
    direction = _make_direction_with_overlay("bottom_right")
    obs_map = {
        "photo1": {
            "negative_space": [
                {"region": "top_center", "size": "medium", "contrast": "high"},
            ]
        }
    }
    violations = validate_overlay_placement(direction, obs_map)
    assert len(violations) == 1
    assert violations[0]["rule"] == "overlay_placement"


def test_overlay_placement_pass_flat_neg_space():
    """Overlay in a region that IS in negative_space (flat format) → PASS."""
    direction = _make_direction_with_overlay("top_center")
    obs_map = {"photo1": {"negative_space": ["top_center", "top_left"]}}
    assert validate_overlay_placement(direction, obs_map) == []


# ── validate_no_guest_names ───────────────────────────────────────────


def test_no_guest_names_article_pass():
    """'— the chef's kitchen' must NOT trigger for guest 'The Hillis Family'."""
    direction = {
        "narration_script": "Step inside — the chef's kitchen awaits.",
        "overlay_register": [],
    }
    violations = validate_no_guest_names(direction, ["The Hillis Family"])
    assert violations == []


def test_no_guest_names_full_name_fail():
    """'— The Hillis Family' MUST trigger for guest 'The Hillis Family'."""
    direction = {
        "narration_script": "Loved by returning guests — The Hillis Family",
        "overlay_register": [],
    }
    violations = validate_no_guest_names(direction, ["The Hillis Family"])
    assert len(violations) >= 1
    assert violations[0]["rule"] == "no_guest_names"


def test_no_guest_names_single_name_fail():
    """'— Eileen' MUST trigger for guest 'Eileen'."""
    direction = {
        "narration_script": "As one guest put it — Eileen",
        "overlay_register": [],
    }
    violations = validate_no_guest_names(direction, ["Eileen"])
    assert len(violations) >= 1
    assert violations[0]["rule"] == "no_guest_names"


def test_no_guest_names_surname_in_text():
    """'Hillis' alone in text catches via part pattern even without full name."""
    direction = {
        "narration_script": "— hillis loved the sunrise view",
        "overlay_register": [],
    }
    violations = validate_no_guest_names(direction, ["The Hillis Family"])
    assert len(violations) >= 1


def test_no_guest_names_empty_list():
    """No guest names → no violations."""
    direction = {"narration_script": "Welcome to paradise.", "overlay_register": []}
    assert validate_no_guest_names(direction, []) == []


def test_no_guest_names_no_match():
    """Guest name not present at all → PASS."""
    direction = {
        "narration_script": "The pool sparkles under the summer sun.",
        "overlay_register": [],
    }
    assert validate_no_guest_names(direction, ["Eileen"]) == []
