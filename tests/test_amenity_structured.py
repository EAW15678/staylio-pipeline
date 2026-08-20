"""
AMENITY-2: Behavioural tests for structured amenities.

Tests verify that the taxonomy, photo-matching, and skill readers all
handle the new {name, category} shape correctly.
"""

import sys
sys.path.insert(0, ".")


# ── Test 1: water_wellness amenity matches correct photo, rejects wrong name ─

def test_water_wellness_matches_correct_photo():
    """A water_wellness amenity claim matches a photo whose located_amenities
    contains the right name within the same category, and does NOT match a
    same-category photo with a different name."""
    from core.page_builder.orchestrate import _find_amenity_photo

    photos = [
        {
            "display_url": "https://r2.dev/pool.jpg",
            "located_amenities": [
                {"name": "swimming pool", "category": "water_wellness", "placement": "outdoor"},
            ],
            "is_setting": False,
            "setting_subject": "",
        },
        {
            "display_url": "https://r2.dev/sauna.jpg",
            "located_amenities": [
                {"name": "sauna", "category": "water_wellness", "placement": "indoor"},
            ],
            "is_setting": False,
            "setting_subject": "",
        },
    ]

    # "Hot tub" should NOT match "swimming pool" or "sauna" — different names
    result = _find_amenity_photo("Hot tub", "water_wellness", photos)
    assert result == "", f"Hot tub should not match pool or sauna, got: {result}"

    # "Private pool" SHOULD match "swimming pool"
    result = _find_amenity_photo("Private pool", "water_wellness", photos)
    assert result == "https://r2.dev/pool.jpg", f"Private pool should match swimming pool, got: {result}"

    # "Sauna" SHOULD match "sauna"
    result = _find_amenity_photo("Sauna", "water_wellness", photos)
    assert result == "https://r2.dev/sauna.jpg", f"Sauna should match sauna, got: {result}"


# ── Test 2: setting amenity matches via is_setting, not located_amenities ────

def test_setting_matches_via_is_setting():
    """A setting amenity claim matches a photo where is_setting=true and
    setting_subject corresponds — NOT via located_amenities."""
    from core.page_builder.orchestrate import _find_amenity_photo

    photos = [
        {
            "display_url": "https://r2.dev/ocean_view.jpg",
            "located_amenities": [],
            "is_setting": True,
            "setting_subject": "Atlantic Ocean beachfront town",
        },
        {
            "display_url": "https://r2.dev/kitchen.jpg",
            "located_amenities": [
                {"name": "ocean view wallpaper", "category": "practical", "placement": "indoor"},
            ],
            "is_setting": False,
            "setting_subject": "",
        },
    ]

    # "Ocean views" should match the setting photo, not the kitchen with ocean wallpaper
    result = _find_amenity_photo("Ocean views", "setting", photos)
    assert result == "https://r2.dev/ocean_view.jpg", f"Ocean views should match via is_setting, got: {result}"

    # "Mountain views" should NOT match — no mountain in setting_subject
    result = _find_amenity_photo("Mountain views", "setting", photos)
    assert result == "", f"Mountain views should not match Atlantic beach, got: {result}"

    # Verify it does NOT use located_amenities for setting items
    photos_only_located = [
        {
            "display_url": "https://r2.dev/wrong.jpg",
            "located_amenities": [
                {"name": "ocean views", "category": "setting", "placement": "outdoor"},
            ],
            "is_setting": False,
            "setting_subject": "",
        },
    ]
    result = _find_amenity_photo("Ocean views", "setting", photos_only_located)
    assert result == "", "Setting amenities must NOT match via located_amenities"


# ── Test 3: skills extract names correctly from structured list ──────────────

def test_skills_extract_names_from_structured():
    """conceive, direct, write_copy each correctly extract just the name
    from a structured amenity list."""
    from core.page_builder.amenity_taxonomy import extract_amenity_names

    structured = [
        {"name": "Private pool", "category": "water_wellness"},
        {"name": "Ocean views", "category": "setting"},
        {"name": "Fire pit", "category": "recreation"},
    ]

    names = extract_amenity_names(structured)
    assert names == ["Private pool", "Ocean views", "Fire pit"]

    # Also handles legacy flat strings
    flat = ["Private pool", "Ocean views", "Fire pit"]
    names_flat = extract_amenity_names(flat)
    assert names_flat == ["Private pool", "Ocean views", "Fire pit"]

    # Mixed (shouldn't happen, but graceful)
    mixed = [{"name": "Hot tub", "category": "water_wellness"}, "Fireplace"]
    names_mixed = extract_amenity_names(mixed)
    assert names_mixed == ["Hot tub", "Fireplace"]

    # Empty / None
    assert extract_amenity_names([]) == []
    assert extract_amenity_names(None) == []

    # The prompt formatting produces the same output
    amenity_str = ", ".join(names[:20])
    amenity_str_flat = ", ".join(names_flat[:20])
    assert amenity_str == amenity_str_flat


# ── Test 4: main.py request model accepts structured shape ───────────────────

def test_main_request_type_accepts_structured():
    """main.py's IntakeSubmissionRequest type annotation accepts the structured
    {name, category} shape — verified by reading the source directly since
    main.py uses Python 3.10+ syntax that can't be imported on 3.9."""
    import ast
    import inspect

    # Read main.py and find the amenities field annotation
    with open("main.py") as f:
        source = f.read()

    tree = ast.parse(source)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "amenities":
                # Check it's typed as list[dict], not list[str]
                ann = ast.dump(node.annotation)
                if "dict" in ann:
                    found = True
                    break

    assert found, "amenities field in IntakeSubmissionRequest should be typed as list[dict]"

    # Also verify the structured shape works with the taxonomy
    from core.page_builder.amenity_taxonomy import extract_amenity_names

    structured = [
        {"name": "Private pool", "category": "water_wellness"},
        {"name": "Ocean views", "category": "setting"},
    ]
    names = extract_amenity_names(structured)
    assert names == ["Private pool", "Ocean views"]
