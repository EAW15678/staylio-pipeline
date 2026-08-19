"""
ENHANCE-2: Behavioural tests for per-photograph enhancement recipes.

Tests exercise _select_recipe, _build_recipe, and validate_operations
with real observation data shapes. No vendor calls, $0.
"""

import io
from unittest.mock import MagicMock, patch

from skills.enhance import _select_recipe, _build_recipe, _apply_edge_stitching


def _photo(w=1200, h=800):
    return {"photo_id": "test", "image_width": w, "image_height": h}


def _obs(**kwargs):
    defaults = {
        "light_quality": "soft", "time_of_day_read": "interior_ambiguous",
        "placement": "indoor", "quality_score": 0.85, "contains_text": False,
        "depth_tier": "medium",
    }
    defaults.update(kwargs)
    return defaults


# ── Test 1: small_weak selected for quality_score 0.4 ────────────────

def test_small_weak_selected_for_low_quality():
    """quality_score 0.4 selects small_weak; payload has upscale smart_enhance,
    resizing 150%, polish true.

    Must fail on beb1ba1 — _select_recipe does not exist.
    """
    photo = _photo(1200, 800)
    obs = _obs(quality_score=0.4)
    name, ops = _select_recipe(photo, obs)
    assert name == "small_weak", f"Expected small_weak, got {name}"
    assert ops["restorations"]["upscale"] == "smart_enhance"
    assert ops["resizing"]["width"] == "150%"
    assert ops["resizing"]["height"] == "150%"
    assert ops["restorations"].get("polish") is True


# ── Test 2: large_clean for high-res high-quality ────────────────────

def test_large_clean_no_upscale():
    """3840x2560 at quality_score 0.9 selects large_clean; payload has
    no upscale and no resizing key.

    Must fail on beb1ba1 — _select_recipe does not exist.
    """
    photo = _photo(3840, 2560)
    obs = _obs(quality_score=0.9)
    name, ops = _select_recipe(photo, obs)
    assert name == "large_clean", f"Expected large_clean, got {name}"
    assert "upscale" not in ops.get("restorations", {})
    assert "resizing" not in ops


# ── Test 3: bright_exterior for outdoor midday ───────────────────────

def test_bright_exterior_outdoor_midday():
    """Outdoor midday selects bright_exterior — hdr 70, exposure -10,
    no polish.

    Must fail on beb1ba1 — _select_recipe does not exist.
    """
    photo = _photo(3840, 2560)
    obs = _obs(placement="outdoor", time_of_day_read="midday")
    name, ops = _select_recipe(photo, obs)
    assert name == "bright_exterior", f"Expected bright_exterior, got {name}"
    assert ops["adjustments"]["hdr"] == 70
    assert ops["adjustments"]["exposure"] == -10
    assert "polish" not in ops.get("restorations", {})


# ── Test 4: text_bearing wins over bright_exterior (precedence) ──────

def test_text_bearing_precedence():
    """contains_text=true AND outdoor midday selects text_bearing,
    not bright_exterior.

    Must fail on beb1ba1 — _select_recipe does not exist.
    """
    photo = _photo(3840, 2560)
    obs = _obs(contains_text=True, placement="outdoor", time_of_day_read="midday")
    name, ops = _select_recipe(photo, obs)
    assert name == "text_bearing", f"Expected text_bearing, got {name}"
    assert "polish" not in ops.get("restorations", {})


# ── Test 5: every recipe passes validate_operations ──────────────────

def test_all_recipes_pass_governance():
    """Every recipe's operations dict passes validate_operations.
    A prohibited operation still raises.

    Regression guard — the prohibited case passes on all commits.
    """
    from agents.agent3.claid_enhancer import validate_operations

    # All five recipes
    for recipe_name in ("small_weak", "large_clean", "flat_light", "bright_exterior", "text_bearing"):
        _, ops = _build_recipe(recipe_name, 1.0)
        validate_operations(ops)  # must not raise

    # text_bearing with is_small (adds upscale + resizing)
    _, ops = _build_recipe("text_bearing", 0.5, is_small=True)
    validate_operations(ops)

    # Prohibited operation must still raise
    import pytest
    with pytest.raises(ValueError, match="GOVERNANCE VIOLATION"):
        validate_operations({"restorations": {"virtual_staging": True}})


# ── Test 6: every recipe includes decompress auto ────────────────────

def test_all_recipes_include_decompress():
    """Every recipe includes decompress: auto.

    Must fail on beb1ba1 — _build_recipe does not exist.
    """
    for recipe_name in ("small_weak", "large_clean", "flat_light", "bright_exterior", "text_bearing"):
        _, ops = _build_recipe(recipe_name, 1.0)
        assert ops.get("restorations", {}).get("decompress") == "auto", \
            f"Recipe {recipe_name} missing decompress: auto"


# ── Test 7: no observation → fallback, never skipped ─────────────────

def test_fallback_no_observation():
    """A photograph with no active observation is still enhanced via
    the fallback and is never skipped.

    Must fail on beb1ba1 — _select_recipe does not exist.
    """
    # Large photo → large_clean fallback
    photo_large = _photo(3840, 2560)
    name_l, ops_l = _select_recipe(photo_large, None)
    assert name_l == "large_clean"
    assert ops_l is not None

    # Small photo → small_weak fallback
    photo_small = _photo(800, 600)
    name_s, ops_s = _select_recipe(photo_small, None)
    assert name_s == "small_weak"
    assert ops_s is not None


# ── Test 8: onboard observe before enhance ───────────────────────────

def test_onboard_observe_before_enhance():
    """onboard invokes observe before enhance, asserted on AST call order.

    Structural assertion — onboard's late imports make full mock-driven
    execution impractical (see PHASH-2 test 4 justification).

    Must fail on beb1ba1 (enhance at step 4, observe at step 5).
    """
    import ast
    import inspect
    from workflows.onboard import onboard

    source = inspect.getsource(onboard)
    tree = ast.parse(source)

    skill_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "_run_skill":
                if node.args and isinstance(node.args[0], ast.Constant):
                    skill_calls.append(node.args[0].value)

    assert "observe" in skill_calls
    assert "enhance" in skill_calls
    obs_idx = skill_calls.index("observe")
    enh_idx = skill_calls.index("enhance")
    assert obs_idx < enh_idx, (
        f"observe (pos {obs_idx}) must come before enhance (pos {enh_idx}). "
        f"Order: {skill_calls}"
    )


# ── Test 9: 24MP photo in polish group omits polish ──────────────────

def test_large_photo_polish_skipped():
    """A 24MP photograph in a polish-eligible group (flat_light) omits
    polish and still enhances with all other operations.

    Must fail on beb1ba1 — _build_recipe does not exist.
    """
    # 24MP = above the 16MP polish ceiling
    _, ops = _build_recipe("flat_light", 24.0)
    assert "polish" not in ops.get("restorations", {}), \
        "polish must be omitted above 16MP"
    # Other operations still present
    assert ops["restorations"]["decompress"] == "auto"
    assert ops["adjustments"]["hdr"] == 100
    assert ops["adjustments"]["contrast"] == 15
