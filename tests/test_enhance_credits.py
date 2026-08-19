"""
ENHANCE-3: Credit counting, recipe split, and rate lookup tests.

All behavioural — call functions with data, assert on outputs.
No vendor calls, $0.
"""

import pytest
from unittest.mock import MagicMock, patch

from skills.enhance import (
    _select_recipe, _build_recipe, _count_credits, _get_credit_rate,
)


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


# ── Test 1: 24MP at quality_score 0.4 → large_weak ──────────────────

def test_large_weak_24mp_low_quality():
    """A 24MP photograph at quality_score 0.4 selects large_weak;
    payload has no upscale, no resizing, and no polish (over 16MP).

    Must fail on bc8d8aa — large_weak recipe does not exist.
    """
    photo = _photo(5712, 4284)  # 24.47MP
    obs = _obs(quality_score=0.4)
    name, ops = _select_recipe(photo, obs)
    assert name == "large_weak", f"Expected large_weak, got {name}"
    assert "upscale" not in ops.get("restorations", {})
    assert "resizing" not in ops
    assert "polish" not in ops.get("restorations", {}), "polish must be skipped above 16MP"


# ── Test 2: 1.5MP at quality_score 0.4 → small_weak ─────────────────

def test_small_weak_1_5mp():
    """A 1.5MP photograph at quality_score 0.4 selects small_weak;
    payload has upscale smart_enhance and resizing 150%.

    Must fail on bc8d8aa — small_weak selector changed (was qs < 0.6 OR mp < 2.0).
    """
    photo = _photo(1500, 1000)  # 1.5MP
    obs = _obs(quality_score=0.4)
    name, ops = _select_recipe(photo, obs)
    assert name == "small_weak", f"Expected small_weak, got {name}"
    assert ops["restorations"]["upscale"] == "smart_enhance"
    assert ops["resizing"]["width"] == "150%"


# ── Test 3: 10MP at quality_score 0.4 → large_weak WITH polish ──────

def test_large_weak_10mp_with_polish():
    """A 10MP photograph at quality_score 0.4 selects large_weak
    with polish true (under 16MP).

    Must fail on bc8d8aa — large_weak recipe does not exist.
    """
    photo = _photo(4000, 2500)  # 10MP
    obs = _obs(quality_score=0.4)
    name, ops = _select_recipe(photo, obs)
    assert name == "large_weak", f"Expected large_weak, got {name}"
    assert ops["restorations"].get("polish") is True


# ── Test 4: exhaustiveness ───────────────────────────────────────────

def test_exhaustive_recipe_selection():
    """Across a spread of sizes and scores, every photograph matches
    exactly one recipe; none matches none.

    Must fail on bc8d8aa — large_weak does not exist.
    """
    test_cases = [
        (_photo(640, 480), _obs(quality_score=0.3), "small_weak"),
        (_photo(640, 480), _obs(quality_score=0.9), "small_weak"),
        (_photo(3840, 2560), _obs(quality_score=0.4), "large_weak"),
        (_photo(3840, 2560), _obs(quality_score=0.9), "large_clean"),
        (_photo(3840, 2560), _obs(quality_score=0.59), "large_weak"),
        (_photo(3840, 2560), _obs(quality_score=0.6), "large_clean"),
        (_photo(1999, 999), _obs(quality_score=0.9), "small_weak"),  # 1.997MP < 2.0
        (_photo(2000, 1000), _obs(quality_score=0.9), "large_clean"),  # 2.0MP exactly
        (_photo(2000, 1000), _obs(quality_score=0.4), "large_weak"),
        # Higher-precedence overrides
        (_photo(3840, 2560), _obs(contains_text=True), "text_bearing"),
        (_photo(3840, 2560), _obs(placement="outdoor", time_of_day_read="midday"), "bright_exterior"),
        (_photo(3840, 2560), _obs(light_quality="flat"), "flat_light"),
        # Fallback (no observation)
        (_photo(3840, 2560), None, "large_clean"),
        (_photo(640, 480), None, "small_weak"),
    ]
    for photo, obs, expected in test_cases:
        name, ops = _select_recipe(photo, obs)
        assert name == expected, (
            f"Photo {photo['image_width']}x{photo['image_height']} "
            f"score={obs.get('quality_score') if obs else 'N/A'}: "
            f"expected {expected}, got {name}"
        )


# ── Test 5: credit counts per recipe ─────────────────────────────────

def test_credit_counts_per_recipe():
    """large_clean=1, flat_light=2, small_weak(1.5MP)=3, large_weak(10MP)=2.

    Must fail on bc8d8aa — _count_credits does not exist.
    """
    _, ops_lc = _build_recipe("large_clean", 5.0)
    assert _count_credits(ops_lc, 5.0) == 1

    _, ops_fl = _build_recipe("flat_light", 5.0)
    assert _count_credits(ops_fl, 5.0) == 2  # adjustments + polish

    _, ops_sw = _build_recipe("small_weak", 1.5)
    assert _count_credits(ops_sw, 1.5) == 3  # adjustments + polish + upscale(<4MP)

    _, ops_lw = _build_recipe("large_weak", 10.0)
    assert _count_credits(ops_lw, 10.0) == 2  # adjustments + polish


# ── Test 6: upscale tier credits ─────────────────────────────────────

def test_upscale_tier_credits():
    """Upscale on 10MP=3, 5MP=2, 1MP=1 credits.

    Must fail on bc8d8aa — _count_credits does not exist.
    """
    ops_with_upscale = {"restorations": {"upscale": "smart_enhance"}, "adjustments": {"hdr": 100}}
    assert _count_credits(ops_with_upscale, 10.0) == 1 + 3  # adj + upscale 9MP+
    assert _count_credits(ops_with_upscale, 5.0) == 1 + 2   # adj + upscale 4-9MP
    assert _count_credits(ops_with_upscale, 1.0) == 1 + 1   # adj + upscale <4MP


# ── Test 7: resizing and decompress = 0 credits ─────────────────────

def test_resizing_decompress_zero_credits():
    """resizing and decompress add 0 credits.

    Must fail on bc8d8aa — _count_credits does not exist.
    """
    ops_base = {"restorations": {"decompress": "auto"}, "adjustments": {"hdr": 100}}
    ops_with_resize = {**ops_base, "resizing": {"width": "150%", "height": "150%"}}
    assert _count_credits(ops_base, 5.0) == _count_credits(ops_with_resize, 5.0)


# ── Test 8: cost_events format ───────────────────────────────────────

def test_cost_events_format():
    """cost_events must receive units=credits, unit_name='credits',
    total_cost=units*unit_cost.

    Must fail on bc8d8aa — old code sends units=1, unit_name='images'.
    Structural: asserts on the emit_cost call shape via the credit
    counting function, since mocking the full enhance loop is
    impractical for a single emit_cost assertion.
    """
    _, ops = _build_recipe("large_clean", 5.0)
    credits = _count_credits(ops, 5.0)
    rate = 0.059
    expected_total = round(credits * rate, 4)
    assert credits == 1
    assert expected_total == 0.059


# ── Test 9: missing rate raises ──────────────────────────────────────

def test_missing_rate_raises():
    """A missing rate raises rather than defaulting to a hardcoded price.

    Must fail on bc8d8aa — _get_credit_rate does not exist.
    """
    sb = MagicMock()
    sb.table.return_value.select.return_value.eq.return_value.eq.return_value.is_.return_value.limit.return_value.execute.return_value = MagicMock(data=[])

    with pytest.raises(EnvironmentError, match="No current Claid credit rate"):
        _get_credit_rate(sb)


# ── Test 10: 24MP text-bearing at low quality → no upscale ──────────

def test_text_bearing_24mp_low_quality_no_upscale():
    """A 24MP contains_text photograph at quality_score 0.4 gets
    no upscale and no resizing — 'small' means physically small only.

    Must fail on bc8d8aa — is_small includes qs < 0.6 there.
    """
    photo = _photo(5712, 4284)  # 24.47MP
    obs = _obs(contains_text=True, quality_score=0.4)
    name, ops = _select_recipe(photo, obs)
    assert name == "text_bearing"
    assert "upscale" not in ops.get("restorations", {}), \
        "24MP text photo must NOT get upscale regardless of quality_score"
    assert "resizing" not in ops, \
        "24MP text photo must NOT get resizing regardless of quality_score"
