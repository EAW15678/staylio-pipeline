"""
ENHANCE-4: Single-definition enforcement tests.

Tests 1-3 are behavioural. Test 4 is a structural guard — the one
case where source inspection is the correct tool, because the
property being tested is "this rule is written once", which is a
property of the source, not of behaviour.
"""

import ast
import inspect
import itertools

from skills.enhance import (
    _is_small, _is_weak, _select_recipe,
    _SMALL_MAX_MP, _WEAK_MAX_SCORE, _WIDE_ASPECT_RATIO,
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


# ── Test 1: is_small is unaffected by quality_score ──────────────────

def test_is_small_unaffected_by_quality():
    """is_small returns true below 2.0MP, false at and above — and is
    unaffected by quality score.

    Must fail on 8dc88ba — _is_small does not exist.
    """
    assert _is_small(1.99) is True
    assert _is_small(2.0) is False
    assert _is_small(0.001) is True
    assert _is_small(24.0) is False
    # is_small takes only mp — quality_score cannot be passed
    assert _is_small.__code__.co_varnames[:1] == ("mp",)


# ── Test 2: is_weak is unaffected by megapixels ─────────────────────

def test_is_weak_unaffected_by_megapixels():
    """is_weak returns true below 0.6, false at and above — and is
    unaffected by megapixels.

    Must fail on 8dc88ba — _is_weak does not exist.
    """
    assert _is_weak(0.59) is True
    assert _is_weak(0.6) is False
    assert _is_weak(0.0) is True
    assert _is_weak(1.0) is False
    # is_weak takes only quality_score
    assert _is_weak.__code__.co_varnames[:1] == ("quality_score",)


# ── Test 3: equivalence with 8dc88ba across full spread ──────────────

def test_recipe_selection_equivalence():
    """Every recipe selection produces the same result as 8dc88ba across
    a spread of sizes, scores, placements and text flags.

    Regression guard — no failing baseline. The refactor must not
    change any selection result.
    """
    # The expected results are what 8dc88ba produces
    cases = [
        # (photo, obs, expected_recipe)
        (_photo(640, 480), _obs(quality_score=0.3), "small_weak"),
        (_photo(640, 480), _obs(quality_score=0.9), "small_weak"),
        (_photo(3840, 2560), _obs(quality_score=0.4), "large_weak"),
        (_photo(3840, 2560), _obs(quality_score=0.9), "large_clean"),
        (_photo(3840, 2560), _obs(quality_score=0.59), "large_weak"),
        (_photo(3840, 2560), _obs(quality_score=0.6), "large_clean"),
        (_photo(1999, 999), _obs(quality_score=0.9), "small_weak"),
        (_photo(2000, 1000), _obs(quality_score=0.9), "large_clean"),
        (_photo(2000, 1000), _obs(quality_score=0.4), "large_weak"),
        (_photo(5712, 4284), _obs(quality_score=0.4), "large_weak"),
        (_photo(5712, 4284), _obs(contains_text=True, quality_score=0.4), "text_bearing"),
        (_photo(640, 480), _obs(contains_text=True, quality_score=0.4), "text_bearing"),
        (_photo(3840, 2560), _obs(placement="outdoor", time_of_day_read="midday"), "bright_exterior"),
        (_photo(3840, 2560), _obs(placement="outdoor", time_of_day_read="morning"), "bright_exterior"),
        (_photo(3840, 2560), _obs(light_quality="hard"), "bright_exterior"),
        (_photo(3840, 2560), _obs(light_quality="flat"), "flat_light"),
        (_photo(3840, 2560), None, "large_clean"),
        (_photo(640, 480), None, "small_weak"),
        # text_bearing + outdoor midday → text_bearing wins (precedence)
        (_photo(3840, 2560), _obs(contains_text=True, placement="outdoor", time_of_day_read="midday"), "text_bearing"),
    ]
    for photo, obs, expected in cases:
        name, _ = _select_recipe(photo, obs)
        w, h = photo["image_width"], photo["image_height"]
        qs = obs.get("quality_score", "N/A") if obs else "N/A"
        assert name == expected, (
            f"{w}x{h} score={qs}: expected {expected}, got {name}"
        )


# ── Test 4: structural guard — no bare literal in logic functions ────

def test_no_bare_rule_literals_in_logic():
    """The literals 2.0, 0.6, and 16 must appear ONLY inside their
    named definitions (_SMALL_MAX_MP, _WEAK_MAX_SCORE, _POLISH_MAX_MP)
    — never as bare numbers in _select_recipe, _build_recipe,
    _count_credits, or _apply_edge_stitching.

    Structural assertion — the property is "this rule is written once",
    which is a property of the source, not of behaviour. This is the
    one case where source inspection is the correct tool.

    Must fail on 8dc88ba — bare 2.0 at lines 81, 90, 107; bare 0.6
    at line 111.
    """
    import skills.enhance as mod
    source = inspect.getsource(mod)

    # Parse the module AST
    tree = ast.parse(source)

    # Find the functions we must NOT have bare literals in
    guarded_functions = {
        "_select_recipe", "_build_recipe", "_count_credits",
        "_apply_edge_stitching",
    }

    # The literal values that must be named
    rule_literals = {2.0, 0.6, 16.0, 16}

    violations = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in guarded_functions:
                for child in ast.walk(node):
                    if isinstance(child, ast.Constant) and isinstance(child.value, (int, float)):
                        if child.value in rule_literals:
                            violations.append(
                                f"Bare literal {child.value} in {node.name}() "
                                f"at line {child.lineno}. Use the named "
                                f"constant instead (_SMALL_MAX_MP, "
                                f"_WEAK_MAX_SCORE, or _POLISH_MAX_MP)."
                            )

    assert not violations, (
        "Rule literals found in logic functions — each rule must be "
        "defined once as a named constant:\n" + "\n".join(violations)
    )
