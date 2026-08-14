"""
DIRECTOR-1: Opening rule validator tests.

All deterministic — no vendor calls, $0.
"""

from skills.direct import validate_opening_establishes, WEAK_OPENER_WIDTH_THRESHOLD


# ── Fixtures ─────────────────────────────────────────────────────────────

OBS_EXTERIOR = {
    "curated_section": "Exterior",
    "is_setting": False,
    "depth_tier": "wide",
    "subject_singularity": "single",
}

OBS_SETTING = {
    "curated_section": "Exterior",
    "is_setting": True,
    "setting_subject": "Atlantic Ocean beach",
    "depth_tier": "wide",
    "subject_singularity": "dual",
}

OBS_POOL = {
    "curated_section": "Pool",
    "is_setting": False,
    "depth_tier": "wide",
    "subject_singularity": "dual",
}

OBS_BEDROOM = {
    "curated_section": "Bedrooms",
    "is_setting": False,
    "depth_tier": "medium",
    "subject_singularity": "single",
}

OBS_BATHROOM = {
    "curated_section": "Bathrooms",
    "is_setting": False,
    "depth_tier": "medium",
    "subject_singularity": "single",
}

OBS_KITCHEN = {
    "curated_section": "Kitchen",
    "is_setting": False,
    "depth_tier": "wide",
    "subject_singularity": "single",
}


def _direction(photo_id, opening_type, motion="push_in"):
    return {
        "beats": [{"ordinal": 1, "photo_id": photo_id, "requested_motion": motion, "duration_seconds": 4}],
        "opening_type": opening_type,
    }


# ── Setting declared on is_setting=false → FAIL ─────────────────────────

def test_setting_declared_but_not_setting_fails():
    obs_map = {"photo-A": OBS_EXTERIOR}
    v = validate_opening_establishes(_direction("photo-A", "setting"), obs_map)
    assert len(v) >= 1
    assert v[0]["rule"] == "opening_establishes"
    assert "is_setting=false" in v[0]["detail"]


def test_setting_declared_on_setting_frame_passes():
    obs_map = {"photo-A": OBS_SETTING}
    v = validate_opening_establishes(_direction("photo-A", "setting"), obs_map)
    assert v == []


# ── Beat 1 on Bedrooms → FAIL regardless of declaration ─────────────────

def test_bedroom_opener_fails():
    obs_map = {"photo-A": OBS_BEDROOM}
    v = validate_opening_establishes(_direction("photo-A", "property"), obs_map)
    assert len(v) >= 1
    assert "interior detail" in v[0]["detail"]


def test_bathroom_opener_fails():
    obs_map = {"photo-A": OBS_BATHROOM}
    v = validate_opening_establishes(_direction("photo-A", "feature"), obs_map)
    assert len(v) >= 1
    assert "interior detail" in v[0]["detail"]


# ── Weak opener: parallax → FAIL; push_in → PASS ────────────────────────

def test_weak_opener_parallax_fails():
    obs_map = {"photo-A": OBS_EXTERIOR}
    widths = {"photo-A": 576}
    v = validate_opening_establishes(
        _direction("photo-A", "property", motion="parallax"),
        obs_map, widths,
    )
    assert len(v) >= 1
    assert "push_in" in v[0]["detail"]


def test_weak_opener_push_in_passes():
    obs_map = {"photo-A": OBS_EXTERIOR}
    widths = {"photo-A": 576}
    v = validate_opening_establishes(
        _direction("photo-A", "property", motion="push_in"),
        obs_map, widths,
    )
    assert v == []


def test_strong_opener_parallax_passes():
    obs_map = {"photo-A": OBS_EXTERIOR}
    widths = {"photo-A": 1024}
    v = validate_opening_establishes(
        _direction("photo-A", "property", motion="parallax"),
        obs_map, widths,
    )
    assert v == []


# ── Valid openers of all three kinds, in several orders → ALL PASS ───────

def test_property_opener_passes():
    obs_map = {"photo-A": OBS_EXTERIOR}
    v = validate_opening_establishes(_direction("photo-A", "property"), obs_map)
    assert v == []


def test_setting_opener_passes():
    obs_map = {"photo-A": OBS_SETTING}
    v = validate_opening_establishes(_direction("photo-A", "setting"), obs_map)
    assert v == []


def test_feature_opener_passes():
    obs_map = {"photo-A": OBS_POOL}
    v = validate_opening_establishes(_direction("photo-A", "feature"), obs_map)
    assert v == []


def test_kitchen_as_feature_passes():
    """Kitchen is NOT a detail section — it can be a standout feature."""
    obs_map = {"photo-A": OBS_KITCHEN}
    v = validate_opening_establishes(_direction("photo-A", "feature"), obs_map)
    assert v == []


# ── Missing opening_type → FAIL ──────────────────────────────────────────

def test_missing_opening_type_fails():
    obs_map = {"photo-A": OBS_EXTERIOR}
    direction = {"beats": [{"ordinal": 1, "photo_id": "photo-A", "requested_motion": "push_in"}]}
    # No opening_type key
    v = validate_opening_establishes(direction, obs_map)
    assert len(v) >= 1
    assert "missing opening_type" in v[0]["detail"]


# ── Order is NOT enforced — different valid openers in any position ──────

def test_order_not_enforced_property_then_setting():
    """Property as beat 1, setting as beat 2 — both valid orders."""
    obs_map = {"photo-A": OBS_EXTERIOR, "photo-B": OBS_SETTING}
    d = {"beats": [
        {"ordinal": 1, "photo_id": "photo-A", "requested_motion": "push_in"},
        {"ordinal": 2, "photo_id": "photo-B", "requested_motion": "pan_left"},
    ], "opening_type": "property"}
    v = validate_opening_establishes(d, obs_map)
    assert v == []


def test_order_not_enforced_setting_then_property():
    """Setting as beat 1, property later — also valid."""
    obs_map = {"photo-A": OBS_SETTING, "photo-B": OBS_EXTERIOR}
    d = {"beats": [
        {"ordinal": 1, "photo_id": "photo-A", "requested_motion": "push_in"},
        {"ordinal": 2, "photo_id": "photo-B", "requested_motion": "pan_right"},
    ], "opening_type": "setting"}
    v = validate_opening_establishes(d, obs_map)
    assert v == []
