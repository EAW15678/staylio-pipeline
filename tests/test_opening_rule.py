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
    assert "may never lead" in v[0]["detail"]


def test_bathroom_opener_fails():
    obs_map = {"photo-A": OBS_BATHROOM}
    v = validate_opening_establishes(_direction("photo-A", "feature"), obs_map)
    assert len(v) >= 1
    assert "may never lead" in v[0]["detail"]


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


def test_kitchen_indoor_as_opener_fails():
    """Indoor kitchen cannot open."""
    obs_map = {"photo-A": {**OBS_KITCHEN, "placement": "indoor"}}
    v = validate_opening_establishes(_direction("photo-A", "feature"), obs_map)
    assert len(v) >= 1
    assert "only outdoor" in v[0]["detail"]


def test_living_areas_indoor_as_opener_fails():
    """Indoor living areas cannot open."""
    obs_map = {"photo-A": {"curated_section": "Living Areas", "is_setting": False, "placement": "indoor"}}
    v = validate_opening_establishes(_direction("photo-A", "feature"), obs_map)
    assert len(v) >= 1


def test_extras_indoor_as_opener_fails():
    """Indoor Extras cannot open."""
    obs_map = {"photo-A": {"curated_section": "Extras", "is_setting": False, "placement": "indoor"}}
    v = validate_opening_establishes(_direction("photo-A", "feature"), obs_map)
    assert len(v) >= 1


def test_extras_unknown_as_opener_fails():
    """Extras with placement='unknown' cannot open."""
    obs_map = {"photo-A": {"curated_section": "Extras", "is_setting": False, "placement": "unknown"}}
    v = validate_opening_establishes(_direction("photo-A", "feature"), obs_map)
    assert len(v) >= 1


def test_extras_outdoor_as_opener_passes():
    """Extras with placement='outdoor' CAN open as an exterior feature."""
    obs_map = {"photo-A": {"curated_section": "Extras", "is_setting": False, "placement": "outdoor"}}
    v = validate_opening_establishes(_direction("photo-A", "feature"), obs_map)
    assert v == []


def test_bedroom_outdoor_still_fails():
    """Bedrooms NEVER open regardless of placement."""
    obs_map = {"photo-A": {**OBS_BEDROOM, "placement": "outdoor"}}
    v = validate_opening_establishes(_direction("photo-A", "property"), obs_map)
    assert len(v) >= 1


# ── Interior sections at beats 2+ → PASS (rule only governs beat 1) ─────

def test_kitchen_at_beat2_passes():
    obs_map = {"photo-A": OBS_EXTERIOR, "photo-B": OBS_KITCHEN}
    d = {"beats": [
        {"ordinal": 1, "photo_id": "photo-A", "requested_motion": "push_in", "duration_seconds": 4},
        {"ordinal": 2, "photo_id": "photo-B", "requested_motion": "pan_left", "duration_seconds": 4},
    ], "opening_type": "property"}
    v = validate_opening_establishes(d, obs_map)
    assert v == []


def test_living_areas_at_beat3_passes():
    obs_map = {"photo-A": OBS_EXTERIOR, "photo-C": {"curated_section": "Living Areas"}}
    d = {"beats": [
        {"ordinal": 1, "photo_id": "photo-A", "requested_motion": "push_in", "duration_seconds": 4},
        {"ordinal": 3, "photo_id": "photo-C", "requested_motion": "pan_right", "duration_seconds": 4},
    ], "opening_type": "property"}
    v = validate_opening_establishes(d, obs_map)
    assert v == []


def test_bedroom_at_beat4_passes():
    obs_map = {"photo-A": OBS_POOL, "photo-D": OBS_BEDROOM}
    d = {"beats": [
        {"ordinal": 1, "photo_id": "photo-A", "requested_motion": "push_in", "duration_seconds": 4},
        {"ordinal": 4, "photo_id": "photo-D", "requested_motion": "pan_left", "duration_seconds": 3},
    ], "opening_type": "feature"}
    v = validate_opening_establishes(d, obs_map)
    assert v == []


def test_bathroom_at_beat5_passes():
    obs_map = {"photo-A": OBS_SETTING, "photo-E": OBS_BATHROOM}
    d = {"beats": [
        {"ordinal": 1, "photo_id": "photo-A", "requested_motion": "push_in", "duration_seconds": 4},
        {"ordinal": 5, "photo_id": "photo-E", "requested_motion": "tilt_up", "duration_seconds": 3},
    ], "opening_type": "setting"}
    v = validate_opening_establishes(d, obs_map)
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
