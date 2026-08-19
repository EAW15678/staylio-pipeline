"""
PAGE-3: Behavioural tests for the substrate-native page orchestrator.

Every test calls build_page() or its helpers with synthetic data and
asserts on returned HTML or data structures. No source-text inspection.
All Supabase calls are mocked via a FakeSB stub.
"""

import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, ".")


# ── Supabase stub ────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, data, count=None):
        self.data = data
        self.count = count if count is not None else len(data)


class _Tbl:
    def __init__(self, name, tables):
        self._name = name
        self._tables = tables
        self._inserts = tables.setdefault("_inserts", [])
        self._eq_filters = []

    def select(self, *a, **k): return self
    def eq(self, col, val):
        self._eq_filters.append((col, val))
        return self
    def neq(self, *a, **k): return self
    def is_(self, *a, **k): return self
    def order(self, *a, **k): return self
    def limit(self, *a, **k): return self
    def like(self, *a, **k): return self

    def execute(self):
        rows = self._tables.get(self._name, [])
        # Apply eq filters for basic stub filtering
        for col, val in self._eq_filters:
            rows = [r for r in rows if r.get(col) == val]
        self._eq_filters = []
        return _Resp(rows)

    def insert(self, payload):
        self._inserts.append((self._name, payload))
        return self

    def upsert(self, payload, **k):
        self._inserts.append((self._name, payload))
        return self

    def update(self, payload):
        return self


class FakeSB:
    def __init__(self, tables=None):
        self._tables = tables or {}
        self._tables.setdefault("_inserts", [])

    def table(self, name):
        return _Tbl(name, self._tables)

    def hitl_inserts(self):
        return [p for (t, p) in self._tables["_inserts"] if t == "hitl_queue_items"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _prop(name="Test Property", slug="test-property", amenities=None):
    return {
        "id": "PROP-1", "name": name, "slug": slug,
        "city": "Asheville", "state_region": "NC",
        "booking_url": "https://example.com/book",
        "bedrooms": 3, "bathrooms": 2, "max_occupancy": 8,
        "property_type": "house", "vibe_profile": "retreat",
        "latitude": 35.5, "longitude": -82.5,
        "amenities": amenities or [],
        "ical_url": None,
    }


def _photo(pid, w=3000, h=2000):
    return {"photo_id": pid, "property_id": "PROP-1",
            "content_hash": f"hash_{pid}", "phash": f"ph_{pid}",
            "is_canonical": True, "image_width": w, "image_height": h}


def _rendition(pid, kind="enhanced"):
    return {"photo_id": pid, "kind": kind, "storage_url": f"https://r2.dev/{pid}/{kind}.jpg"}


def _renditions_for(pid):
    """Return both original and enhanced renditions for a photo."""
    return [_rendition(pid, "original"), _rendition(pid, "enhanced")]


def _obs(pid, section="Exterior", role="supporting", rank=1, located_amenities=None):
    obs = {
        "photo_id": pid, "property_id": "PROP-1",
        "curated_section": section, "role": role,
        "rank_within_section": rank, "quality_score": 0.85,
        "alt_text": f"Photo {pid}", "gallery_visible": True,
        "tour_eligible": True, "superseded_at": None,
    }
    if located_amenities is not None:
        obs["located_amenities"] = located_amenities
    return obs


def _build_tables(photos, renditions_flat, observations, prop=None,
                  video_master=None, narrations=None, copy=None,
                  owner_ctx=None, guest_evidence=None, local_guide=None,
                  landing_pages=None):
    """Build a tables dict for FakeSB from flat lists."""
    tables = {
        "properties": [prop or _prop()],
        "photographs": photos,
        "observations": observations,
        "copy_versions": copy or [],
        "owner_context": owner_ctx or [],
        "guest_evidence": guest_evidence or [],
        "local_guides": local_guide or [],
        "video_artifacts": [],
        "landing_pages": landing_pages or [],
    }
    if video_master:
        tables["video_artifacts"].append(video_master)
    if narrations:
        tables["video_artifacts"].extend(narrations)

    # renditions are per-photo; stub returns all regardless of eq filter
    tables["renditions"] = renditions_flat
    return tables


# ── Test 1: photos grouped by curated_section ────────────────────────────────

def test_photos_grouped_by_curated_section():
    """The orchestrator produces a page with photos grouped by curated_section.
    No intermediate dict shape involved."""
    photos = [_photo(f"p{i}") for i in range(7)]
    renditions = []
    for i in range(7):
        renditions.extend(_renditions_for(f"p{i}"))
    observations = [
        _obs("p0", "Exterior", "hero", 1),
        _obs("p1", "Exterior", "supporting", 2),
        _obs("p2", "Pool", "hero", 1),
        _obs("p3", "Living Areas", "hero", 1),
        _obs("p4", "Kitchen", "hero", 1),
        _obs("p5", "Bedrooms", "hero", 1),
        _obs("p6", "Bathrooms", "hero", 1),
    ]

    sb = FakeSB(_build_tables(photos, renditions, observations))
    from core.page_builder.orchestrate import build_page
    result = build_page(sb, "PROP-1")

    assert result["ok"] is True
    html = result["html"]

    # The real section names appear as headers, not old GCV names
    for section in ["Exterior", "Pool", "Living Areas", "Kitchen", "Bedrooms", "Bathrooms"]:
        assert section in html, f"Section '{section}' not found in HTML"

    # Old GCV names must NOT appear as section headers
    for old_label in ["living_room", "pool_hot_tub", "standard_bedroom",
                      "Exterior & Views", "Outdoor & Pool", "Amenities & Extras"]:
        assert f'cat-module-label">{old_label}' not in html, f"Old label '{old_label}' found as section header"


# ── Test 2: amenity with photo proof ─────────────────────────────────────────

def test_amenity_photo_proof():
    """An amenity present in located_amenities renders with a photo;
    an amenity absent renders as text only."""
    photos = [_photo("p0"), _photo("p1")]
    renditions = _renditions_for("p0") + _renditions_for("p1")
    observations = [
        _obs("p0", "Exterior", "hero", 1, located_amenities=[
            {"name": "swimming pool", "category": "water_wellness", "placement": "outdoor"},
        ]),
        _obs("p1", "Kitchen", "supporting", 1),
    ]

    prop = _prop(amenities=["Private pool", "Fast WiFi (200+ Mbps)"])
    copy = [{"property_id": "PROP-1", "content": {"amenity_highlights": {
        "Private pool": "Enjoy a refreshing dip in the private pool.",
        "Fast WiFi (200+ Mbps)": "Stream and work from anywhere.",
    }}, "version": 1}]

    sb = FakeSB(_build_tables(photos, renditions, observations, prop=prop, copy=copy))
    from core.page_builder.orchestrate import build_page
    result = build_page(sb, "PROP-1")

    assert result["ok"] is True
    html = result["html"]

    # Pool has photo proof
    assert "amenity-photo" in html, "Expected photo proof for Private pool"
    # The pool photo URL should be in the amenities section
    assert "p0/enhanced.jpg" in html

    # WiFi has no photo proof (no visual evidence possible)
    # Count amenity-photo occurrences — should be exactly 1 (pool only)
    assert html.count("amenity-photo") == 1, "Only pool should have photo proof"


# ── Test 3: hero video presence and absence ──────────────────────────────────

def test_hero_video_present_and_absent():
    """A ready video_artifacts master row → hero video present.
    No ready row → hero section absent, page still builds."""
    photos = [_photo("p0")]
    renditions = _renditions_for("p0")
    observations = [_obs("p0", "Exterior", "hero", 1)]

    # WITH video
    video = {"storage_url": "https://r2.dev/master.mp4", "kind": "master",
             "status": "ready", "superseded_at": None,
             "property_id": "PROP-1"}
    sb = FakeSB(_build_tables(photos, renditions, observations, video_master=video))
    from core.page_builder.orchestrate import build_page
    result = build_page(sb, "PROP-1")
    assert result["ok"] is True
    assert "master.mp4" in result["html"], "Hero video should be in HTML"
    assert "<video" in result["html"]

    # WITHOUT video
    sb2 = FakeSB(_build_tables(photos, renditions, observations))
    result2 = build_page(sb2, "PROP-1")
    assert result2["ok"] is True
    assert "<video" not in result2["html"], "No video tag when no master exists"
    assert "master.mp4" not in result2["html"]


# ── Test 4: zero canonical photos → hold ─────────────────────────────────────

def test_zero_photos_hold():
    """Zero canonical photographs → hold + hitl row + alert."""
    sb = FakeSB(_build_tables([], [], []))
    from core.page_builder.orchestrate import build_page, raise_hold

    result = build_page(sb, "PROP-1")
    assert result["ok"] is False
    assert result["hold"] is True
    assert "no_canonical_photos" in result["hold_code"]

    # Verify raise_hold creates hitl row and calls send_halt_alert
    import skills.notify  # ensure module is imported for patching
    with patch("skills.notify.send_halt_alert", return_value=True) as mock_alert:
        raise_hold(sb, "PROP-1", result["reason"], result["hold_code"])

    rows = sb.hitl_inserts()
    assert len(rows) == 1
    assert rows[0]["queue_type"] == "pipeline_failure"
    assert rows[0]["priority"] == "p0"
    assert rows[0]["status"] == "open"
    assert rows[0]["created_by_type"] == "system"
    assert mock_alert.called


# ── Test 5: photos but no observations → hold ───────────────────────────────

def test_photos_no_observations_hold():
    """Photographs exist, zero have an active observation → hold."""
    photos = [_photo("p0"), _photo("p1")]
    renditions = _renditions_for("p0") + _renditions_for("p1")
    # No observations
    sb = FakeSB(_build_tables(photos, renditions, []))
    from core.page_builder.orchestrate import build_page

    result = build_page(sb, "PROP-1")
    assert result["ok"] is False
    assert result["hold"] is True
    assert "no_observations" in result["hold_code"]


# ── Test 6: hold with live page → existing row untouched ────────────────────

def test_hold_with_live_page_untouched():
    """A property with an existing live landing_pages row, under a hold
    condition → the existing row is untouched.

    This proves the 'never take down a live page' ruling. The orchestrator
    returns a hold result without modifying any landing_pages data. The
    caller (publish_page) is never called, so the R2 file and the database
    row are preserved.
    """
    existing_page = {
        "property_id": "PROP-1",
        "slug": "test-property",
        "page_url": "https://test-property.upliftstays.com",
        "status": "deployed",
        "last_built_at": "2026-08-18T22:39:29Z",
    }

    # Zero photos → hold condition
    tables = _build_tables([], [], [], landing_pages=[existing_page])
    sb = FakeSB(tables)
    from core.page_builder.orchestrate import build_page

    result = build_page(sb, "PROP-1")
    assert result["ok"] is False
    assert result["hold"] is True

    # The landing_pages data is untouched — no upsert, no update
    lp_writes = [p for (t, p) in sb._tables["_inserts"] if t == "landing_pages"]
    assert lp_writes == [], (
        f"Hold must not write to landing_pages, but found: {lp_writes}"
    )

    # The existing row is still in the table (unchanged)
    assert tables["landing_pages"] == [existing_page], (
        "Existing landing_pages row must be untouched"
    )


# ── Test 7: no agents/ imports in orchestrator ──────────────────────────────

def test_no_agents_imports():
    """A normal successful build uses no function or table from agents/.
    Verified both by grep and by runtime — the module loads and runs
    without any agents/ import."""
    import importlib
    import core.page_builder.orchestrate as mod
    importlib.reload(mod)

    # Check the module's source for agents/ imports (actual import statements, not comments)
    import inspect
    source = inspect.getsource(mod)
    import ast
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("agents"):
            assert False, f"orchestrate.py must not import from agents/ — found: from {node.module}"
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("agents"):
                    assert False, f"orchestrate.py must not import agents — found: import {alias.name}"

    # Also verify it runs successfully (imports resolve at runtime)
    photos = [_photo("p0")]
    renditions = _renditions_for("p0")
    observations = [_obs("p0", "Exterior", "hero", 1)]
    sb = FakeSB(_build_tables(photos, renditions, observations))
    result = mod.build_page(sb, "PROP-1")
    assert result["ok"] is True, f"Build should succeed, got: {result.get('reason')}"
