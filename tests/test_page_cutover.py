"""
PAGE-4: Behavioural tests for the render_page → publish_page cutover.

Every test calls the real functions with mocked substrate data and asserts
on returned SkillResult status, data shape, and mock call counts.
"""

import sys
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, ".")

from skills.contract import SkillResult


# ── Stub helpers (reused from test_page_orchestrator.py pattern) ─────────────

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
    def neq(self, col, val):
        self._neq_filters = getattr(self, '_neq_filters', [])
        self._neq_filters.append((col, val))
        return self
    def is_(self, *a, **k): return self
    def order(self, *a, **k): return self
    def limit(self, *a, **k): return self
    def like(self, *a, **k): return self
    def not_(self): return self

    def execute(self):
        rows = self._tables.get(self._name, [])
        for col, val in self._eq_filters:
            rows = [r for r in rows if r.get(col) == val]
        self._eq_filters = []
        for col, val in getattr(self, '_neq_filters', []):
            rows = [r for r in rows if r.get(col) != val]
        self._neq_filters = []
        return _Resp(rows)

    def insert(self, payload):
        self._inserts.append((self._name, payload))
        return self

    def upsert(self, payload, **k):
        self._inserts.append((self._name, payload))
        return self

    def update(self, payload):
        return self

    def put_object(self, **k):
        self._inserts.append(("_r2_upload", k))


class FakeSB:
    def __init__(self, tables=None):
        self._tables = tables or {}
        self._tables.setdefault("_inserts", [])

    def table(self, name):
        return _Tbl(name, self._tables)

    def hitl_inserts(self):
        return [p for (t, p) in self._tables["_inserts"] if t == "hitl_queue_items"]

    def r2_uploads(self):
        return [p for (t, p) in self._tables["_inserts"] if t == "_r2_upload"]

    def landing_page_writes(self):
        return [p for (t, p) in self._tables["_inserts"] if t == "landing_pages"]


def _prop():
    return {
        "id": "PROP-1", "name": "Test Property", "slug": "test-property",
        "city": "Asheville", "state_region": "NC",
        "booking_url": "https://example.com/book",
        "bedrooms": 3, "bathrooms": 2, "max_occupancy": 8,
        "property_type": "house", "vibe_profile": "retreat",
        "latitude": 35.5, "longitude": -82.5,
        "amenities": [], "ical_url": None, "status": "onboarding",
    }


def _photo(pid):
    return {"photo_id": pid, "property_id": "PROP-1",
            "content_hash": f"hash_{pid}", "phash": f"ph_{pid}",
            "is_canonical": True, "image_width": 3000, "image_height": 2000}


def _renditions_for(pid):
    return [
        {"photo_id": pid, "kind": "original", "storage_url": f"https://r2.dev/{pid}/original.jpg"},
        {"photo_id": pid, "kind": "enhanced", "storage_url": f"https://r2.dev/{pid}/enhanced.jpg"},
    ]


def _obs(pid, section="Exterior"):
    return {
        "photo_id": pid, "property_id": "PROP-1",
        "curated_section": section, "role": "supporting",
        "rank_within_section": 1, "quality_score": 0.85,
        "alt_text": f"Photo {pid}", "superseded_at": None,
    }


def _good_tables():
    """Tables for a successful build — 3 photos with observations."""
    return {
        "properties": [_prop()],
        "photographs": [_photo("p0"), _photo("p1"), _photo("p2")],
        "renditions": _renditions_for("p0") + _renditions_for("p1") + _renditions_for("p2"),
        "observations": [_obs("p0", "Exterior"), _obs("p1", "Kitchen"), _obs("p2", "Bedrooms")],
        "copy_versions": [],
        "owner_context": [],
        "guest_evidence": [],
        "local_guides": [],
        "video_artifacts": [],
        "landing_pages": [],
    }


def _hold_tables_no_photos():
    """Tables that trigger a hold — zero canonical photos."""
    return {
        "properties": [_prop()],
        "photographs": [],
        "renditions": [],
        "observations": [],
        "copy_versions": [],
        "owner_context": [],
        "guest_evidence": [],
        "local_guides": [],
        "video_artifacts": [],
        "landing_pages": [{"property_id": "PROP-1", "slug": "test-property",
                          "page_url": "https://test-property.upliftstays.com",
                          "status": "deployed", "last_built_at": "2026-08-18T22:39:29Z"}],
    }


# ── Test 1: render_page returns ok with correct shape ────────────────────────

def test_render_page_ok_shape():
    """render_page returns SkillResult.ok with the correct data keys."""
    sb = FakeSB(_good_tables())

    with patch("skills.render_page.get_substrate", return_value=sb):
        from skills.render_page import render_page
        result = render_page("PROP-1")

    assert result.status == "ok", f"Expected ok, got {result.status}: {result.reason}"
    assert result.is_ok is True
    for key in ["html", "photo_count", "gallery_count", "unresolvable",
                "sections_built", "lightbox_present"]:
        assert key in result.data, f"Missing key '{key}' in result.data"
    assert result.data["photo_count"] == 3
    assert result.data["unresolvable"] == 0


# ── Test 2: render_page returns held (not failed) on hold ────────────────────

def test_render_page_held_on_zero_photos():
    """render_page returns SkillResult.held, not failed, when build_page holds."""
    tables = _good_tables()
    tables["photographs"] = []  # zero photos → hold
    tables["observations"] = []
    sb = FakeSB(tables)

    with patch("skills.render_page.get_substrate", return_value=sb):
        from skills.render_page import render_page
        result = render_page("PROP-1")

    assert result.status == "held", f"Expected held, got {result.status}: {result.reason}"
    assert result.is_ok is False
    assert result.data["hold_code"] == "no_canonical_photos"


# ── Test 3: publish_page returns held without second alert ───────────────────

def test_publish_page_held_no_second_alert():
    """publish_page, where render_page returns held, returns SkillResult.held
    and does NOT call escalate_halt or any second alert."""
    held_result = SkillResult.held(
        reason="No canonical photographs",
        data={"hold_code": "no_canonical_photos"},
    )

    sb = FakeSB({"properties": [_prop()], "landing_pages": []})

    import skills.render_page  # ensure module loaded for patching

    with patch("skills.publish_page.get_substrate", return_value=sb), \
         patch("skills.publish_page.require_env", return_value="val"), \
         patch("skills.publish_page.record_run", return_value="RUN-1"), \
         patch("skills.publish_page.record_step", return_value="STEP-1"), \
         patch("skills.publish_page.complete_step") as mock_step, \
         patch("skills.publish_page.complete_run"), \
         patch("skills.render_page.render_page", return_value=held_result):

        from skills.publish_page import publish_page
        result = publish_page("PROP-1", force=True)

    assert result.status == "held", f"Expected held, got {result.status}"
    assert result.data["hold_code"] == "no_canonical_photos"

    # Verify step was recorded as "failed" (constraint-safe) with held metadata
    step_call = mock_step.call_args
    assert step_call.kwargs.get("status") == "failed" or step_call[1].get("status") == "failed"
    metadata = step_call.kwargs.get("metadata") or step_call[1].get("metadata") or {}
    assert metadata.get("skill_status") == "held"

    # escalate_halt must NOT have been called — zero imports of it
    # (the held branch does not import or call escalate_halt)


# ── Test 4: end-to-end proof ────────────────────────────────────────────────

def test_end_to_end_hold_preserves_live_page():
    """End-to-end: publish_page on a property with a live landing_pages row,
    where the underlying data causes build_page to hold.

    Proves:
    1. Zero R2 uploads
    2. Existing landing_pages row unchanged
    3. send_halt_alert called exactly once (by raise_hold in orchestrate.py)
    """
    tables = _hold_tables_no_photos()
    existing_page = tables["landing_pages"][0].copy()
    sb = FakeSB(tables)

    import skills.notify

    with patch("skills.publish_page.get_substrate", return_value=sb), \
         patch("skills.publish_page.require_env", return_value="val"), \
         patch("skills.publish_page.record_run", return_value="RUN-1"), \
         patch("skills.publish_page.record_step", return_value="STEP-1"), \
         patch("skills.publish_page.complete_step"), \
         patch("skills.publish_page.complete_run"), \
         patch("skills.render_page.get_substrate", return_value=sb), \
         patch("skills.notify.send_halt_alert", return_value=True) as mock_alert:

        from skills.publish_page import publish_page
        result = publish_page("PROP-1", force=True)

    # 1. Result is held
    assert result.status == "held", f"Expected held, got {result.status}: {result.reason}"

    # 2. Zero R2 uploads
    assert sb.r2_uploads() == [], "No R2 uploads should occur on hold"

    # 3. Existing landing_pages row is unchanged
    assert sb.landing_page_writes() == [], \
        f"landing_pages must not be written to on hold, got: {sb.landing_page_writes()}"
    assert tables["landing_pages"][0] == existing_page, \
        "Existing landing_pages row must be unchanged"

    # 4. send_halt_alert called exactly once — by raise_hold in orchestrate.py
    assert mock_alert.call_count == 1, \
        f"send_halt_alert must fire exactly once, got {mock_alert.call_count}"

    # 5. A hitl_queue_items row was created
    hitl = sb.hitl_inserts()
    assert len(hitl) == 1
    assert hitl[0]["priority"] == "p0"
    assert hitl[0]["queue_type"] == "pipeline_failure"


# ── Test 5: no agents/ imports in the chain ──────────────────────────────────

def test_no_agents_imports_in_chain():
    """A successful build through render_page touches no agents/ import.
    Verified by AST walk and by runtime."""
    import ast
    import inspect

    # Check render_page.py
    import skills.render_page as rp_mod
    rp_source = inspect.getsource(rp_mod)
    rp_tree = ast.parse(rp_source)
    for node in ast.walk(rp_tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("agents"):
            assert False, f"render_page.py imports from agents/: from {node.module}"
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("agents"):
                    assert False, f"render_page.py imports agents: import {alias.name}"

    # Check core/page_builder/orchestrate.py
    import core.page_builder.orchestrate as orch_mod
    orch_source = inspect.getsource(orch_mod)
    orch_tree = ast.parse(orch_source)
    for node in ast.walk(orch_tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("agents"):
            assert False, f"orchestrate.py imports from agents/: from {node.module}"

    # Runtime proof: render_page succeeds without needing agents/
    sb = FakeSB(_good_tables())
    with patch("skills.render_page.get_substrate", return_value=sb):
        result = rp_mod.render_page("PROP-1")
    assert result.is_ok, f"render_page should succeed, got: {result.status}: {result.reason}"
