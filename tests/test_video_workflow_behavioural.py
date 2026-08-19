"""
VIDEO-WORKFLOW-1B: Behavioural tests for the video pipeline in onboard.

Every test here CALLS onboard() with all sixteen skills patched and asserts on
what actually happened — mock call records, not source text.

Erick's rulings under test:
  - A video failure must never cost a customer their page (tests 5, 6, 8, 10).
  - A video failure must raise a p0 queue row and alert erick@staylio.ai (test 7).
  - A clean run must not raise a false alarm (test 9).

Patch targets, read from workflows/onboard.py:
  - Contract helpers are imported at MODULE level (onboard.py:26) →
    patched on workflows.onboard.
  - Every skill is imported INSIDE the function → patched at source module.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, ".")

from skills.contract import SkillResult  # noqa: E402

# patch() resolves dotted targets by attribute lookup, so each submodule must be
# imported before it can be patched.
import skills.ingest_intake  # noqa: E402,F401
import skills.acquire_listing  # noqa: E402,F401
import skills.deduplicate  # noqa: E402,F401
import skills.observe  # noqa: E402,F401
import skills.enhance  # noqa: E402,F401
import skills.write_copy  # noqa: E402,F401
import skills.build_guide  # noqa: E402,F401
import skills.conceive  # noqa: E402,F401
import skills.direct  # noqa: E402,F401
import skills.narrate  # noqa: E402,F401
import skills.narrate_guest_cards  # noqa: E402,F401
import skills.score_music  # noqa: E402,F401
import skills.generate_motion  # noqa: E402,F401
import skills.assemble  # noqa: E402,F401
import skills.publish_page  # noqa: E402,F401
import skills.notify  # noqa: E402,F401


# ── Supabase stub ────────────────────────────────────────────────────────────
# onboard reads properties and intake_answers directly. Return empty data so no
# acquire URLs and no owner photos are found — those paths are not under test.

class _Resp:
    def __init__(self, data):
        self.data = data


class _Tbl:
    """Records inserts so tests can assert on the hitl_queue_items payload."""

    def __init__(self, name, recorder):
        self.name = name
        self.recorder = recorder

    def select(self, *a, **k):
        return self

    def eq(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def order(self, *a, **k):
        return self

    def execute(self):
        return _Resp([])

    def insert(self, payload):
        self.recorder.append((self.name, payload))
        return self

    def update(self, payload):
        return self


class FakeSB:
    def __init__(self, insert_raises=False):
        self.inserts = []
        self.insert_raises = insert_raises

    def table(self, name):
        if self.insert_raises and name == "hitl_queue_items":
            raise RuntimeError("simulated hitl insert failure")
        return _Tbl(name, self.inserts)


# ── Fixture: every skill patched, all succeeding ─────────────────────────────

_EARLY = [
    ("skills.ingest_intake", "ingest_intake"),
    ("skills.acquire_listing", "acquire_listing"),
    ("skills.acquire_listing", "acquire_owner_photos"),
    ("skills.deduplicate", "deduplicate"),
    ("skills.observe", "observe"),
    ("skills.enhance", "enhance"),
    ("skills.write_copy", "write_copy"),
    ("skills.build_guide", "build_guide"),
]

_VIDEO = [
    ("skills.conceive", "conceive"),
    ("skills.direct", "direct"),
    ("skills.narrate", "narrate"),
    ("skills.narrate_guest_cards", "narrate_guest_cards"),
    ("skills.score_music", "score_music"),
    ("skills.generate_motion", "generate_motion"),
    ("skills.assemble", "assemble"),
]


class Harness:
    """Patches all sixteen skills plus the contract helpers, and runs onboard."""

    def __init__(self, insert_raises=False):
        self.mocks = {}
        self.sb = FakeSB(insert_raises=insert_raises)
        self._patchers = []

    def __enter__(self):
        for mod, name in _EARLY + _VIDEO:
            m = MagicMock(return_value=SkillResult.ok({}))
            p = patch(f"{mod}.{name}", m)
            p.start()
            self._patchers.append(p)
            self.mocks[name] = m

        # conceive and direct hand ids downstream
        self.mocks["conceive"].return_value = SkillResult.ok({"concept_id": "CID-1"})
        self.mocks["direct"].return_value = SkillResult.ok({"direction_id": "DID-1"})

        pub = MagicMock(return_value=SkillResult.ok({"page_url": "https://x"}))
        p = patch("skills.publish_page.publish_page", pub)
        p.start()
        self._patchers.append(p)
        self.mocks["publish_page"] = pub

        alert = MagicMock(return_value=True)
        p = patch("skills.notify.send_halt_alert", alert)
        p.start()
        self._patchers.append(p)
        self.mocks["send_halt_alert"] = alert

        for helper, val in [
            ("get_substrate", self.sb),
            ("record_run", "RUN-1"),
            ("record_step", "STEP-1"),
            ("complete_step", None),
            ("complete_run", None),
        ]:
            p = patch(f"workflows.onboard.{helper}", MagicMock(return_value=val))
            p.start()
            self._patchers.append(p)

        return self

    def __exit__(self, *exc):
        for p in reversed(self._patchers):
            p.stop()
        return False

    def run(self):
        from workflows.onboard import onboard
        return onboard("PROP-1")

    def hitl_inserts(self):
        return [p for (t, p) in self.sb.inserts if t == "hitl_queue_items"]


# ── Test 4: ids chain from conceive → direct → the rest ──────────────────────

def test_ids_chain_through_video_steps():
    with Harness() as h:
        h.run()

        assert h.mocks["direct"].call_args.kwargs["concept_id"] == "CID-1", \
            "direct must receive the concept_id conceive returned"

        for name in ("narrate", "score_music", "generate_motion", "assemble"):
            assert h.mocks[name].call_args.kwargs["direction_id"] == "DID-1", \
                f"{name} must receive the direction_id direct returned"


# ── Test 5: assemble fails → the page is still published ─────────────────────

def test_assemble_failure_still_publishes():
    with Harness() as h:
        h.mocks["assemble"].return_value = SkillResult.failed(reason="render exploded")
        result = h.run()

        assert h.mocks["publish_page"].called is True, \
            "a video failure must never cost a customer their page"
        assert result.is_ok, "onboard must not fail because assembly failed"


# ── Test 6: conceive fails → later steps skipped, page still published ───────

def test_conceive_failure_skips_video_but_publishes():
    with Harness() as h:
        h.mocks["conceive"].return_value = SkillResult.failed(reason="no premise")
        h.run()

        for name in ("direct", "narrate", "score_music", "generate_motion", "assemble"):
            assert h.mocks[name].called is False, \
                f"{name} must not run when conceive failed"

        assert h.mocks["publish_page"].called is True


# ── Test 7: a video failure raises a p0 queue row and alerts ─────────────────

def test_video_failure_raises_queue_row_and_alert():
    with Harness() as h:
        h.mocks["generate_motion"].return_value = SkillResult.failed(reason="runway 500")
        h.run()

        rows = h.hitl_inserts()
        assert len(rows) == 1, f"expected exactly one hitl row, got {len(rows)}"

        row = rows[0]
        assert row["queue_type"] == "pipeline_failure"
        assert row["priority"] == "p0"
        assert row["status"] == "open"
        assert row["created_by_type"] == "system"
        assert "CRITICAL" in row["title"]

        assert h.mocks["send_halt_alert"].called is True
        detail = h.mocks["send_halt_alert"].call_args.kwargs.get("detail", "")
        assert "without a hero video" in detail


# ── Test 8: the alert itself fails → the page is still published ─────────────

def test_alert_failure_still_publishes():
    with Harness() as h:
        h.mocks["assemble"].return_value = SkillResult.failed(reason="boom")
        h.mocks["send_halt_alert"].side_effect = RuntimeError("resend down")
        h.run()

        assert h.mocks["publish_page"].called is True, \
            "an alerting fault must never cost a customer their page"


# ── Test 9: everything succeeds → no queue row, no alert ─────────────────────

def test_clean_run_raises_no_alert():
    with Harness() as h:
        h.run()

        assert h.hitl_inserts() == [], "a clean run must not raise a queue row"
        assert h.mocks["send_halt_alert"].called is False, \
            "a clean run must not send a false alarm"


# ── Test 10: the queue insert fails → the page is still published ────────────
# onboard.py wraps the hitl insert AND send_halt_alert in one try/except, so an
# insert failure silently suppresses the alert. Publishing must survive it.

def test_queue_insert_failure_still_publishes():
    with Harness(insert_raises=True) as h:
        h.mocks["assemble"].return_value = SkillResult.failed(reason="boom")
        h.run()

        assert h.mocks["publish_page"].called is True, \
            "a failed queue insert must never cost a customer their page"
        assert h.mocks["send_halt_alert"].called is False, \
            "documents current behaviour: an insert failure suppresses the alert"
