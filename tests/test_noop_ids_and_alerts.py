"""
B1 FIX: Tests for noop-blind ID extraction and broken alert inserts.

Bug class 1: conceive/direct noop responses must carry concept_id/direction_id.
Bug class 2: all hitl_queue_items inserts must have valid priority, status,
queue_type, and a real account_id.

Every test must fail on the pre-fix commit.
"""

import sys
import json
from unittest.mock import MagicMock, patch

sys.path.insert(0, ".")

from skills.contract import SkillResult


# ── Test 1: conceive noop carries concept_id ────────────────────────────────

def test_conceive_noop_carries_concept_id():
    """conceive with an existing active concept and force=False returns noop
    with a real concept_id matching the actual row."""
    from skills.conceive import conceive

    sb = MagicMock()
    # Existing active concept — returns a real row with concept_id
    sb.table.return_value.select.return_value.eq.return_value.eq.return_value.is_.return_value.limit.return_value.execute.return_value = MagicMock(
        data=[{"concept_id": "EXISTING-CID-123"}]
    )

    with patch("skills.conceive.get_substrate", return_value=sb):
        result = conceive("test-prop", force=False)

    assert result.status == "ok", f"Expected noop (status=ok), got {result.status}"
    assert result.data.get("noop") is True, "Should be a noop"
    assert result.data.get("concept_id") == "EXISTING-CID-123", (
        f"Noop must carry concept_id, got: {result.data}"
    )


# ── Test 2: direct noop carries direction_id ────────────────────────────────

def test_direct_noop_carries_direction_id():
    """direct with an existing direction and force=False returns noop
    with a real direction_id."""
    from skills.direct import direct

    sb = MagicMock()
    # get_substrate
    # Existing direction — returns a real row with direction_id
    sb.table.return_value.select.return_value.eq.return_value.is_.return_value.limit.return_value.execute.return_value = MagicMock(
        data=[{"direction_id": "EXISTING-DID-456"}]
    )

    with patch("skills.direct.get_substrate", return_value=sb), \
         patch("skills.direct.require_env", return_value="key"):
        result = direct("test-prop", concept_id="CID-1", force=False)

    assert result.status == "ok", f"Expected noop (status=ok), got {result.status}"
    assert result.data.get("noop") is True, "Should be a noop"
    assert result.data.get("direction_id") == "EXISTING-DID-456", (
        f"Noop must carry direction_id, got: {result.data}"
    )


# ── Test 3: onboard conceive-noop → direct gets correct concept_id ──────────

def test_onboard_conceive_noop_chains_concept_id():
    """A full onboard() run where conceive noops does not raise — direct is
    called with the correct concept_id from the noop."""
    import skills.ingest_intake
    import skills.acquire_listing
    import skills.deduplicate
    import skills.observe
    import skills.enhance
    import skills.write_copy
    import skills.build_guide
    import skills.conceive
    import skills.direct
    import skills.narrate
    import skills.narrate_guest_cards
    import skills.score_music
    import skills.generate_motion
    import skills.assemble
    import skills.publish_page
    import skills.notify

    # Conceive returns noop WITH concept_id
    conceive_noop = SkillResult.noop(
        "Active concept already exists.",
        {"concepts_existing": 1, "concept_id": "NOOP-CID"},
    )
    direct_ok = SkillResult.ok({"direction_id": "DID-1"})

    patchers = []
    mocks = {}

    for mod, name in [
        ("skills.ingest_intake", "ingest_intake"),
        ("skills.acquire_listing", "acquire_listing"),
        ("skills.deduplicate", "deduplicate"),
        ("skills.observe", "observe"),
        ("skills.enhance", "enhance"),
        ("skills.write_copy", "write_copy"),
        ("skills.build_guide", "build_guide"),
        ("skills.narrate", "narrate"),
        ("skills.narrate_guest_cards", "narrate_guest_cards"),
        ("skills.score_music", "score_music"),
        ("skills.generate_motion", "generate_motion"),
        ("skills.assemble", "assemble"),
    ]:
        m = MagicMock(return_value=SkillResult.ok({}))
        p = patch(f"{mod}.{name}", m)
        p.start()
        patchers.append(p)
        mocks[name] = m

    # conceive returns noop with concept_id
    p = patch("skills.conceive.conceive", MagicMock(return_value=conceive_noop))
    p.start()
    patchers.append(p)

    # direct returns ok with direction_id
    direct_mock = MagicMock(return_value=direct_ok)
    p = patch("skills.direct.direct", direct_mock)
    p.start()
    patchers.append(p)
    mocks["direct"] = direct_mock

    pub = MagicMock(return_value=SkillResult.ok({"page_url": "https://x"}))
    p = patch("skills.publish_page.publish_page", pub)
    p.start()
    patchers.append(p)

    p = patch("skills.notify.send_halt_alert", MagicMock(return_value=True))
    p.start()
    patchers.append(p)

    sb = MagicMock()
    sb.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[])
    sb.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
    sb.table.return_value.select.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
    sb.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])

    for helper, val in [
        ("get_substrate", sb),
        ("record_run", "RUN-1"),
        ("record_step", "STEP-1"),
        ("complete_step", None),
        ("complete_run", None),
    ]:
        p = patch(f"workflows.onboard.{helper}", MagicMock(return_value=val))
        p.start()
        patchers.append(p)

    try:
        from workflows.onboard import onboard
        result = onboard("PROP-1")

        # direct must have been called with concept_id from the noop
        assert mocks["direct"].called, "direct must be called when conceive noops"
        assert mocks["direct"].call_args.kwargs.get("concept_id") == "NOOP-CID", (
            f"direct must receive concept_id from noop, got: {mocks['direct'].call_args}"
        )
    finally:
        for p in reversed(patchers):
            p.stop()


# ── Test 4: onboard direct-noop → narrate gets correct direction_id ─────────

def test_onboard_direct_noop_chains_direction_id():
    """A full onboard() where direct noops — narrate receives the correct
    direction_id."""
    import skills.ingest_intake
    import skills.acquire_listing
    import skills.deduplicate
    import skills.observe
    import skills.enhance
    import skills.write_copy
    import skills.build_guide
    import skills.conceive
    import skills.direct
    import skills.narrate
    import skills.narrate_guest_cards
    import skills.score_music
    import skills.generate_motion
    import skills.assemble
    import skills.publish_page
    import skills.notify

    conceive_ok = SkillResult.ok({"concept_id": "CID-1"})
    direct_noop = SkillResult.noop(
        "Direction already exists.",
        {"directions_existing": 1, "direction_id": "NOOP-DID"},
    )

    patchers = []
    mocks = {}

    for mod, name in [
        ("skills.ingest_intake", "ingest_intake"),
        ("skills.acquire_listing", "acquire_listing"),
        ("skills.deduplicate", "deduplicate"),
        ("skills.observe", "observe"),
        ("skills.enhance", "enhance"),
        ("skills.write_copy", "write_copy"),
        ("skills.build_guide", "build_guide"),
        ("skills.narrate_guest_cards", "narrate_guest_cards"),
        ("skills.score_music", "score_music"),
        ("skills.generate_motion", "generate_motion"),
        ("skills.assemble", "assemble"),
    ]:
        m = MagicMock(return_value=SkillResult.ok({}))
        p = patch(f"{mod}.{name}", m)
        p.start()
        patchers.append(p)
        mocks[name] = m

    p = patch("skills.conceive.conceive", MagicMock(return_value=conceive_ok))
    p.start()
    patchers.append(p)

    p = patch("skills.direct.direct", MagicMock(return_value=direct_noop))
    p.start()
    patchers.append(p)

    narrate_mock = MagicMock(return_value=SkillResult.ok({}))
    p = patch("skills.narrate.narrate", narrate_mock)
    p.start()
    patchers.append(p)
    mocks["narrate"] = narrate_mock

    pub = MagicMock(return_value=SkillResult.ok({"page_url": "https://x"}))
    p = patch("skills.publish_page.publish_page", pub)
    p.start()
    patchers.append(p)

    p = patch("skills.notify.send_halt_alert", MagicMock(return_value=True))
    p.start()
    patchers.append(p)

    sb = MagicMock()
    sb.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[])
    sb.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
    sb.table.return_value.select.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
    sb.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])

    for helper, val in [
        ("get_substrate", sb),
        ("record_run", "RUN-1"),
        ("record_step", "STEP-1"),
        ("complete_step", None),
        ("complete_run", None),
    ]:
        p = patch(f"workflows.onboard.{helper}", MagicMock(return_value=val))
        p.start()
        patchers.append(p)

    try:
        from workflows.onboard import onboard
        result = onboard("PROP-1")

        assert mocks["narrate"].called, "narrate must be called when direct noops"
        assert mocks["narrate"].call_args.kwargs.get("direction_id") == "NOOP-DID", (
            f"narrate must receive direction_id from noop, got: {mocks['narrate'].call_args}"
        )
    finally:
        for p in reversed(patchers):
            p.stop()


# ── Test 5: raise_hold inserts real account_id ──────────────────────────────

def test_raise_hold_inserts_account_id():
    """raise_hold inserts a real, non-null account_id matching the property's
    actual account."""
    sb = MagicMock()
    sb.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(
        data=[{"name": "Test Property", "account_id": "ACCT-789"}]
    )
    sb.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[])

    import skills.notify
    with patch("skills.notify.send_halt_alert", return_value=True):
        from core.page_builder.orchestrate import raise_hold
        raise_hold(sb, "PROP-1", "test reason", "test_hold")

    insert_calls = sb.table.return_value.insert.call_args_list
    assert len(insert_calls) >= 1
    inserted = insert_calls[0][0][0]
    assert inserted.get("account_id") == "ACCT-789", (
        f"Expected account_id='ACCT-789', got: {inserted.get('account_id')}"
    )


# ── Test 6: onboard video-failure alert inserts account_id ──────────────────

def test_onboard_video_alert_inserts_account_id():
    """onboard.py's video-failure alert insert includes a real account_id."""
    import skills.ingest_intake
    import skills.acquire_listing
    import skills.deduplicate
    import skills.observe
    import skills.enhance
    import skills.write_copy
    import skills.build_guide
    import skills.conceive
    import skills.direct
    import skills.narrate
    import skills.narrate_guest_cards
    import skills.score_music
    import skills.generate_motion
    import skills.assemble
    import skills.publish_page
    import skills.notify

    patchers = []
    for mod, name in [
        ("skills.ingest_intake", "ingest_intake"),
        ("skills.acquire_listing", "acquire_listing"),
        ("skills.deduplicate", "deduplicate"),
        ("skills.observe", "observe"),
        ("skills.enhance", "enhance"),
        ("skills.write_copy", "write_copy"),
        ("skills.build_guide", "build_guide"),
        ("skills.narrate_guest_cards", "narrate_guest_cards"),
        ("skills.score_music", "score_music"),
        ("skills.generate_motion", "generate_motion"),
    ]:
        p = patch(f"{mod}.{name}", MagicMock(return_value=SkillResult.ok({})))
        p.start()
        patchers.append(p)

    # conceive ok, direct ok, narrate ok, but assemble fails → video alert
    p = patch("skills.conceive.conceive", MagicMock(return_value=SkillResult.ok({"concept_id": "C1"})))
    p.start()
    patchers.append(p)
    p = patch("skills.direct.direct", MagicMock(return_value=SkillResult.ok({"direction_id": "D1"})))
    p.start()
    patchers.append(p)
    p = patch("skills.narrate.narrate", MagicMock(return_value=SkillResult.ok({})))
    p.start()
    patchers.append(p)
    p = patch("skills.assemble.assemble", MagicMock(return_value=SkillResult.failed(reason="boom")))
    p.start()
    patchers.append(p)

    pub = MagicMock(return_value=SkillResult.ok({"page_url": "https://x"}))
    p = patch("skills.publish_page.publish_page", pub)
    p.start()
    patchers.append(p)

    p = patch("skills.notify.send_halt_alert", MagicMock(return_value=True))
    p.start()
    patchers.append(p)

    # sb returns account_id from properties lookup
    sb = MagicMock()
    sb.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[])
    sb.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(
        data=[{"name": "Test", "account_id": "ACCT-VIDEO"}]
    )
    sb.table.return_value.select.return_value.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
    sb.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
    sb.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[])

    for helper, val in [
        ("get_substrate", sb),
        ("record_run", "RUN-1"),
        ("record_step", "STEP-1"),
        ("complete_step", None),
        ("complete_run", None),
    ]:
        p = patch(f"workflows.onboard.{helper}", MagicMock(return_value=val))
        p.start()
        patchers.append(p)

    try:
        from workflows.onboard import onboard
        onboard("PROP-1")

        # Find the hitl insert call
        insert_calls = sb.table.return_value.insert.call_args_list
        hitl_inserts = [c[0][0] for c in insert_calls if isinstance(c[0][0], dict) and c[0][0].get("queue_type") == "pipeline_failure"]
        assert len(hitl_inserts) >= 1, "Expected at least one hitl insert"
        assert hitl_inserts[0].get("account_id") == "ACCT-VIDEO", (
            f"Expected account_id='ACCT-VIDEO', got: {hitl_inserts[0].get('account_id')}"
        )
    finally:
        for p in reversed(patchers):
            p.stop()


# ── Test 7: write_copy review insert has valid values ───────────────────────

def test_write_copy_review_insert_valid():
    """write_copy's review insert: valid priority (p0-p3), valid status
    (open/in_progress/resolved/dismissed), valid queue_type, real account_id."""
    from skills.write_copy import write_copy

    sb = MagicMock()
    # Default: all queries return empty with count=0
    default_resp = MagicMock(data=[], count=0)
    sb.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(
        data=[{"name": "Test", "city": "Test City", "state_region": "NC",
               "booking_url": "#", "bedrooms": 3, "bathrooms": 2,
               "account_id": "ACCT-WC"}]
    )
    sb.table.return_value.select.return_value.eq.return_value.execute.return_value = MagicMock(data=[], count=0)
    sb.table.return_value.select.return_value.eq.return_value.is_.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
    sb.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[], count=0)
    sb.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[])
    sb.table.return_value.upsert.return_value.execute.return_value = MagicMock(data=[])

    # Mock the LLM to return a low-quality result that triggers review
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"property_description": "Test desc", "tagline": "Test", "spotlights": [], "faqs": [], "amenity_highlights": {}}')]

    mock_quality = MagicMock()
    mock_quality.content = [MagicMock(text='{"score": 2, "result": "needs_review", "issues": ["too short"]}')]

    with patch("skills.write_copy.get_substrate", return_value=sb), \
         patch("skills.write_copy.require_env", return_value="key"), \
         patch("skills.write_copy.record_run", return_value="RUN-1"), \
         patch("skills.write_copy.record_step", return_value="STEP-1"), \
         patch("skills.write_copy.complete_step"), \
         patch("skills.write_copy.complete_run"), \
         patch("skills.write_copy.emit_cost"), \
         patch("anthropic.Anthropic") as MockAnthropic:

        MockAnthropic.return_value.messages.create.side_effect = [mock_response, mock_quality]
        result = write_copy("test-prop", force=True)

    # Find the hitl insert
    insert_calls = sb.table.return_value.insert.call_args_list
    hitl_inserts = [c[0][0] for c in insert_calls
                    if isinstance(c[0][0], dict) and "queue_type" in c[0][0]]

    assert len(hitl_inserts) >= 1, f"Expected a hitl insert, got {len(hitl_inserts)}"
    row = hitl_inserts[0]

    VALID_PRIORITIES = {"p0", "p1", "p2", "p3"}
    VALID_STATUSES = {"open", "in_progress", "resolved", "dismissed"}
    VALID_QUEUE_TYPES = {"content_review", "photo_override", "onboarding_stalled",
                         "customer_health", "pipeline_failure", "break_glass"}

    assert row["priority"] in VALID_PRIORITIES, f"Invalid priority: {row['priority']}"
    assert row["status"] in VALID_STATUSES, f"Invalid status: {row['status']}"
    assert row["queue_type"] in VALID_QUEUE_TYPES, f"Invalid queue_type: {row['queue_type']}"
    assert row.get("account_id") == "ACCT-WC", f"Expected account_id='ACCT-WC', got: {row.get('account_id')}"
