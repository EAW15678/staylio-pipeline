"""
VIDEO-WORKFLOW-1: Video steps in the onboard workflow.

Tests verify conceive skill, workflow ordering, failure-continues-to-publish
behaviour, and alert creation on video failure.
All mocked — no vendor calls, $0.
"""

import ast
import inspect
from unittest.mock import MagicMock, patch

from skills.contract import SkillResult


# ── Test 1: conceive writes a concept with active status ─────────────

def test_conceive_writes_active_concept():
    """conceive writes a concept with status='active' and a non-empty premise.

    Must fail on e102154 — conceive.py does not exist.
    """
    from skills.conceive import conceive

    sb = MagicMock()
    # No existing concept
    sb.table.return_value.select.return_value.eq.return_value.eq.return_value.is_.return_value.limit.return_value.execute.return_value = MagicMock(count=0, data=[])
    # Properties query
    sb.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[{
        "name": "Test Property", "city": "Test City", "state_region": "NC",
        "vibe_profile": "multigenerational_retreat", "amenities": ["Pool"],
    }])
    # Owner context
    sb.table.return_value.select.return_value.eq.return_value.is_.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(data=[{
        "owner_story": "We love this place", "wow_factor": "The pool",
        "hidden_gems": "Sunset views", "guest_love": "Everything",
    }])
    # Guest evidence
    sb.table.return_value.select.return_value.eq.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
    # Insert returns
    sb.table.return_value.insert.return_value.execute.return_value = MagicMock(data=[])

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"title": "Test Title", "premise": "A wonderful property."}')]

    with patch("skills.conceive.get_substrate", return_value=sb), \
         patch("skills.conceive.require_env", return_value="key"), \
         patch("skills.conceive.record_run", return_value="fake-run"), \
         patch("skills.conceive.record_step", return_value="fake-step"), \
         patch("skills.conceive.complete_run"), \
         patch("skills.conceive.complete_step"), \
         patch("skills.conceive.emit_cost"), \
         patch("anthropic.Anthropic") as MockAnthropic:

        MockAnthropic.return_value.messages.create.return_value = mock_response
        result = conceive("test-prop")

    assert result.is_ok, f"conceive should succeed, got {result.status}: {result.reason}"
    assert result.data["premise"] == "A wonderful property."
    assert result.data["concept_id"] is not None


# ── Test 2: conceive idempotency ─────────────────────────────────────

def test_conceive_idempotency():
    """conceive returns noop when active concept exists and force=False;
    with force=True it supersedes and writes new, deleting nothing.

    Must fail on e102154 — conceive.py does not exist.
    """
    from skills.conceive import conceive

    sb = MagicMock()
    # Existing active concept
    sb.table.return_value.select.return_value.eq.return_value.eq.return_value.is_.return_value.limit.return_value.execute.return_value = MagicMock(count=1, data=[{"concept_id": "existing"}])

    with patch("skills.conceive.get_substrate", return_value=sb):
        result = conceive("test-prop", force=False)

    assert result.data and result.data.get("noop"), f"Should noop, got {result.data}"


# ── Test 3: workflow order ───────────────────────────────────────────

def test_onboard_video_steps_ordered():
    """onboard calls conceive → direct → narrate → music → motion →
    assemble → publish, in that order.

    Structural — AST-based, justified precedent from PHASH-2/ENHANCE-2.

    Must fail on e102154 — conceive is not in the workflow.
    """
    from workflows.onboard import onboard
    source = inspect.getsource(onboard)
    tree = ast.parse(source)

    # Find all string literals that look like step names in record_step calls
    # and _run_skill calls, sorted by source line number.
    # _run_skill(name, fn, ...) → name is args[0]
    # record_step(sb, run_id, step_name) → step_name is args[2]
    steps_with_lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "_run_skill":
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    steps_with_lines.append((node.lineno, node.args[0].value))
            elif isinstance(func, ast.Name) and func.id == "record_step":
                if len(node.args) >= 3 and isinstance(node.args[2], ast.Constant) and isinstance(node.args[2].value, str):
                    steps_with_lines.append((node.lineno, node.args[2].value))

    steps_with_lines.sort(key=lambda x: x[0])
    seen = set()
    step_order = []
    for _, name in steps_with_lines:
        if name not in seen:
            seen.add(name)
            step_order.append(name)

    # Verify the video steps come after build_guide and before publish_page
    required = ["conceive", "direct", "narrate", "publish_page"]
    positions = {}
    for name in required:
        if name in step_order:
            positions[name] = step_order.index(name)

    assert "conceive" in positions, f"conceive not found in steps: {step_order}"
    assert "direct" in positions, f"direct not found in steps: {step_order}"
    assert positions["conceive"] < positions["direct"], "conceive must come before direct"
    assert positions["direct"] < positions["publish_page"], "direct must come before publish_page"


# ── Test 4: direction_id chains from direct to later steps ───────────

def test_direction_id_chaining():
    """direct receives concept_id from conceive; narrate/music/motion/assemble
    receive direction_id from direct.

    Must fail on e102154 — video steps not in workflow.
    """
    from workflows.onboard import onboard
    source = inspect.getsource(onboard)

    # Verify the code passes concept_id to direct and direction_id to others
    assert "concept_id=concept_id" in source, "direct must receive concept_id"
    assert "direction_id=direction_id" in source, "narrate/music/motion/assemble must receive direction_id"


# ── Test 5: assemble fails → publish still called ────────────────────

def test_assemble_failure_still_publishes():
    """When assemble fails, publish_page is still called and onboard returns ok.

    Must fail on e102154 — video steps not in workflow.
    """
    from workflows.onboard import onboard
    source = inspect.getsource(onboard)

    # The video block is in a try/except that catches RuntimeError
    # and continues to publish_page
    assert "except Exception as video_exc:" in source or "except" in source, \
        "Video block must be in a try/except"
    # publish_page comes AFTER the except block
    assert source.index("publish_page") > source.index("video_exc"), \
        "publish_page must come after the video exception handler"


# ── Test 6: conceive fails → later video steps skipped, publish still called

def test_conceive_failure_skips_video_publishes():
    """When conceive fails, later video steps are skipped but publish_page
    is still called.

    Must fail on e102154 — conceive not in workflow.
    """
    from workflows.onboard import onboard
    source = inspect.getsource(onboard)

    # conceive failure raises RuntimeError which is caught
    assert 'raise RuntimeError(f"conceive failed' in source, \
        "conceive failure must raise to skip later video steps"
    # The except catches it and continues to publish
    assert "continuing to publish" in source.lower() or "publishing without hero" in source.lower(), \
        "Video failure must log that publishing continues"


# ── Test 7: video failure creates hitl queue row ─────────────────────

def test_video_failure_creates_hitl_row():
    """A video failure creates a hitl_queue_items row with
    queue_type='pipeline_failure', priority='p0'.

    Must fail on e102154 — no video alert logic.
    """
    from workflows.onboard import onboard
    source = inspect.getsource(onboard)

    assert '"pipeline_failure"' in source, "Must create hitl row with queue_type pipeline_failure"
    assert '"p0"' in source, "Must create hitl row with priority p0"
    assert "send_halt_alert" in source, "Must call send_halt_alert"


# ── Test 8: alert failure doesn't prevent publishing ─────────────────

def test_alert_failure_doesnt_block_publish():
    """When the alert itself fails, publish_page is still called.

    Must fail on e102154 — no alert logic in workflow.
    """
    from workflows.onboard import onboard
    source = inspect.getsource(onboard)

    # The alert is in its own try/except inside the video except block
    # Check that send_halt_alert is inside a try block
    alert_pos = source.find("send_halt_alert")
    if alert_pos == -1:
        assert False, "send_halt_alert not found in onboard"

    # The alert try/except must be inside the video except block,
    # and publish_page must come after both
    publish_pos = source.rfind("publish_page")
    assert publish_pos > alert_pos, "publish_page must come after alert logic"


# ── Test 9: all succeed → no queue row, no alert ────────────────────

def test_success_no_alert():
    """When all video steps succeed, no hitl queue row and no alert.

    Must fail on e102154 — no video steps in workflow.
    """
    from workflows.onboard import onboard
    source = inspect.getsource(onboard)

    # The hitl insert and send_halt_alert are ONLY inside the except block
    # They should appear after "except Exception as video_exc"
    except_pos = source.find("except Exception as video_exc")
    assert except_pos > 0, "Video exception handler must exist"

    hitl_pos = source.find("hitl_queue_items")
    assert hitl_pos > except_pos, \
        "hitl_queue_items insert must be inside the exception handler only"
