"""
FIX: escalate_halt must include created_by_type in hitl_queue_items insert.

Crashed Vista Azule's launch on 2026-08-20 — created_by_type is NOT NULL.
"""

import sys
from unittest.mock import MagicMock

sys.path.insert(0, ".")


def test_escalate_halt_includes_created_by_type():
    """escalate_halt's hitl_queue_items insert includes created_by_type='system'."""
    from skills.contract import escalate_halt

    sb = MagicMock()
    # properties lookup returns empty
    sb.table.return_value.select.return_value.eq.return_value.limit.return_value.execute.return_value = MagicMock(data=[])
    # insert returns a row with an id
    sb.table.return_value.insert.return_value.execute.return_value = MagicMock(
        data=[{"id": "test-hitl-id"}]
    )

    escalate_halt(
        sb, "prop-1",
        queue_type="publish_halt",
        reason_code="slug_collision",
        title="Test halt",
        detail="Testing created_by_type",
    )

    # Find the insert call on hitl_queue_items
    insert_calls = [
        call for call in sb.table.return_value.insert.call_args_list
    ]
    assert len(insert_calls) >= 1, "Expected at least one insert call"

    inserted_dict = insert_calls[0][0][0]  # first positional arg of first insert call
    assert "created_by_type" in inserted_dict, \
        f"created_by_type missing from insert dict: {inserted_dict}"
    assert inserted_dict["created_by_type"] == "system", \
        f"Expected 'system', got '{inserted_dict['created_by_type']}'"
