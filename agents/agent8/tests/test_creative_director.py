"""
Tests for Agent 8 Stage 3 creative director.
All mocked -- $0.00 test cost.
"""

import json
from datetime import date
from unittest.mock import patch, MagicMock, call

import pytest

from agents.agent8.creative_director import (
    direct_concept,
    validate_motion_affordance,
    validate_space_direction,
    validate_depth_rhythm,
    validate_continuity,
    validate_overlay_placement,
    validate_amenity_references,
    validate_no_ota,
    validate_guest_text_verbatim,
    validate_music_brief_prohibited,
    validate_duration,
    _run_validators,
)

PROPERTY_ID = "a1b2c3d4-0001-0001-0001-000000000001"
CONCEPT_ID = "c0c0c0c0-0001-0001-0001-000000000001"

MOCK_CONCEPT = {
    "concept_id": CONCEPT_ID,
    "property_id": PROPERTY_ID,
    "title": "Three generations, one sunset",
    "premise": "A multi-generational family watches the sunset.",
    "source_material": ["wow_factor", "amenity: Fire pit"],
    "status": "draft",
}

SHOT_ID_1 = "shot-0001-0001-0001-000000000001"
SHOT_ID_2 = "shot-0002-0002-0002-000000000002"
SHOT_ID_3 = "shot-0003-0003-0003-000000000003"
SHOT_ID_4 = "shot-0004-0004-0004-000000000004"

MOCK_INVENTORY = [
    {
        "shot_id": SHOT_ID_1,
        "property_id": PROPERTY_ID,
        "motion_affordance": ["push_in", "static"],
        "depth_structure": "layered",
        "depth_tier": "mid",
        "space_direction": "left",
        "light_direction": "side",
        "time_of_day_read": "golden_hour",
        "negative_space": ["top_left", "top_right"],
        "subject_singularity": True,
        "focal_point": "fire pit",
        "foreground_elements": ["railing"],
        "frame_element": "deck",
        "beyond_frame_element": "mountains",
        "tonal_signature": "warm",
        "located_amenities": ["Fire pit"],
        "motion_risk": "low",
        "superseded_at": None,
    },
    {
        "shot_id": SHOT_ID_2,
        "property_id": PROPERTY_ID,
        "motion_affordance": ["pan_left", "static"],
        "depth_structure": "flat",
        "depth_tier": "close",
        "space_direction": "center",
        "light_direction": "front",
        "time_of_day_read": "golden_hour",
        "negative_space": ["bottom_right"],
        "subject_singularity": False,
        "focal_point": "family seating",
        "foreground_elements": [],
        "frame_element": "living room",
        "beyond_frame_element": None,
        "tonal_signature": "neutral",
        "located_amenities": [],
        "motion_risk": "low",
        "superseded_at": None,
    },
    {
        "shot_id": SHOT_ID_3,
        "property_id": PROPERTY_ID,
        "motion_affordance": ["dolly_out", "static"],
        "depth_structure": "deep",
        "depth_tier": "far",
        "space_direction": "right",
        "light_direction": "back",
        "time_of_day_read": "sunset",
        "negative_space": ["top_left"],
        "subject_singularity": True,
        "focal_point": "sunset view",
        "foreground_elements": ["tree branch"],
        "frame_element": "yard",
        "beyond_frame_element": "horizon",
        "tonal_signature": "warm",
        "located_amenities": [],
        "motion_risk": "medium",
        "superseded_at": None,
    },
    {
        "shot_id": SHOT_ID_4,
        "property_id": PROPERTY_ID,
        "motion_affordance": ["push_in", "tilt_up"],
        "depth_structure": "layered",
        "depth_tier": "mid",
        "space_direction": "left",
        "light_direction": "side",
        "time_of_day_read": "midday",
        "negative_space": ["top_right"],
        "subject_singularity": False,
        "focal_point": "pool",
        "foreground_elements": [],
        "frame_element": "pool area",
        "beyond_frame_element": "garden",
        "tonal_signature": "bright",
        "located_amenities": ["Private pool"],
        "motion_risk": "low",
        "superseded_at": None,
    },
]

MOCK_KB = {
    "name": {"value": "Vista Azule", "source": "intake_portal", "confidence": 1.0},
    "slug": "vista-azule",
    "vibe_profile": "multigenerational_retreat",
    "city": {"value": "Carolina Beach", "source": "vrbo", "confidence": 0.8},
    "state": {"value": "NC", "source": "vrbo", "confidence": 0.8},
    "bedrooms": {"value": 5, "source": "airbnb", "confidence": 0.8},
    "bathrooms": {"value": 4.5, "source": "airbnb", "confidence": 0.8},
    "amenities": [
        {"value": "Hot tub", "source": "airbnb", "confidence": 0.8},
        {"value": "Private pool", "source": "vrbo", "confidence": 0.8},
        {"value": "Fire pit", "source": "airbnb", "confidence": 0.8},
    ],
    "guest_reviews": [
        {
            "text": "Nobody felt like they got the bad room.",
            "source": "guest_book",
            "reviewer_name": "The Williamson Family",
            "stay_date": "August 2024",
            "is_guest_book": True,
        },
    ],
}

MOCK_SPEC = {
    "beats": [
        {
            "ordinal": 1,
            "shot_id": SHOT_ID_1,
            "motion": "push_in",
            "duration_sec": 10,
            "motion_technique": "parallax",
            "overlay_ref": None,
            "intent": "establish the approach",
        },
        {
            "ordinal": 2,
            "shot_id": SHOT_ID_2,
            "motion": "static",
            "duration_sec": 8,
            "motion_technique": "hold",
            "overlay_ref": None,
            "intent": "family gathering",
        },
        {
            "ordinal": 3,
            "shot_id": SHOT_ID_3,
            "motion": "dolly_out",
            "duration_sec": 12,
            "motion_technique": "reveal",
            "overlay_ref": None,
            "intent": "sunset reveal",
        },
        {
            "ordinal": 4,
            "shot_id": SHOT_ID_1,
            "motion": "static",
            "duration_sec": 8,
            "motion_technique": "hold",
            "overlay_ref": None,
            "intent": "fire pit moment",
        },
    ],
    "narrative_order": "arrive, gather, discover, rest",
    "continuity_notes": [{"kind": "LIGHT", "detail": "golden hour throughout", "beats": [1, 2, 3, 4]}],
    "narration_brief": "A family finds their rhythm at Vista Azule.",
    "narration_provenance": "original",
    "music_brief": {"tempo": "slow", "mood": "warm", "arc": "builds gently"},
    "overlay_register": [
        {"beat_ordinal": 1, "text": "Vista Azule", "grid_region": "top_left", "provenance": "original"},
    ],
    "director_rationale": "Golden hour continuity, depth variation, left-center-right flow.",
    "vibe_drift": None,
    "target_duration_sec": 45,
}

MOCK_EXISTING_SPEC = {
    "spec_id": "spec-existing",
    "concept_id": CONCEPT_ID,
    "property_id": PROPERTY_ID,
    "spec_version": 1,
    "beats": MOCK_SPEC["beats"],
    "status": "draft",
}

INVENTORY_MAP = {r["shot_id"]: r for r in MOCK_INVENTORY}


# -- Helpers for mocking Claude responses -----------------------------------

def _make_mock_claude_response(spec_dict: dict) -> MagicMock:
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = json.dumps(spec_dict)
    return mock_response


def _patch_all_deps(
    concept=MOCK_CONCEPT,
    inventory=MOCK_INVENTORY,
    kb=MOCK_KB,
    existing_spec=None,
    claude_responses=None,
):
    """
    Return a list of patches, mock_client, and a persist tracker.
    claude_responses: list of spec dicts for sequential Claude calls.

    Patches internal functions (_load_concept, _load_shot_inventory, etc.)
    to avoid importing core.supabase_store which requires real env vars.
    """
    if claude_responses is None:
        claude_responses = [MOCK_SPEC]

    mock_client = MagicMock()
    responses = [_make_mock_claude_response(s) for s in claude_responses]
    mock_client.messages.create.side_effect = responses

    # Track persisted specs
    persisted = []

    def _mock_persist(concept_arg, spec, inv_rows, kb_arg, *, status="draft", rejection_reasons=None):
        row = {
            "property_id": concept_arg["property_id"],
            "concept_id": concept_arg["concept_id"],
            "spec_version": 1,
            "beats": spec.get("beats", []),
            "beat_count": len(spec.get("beats", [])),
            "target_duration_sec": spec.get("target_duration_sec", 45),
            "narrative_order": spec.get("narrative_order"),
            "continuity_notes": spec.get("continuity_notes"),
            "narration_brief": spec.get("narration_brief"),
            "narration_provenance": spec.get("narration_provenance"),
            "music_brief": spec.get("music_brief"),
            "overlay_register": spec.get("overlay_register"),
            "director_rationale": spec.get("director_rationale"),
            "vibe_drift": spec.get("vibe_drift"),
            "director_model": "claude-sonnet-4-6",
            "status": status,
            "rejection_reasons": rejection_reasons,
            "created_by_agent": "agent8_stage3",
        }
        persisted.append(row)
        return row

    patches = [
        patch(
            "agents.agent8.creative_director._load_concept",
            return_value=concept,
        ),
        patch(
            "agents.agent8.creative_director._load_shot_inventory",
            return_value=inventory,
        ),
        patch(
            "agents.agent8.creative_director._load_kb",
            return_value=kb,
        ),
        patch(
            "agents.agent8.creative_director._load_kb_from_supabase",
            return_value=None,
        ),
        patch(
            "agents.agent8.creative_director._load_existing_spec",
            return_value=existing_spec,
        ),
        patch(
            "agents.agent8.creative_director._persist_spec",
            side_effect=_mock_persist,
        ),
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        patch(
            "agents.agent8.creative_director.anthropic.Anthropic",
            return_value=mock_client,
        ),
        patch("agents.agent8.retry.time.sleep"),
    ]
    return patches, mock_client, persisted


# -- Integration-level tests ------------------------------------------------

class TestDirectConceptGather:
    def test_refuses_when_no_shot_inventory(self):
        """Raises ValueError naming Stage 1 when no inventory rows exist."""
        patches, _, _ = _patch_all_deps(inventory=[])
        with pytest.raises(ValueError, match="Stage 1"):
            for p in patches:
                p.start()
            try:
                direct_concept(CONCEPT_ID)
            finally:
                for p in patches:
                    p.stop()

    def test_returns_existing_spec_when_not_force(self):
        """Existing active spec returned without model call."""
        patches, mock_client, _ = _patch_all_deps(existing_spec=MOCK_EXISTING_SPEC)
        for p in patches:
            p.start()
        try:
            result = direct_concept(CONCEPT_ID)
            mock_client.messages.create.assert_not_called()
            assert result["spec_id"] == "spec-existing"
        finally:
            for p in patches:
                p.stop()

    def test_force_bypasses_existing_check(self):
        """force=True regenerates even when existing spec is present."""
        patches, mock_client, _ = _patch_all_deps(existing_spec=MOCK_EXISTING_SPEC)
        for p in patches:
            p.start()
        try:
            result = direct_concept(CONCEPT_ID, force=True)
            mock_client.messages.create.assert_called()
            # Should have generated a new spec, not returned existing
            assert result.get("spec_id") != "spec-existing"
            assert result.get("created_by_agent") == "agent8_stage3"
        finally:
            for p in patches:
                p.stop()

    def test_dry_run_with_existing_makes_no_model_call(self):
        """$0 path: dry_run with existing spec returns it, no model call."""
        patches, mock_client, _ = _patch_all_deps(existing_spec=MOCK_EXISTING_SPEC)
        for p in patches:
            p.start()
        try:
            result = direct_concept(CONCEPT_ID, dry_run=True)
            mock_client.messages.create.assert_not_called()
            assert result["spec_id"] == "spec-existing"
        finally:
            for p in patches:
                p.stop()


# -- Validator unit tests ---------------------------------------------------

class TestValidateMotionAffordance:
    def test_pass(self):
        beats = [
            {"ordinal": 1, "shot_id": SHOT_ID_1, "motion": "push_in"},
            {"ordinal": 2, "shot_id": SHOT_ID_2, "motion": "static"},
        ]
        assert validate_motion_affordance(beats, INVENTORY_MAP) == []

    def test_fail(self):
        beats = [
            {"ordinal": 1, "shot_id": SHOT_ID_1, "motion": "dolly_out"},
        ]
        violations = validate_motion_affordance(beats, INVENTORY_MAP)
        assert len(violations) == 1
        assert violations[0]["rule"] == "motion_affordance"
        assert 1 in violations[0]["beats"]


class TestValidateSpaceDirection:
    def test_pass(self):
        """left -> center is fine."""
        beats = [
            {"ordinal": 1, "shot_id": SHOT_ID_1},  # left
            {"ordinal": 2, "shot_id": SHOT_ID_2},  # center
        ]
        assert validate_space_direction(beats, INVENTORY_MAP) == []

    def test_fail(self):
        """left -> right is a conflict."""
        beats = [
            {"ordinal": 1, "shot_id": SHOT_ID_1},  # left
            {"ordinal": 2, "shot_id": SHOT_ID_3},  # right
        ]
        violations = validate_space_direction(beats, INVENTORY_MAP)
        assert len(violations) == 1
        assert violations[0]["rule"] == "space_direction"


class TestValidateDepthRhythm:
    def test_pass(self):
        """mid, close, far -- all different, no violation."""
        beats = [
            {"ordinal": 1, "shot_id": SHOT_ID_1},  # mid
            {"ordinal": 2, "shot_id": SHOT_ID_2},  # close
            {"ordinal": 3, "shot_id": SHOT_ID_3},  # far
        ]
        assert validate_depth_rhythm(beats, INVENTORY_MAP) == []

    def test_fail(self):
        """mid, mid, mid -- three consecutive same depth_tier."""
        beats = [
            {"ordinal": 1, "shot_id": SHOT_ID_1},  # mid
            {"ordinal": 2, "shot_id": SHOT_ID_4},  # mid
            {"ordinal": 3, "shot_id": SHOT_ID_1},  # mid
        ]
        violations = validate_depth_rhythm(beats, INVENTORY_MAP)
        assert len(violations) == 1
        assert violations[0]["rule"] == "depth_rhythm"


class TestValidateNoOta:
    def test_pass(self):
        spec = {"narration_brief": "A family vacation at the property."}
        assert validate_no_ota(spec) == []

    def test_fail(self):
        spec = {"narration_brief": "Book on Airbnb today!"}
        violations = validate_no_ota(spec)
        assert len(violations) >= 1
        assert violations[0]["rule"] == "no_ota"


class TestValidateMusicBriefProhibited:
    def test_pass(self):
        spec = {"music_brief": {"tempo": "slow", "mood": "warm", "arc": "builds"}}
        assert validate_music_brief_prohibited(spec) == []

    def test_fail(self):
        spec = {"music_brief": {"tempo": "slow", "mood": "like Bon Iver", "arc": "builds"}}
        violations = validate_music_brief_prohibited(spec)
        assert len(violations) >= 1
        assert violations[0]["rule"] == "music_brief_prohibited"


class TestValidateDuration:
    def test_pass(self):
        spec = {
            "beats": [
                {"duration_sec": 10},
                {"duration_sec": 12},
                {"duration_sec": 11},
                {"duration_sec": 10},
            ],
            "target_duration_sec": 45,
        }
        assert validate_duration(spec) == []

    def test_fail(self):
        spec = {
            "beats": [
                {"duration_sec": 5},
                {"duration_sec": 5},
            ],
            "target_duration_sec": 45,
        }
        violations = validate_duration(spec)
        assert len(violations) == 1
        assert violations[0]["rule"] == "duration"


class TestValidateContinuity:
    def test_pass_consistent_time(self):
        """golden_hour -> golden_hour is fine."""
        beats = [
            {"ordinal": 1, "shot_id": SHOT_ID_1, "intent": ""},   # golden_hour
            {"ordinal": 2, "shot_id": SHOT_ID_2, "intent": ""},   # golden_hour
        ]
        assert validate_continuity(beats, INVENTORY_MAP) == []

    def test_fail_time_jump(self):
        """midday -> night with no intent is a violation."""
        inv = dict(INVENTORY_MAP)
        # Create a fake night frame
        night_id = "shot-night"
        inv[night_id] = {"time_of_day_read": "night"}
        beats = [
            {"ordinal": 1, "shot_id": SHOT_ID_4, "intent": ""},   # midday
            {"ordinal": 2, "shot_id": night_id, "intent": ""},     # night
        ]
        violations = validate_continuity(beats, inv)
        assert len(violations) == 1
        assert violations[0]["rule"] == "continuity"


class TestValidateOverlayPlacement:
    def test_pass(self):
        spec = {
            "beats": [{"ordinal": 1, "shot_id": SHOT_ID_1}],
            "overlay_register": [
                {"beat_ordinal": 1, "text": "Hello", "grid_region": "top_left"},
            ],
        }
        assert validate_overlay_placement(spec, INVENTORY_MAP) == []

    def test_fail(self):
        spec = {
            "beats": [{"ordinal": 1, "shot_id": SHOT_ID_1}],
            "overlay_register": [
                {"beat_ordinal": 1, "text": "Hello", "grid_region": "bottom_left"},
            ],
        }
        violations = validate_overlay_placement(spec, INVENTORY_MAP)
        assert len(violations) == 1
        assert violations[0]["rule"] == "overlay_placement"


class TestValidateGuestTextVerbatim:
    def test_pass(self):
        spec = {
            "overlay_register": [
                {
                    "text": "Nobody felt like they got the bad room.",
                    "provenance": "guest_book",
                    "beat_ordinal": 1,
                },
            ],
        }
        assert validate_guest_text_verbatim(spec, MOCK_KB) == []

    def test_fail(self):
        spec = {
            "overlay_register": [
                {
                    "text": "Everyone loved their room.",
                    "provenance": "guest_book",
                    "beat_ordinal": 1,
                },
            ],
        }
        violations = validate_guest_text_verbatim(spec, MOCK_KB)
        assert len(violations) == 1
        assert violations[0]["rule"] == "guest_text_verbatim"


# -- Revision loop tests ---------------------------------------------------

class TestRevisionLoop:
    def test_revision_loop_triggers_on_violation(self):
        """Mock Claude to return invalid spec then valid spec."""
        # First response has a motion not in affordance
        bad_beats = [
            dict(MOCK_SPEC["beats"][0], motion="crane_up"),  # invalid motion
            MOCK_SPEC["beats"][1],
            MOCK_SPEC["beats"][2],
            MOCK_SPEC["beats"][3],
        ]
        bad_spec = dict(MOCK_SPEC, beats=bad_beats)

        # Second response: all valid -- use only shots that don't create
        # space_direction conflicts (left->center->left->center)
        good_beats = [
            {"ordinal": 1, "shot_id": SHOT_ID_1, "motion": "push_in", "duration_sec": 12,
             "motion_technique": "parallax", "overlay_ref": None, "intent": "approach"},
            {"ordinal": 2, "shot_id": SHOT_ID_2, "motion": "static", "duration_sec": 10,
             "motion_technique": "hold", "overlay_ref": None, "intent": "gather"},
            {"ordinal": 3, "shot_id": SHOT_ID_4, "motion": "push_in", "duration_sec": 12,
             "motion_technique": "parallax", "overlay_ref": None, "intent": "discover"},
            {"ordinal": 4, "shot_id": SHOT_ID_2, "motion": "pan_left", "duration_sec": 10,
             "motion_technique": "slow", "overlay_ref": None, "intent": "rest"},
        ]
        good_spec = dict(MOCK_SPEC, beats=good_beats, overlay_register=[])

        patches, mock_client, _ = _patch_all_deps(
            claude_responses=[bad_spec, good_spec],
        )
        for p in patches:
            p.start()
        try:
            result = direct_concept(CONCEPT_ID)
            # Should have called Claude twice (propose + revise)
            assert mock_client.messages.create.call_count == 2
            assert result.get("created_by_agent") == "agent8_stage3"
        finally:
            for p in patches:
                p.stop()

    def test_revision_attempt_limit(self):
        """Mock Claude to always return invalid, verify status='draft' with violations."""
        # All responses have a motion not in affordance
        bad_spec = dict(MOCK_SPEC)
        bad_beats = list(MOCK_SPEC["beats"])
        bad_beats[0] = dict(bad_beats[0])
        bad_beats[0]["motion"] = "crane_up"  # not in affordance
        bad_spec = dict(MOCK_SPEC, beats=bad_beats)

        # 1 initial + 2 revision attempts = 3 responses needed
        patches, mock_client, _ = _patch_all_deps(
            claude_responses=[bad_spec, bad_spec, bad_spec],
        )
        for p in patches:
            p.start()
        try:
            result = direct_concept(CONCEPT_ID)
            # 1 propose + 2 revisions = 3 calls
            assert mock_client.messages.create.call_count == 3
            assert result.get("status") == "draft"
            assert result.get("rejection_reasons") is not None
            assert len(result["rejection_reasons"]) > 0
        finally:
            for p in patches:
                p.stop()
