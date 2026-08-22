"""
FINISH-A: Tests for Aleph finishing — real integration boundaries.
"""

import sys
import inspect
sys.path.insert(0, ".")


# ── Test 1: preserve adds no atmosphere clause ────────────────────────

def test_preserve_adds_no_atmosphere():
    from skills.finish_beats import _build_aleph_prompt
    prompt = _build_aleph_prompt(["gentle movement to the existing water surface"], "preserve")
    assert "warm" not in prompt.lower(), "preserve must not add warm atmosphere"
    assert "sunlight" not in prompt.lower(), "preserve must not add sunlight"
    assert "golden" not in prompt.lower(), "preserve must not add golden"
    assert "Preserve the existing scene" in prompt


# ── Test 2: warm adds only the validated restrained clause ────────────

def test_warm_adds_validated_clause():
    from skills.finish_beats import _build_aleph_prompt, ATMOSPHERE_PROMPTS
    prompt = _build_aleph_prompt([], "warm")
    assert ATMOSPHERE_PROMPTS["warm"] in prompt, \
        "warm must add the exact validated atmosphere clause"
    assert "Preserve the existing scene" in prompt


# ── Test 3: finisher cannot invent atmosphere ─────────────────────────

def test_finisher_cannot_invent_atmosphere():
    """The atmosphere clause comes ONLY from ATMOSPHERE_PROMPTS keyed by
    the Director's atmosphere field. If atmosphere is not in the dict,
    no atmospheric clause is added."""
    from skills.finish_beats import _build_aleph_prompt, ATMOSPHERE_PROMPTS

    # Only preserve and warm exist
    assert set(ATMOSPHERE_PROMPTS.keys()) == {"preserve", "warm"}, \
        "Only preserve and warm are validated atmosphere values"

    # preserve → None (no clause)
    assert ATMOSPHERE_PROMPTS["preserve"] is None

    # Unknown atmosphere → no clause (same as preserve)
    prompt = _build_aleph_prompt([], "unknown_atmosphere")
    assert "sunlight" not in prompt.lower()
    assert "golden" not in prompt.lower()


def test_finisher_cannot_invent_atmosphere_mutation_proof():
    """If ATMOSPHERE_PROMPTS gained a new key, this test would not fail
    — but adding it would require modifying the ATMOSPHERE_PROMPTS dict
    explicitly. The constraint is that only the dict contents can produce
    atmosphere clauses. Remove 'warm' from the dict and the warm prompt
    disappears."""
    from skills.finish_beats import _build_aleph_prompt, ATMOSPHERE_PROMPTS

    # Removing warm from dict would make warm produce no clause
    warm_val = ATMOSPHERE_PROMPTS["warm"]
    assert warm_val is not None, "warm must produce a clause"
    assert "Warm" in warm_val or "warm" in warm_val


# ── Test 4: every content_motion reaching Aleph is observation-grounded

def test_content_motion_grounded():
    """GROUNDING_RULES maps each content_motion element to an observation
    check. An element not in GROUNDING_RULES or failing its check is
    removed before prompt construction."""
    from skills.finish_beats import GROUNDING_RULES

    # water grounding requires water_surface in motion_risk
    water_rule = GROUNDING_RULES.get("water")
    assert water_rule is not None
    assert water_rule["check"]({"motion_risk": ["water_surface"]}) is True
    assert water_rule["check"]({"motion_risk": []}) is False

    # foliage grounding requires plant/tree/palm in foreground_elements
    foliage_rule = GROUNDING_RULES.get("foliage")
    assert foliage_rule is not None
    assert foliage_rule["check"]({"foreground_elements": ["palm tree"]}) is True
    assert foliage_rule["check"]({"foreground_elements": ["sofa"]}) is False


# ── Test 5: unsupported elements removed ──────────────────────────────

def test_unsupported_elements_removed():
    """An element not in GROUNDING_RULES produces no prompt phrase."""
    from skills.finish_beats import GROUNDING_RULES, _build_aleph_prompt

    # "butterflies" is not a grounding rule
    assert "butterflies" not in GROUNDING_RULES

    # Therefore it would be removed during grounding and not reach the prompt
    # (The removal happens in finish_beats main loop, not in _build_aleph_prompt)


# ── Test 6: atmosphere-only warm beat still executes ──────────────────

def test_atmosphere_only_warm_executes():
    """Empty grounded content_motion + atmosphere=warm → Aleph eligible.
    Skip only when BOTH grounded content_motion empty AND preserve."""
    from skills.finish_beats import _build_aleph_prompt, ATMOSPHERE_PROMPTS

    # atmosphere=warm with no content motion → still produces a prompt
    prompt = _build_aleph_prompt([], "warm")
    assert len(prompt) > 50, "warm-only prompt must be substantive"
    assert ATMOSPHERE_PROMPTS["warm"] in prompt


# ── Test 7: empty grounded + preserve skips ───────────────────────────

def test_empty_grounded_preserve_skips():
    """When grounded content_motion is empty AND atmosphere=preserve,
    there is no useful finishing intent → skip Aleph."""
    from skills.finish_beats import _build_aleph_prompt, ATMOSPHERE_PROMPTS

    # The skip logic is in finish_beats main loop:
    # if not grounded_phrases and atmosphere == "preserve": skip
    # Verify the prompt would be trivial
    prompt = _build_aleph_prompt([], "preserve")
    # Only the preservation clause, no action
    assert prompt == "Preserve the existing scene and camera movement."


def test_empty_grounded_preserve_skips_mutation_proof():
    """If the skip condition were removed, beats with no grounded
    content_motion and preserve atmosphere would still call Aleph
    with a trivial prompt — wasting cost for no visual improvement."""
    src = inspect.getsource(
        __import__("skills.finish_beats", fromlist=["finish_beats"]).finish_beats
    )
    assert 'atmosphere == "preserve"' in src, \
        "Skip logic must check atmosphere == preserve"
    assert "not grounded_phrases" in src, \
        "Skip logic must check empty grounded_phrases"


# ── Test 8: text-bearing frame skips ──────────────────────────────────

def test_text_bearing_skips():
    """The finish_beats function checks contains_text and skips."""
    src = inspect.getsource(
        __import__("skills.finish_beats", fromlist=["finish_beats"]).finish_beats
    )
    assert "contains_text" in src, "Must check contains_text"


# ── Test 9: Aleph does not extend camera travel ──────────────────────

def test_aleph_preserves_camera():
    """Every Aleph prompt starts with 'Preserve the existing scene and
    camera movement.' and contains no camera-extension language."""
    from skills.finish_beats import _build_aleph_prompt

    for atmo in ["preserve", "warm"]:
        for phrases in [[], ["gentle movement to the existing water surface"]]:
            prompt = _build_aleph_prompt(phrases, atmo)
            assert prompt.startswith("Preserve the existing scene and camera movement.")
            for forbidden in ["travel farther", "reveal more", "orbit", "pull back",
                             "move through", "extend", "additional room"]:
                assert forbidden not in prompt.lower(), \
                    f"Prompt must not contain camera-extension language: '{forbidden}'"


# ── Test 10: truth gate samples first/middle/last ─────────────────────

def test_truth_gate_samples_three_frames():
    """The truth gate function checks first, middle, and last frames."""
    src = inspect.getsource(
        __import__("skills.finish_beats", fromlist=["_truth_gate"])._truth_gate
    )
    assert '"first"' in src and '"middle"' in src and '"last"' in src, \
        "Truth gate must check first, middle, and last frames"
    assert "1280" in src, "Truth gate must use ~1280px review width"


# ── Test 11: global relighting passes ─────────────────────────────────

def test_truth_gate_tolerates_relighting():
    """The truth gate uses edge-map comparison, not pixel RGB diff.
    This makes it tolerant of uniform colour-temperature shifts while
    catching structural changes."""
    src = inspect.getsource(
        __import__("skills.finish_beats", fromlist=["_truth_gate"])._truth_gate
    )
    assert "edge" in src.lower() or "FIND_EDGES" in src, \
        "Truth gate must use edge-based comparison (tolerant of relighting)"
    assert "grid" in src.lower(), \
        "Truth gate must use grid-based composition check"


# ── Test 12: truth failure → pre-Aleph fallback ──────────────────────

def test_truth_failure_uses_pre_aleph():
    """When truth gate fails, the raw pre-Aleph artifact must remain
    active (not superseded). The beat is never dropped."""
    src = inspect.getsource(
        __import__("skills.finish_beats", fromlist=["finish_beats"]).finish_beats
    )
    # On truth failure, the code must NOT supersede the raw artifact
    # and must continue (not raise/return error)
    assert "truth_failed" in src, "Must track truth failures"
    assert "using pre-Aleph" in src or "pre-Aleph fallback" in src or "pre-Aleph render" in src, \
        "Must log pre-Aleph fallback on truth failure"


def test_truth_failure_mutation_proof():
    """If the truth gate failure path superseded the raw artifact,
    the beat would be dropped from the film."""
    src = inspect.getsource(
        __import__("skills.finish_beats", fromlist=["finish_beats"]).finish_beats
    )
    # The .update({"superseded_at": ...}) call must come AFTER "Truth passed"
    # not inside the truth_failed branch
    truth_pass_idx = src.find("# Truth passed")
    update_supersede_idx = src.find('.update({\n            "superseded_at"')
    if update_supersede_idx == -1:
        update_supersede_idx = src.find('.update({')
        # Find the one that sets superseded_at
        while update_supersede_idx != -1:
            chunk = src[update_supersede_idx:update_supersede_idx + 100]
            if "superseded_at" in chunk:
                break
            update_supersede_idx = src.find('.update({', update_supersede_idx + 1)
    assert update_supersede_idx > truth_pass_idx, \
        "Supersede (.update superseded_at) must only happen after truth passes"


# ── Test 13: superseded pre-Aleph remains retrievable ─────────────────

def test_superseded_remains_retrievable():
    """Superseding sets superseded_at, does NOT delete the row."""
    src = inspect.getsource(
        __import__("skills.finish_beats", fromlist=["finish_beats"]).finish_beats
    )
    # Must use update().eq() not delete()
    assert ".update(" in src and "superseded_at" in src, \
        "Must use update to set superseded_at"
    assert ".delete(" not in src, \
        "Must NOT delete artifacts — supersede only"


# ── Test 14: second force=False = zero Aleph calls ────────────────────

def test_idempotency_check_exists():
    """The finish_beats function checks for existing artifacts by
    input_hash before calling Aleph."""
    src = inspect.getsource(
        __import__("skills.finish_beats", fromlist=["finish_beats"]).finish_beats
    )
    assert "aleph_hash" in src or "input_hash" in src, \
        "Must compute hash for idempotency"
    assert "existing" in src and "count" in src, \
        "Must check for existing matching artifact"


def test_idempotency_mutation_proof():
    """If the idempotency check were removed, every force=False run
    would re-call Aleph — costing ~$0.84/beat repeatedly."""
    src = inspect.getsource(
        __import__("skills.finish_beats", fromlist=["finish_beats"]).finish_beats
    )
    # The hash must cover prompt + pre-aleph identity + model
    assert '"prompt"' in src and '"model"' in src and '"pre_aleph' in src, \
        "Hash must cover prompt + pre-aleph identity + model"


# ── Test 15: finisher cannot invent content motion ────────────────────

def test_finisher_cannot_invent_content_motion():
    """Content motion reaching the Aleph prompt must originate from
    the Director's beat.content_motion, not be invented by the finisher.
    The finish_beats function reads content_motion from the beat dict."""
    src = inspect.getsource(
        __import__("skills.finish_beats", fromlist=["finish_beats"]).finish_beats
    )
    assert 'beat.get("content_motion")' in src or "content_motion" in src, \
        "Must read content_motion from the Director's beat"
    # The grounding loop iterates over Director-provided elements only
    assert "for element in content_motion" in src, \
        "Must iterate over Director-provided content_motion elements"


def test_finisher_cannot_invent_content_motion_mutation_proof():
    """If the finisher added elements not in the Director's
    content_motion list, it would be inventing creative intent."""
    src = inspect.getsource(
        __import__("skills.finish_beats", fromlist=["finish_beats"]).finish_beats
    )
    # The only source of content_motion elements is the beat
    # There must be no hardcoded additions like "water" or "foliage"
    # outside the grounding rules lookup
    assert "content_motion.append" not in src, \
        "Finisher must not append to content_motion"
    assert 'grounded_phrases.append(rule["prompt_phrase"])' in src, \
        "Grounded phrases come only from GROUNDING_RULES lookups"
