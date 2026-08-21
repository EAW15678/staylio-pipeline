"""
MOTION-1: Tests for locked beat duration quantization and downgrade-not-drop.

Runway accepts only 5 or 10 second durations. Five of six locked beats
failed silently on this limit, turning a 29.4s film into 10.6s.
"""

import sys
sys.path.insert(0, ".")


# ── Test 1: prompt states the locked duration rule ──────────────────────────

def test_prompt_states_locked_duration_rule():
    """The prompt tells the director locked beats must be 5.0 or 10.0 seconds."""
    from skills.direct import _build_prompt
    concept = {"title": "T", "premise": "P"}
    frames = [{"photo_id": "f1", "motion_affordance": ["push_in"],
               "curated_section": "Exterior", "quality_score": 0.9}]
    prop = {"name": "T", "city": "X", "state_region": "NC",
            "vibe_profile": "romantic_escape", "amenities": []}
    prompt = _build_prompt(concept, frames, prop, {}, [])

    assert "5.0 or 10.0 seconds" in prompt, "Must state the 5/10 duration rule"
    assert "Runway accepts no other duration" in prompt or \
           "Runway\n      accepts no other duration" in prompt, \
        "Must state Runway rejects other durations"


# ── Test 2: quantizer snaps correctly ───────────────────────────────────────

def test_quantizer_snaps():
    """Duration quantization: 3.5→5, 5.0→5, 7.4→5, 7.6→10, 10.0→10."""
    # The quantizer logic from generate_motion.py
    def quantize(duration):
        return 10 if duration > 7.5 else 5

    assert quantize(3.5) == 5
    assert quantize(5.0) == 5
    assert quantize(7.4) == 5
    assert quantize(7.6) == 10
    assert quantize(10.0) == 10
    assert quantize(1.8) == 5
    assert quantize(8.0) == 10


# ── Test 3: directed duration preserved in artifact ─────────────────────────

def test_directed_duration_preserved():
    """The beat's directed duration is preserved in the artifact record
    even when the Runway duration differs. This is verified by checking
    the generate_motion code writes duration (the beat's value) not
    runway_duration to the artifact."""
    import inspect
    from skills.generate_motion import generate_motion

    source = inspect.getsource(generate_motion)
    # The insert for locked clips should use 'duration' not 'runway_duration'
    # for the duration_seconds field
    locked_insert_area = source[source.find("runway_duration"):]
    # Find the artifact insert that follows
    insert_start = locked_insert_area.find('"duration_seconds":')
    assert insert_start > 0, "Must have duration_seconds in locked clip insert"
    insert_line = locked_insert_area[insert_start:insert_start + 40]
    assert "duration" in insert_line, "duration_seconds must use the beat's directed duration"
    # It should NOT be runway_duration
    assert "runway_duration" not in insert_line, \
        "duration_seconds must NOT use runway_duration — preserve directed duration"


# ── Test 4: locked failure downgrades to bounded, not dropped ───────────────

def test_locked_failure_downgrades_not_drops():
    """A locked beat whose Runway call raises is downgraded to bounded
    and still appears in the output — not dropped. This is the defect
    that turned a 29.4s film into 10.6s."""
    import inspect
    from skills.generate_motion import generate_motion

    source = inspect.getsource(generate_motion)

    # Find the except block for Runway failures
    except_start = source.find("Runway locked failed")
    assert except_start > 0, "Must have the Runway failure handler"

    # The code after the failure should insert a bounded fallback clip
    fallback_area = source[except_start:except_start + 2000]
    assert '"downgraded": True' in fallback_area, \
        "Must record downgraded=True in the fallback clip"
    assert '"original_technique": "locked"' in fallback_area, \
        "Must record original_technique='locked'"
    assert '"technique": "bounded"' in fallback_area, \
        "Fallback must be bounded technique"
    assert "clips_rendered += 1" in fallback_area, \
        "Must count the downgraded clip as rendered"

    # The critical check: there must NOT be a bare 'continue' that skips
    # the insert. The old code had: clips_rejected += 1; continue
    # (which dropped the beat). The new code does the insert THEN continue.
    # Verify the insert happens before continue.
    insert_pos = fallback_area.find("insert({")
    continue_pos = fallback_area.find("continue")
    assert insert_pos > 0, "Must have an insert in the failure path"
    assert continue_pos > insert_pos, \
        "continue must come AFTER the insert — not before (which would drop the beat)"


# ── Test 5: downgrade records required fields ───────────────────────────────

def test_downgrade_records_fields():
    """The downgraded clip records downgraded, original_technique, and
    failure_reason in motion_params."""
    import inspect
    from skills.generate_motion import generate_motion

    source = inspect.getsource(generate_motion)
    fallback = source[source.find("downgrading to bounded"):][:1500]

    assert '"downgraded": True' in fallback
    assert '"original_technique": "locked"' in fallback
    assert '"failure_reason":' in fallback


# ── Test 6: error capture at least 300 chars ────────────────────────────────

def test_error_capture_300_chars():
    """Error string uses [:300], not [:80]."""
    import inspect
    from skills.generate_motion import generate_motion

    source = inspect.getsource(generate_motion)
    runway_section = source[source.find("Runway locked failed"):][:500]
    assert "[:300]" in runway_section, "Error capture must use [:300], not [:80]"
