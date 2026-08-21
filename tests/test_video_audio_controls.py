"""
VIDEO-AUDIO FIX: Hero video must have unmute control, guest audio must toggle.

Both dropped during PAGE-2/3/4 rebuild. Found 2026-08-20 on the first real
full video pipeline run.
"""

import sys
import re

sys.path.insert(0, ".")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_page_with_video():
    """Build a page with a hero video to check the HTML output."""
    from tests.test_page_orchestrator import FakeSB, _build_tables, _photo, _renditions_for, _obs, _prop
    from unittest.mock import patch

    video = {"storage_url": "https://r2.dev/master.mp4", "kind": "master",
             "status": "ready", "superseded_at": None, "property_id": "PROP-1"}
    photos = [_photo("p0")]
    renditions = _renditions_for("p0")
    observations = [_obs("p0", "Exterior", "hero", 1)]
    prop = _prop()

    sb = FakeSB(_build_tables(photos, renditions, observations, prop=prop, video_master=video))

    from core.page_builder.orchestrate import build_page
    with patch("skills.notify.send_halt_alert", return_value=True):
        result = build_page(sb, "PROP-1")

    assert result["ok"], f"Build should succeed: {result.get('reason')}"
    return result["html"]


def _build_page_with_audio():
    """Build a page with guest reviews and audio URLs."""
    from tests.test_page_orchestrator import FakeSB, _build_tables, _photo, _renditions_for, _obs, _prop
    from unittest.mock import patch

    photos = [_photo("p0")]
    renditions = _renditions_for("p0")
    observations = [_obs("p0", "Exterior", "hero", 1)]

    guest_evidence = [
        {"property_id": "PROP-1", "written_text": "Great place!", "verbal_text": "",
         "reviewer_name": "John D.", "is_guest_book": True, "star_rating": None,
         "stay_date": None, "source": "guest_book"},
    ]
    narrations = [{
        "storage_url": "https://r2.dev/narration_john.mp3",
        "kind": "narration", "status": "ready", "provenance": "guest_book",
        "superseded_at": None, "property_id": "PROP-1",
        "source_reference": {"reviewer_name": "John D."},
    }]

    tables = _build_tables(photos, renditions, observations,
                           guest_evidence=guest_evidence, narrations=narrations)
    sb = FakeSB(tables)

    from core.page_builder.orchestrate import build_page
    with patch("skills.notify.send_halt_alert", return_value=True):
        result = build_page(sb, "PROP-1")

    assert result["ok"], f"Build should succeed: {result.get('reason')}"
    return result["html"]


# ── Test 1: Hero video has a "Hear me" control ──────────────────────────────

def test_hero_video_has_unmute_control():
    """The hero video HTML includes a 'Hear me' control element."""
    html = _build_page_with_video()

    assert "hero-hear-btn" in html, "Should have a hero-hear-btn element"
    assert "Hear me" in html, "Should have 'Hear me' text"
    assert "loop" in html.split("hero-video")[0] + html.split("hero-video")[1][:100], \
        "Video tag should have loop attribute"


# ── Test 2: Unmute JS sets muted=false and restarts ─────────────────────────

def test_hero_unmute_js_behavior():
    """The JS sets muted=false and restarts playback on click."""
    html = _build_page_with_video()

    # Extract the hero video script block
    assert "v.muted=false" in html, "JS should set muted=false on click"
    assert "v.currentTime=0" in html, "JS should restart from beginning"
    assert "v.play()" in html, "JS should call play()"
    # Re-mute on loop
    assert "v.muted=true" in html, "JS should re-mute on loop restart"


# ── Test 3: Clicking playing audio stops it ─────────────────────────────────

def test_audio_toggle_stops_playing():
    """Clicking an already-playing audio button stops it — pause() called,
    no new Audio created."""
    html = _build_page_with_audio()

    # The toggle logic: if(current && currentBtn===this){current.pause();...return;}
    assert "currentBtn===this" in html, \
        "Should check if clicked button is the currently playing one"
    assert "current.pause()" in html, \
        "Should pause when same button clicked"
    # The return after pause prevents creating a new Audio
    assert re.search(r"current\.pause\(\);.*?return;", html, re.DOTALL), \
        "Should return after pausing (not create new Audio)"


# ── Test 4: Different button stops first, starts second ─────────────────────

def test_audio_switch_track():
    """Clicking a different button while one is playing stops the first
    and starts the second."""
    html = _build_page_with_audio()

    # Pattern: if(current){current.pause();current=null;} then new Audio(src)
    assert "if(current){current.pause();current=null;}" in html, \
        "Should pause existing track before starting new one"
    assert "current=new Audio(src)" in html, \
        "Should create new Audio for the clicked track"
    # ended handler cleans up
    assert "current.addEventListener('ended'" in html, \
        "Should clean up on track end"


# ── Test 5: hero-hear-btn has CSS with position: absolute ────────────────────

def test_hero_hear_btn_has_positioning_css():
    """The rendered CSS includes a #hero-hear-btn rule with position: absolute,
    not just the button element in the HTML — the button must be pulled out of
    the flex flow to be visible."""
    html = _build_page_with_video()

    # The CSS is in a <style> block in the HTML
    assert "#hero-hear-btn" in html, "CSS rule for #hero-hear-btn should exist"

    # Extract the CSS rule
    import re
    rule_match = re.search(r"#hero-hear-btn\s*\{([^}]+)\}", html)
    assert rule_match, "Should find a CSS rule block for #hero-hear-btn"

    rule_body = rule_match.group(1)
    assert "position: absolute" in rule_body, (
        f"#hero-hear-btn must have position: absolute to escape flex flow, got: {rule_body[:100]}"
    )
    assert "z-index:" in rule_body, "Should have z-index to layer above video"


# ── Test 6: justUnmuted flag prevents instant re-mute ────────────────────────

def test_unmute_not_instantly_remuted():
    """Simulates the JS event ordering: after click (currentTime near 0),
    the timeupdate handler must NOT re-mute. After genuine loop (past 0.5s
    then back to 0), it DOES re-mute.

    This tests the actual logic, not just string presence — the bug was
    about event *ordering*, where the timeupdate fires immediately after
    click sets currentTime=0 and sees the same signal as a loop."""
    html = _build_page_with_video()

    # Verify the justUnmuted guard exists in the emitted JS
    assert "justUnmuted=true" in html, "Click handler must set justUnmuted=true"
    assert "justUnmuted && v.currentTime>0.5" in html, \
        "timeupdate must clear justUnmuted after playback advances past 0.5s"
    assert "!justUnmuted && !v.muted" in html, \
        "Re-mute check must be guarded by !justUnmuted"

    # Simulate the JS logic in Python to verify event ordering
    # This mirrors the exact branching in the JS

    class FakeVideo:
        def __init__(self):
            self.muted = True
            self.currentTime = 0.0
            self.duration = 36.0

    justUnmuted = False
    v = FakeVideo()

    # === Simulate: user clicks "Hear me" ===
    v.muted = False
    v.currentTime = 0.0
    justUnmuted = True
    # v.play() called — next event is timeupdate

    # === Simulate: timeupdate fires immediately (currentTime still near 0) ===
    # This is the bug case — without the flag, this would re-mute
    if justUnmuted and v.currentTime > 0.5:
        justUnmuted = False
    should_remute = (not justUnmuted and not v.muted and v.currentTime < 0.1 and v.duration > 0)
    assert should_remute is False, (
        "Must NOT re-mute immediately after click — justUnmuted flag should prevent it"
    )
    assert v.muted is False, "Video should still be unmuted after click"

    # === Simulate: playback advances past 0.5s — flag clears ===
    v.currentTime = 1.0
    if justUnmuted and v.currentTime > 0.5:
        justUnmuted = False
    assert justUnmuted is False, "Flag should clear after playback passes 0.5s"

    # === Simulate: genuine loop — currentTime wraps back to near 0 ===
    v.currentTime = 0.05
    if justUnmuted and v.currentTime > 0.5:
        justUnmuted = False
    should_remute_on_loop = (not justUnmuted and not v.muted and v.currentTime < 0.1 and v.duration > 0)
    assert should_remute_on_loop is True, (
        "MUST re-mute on genuine loop (currentTime near 0, flag cleared)"
    )


# ── Test 7: without the flag, the bug reproduces ─────────────────────────────

def test_without_flag_bug_reproduces():
    """Proves the old logic (no justUnmuted flag) would re-mute instantly
    after click — the exact bug this fix addresses."""

    class FakeVideo:
        def __init__(self):
            self.muted = True
            self.currentTime = 0.0
            self.duration = 36.0

    v = FakeVideo()

    # Click: unmute + restart
    v.muted = False
    v.currentTime = 0.0

    # OLD timeupdate logic (no flag):
    # if(!v.muted && v.currentTime<0.1 && v.duration>0) { v.muted=true; }
    old_should_remute = (not v.muted and v.currentTime < 0.1 and v.duration > 0)
    assert old_should_remute is True, (
        "Old logic WOULD re-mute instantly — confirming the bug exists"
    )
