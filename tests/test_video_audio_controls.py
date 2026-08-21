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
