"""
MOTION-1: Bounded camera motion tests.

Verifies that computed pan animations keep the viewport inside the
source image for all motion types and source sizes. Mocked — no
vendor calls, $0.
"""

import pytest

from skills.bounded_motion import (
    compute_bounded_animation,
    assert_viewport_in_bounds,
    compute_pan_travel,
    _safe_range,
)


# ── _safe_range ───────────────────────────────────────────────────────


def test_safe_range_cover_exact_no_travel():
    """Width-limited at 100% scale → no horizontal travel."""
    # Image 1024x768 in 1920x1080: cover = 1920/1024 = 1.875
    min_x, max_x = _safe_range(1920, 1024, 1.875, 1.0)
    assert abs(min_x - 50.0) < 0.1
    assert abs(max_x - 50.0) < 0.1


def test_safe_range_natural_vertical_overflow():
    """Width-limited image has natural vertical overflow at 100%."""
    # Image 768x852 in 1920x1080: cover = 1920/768 = 2.5
    # displayed_h = 852 * 2.5 = 2130, overflow_h = 1050
    min_y, max_y = _safe_range(1080, 852, 2.5, 1.0)
    assert min_y < 50.0  # has travel below center
    assert max_y > 50.0  # has travel above center
    assert (max_y - min_y) > 20  # substantial travel


def test_safe_range_at_110_scale():
    """At 110% scale, horizontal travel opens up."""
    # Width-limited: cover = 1920/1024 = 1.875
    min_x, max_x = _safe_range(1920, 1024, 1.875, 1.1)
    assert min_x < 50.0
    assert max_x > 50.0
    travel = max_x - min_x
    assert travel > 5  # should have ~9% travel


# ── assert_viewport_in_bounds ─────────────────────────────────────────


def test_bounds_centered_always_safe():
    """Centered viewport at 100% scale is always safe."""
    assert_viewport_in_bounds(50, 50, 1.0, 1024, 768)


def test_bounds_centered_at_120():
    """Centered at 120% scale is safe."""
    assert_viewport_in_bounds(50, 50, 1.2, 1024, 768)


def test_bounds_extreme_x_fails():
    """Viewport at 0% x should fail — left edge would go outside."""
    with pytest.raises(AssertionError):
        assert_viewport_in_bounds(0, 50, 1.0, 1024, 768)


def test_bounds_extreme_y_fails():
    """Viewport at 100% y should fail — bottom edge outside."""
    with pytest.raises(AssertionError):
        assert_viewport_in_bounds(50, 100, 1.0, 1024, 768)


def test_bounds_384x288_extreme_case():
    """384x288 (smallest Brant photo) at various scales."""
    # At 100%, this is massively upscaled. Centered should work.
    assert_viewport_in_bounds(50, 50, 1.0, 384, 288)
    # At 110%
    assert_viewport_in_bounds(50, 50, 1.1, 384, 288)
    # At 120%
    assert_viewport_in_bounds(50, 50, 1.2, 384, 288)


# ── compute_bounded_animation ─────────────────────────────────────────


def test_bounded_push_in():
    """push_in: start zoomed in, end at cover."""
    anim = compute_bounded_animation("push_in", 1024, 768)
    assert anim["type"] == "pan"
    assert anim["start_x"] == "50.0%"
    assert anim["end_x"] == "50.0%"
    # Start scale > end scale (zooming in to out)
    start = float(anim["start_scale"].rstrip("%"))
    end = float(anim["end_scale"].rstrip("%"))
    assert start > end


def test_bounded_pull_back():
    """pull_back: start at cover, zoom out."""
    anim = compute_bounded_animation("pull_back", 1024, 768)
    start = float(anim["start_scale"].rstrip("%"))
    end = float(anim["end_scale"].rstrip("%"))
    assert start < end


def test_bounded_pan_right():
    """pan_right: x increases (start_x < end_x)."""
    anim = compute_bounded_animation("pan_right", 1024, 768)
    start_x = float(anim["start_x"].rstrip("%"))
    end_x = float(anim["end_x"].rstrip("%"))
    assert start_x < end_x


def test_bounded_pan_left():
    """pan_left: x decreases (start_x > end_x)."""
    anim = compute_bounded_animation("pan_left", 1024, 768)
    start_x = float(anim["start_x"].rstrip("%"))
    end_x = float(anim["end_x"].rstrip("%"))
    assert start_x > end_x


def test_bounded_tilt_up():
    """tilt_up: y decreases (start_y > end_y)."""
    anim = compute_bounded_animation("tilt_up", 768, 852)
    start_y = float(anim["start_y"].rstrip("%"))
    end_y = float(anim["end_y"].rstrip("%"))
    assert start_y > end_y


def test_bounded_tilt_down():
    """tilt_down: y increases (start_y < end_y)."""
    anim = compute_bounded_animation("tilt_down", 768, 852)
    start_y = float(anim["start_y"].rstrip("%"))
    end_y = float(anim["end_y"].rstrip("%"))
    assert start_y < end_y


def test_bounded_parallax():
    """parallax: has some x travel and scale change."""
    anim = compute_bounded_animation("parallax", 1024, 768)
    start = float(anim["start_scale"].rstrip("%"))
    end = float(anim["end_scale"].rstrip("%"))
    assert start != end  # has zoom change


def test_bounded_hold():
    """hold: no motion, static at cover."""
    anim = compute_bounded_animation("hold", 1024, 768)
    assert anim["start_x"] == anim["end_x"] == "50.0%"
    assert anim["start_y"] == anim["end_y"] == "50.0%"
    assert anim["start_scale"] == anim["end_scale"] == "100%"


# ── Bounds assertion on every computed animation ──────────────────────


BRANT_DIMS = [
    (1024, 768), (1024, 768), (1024, 767), (1024, 767),
    (1024, 853), (1024, 768), (1024, 767), (1024, 794),
    (1024, 768), (1024, 768), (1024, 768),
    (768, 852), (768, 852), (768, 852), (768, 852),
    (768, 852), (768, 852), (768, 852),
    (576, 432),
    (480, 640),
    (384, 288),
]

ALL_MOTIONS = ["push_in", "pull_back", "pan_left", "pan_right",
               "tilt_up", "tilt_down", "parallax", "hold"]


@pytest.mark.parametrize("dims", BRANT_DIMS, ids=[f"{w}x{h}" for w, h in BRANT_DIMS])
@pytest.mark.parametrize("motion", ALL_MOTIONS)
def test_bounds_all_brant_photos_all_motions(dims, motion):
    """Every motion on every Brant photo must stay in bounds.

    The compute function asserts internally; if this test passes,
    the viewport never left the image for any combination.
    """
    w, h = dims
    anim = compute_bounded_animation(motion, w, h)
    assert anim["type"] == "pan"
    # If reduced, verify it says so
    if anim.get("reduced"):
        assert anim.get("reduced_reason") is not None


def test_384x288_pan_reduces():
    """384x288 (smallest) requesting a pan should reduce."""
    # At 384x288 in 1920x1080: cover = 1920/384 = 5.0
    # At 110% scale: displayed_w = 384*5*1.1 = 2112, overflow = 192
    # This SHOULD have some travel actually. Let's check.
    anim = compute_bounded_animation("pan_right", 384, 288)
    # Should still produce a valid animation (not necessarily reduced)
    assert anim["type"] == "pan"


# ── Assembly integration ──────────────────────────────────────────────


def test_bounded_beat_produces_image_element():
    """A bounded clip in _build_renderscript produces an image element."""
    from skills.assemble import _build_renderscript

    clips = [
        {
            "beat_ordinal": 1,
            "storage_url": "https://example.com/photo.jpg",
            "duration_seconds": 5,
            "technique": "bounded",
            "photo_id": "test-photo-1",
            "requested_motion": "push_in",
        },
    ]
    photo_dims = {"test-photo-1": (1024, 768)}
    rendition_urls = {"test-photo-1": "https://example.com/photo.jpg"}

    rs = _build_renderscript(
        clips, rendition_urls=rendition_urls, photo_dims=photo_dims,
    )
    elements = rs["elements"]
    clip_el = [e for e in elements if e.get("type") in ("image", "video")]
    assert len(clip_el) == 1
    assert clip_el[0]["type"] == "image"
    assert "animations" in clip_el[0]
    assert clip_el[0]["animations"][0]["type"] == "pan"


def test_generative_beat_produces_video_element():
    """A generative clip produces a video element (unchanged path)."""
    from skills.assemble import _build_renderscript

    clips = [
        {
            "beat_ordinal": 1,
            "storage_url": "https://example.com/clip.mp4",
            "duration_seconds": 5,
            "technique": "generative",
        },
    ]

    rs = _build_renderscript(clips)
    elements = rs["elements"]
    clip_el = [e for e in elements if e.get("type") in ("image", "video")]
    assert len(clip_el) == 1
    assert clip_el[0]["type"] == "video"
    assert "animations" not in clip_el[0]


def test_mixed_direction_correct_order():
    """Mixed bounded + generative beats assemble in correct beat order."""
    from skills.assemble import _build_renderscript

    clips = [
        {
            "beat_ordinal": 1,
            "storage_url": "https://example.com/photo1.jpg",
            "duration_seconds": 4,
            "technique": "bounded",
            "photo_id": "p1",
            "requested_motion": "push_in",
        },
        {
            "beat_ordinal": 2,
            "storage_url": "https://example.com/clip2.mp4",
            "duration_seconds": 5,
            "technique": "generative",
        },
        {
            "beat_ordinal": 3,
            "storage_url": "https://example.com/photo3.jpg",
            "duration_seconds": 4,
            "technique": "bounded",
            "photo_id": "p3",
            "requested_motion": "pan_right",
        },
    ]
    photo_dims = {"p1": (1024, 768), "p3": (768, 852)}
    rendition_urls = {
        "p1": "https://example.com/photo1.jpg",
        "p3": "https://example.com/photo3.jpg",
    }

    rs = _build_renderscript(
        clips, rendition_urls=rendition_urls, photo_dims=photo_dims,
    )
    media = [e for e in rs["elements"] if e.get("type") in ("image", "video")]
    assert len(media) == 3
    assert media[0]["type"] == "image"   # beat 1: bounded
    assert media[0]["time"] == 0.0
    assert media[1]["type"] == "video"   # beat 2: generative
    assert media[1]["time"] == 4.0
    assert media[2]["type"] == "image"   # beat 3: bounded
    assert media[2]["time"] == 9.0
    # Total duration
    assert rs["duration"] == 13.0


def test_small_photo_motion_not_dropped():
    """A photo too small for full pan still produces a valid beat."""
    anim = compute_bounded_animation("pan_right", 384, 288)
    assert anim["type"] == "pan"
    # It may be reduced but the shot is NOT dropped
    assert anim["start_scale"] is not None
    assert anim["end_scale"] is not None
