"""
Bounded camera motion — the camera moves WITHIN the photograph.

The pan animation is rendered by Creatomate on the still image.
No Runway generation, no invented content, no cost per clip.

The viewport must never exceed the source image bounds. This module
computes the animation and asserts that constraint.

Creatomate pan animation properties (from creatomate-node Pan.ts):
  type: "pan"
  start_x, end_x: horizontal focal point (0–100%, 50% = centered)
  start_y, end_y: vertical focal point (0–100%, 50% = centered)
  start_scale, end_scale: zoom level (100% = cover, >100% = zoomed in)
  easing: interpolation curve
  scope: "element"

Source: github.com/Creatomate/creatomate-node/blob/main/src/animations/pan/Pan.ts
Verified: blog example at creatomate.com/blog/json-to-video-practical-examples

Maths:
  cover_factor = max(out_w / src_w, out_h / src_h)

  At pan scale s (ratio >= 1.0):
    displayed_w = src_w * cover_factor * s
    displayed_h = src_h * cover_factor * s
    overflow_w  = displayed_w - out_w
    overflow_h  = displayed_h - out_h

  Viewport centre in x must satisfy:
    min_x = 50 * out_w / (src_w * cover_factor * s)
    max_x = 100 - min_x

  Same for y with out_h / (src_h * cover_factor * s).

  If min_x > max_x (or min_y > max_y), there is no travel in that
  dimension at the chosen scale — reduce the move.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Default pan scale for moves that need travel room.
# 110% provides ~10% of frame width as horizontal travel on a
# width-limited image — enough for a visible 5-second pan without
# excessive zoom.
_DEFAULT_PAN_SCALE = 1.10

# Push-in / pull-back zoom range (start → end).
_PUSH_IN_START = 1.12
_PUSH_IN_END = 1.00
_PULL_BACK_START = 1.00
_PULL_BACK_END = 1.12

# Parallax: subtle combined pan + zoom.
_PARALLAX_PAN_FRAC = 0.4   # use 40% of available travel
_PARALLAX_SCALE_START = 1.08
_PARALLAX_SCALE_END = 1.02


def _safe_range(out_dim: int, src_dim: int, cover: float, s: float):
    """Return (min_pct, max_pct) for one dimension at scale s."""
    displayed = src_dim * cover * s
    if displayed <= 0:
        return (50.0, 50.0)
    half_viewport_frac = out_dim / (2 * displayed)
    min_pct = half_viewport_frac * 100
    max_pct = 100 - min_pct
    if min_pct > max_pct:
        return (50.0, 50.0)
    return (round(min_pct, 2), round(max_pct, 2))


def assert_viewport_in_bounds(
    x: float, y: float, scale: float,
    src_w: int, src_h: int,
    out_w: int = 1920, out_h: int = 1080,
) -> None:
    """Assert the viewport at (x%, y%, scale) is inside the source image.

    Raises AssertionError with detail if the viewport exceeds bounds.
    """
    cover = max(out_w / src_w, out_h / src_h)
    s = scale  # ratio, e.g. 1.1
    min_x, max_x = _safe_range(out_w, src_w, cover, s)
    min_y, max_y = _safe_range(out_h, src_h, cover, s)
    assert x >= min_x - 0.01 and x <= max_x + 0.01, (
        f"x={x:.2f}% outside safe range [{min_x:.2f}%, {max_x:.2f}%] "
        f"at scale {scale:.0%} for {src_w}×{src_h} in {out_w}×{out_h}"
    )
    assert y >= min_y - 0.01 and y <= max_y + 0.01, (
        f"y={y:.2f}% outside safe range [{min_y:.2f}%, {max_y:.2f}%] "
        f"at scale {scale:.0%} for {src_w}×{src_h} in {out_w}×{out_h}"
    )


def compute_pan_travel(
    src_w: int, src_h: int,
    out_w: int = 1920, out_h: int = 1080,
    scale: float = 1.10,
) -> dict:
    """Compute available pan travel for a source image at a given scale.

    Returns dict with travel_x_pct, travel_y_pct (range of movement in
    pan coordinate %), travel_x_px, travel_y_px (output pixels),
    upscale_factor (how much the source is magnified), and
    source_px_visible_w, source_px_visible_h.
    """
    cover = max(out_w / src_w, out_h / src_h)
    s = scale

    min_x, max_x = _safe_range(out_w, src_w, cover, s)
    min_y, max_y = _safe_range(out_h, src_h, cover, s)

    displayed_w = src_w * cover * s
    displayed_h = src_h * cover * s
    overflow_w = max(0, displayed_w - out_w)
    overflow_h = max(0, displayed_h - out_h)

    # Source pixels visible in the output (how much of the source we see)
    visible_w = out_w / (cover * s)
    visible_h = out_h / (cover * s)

    return {
        "travel_x_pct": round(max_x - min_x, 2),
        "travel_y_pct": round(max_y - min_y, 2),
        "travel_x_px": round(overflow_w, 1),
        "travel_y_px": round(overflow_h, 1),
        "upscale_factor": round(cover * s, 3),
        "source_px_visible_w": round(visible_w, 1),
        "source_px_visible_h": round(visible_h, 1),
    }


def compute_bounded_animation(
    requested_motion: str,
    src_w: int, src_h: int,
    out_w: int = 1920, out_h: int = 1080,
    duration: float = 5.0,
) -> dict:
    """Compute a Creatomate pan animation for a bounded beat.

    Maps the director's requested_motion to pan animation parameters
    that keep the viewport inside the source image. If the source is
    too small for the requested travel, the move is REDUCED (shorter
    travel, lower scale) — never dropped.

    Returns a dict ready for Creatomate's animations array:
        {type, easing, scope, start_x, end_x, start_y, end_y,
         start_scale, end_scale, reduced, reduced_reason}

    'reduced' and 'reduced_reason' are metadata (not sent to Creatomate).
    """
    cover = max(out_w / src_w, out_h / src_h)
    reduced = False
    reduced_reason = None

    if requested_motion == "push_in":
        start_s, end_s = _PUSH_IN_START, _PUSH_IN_END
        start_x, end_x = 50.0, 50.0
        start_y, end_y = 50.0, 50.0

    elif requested_motion == "pull_back":
        start_s, end_s = _PULL_BACK_START, _PULL_BACK_END
        start_x, end_x = 50.0, 50.0
        start_y, end_y = 50.0, 50.0

    elif requested_motion in ("pan_left", "pan_right"):
        s = _DEFAULT_PAN_SCALE
        min_x, max_x = _safe_range(out_w, src_w, cover, s)
        travel = max_x - min_x

        if travel < 1.0:
            # Not enough room — reduce scale or give minimal travel
            reduced = True
            reduced_reason = (
                f"Horizontal travel {travel:.1f}% at {s:.0%} scale too small "
                f"for {src_w}×{src_h}; reduced to subtle push_in"
            )
            logger.info("[bounded] %s", reduced_reason)
            # Fall back to a subtle push_in
            start_s, end_s = 1.05, 1.00
            start_x, end_x = 50.0, 50.0
            start_y, end_y = 50.0, 50.0
        else:
            start_s, end_s = s, s
            if requested_motion == "pan_right":
                start_x, end_x = min_x, max_x
            else:
                start_x, end_x = max_x, min_x
            start_y, end_y = 50.0, 50.0
            # Clamp y to safe range
            min_y, max_y = _safe_range(out_h, src_h, cover, s)
            start_y = max(min_y, min(max_y, 50.0))
            end_y = start_y

    elif requested_motion in ("tilt_up", "tilt_down"):
        s = _DEFAULT_PAN_SCALE
        min_y, max_y = _safe_range(out_h, src_h, cover, s)
        travel = max_y - min_y

        # Some images have natural vertical overflow — try 100% first
        if travel < 3.0:
            # Check natural overflow at 100%
            min_y_100, max_y_100 = _safe_range(out_h, src_h, cover, 1.0)
            travel_100 = max_y_100 - min_y_100
            if travel_100 >= 3.0:
                # Use natural overflow without additional zoom
                s = 1.0
                min_y, max_y = min_y_100, max_y_100
                travel = travel_100

        if travel < 1.0:
            reduced = True
            reduced_reason = (
                f"Vertical travel {travel:.1f}% at {s:.0%} scale too small "
                f"for {src_w}×{src_h}; reduced to subtle push_in"
            )
            logger.info("[bounded] %s", reduced_reason)
            start_s, end_s = 1.05, 1.00
            start_x, end_x = 50.0, 50.0
            start_y, end_y = 50.0, 50.0
        else:
            start_s, end_s = s, s
            start_x, end_x = 50.0, 50.0
            # Clamp x to safe range
            min_x, max_x = _safe_range(out_w, src_w, cover, s)
            start_x = max(min_x, min(max_x, 50.0))
            end_x = start_x
            if requested_motion == "tilt_up":
                start_y, end_y = max_y, min_y
            else:
                start_y, end_y = min_y, max_y

    elif requested_motion == "parallax":
        start_s = _PARALLAX_SCALE_START
        end_s = _PARALLAX_SCALE_END
        # Use average scale for position computation
        avg_s = (start_s + end_s) / 2
        min_x, max_x = _safe_range(out_w, src_w, cover, end_s)
        travel = max_x - min_x
        if travel < 1.0:
            # No horizontal room — just do the zoom
            start_x, end_x = 50.0, 50.0
        else:
            shift = travel * _PARALLAX_PAN_FRAC / 2
            start_x = 50.0 - shift
            end_x = 50.0 + shift
            # Clamp to safe range at each scale
            min_x_s, max_x_s = _safe_range(out_w, src_w, cover, start_s)
            min_x_e, max_x_e = _safe_range(out_w, src_w, cover, end_s)
            start_x = max(min_x_s, min(max_x_s, start_x))
            end_x = max(min_x_e, min(max_x_e, end_x))
        start_y, end_y = 50.0, 50.0

    elif requested_motion == "hold":
        # No motion at all — static image at cover
        start_s, end_s = 1.0, 1.0
        start_x, end_x = 50.0, 50.0
        start_y, end_y = 50.0, 50.0

    else:
        # Unknown motion type — default to subtle push_in
        reduced = True
        reduced_reason = f"Unknown motion '{requested_motion}'; defaulting to push_in"
        logger.info("[bounded] %s", reduced_reason)
        start_s, end_s = 1.05, 1.00
        start_x, end_x = 50.0, 50.0
        start_y, end_y = 50.0, 50.0

    # ── Assert bounds at start AND end ─────────────────────────────────
    assert_viewport_in_bounds(start_x, start_y, start_s, src_w, src_h, out_w, out_h)
    assert_viewport_in_bounds(end_x, end_y, end_s, src_w, src_h, out_w, out_h)

    # Also assert at midpoint (linear interpolation) for pans where
    # scale changes during travel.
    if start_s != end_s:
        mid_s = (start_s + end_s) / 2
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        assert_viewport_in_bounds(mid_x, mid_y, mid_s, src_w, src_h, out_w, out_h)

    return {
        "type": "pan",
        "easing": "linear",
        "scope": "element",
        "start_x": f"{start_x:.1f}%",
        "end_x": f"{end_x:.1f}%",
        "start_y": f"{start_y:.1f}%",
        "end_y": f"{end_y:.1f}%",
        "start_scale": f"{start_s * 100:.0f}%",
        "end_scale": f"{end_s * 100:.0f}%",
        # Metadata — not sent to Creatomate
        "reduced": reduced,
        "reduced_reason": reduced_reason,
    }
