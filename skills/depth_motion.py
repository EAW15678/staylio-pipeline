"""
MOTION-D: Depth-based parallax motion rendering.

Translates the director's creative intent into DepthFlow parameters,
overscan maths, and crop geometry. Called by generate_motion when
technique='depth'.

The director expresses:
    camera intent:      lateral_right | lateral_left | push_in
    intensity:          restrained (4%) | moderate (8%)
    hero focal point:   from observation.focal_point

This module translates that into:
    DepthFlow state variables, render dimensions, crop coordinates.

The director never sees Modal, DepthFlow, or overscan arithmetic.
"""

import logging
import math

logger = logging.getLogger(__name__)

# ── Magnitude mapping ─────────────────────────────────────────────────
# intensity name → (amp, description)
# 4% = amp 2.0 (Phase 1-5 "restrained"), 8% = amp 4.0 (Phase 1-5 "moderate")
INTENSITY_AMP = {
    "restrained": 2.0,   # 4% — validated as "best" across all phases
    "moderate": 4.0,     # 8% — validated as "usable" to "best"
}
DEFAULT_INTENSITY = "restrained"

# Delivery resolution
DELIVERY_W, DELIVERY_H = 1920, 1080

# ── Thin-structure risk signals from observations ─────────────────────
# These motion_risk values indicate foreground elements that smear under
# depth reprojection. Overscan does not fix this — the distortion is in
# the interior of the frame, not at the edges.
THIN_STRUCTURE_RISKS = {"thin_railings", "thin_furniture", "pendant_lights"}

# ── Minimum overscan margin (as fraction) ─────────────────────────────
MIN_MARGIN = 0.10  # 10% absolute minimum
DEFAULT_MARGIN = 0.30  # 30% validated default


def check_depth_eligibility(
    photo_id: str,
    image_width: int,
    image_height: int,
    has_depth_map: bool,
    depth_structure: str,
    motion_risks: list,
    requested_motion: str,
    intensity: str = DEFAULT_INTENSITY,
) -> dict:
    """Check whether a photograph is eligible for depth motion.

    Returns:
        dict with:
            eligible: bool
            reason: str (why ineligible, or "eligible")
            margin: float (the margin fraction to use, if eligible)
            render_w: int
            render_h: int
            amp: float
    """
    amp = INTENSITY_AMP.get(intensity, INTENSITY_AMP[DEFAULT_INTENSITY])

    # 1. Must have a cached depth map
    if not has_depth_map:
        return {"eligible": False, "reason": "no cached depth map"}

    # 2. Depth structure must be deep enough for parallax
    if depth_structure in ("shallow", "flat"):
        return {"eligible": False, "reason": f"depth_structure={depth_structure} — insufficient parallax"}

    # 3. Thin-structure risk check
    risk_set = set(motion_risks or [])
    thin_risks = risk_set & THIN_STRUCTURE_RISKS
    if thin_risks:
        return {"eligible": False, "reason": f"thin-structure risk: {', '.join(thin_risks)}"}

    # 4. Calculate required overscan margin from source dimensions
    # The source must fit the oversized render canvas at 16:9 aspect ratio.
    margin = DEFAULT_MARGIN

    # Required render dimensions
    render_w = int(DELIVERY_W * (1 + margin))
    render_h = int(DELIVERY_H * (1 + margin))
    render_w += render_w % 2  # round to even
    render_h += render_h % 2

    # Check if source can cover the render canvas
    # Source aspect ratio may differ from 16:9 — the limiting dimension matters.
    source_aspect = image_width / max(image_height, 1)
    render_aspect = render_w / render_h

    if source_aspect >= render_aspect:
        # Source is wider — height is the limiting dimension
        effective_width = int(image_height * render_aspect)
        if effective_width < render_w:
            # Try minimum margin
            render_w_min = int(DELIVERY_W * (1 + MIN_MARGIN))
            render_h_min = int(DELIVERY_H * (1 + MIN_MARGIN))
            render_w_min += render_w_min % 2
            render_h_min += render_h_min % 2
            eff_w_min = int(image_height * (render_w_min / render_h_min))
            if eff_w_min < render_w_min:
                return {"eligible": False, "reason": f"source too small ({image_width}x{image_height}) for even {int(MIN_MARGIN*100)}% margin"}
            margin = MIN_MARGIN
            render_w, render_h = render_w_min, render_h_min
    else:
        # Source is taller — width is the limiting dimension
        if image_width < render_w:
            render_w_min = int(DELIVERY_W * (1 + MIN_MARGIN))
            render_h_min = int(DELIVERY_H * (1 + MIN_MARGIN))
            render_w_min += render_w_min % 2
            render_h_min += render_h_min % 2
            if image_width < render_w_min:
                return {"eligible": False, "reason": f"source width {image_width}px insufficient for {int(MIN_MARGIN*100)}% margin (need {render_w_min}px)"}
            margin = MIN_MARGIN
            render_w, render_h = render_w_min, render_h_min

    return {
        "eligible": True,
        "reason": "eligible",
        "margin": margin,
        "render_w": render_w,
        "render_h": render_h,
        "amp": amp,
    }


def build_depthflow_script(
    requested_motion: str,
    amp: float,
) -> str:
    """Build the DepthFlow custom scene script for the given motion.

    All trajectories are one-way (self.tau, 0→1). Never sinusoidal.
    """
    if requested_motion in ("lateral_right", "pan_right", "pan_left_to_right"):
        offset_expr = f"({amp} * t, 0.0)"
    elif requested_motion in ("lateral_left", "pan_left", "pan_right_to_left"):
        offset_expr = f"(-{amp} * t, 0.0)"
    else:
        # Default: gentle lateral right for push_in, tilt, or unrecognized
        offset_expr = f"({amp} * t, 0.0)"

    update_code = f"""
    def update(self):
        import math
        t = self.tau  # 0 -> 1 linear, one-way
        self.state.offset = {offset_expr}
        self.state.isometric = 0.60
        self.state.height = 0.15
        self.state.steady = 0.50
"""

    return f"""
from attrs import define
from depthflow.scene import DepthScene

@define
class CustomScene(DepthScene):
{update_code}

if __name__ == "__main__":
    import sys
    scene = CustomScene()
    scene.cli.meta(sys.argv[1:])
"""


def render_depth_beat(
    image_url: str,
    depth_map_url: str,
    requested_motion: str,
    render_w: int,
    render_h: int,
    amp: float,
    duration: float = 3.0,
    fps: int = 30,
) -> dict:
    """Render a depth motion beat via Modal and return the result.

    This calls the Modal depth-render function which:
    1. Downloads image + depth map
    2. Renders at oversized canvas (render_w × render_h)
    3. Crops centre DELIVERY_W × DELIVERY_H with ffmpeg
    4. Uploads to R2

    Returns dict with storage_url, render_seconds, file_size.
    """
    import modal

    render_fn = modal.Function.from_name("depth-render", "render_depth_beat")
    script = build_depthflow_script(requested_motion, amp)

    result = render_fn.remote(
        image_url=image_url,
        depth_map_url=depth_map_url,
        script_code=script,
        render_w=render_w,
        render_h=render_h,
        delivery_w=DELIVERY_W,
        delivery_h=DELIVERY_H,
        duration=duration,
        fps=fps,
    )
    return result
