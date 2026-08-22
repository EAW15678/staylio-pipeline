"""
MOTION-D1: Depth-based parallax motion rendering.

Orthogonal model — camera treatment and content motion are independent:
    camera_treatment:   static | bounded | depth | generative
    content_motion:     [] (list of what should move — water, foliage, fire...)

The director decides both separately. This module handles the camera
treatment side: eligibility, script generation, Modal render.

Thin-structure risk is contextual, not binary. A photograph with
peripheral thin structures at restrained magnitude is eligible;
one with a central chandelier crossing a major depth boundary is not.
"""

import logging

logger = logging.getLogger(__name__)

# ── Magnitude mapping ─────────────────────────────────────────────────
INTENSITY_AMP = {
    "restrained": 2.0,   # 4% — validated as "best" across all phases
    "moderate": 4.0,     # 8% — validated as "usable" to "best"
}
DEFAULT_INTENSITY = "restrained"

# Delivery resolution
DELIVERY_W, DELIVERY_H = 1920, 1080

# ── Minimum overscan margin (as fraction) ─────────────────────────────
MIN_MARGIN = 0.10  # 10% absolute minimum
DEFAULT_MARGIN = 0.30  # 30% validated default

# ── Thin-structure risk: contextual, not binary ───────────────────────
# These motion_risk values indicate thin foreground elements. Whether
# they actually block depth depends on context — are they peripheral
# or central? At what magnitude?
#
# HIGH-CONFIDENCE rejection: thin structures that typically occupy
# the central/hero region and cross major depth boundaries.
# These smear visibly regardless of magnitude or trajectory.
CENTRAL_THIN_STRUCTURE_RISKS = {"pendant_lights", "chandelier"}

# CONTEXTUAL: thin structures that may be peripheral (e.g. railings
# along the frame edge). At restrained magnitude with lateral motion,
# peripheral thin structures are tolerable — Phase 1 proved this
# across dozens of blind-rated clips with f81ead38's railings.
PERIPHERAL_THIN_STRUCTURE_RISKS = {"thin_railings", "thin_furniture"}

# ── Empirical overrides ──────────────────────────────────────────────
# Validated (camera_family, magnitude_range) combinations that override
# generic risk labels. These encode the experimental finding, not the
# specific photo — the same photo at a different trajectory could fail.
#
# Each entry: (camera_families, max_intensity) that are safe despite
# the presence of the named risk.
EMPIRICAL_RISK_OVERRIDES = {
    # Phase 1: thin_railings + restrained lateral → rated "best"
    # across f81ead38 and similar rooftop/deck frames
    "thin_railings": {
        "safe_camera_families": {"lateral_right", "lateral_left", "pan_right",
                                  "pan_left", "pan_right_to_left", "pan_left_to_right"},
        "max_intensity": "restrained",
    },
}


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

    Thin-structure risk is contextual:
    - Central thin structures (chandeliers, pendants) → always reject
    - Peripheral thin structures (railings) → reject at moderate+,
      allow at restrained with lateral trajectories (empirically validated)

    Returns dict with eligible, reason, margin, render_w, render_h, amp,
    and optionally constraints (e.g. max_intensity forced down).
    """
    amp = INTENSITY_AMP.get(intensity, INTENSITY_AMP[DEFAULT_INTENSITY])
    constraints = []

    # 1. Must have a cached depth map
    if not has_depth_map:
        return {"eligible": False, "reason": "no cached depth map"}

    # 2. Depth structure must be deep enough for parallax
    if depth_structure in ("shallow", "flat"):
        return {"eligible": False, "reason": f"depth_structure={depth_structure} — insufficient parallax"}

    # 3. Thin-structure risk — contextual assessment
    risk_set = set(motion_risks or [])

    # 3a. Central thin structures → always reject
    central_risks = risk_set & CENTRAL_THIN_STRUCTURE_RISKS
    if central_risks:
        return {"eligible": False, "reason": f"central thin-structure risk: {', '.join(central_risks)}"}

    # 3b. Peripheral thin structures → check empirical overrides
    peripheral_risks = risk_set & PERIPHERAL_THIN_STRUCTURE_RISKS
    if peripheral_risks:
        all_overridden = True
        for risk in peripheral_risks:
            override = EMPIRICAL_RISK_OVERRIDES.get(risk)
            if not override:
                all_overridden = False
                break
            if requested_motion not in override["safe_camera_families"]:
                all_overridden = False
                break
            # Check intensity constraint
            max_int = override["max_intensity"]
            if max_int == "restrained" and intensity != "restrained":
                # Constrain to restrained
                intensity = "restrained"
                amp = INTENSITY_AMP["restrained"]
                constraints.append(f"intensity constrained to restrained due to {risk}")

        if not all_overridden:
            return {"eligible": False,
                    "reason": f"thin-structure risk: {', '.join(peripheral_risks)} "
                              f"(no empirical override for {requested_motion}/{intensity})"}

    # 4. Calculate required overscan margin from source dimensions
    margin = DEFAULT_MARGIN
    render_w = int(DELIVERY_W * (1 + margin))
    render_h = int(DELIVERY_H * (1 + margin))
    render_w += render_w % 2
    render_h += render_h % 2

    source_aspect = image_width / max(image_height, 1)
    render_aspect = render_w / render_h

    if source_aspect >= render_aspect:
        effective_width = int(image_height * render_aspect)
        if effective_width < render_w:
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
        if image_width < render_w:
            render_w_min = int(DELIVERY_W * (1 + MIN_MARGIN))
            render_h_min = int(DELIVERY_H * (1 + MIN_MARGIN))
            render_w_min += render_w_min % 2
            render_h_min += render_h_min % 2
            if image_width < render_w_min:
                return {"eligible": False, "reason": f"source width {image_width}px insufficient for {int(MIN_MARGIN*100)}% margin (need {render_w_min}px)"}
            margin = MIN_MARGIN
            render_w, render_h = render_w_min, render_h_min

    result = {
        "eligible": True,
        "reason": "eligible",
        "margin": margin,
        "render_w": render_w,
        "render_h": render_h,
        "amp": amp,
    }
    if constraints:
        result["constraints"] = constraints
    return result


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
    """Render a depth motion beat via Modal and return the result."""
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
