"""
Throwaway: resolution comparison for expanded curation prompt.
Tests 3 Vista Azule frames at 320x240 (current) vs 640x480 (2x).
Compares the 10 new shot_inventory fields to see where resolution matters.

SPENDS REAL MONEY — ~$2 for 6 Claude Sonnet vision calls.
"""

import asyncio
import io
import json
import os
import sys

import anthropic
import httpx
from PIL import Image

R2_BASE = "https://pub-dca6781384314541a5329feab1da5de4.r2.dev/a1b2c3d4-0001-0001-0001-000000000001/enhanced"

# Three frames chosen to stress different fields:
# - exterior: depth_structure, foreground_elements, motion_risk (railings, architecture)
# - pool: motion_risk (reflections, water_surface), depth_structure
# - living_room: physical_room_id test, negative_space
FRAMES = [
    ("exterior_b", "photo_093_2cfa4018.jpg"),
    ("pool", "photo_057_243e2320.jpg"),
    ("living_room", "photo_020_e70156f2.jpg"),
]

NEW_FIELDS = [
    "depth_structure", "foreground_elements", "frame_element",
    "beyond_frame_element", "space_direction", "light_direction",
    "light_quality", "time_of_day_read", "negative_space",
    "depth_tier", "motion_affordance", "motion_risk",
    "physical_room_id", "visual_duplicate_of",
]

PROMPT = """\
You are inspecting a vacation rental property photo.

For this image, report these factual visual attributes as JSON:

{
  "depth_structure": "flat | shallow | deep",
  "foreground_elements": ["list of near-field objects that would separate from background under camera translation"],
  "frame_element": "doorway, arch, window or opening visible in frame, or null",
  "beyond_frame_element": "what is visible through frame_element, or null",
  "space_direction": "into_frame | left | right | up | down | none",
  "light_direction": "front | back | side_left | side_right | top | diffuse",
  "light_quality": "hard | soft | mixed | flat",
  "time_of_day_read": "dawn | morning | midday | golden | dusk | night | interior_ambiguous",
  "negative_space": [{"region": "top_left|top_center|...|bottom_right", "size": "small|medium|large", "contrast": "high|medium|low"}],
  "depth_tier": "wide | medium | detail",
  "motion_affordance": ["push_in", "pull_back", "pan_left", "pan_right", "tilt_up", "tilt_down", "parallax", "orbit", "hold"],
  "motion_risk": ["repeating_geometry", "straight_architectural_lines", "reflections", "water_surface", "fine_text", "thin_railings", "patterned_fabric"],
  "physical_room_id": "e.g. bedroom_1, kitchen, pool_deck",
  "visual_duplicate_of": null
}

Rules for motion_affordance:
- push_in requires depth_structure != "flat"
- parallax requires non-empty foreground_elements
- Only list moves the frame genuinely supports

Rules for motion_risk:
- MUST include "reflections" if reflective surfaces visible (glass, polished stone, mirrors)
- MUST include "water_surface" if water is visible

Return ONLY valid JSON, no markdown."""


def resize_image(img_bytes: bytes, cell_w: int, cell_h: int) -> bytes:
    """Resize image to a specific cell size, return JPEG bytes."""
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((cell_w, cell_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def call_claude_with_image(image_bytes: bytes, label: str) -> dict:
    """Single Claude Sonnet vision call with one image."""
    import base64
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    b64 = base64.b64encode(image_bytes).decode()

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64,
                    },
                },
                {
                    "type": "text",
                    "text": PROMPT,
                },
            ],
        }],
    )

    raw = resp.content[0].text.strip()
    # Strip code fences
    import re
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def main():
    results = {}

    for name, filename in FRAMES:
        url = f"{R2_BASE}/{filename}"
        print(f"\nFetching {name}: {filename}...")

        with httpx.Client(timeout=30) as client:
            resp = client.get(url)
            resp.raise_for_status()
            original_bytes = resp.content

        print(f"  Original: {len(original_bytes)//1024}KB")

        for label, (w, h) in [("320x240", (320, 240)), ("640x480", (640, 480))]:
            resized = resize_image(original_bytes, w, h)
            print(f"  Calling Claude with {label} ({len(resized)//1024}KB)...")
            try:
                result = call_claude_with_image(resized, f"{name}_{label}")
                results[f"{name}_{label}"] = result
                print(f"  OK: {json.dumps(result, indent=2)[:200]}...")
            except Exception as exc:
                print(f"  FAILED: {exc}")
                results[f"{name}_{label}"] = {"error": str(exc)}

    # Comparison
    print("\n" + "=" * 70)
    print("  RESOLUTION COMPARISON: 320x240 vs 640x480")
    print("=" * 70)

    for name, _ in FRAMES:
        low = results.get(f"{name}_320x240", {})
        high = results.get(f"{name}_640x480", {})

        if "error" in low or "error" in high:
            print(f"\n  {name}: SKIPPED (error in one or both)")
            continue

        print(f"\n  {name}:")
        disagreements = 0
        for field in NEW_FIELDS:
            lv = low.get(field)
            hv = high.get(field)
            if json.dumps(lv, sort_keys=True) != json.dumps(hv, sort_keys=True):
                disagreements += 1
                print(f"    DISAGREE {field}:")
                print(f"      320x240: {json.dumps(lv)[:100]}")
                print(f"      640x480: {json.dumps(hv)[:100]}")
            else:
                print(f"    agree    {field}: {json.dumps(lv)[:80]}")

        print(f"    --- {disagreements} disagreements out of {len(NEW_FIELDS)} fields")

    # Save full results
    out_path = os.path.expanduser("~/Desktop/resolution_comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
