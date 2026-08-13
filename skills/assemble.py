"""
Skill: assemble — composite clips + narration + music into a master video.

Ports Agent 8 Stage 7. The only genuinely new code: source-based
RenderScript assembly, NOT Creatomate templates (Erick's standing
ruling — no Creatomate templates ever).

⚠️ RENDERSCRIPT ASSESSMENT:
The existing Agent 8 assembly.py uses Creatomate's template_id-based
rendering. The substrate port MUST use source-based RenderScript
(JSON that describes the composition without a pre-built template).

Creatomate's RenderScript API accepts a JSON source object that defines
clips, audio tracks, timing, and overlays programmatically. This is
the correct path — it gives us full control over composition without
template dependency.

HOWEVER: the exact RenderScript timing/format for multi-clip + narration
+ music + audio ducking has NOT been validated against Creatomate's
API. The structure below is based on Creatomate's documented JSON source
format but may need adjustment when the first real render is attempted.

Usage:
    from skills.assemble import assemble
    result = assemble("property_id", direction_id="uuid")
"""

import hashlib
import json
import logging
import os
import uuid

import httpx

from skills.contract import (
    SkillResult, get_substrate, require_env,
    record_run, record_step, complete_step, complete_run, emit_cost,
    skills_r2_upload,
)

logger = logging.getLogger(__name__)

CREATOMATE_API_BASE = "https://api.creatomate.com/v2"
CREATOMATE_COST_PER_RENDER_MINUTE = 0.84

# Canvas dimensions by aspect ratio
ASPECT_DIMENSIONS = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
}


def _build_renderscript(clips: list, narration=None, music=None,
                        aspect_ratio="16:9", title_text=None, location_text=None) -> dict:
    """Build a Creatomate source-based RenderScript from artifacts.

    Source-based (no template_id): the JSON source object defines the
    full composition. This is Erick's standing ruling — no Creatomate
    templates ever.

    ⚠️ This RenderScript structure is based on Creatomate's documented
    JSON source format. Timing and layering may need adjustment on
    first real render.
    """
    # Build clip elements in beat order
    clip_elements = []
    time_offset = 0.0
    for clip in sorted(clips, key=lambda c: c.get("beat_ordinal", 0)):
        duration = clip.get("duration_seconds", 5)
        clip_elements.append({
            "type": "video",
            "source": clip["storage_url"],
            "time": time_offset,
            "duration": duration,
            "fit": "cover",
        })
        time_offset += duration

    total_duration = time_offset

    # Build audio layers
    audio_elements = []
    if narration and narration.get("storage_url"):
        audio_elements.append({
            "type": "audio",
            "source": narration["storage_url"],
            "time": 0,
            "duration": min(narration.get("duration_seconds", total_duration), total_duration),
            "volume": "100%",
        })

    if music and music.get("storage_url"):
        # Music at lower volume when narration is present (ducking)
        music_volume = "30%" if narration else "80%"
        audio_elements.append({
            "type": "audio",
            "source": music["storage_url"],
            "time": 0,
            "duration": total_duration,
            "volume": music_volume,
        })

    width, height = ASPECT_DIMENSIONS.get(aspect_ratio, (1920, 1080))

    # ── Title annotations (landing page hero only) ──────────────────────
    # Opens and closes with property name + location.
    # Closing treatment is lighter/more transparent per Erick's note.
    title_elements = []
    if title_text:
        location_line = location_text or ""
        # Opening title: first 3 seconds, full opacity
        title_elements.append({
            "type": "text",
            "text": title_text,
            "time": 0,
            "duration": 3,
            "x": "50%",
            "y": "85%",
            "width": "80%",
            "font_family": "Open Sans",
            "font_weight": "700",
            "font_size": "48",
            "color": "#ffffff",
            "shadow_color": "rgba(0,0,0,0.6)",
            "shadow_blur": "8",
            "x_alignment": "50%",
            "y_alignment": "50%",
        })
        if location_line:
            title_elements.append({
                "type": "text",
                "text": location_line,
                "time": 0,
                "duration": 3,
                "x": "50%",
                "y": "92%",
                "width": "80%",
                "font_family": "Open Sans",
                "font_weight": "400",
                "font_size": "28",
                "color": "rgba(255,255,255,0.9)",
                "shadow_color": "rgba(0,0,0,0.5)",
                "shadow_blur": "6",
                "x_alignment": "50%",
                "y_alignment": "50%",
            })
        # Closing title: last 3 seconds, more transparent
        title_elements.append({
            "type": "text",
            "text": title_text,
            "time": total_duration - 3,
            "duration": 3,
            "x": "50%",
            "y": "85%",
            "width": "80%",
            "font_family": "Open Sans",
            "font_weight": "700",
            "font_size": "42",
            "color": "rgba(255,255,255,0.7)",
            "shadow_color": "rgba(0,0,0,0.4)",
            "shadow_blur": "6",
            "x_alignment": "50%",
            "y_alignment": "50%",
        })
        if location_line:
            title_elements.append({
                "type": "text",
                "text": location_line,
                "time": total_duration - 3,
                "duration": 3,
                "x": "50%",
                "y": "92%",
                "width": "80%",
                "font_family": "Open Sans",
                "font_weight": "400",
                "font_size": "24",
                "color": "rgba(255,255,255,0.5)",
                "shadow_color": "rgba(0,0,0,0.3)",
                "shadow_blur": "4",
                "x_alignment": "50%",
                "y_alignment": "50%",
            })

    return {
        "output_format": "mp4",
        "width": width,
        "height": height,
        "duration": total_duration,
        "elements": clip_elements + audio_elements + title_elements,
    }


def assemble(
    property_id: str,
    direction_id: str,
    *,
    aspect_ratio: str = "16:9",
    force: bool = False,
) -> SkillResult:
    """Assemble a master video from clips + narration + music.

    aspect_ratio: "16:9" for landing page hero, "9:16" for social variants.
    Title annotations (property name + location) are added for hero only.

    Returns SkillResult.ok({artifact_id, duration_seconds, cost_usd})
    """
    try:
        sb = get_substrate()
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    try:
        creatomate_key = require_env("CREATOMATE_API_KEY", "Creatomate video rendering")
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    # ── Load clips for this direction ───────────────────────────────────
    clips_resp = sb.table("video_artifacts").select("*").eq(
        "direction_id", direction_id
    ).eq("kind", "clip").eq("status", "ready").is_(
        "superseded_at", "null"
    ).order("beat_ordinal").execute()
    clips = clips_resp.data or []

    if not clips:
        return SkillResult.failed(
            reason="No ready clips for this direction — run generate_motion first",
            attempted=0, succeeded=0, failed_count=0,
        )

    # ── Load narration (optional) ───────────────────────────────────────
    narr_resp = sb.table("video_artifacts").select("*").eq(
        "direction_id", direction_id
    ).eq("kind", "narration").eq("status", "ready").is_(
        "superseded_at", "null"
    ).limit(1).execute()
    narration = narr_resp.data[0] if narr_resp.data else None

    # ── Load music (optional) ───────────────────────────────────────────
    music_resp = sb.table("video_artifacts").select("*").eq(
        "direction_id", direction_id
    ).eq("kind", "music").eq("status", "ready").is_(
        "superseded_at", "null"
    ).limit(1).execute()
    music = music_resp.data[0] if music_resp.data else None

    # ── Input hash ──────────────────────────────────────────────────────
    hash_input = json.dumps({
        "clip_ids": sorted([c["artifact_id"] for c in clips]),
        "narration_id": narration["artifact_id"] if narration else None,
        "music_id": music["artifact_id"] if music else None,
    }, sort_keys=True)
    input_hash = hashlib.sha256(hash_input.encode()).hexdigest()

    if not force:
        existing = sb.table("video_artifacts").select("artifact_id", count="exact").eq(
            "input_hash", input_hash
        ).eq("kind", "master").eq("status", "ready").is_(
            "superseded_at", "null"
        ).limit(0).execute()
        if existing.count > 0:
            return SkillResult.noop("Master already assembled (input_hash match).", {})

    # ── Record run ──────────────────────────────────────────────────────
    run_id = record_run(sb, property_id, "monthly_cycle")
    step_id = record_step(sb, run_id, "assemble")

    # ── Load property name + location for title annotations ────────────
    title_text = None
    location_text = None
    if aspect_ratio == "16:9":  # Title annotations on hero only
        prop_resp = sb.table("properties").select("name, city, state_region").eq(
            "id", property_id).limit(1).execute()
        if prop_resp.data:
            p = prop_resp.data[0]
            title_text = p.get("name")
            city = p.get("city") or ""
            state = p.get("state_region") or ""
            location_text = ", ".join(filter(None, [city, state])) or None

    # ── Build RenderScript ──────────────────────────────────────────────
    renderscript = _build_renderscript(
        clips, narration, music,
        aspect_ratio=aspect_ratio, title_text=title_text, location_text=location_text,
    )
    total_duration = renderscript["duration"]

    # ── Render via Creatomate ───────────────────────────────────────────
    try:
        resp = httpx.post(
            f"{CREATOMATE_API_BASE}/renders",
            headers={
                "Authorization": f"Bearer {creatomate_key}",
                "Content-Type": "application/json",
            },
            json={"source": renderscript},
            timeout=30,
        )
        resp.raise_for_status()
        render_data = resp.json()

        # Creatomate returns a list; take the first render
        render = render_data[0] if isinstance(render_data, list) else render_data
        render_id = render.get("id")

        # Poll for completion
        import time
        for _ in range(60):
            time.sleep(5)
            status_resp = httpx.get(
                f"{CREATOMATE_API_BASE}/renders/{render_id}",
                headers={"Authorization": f"Bearer {creatomate_key}"},
                timeout=15,
            )
            status_resp.raise_for_status()
            render_status = status_resp.json()

            if render_status.get("status") == "succeeded":
                break
            elif render_status.get("status") in ("failed", "cancelled"):
                raise RuntimeError(
                    f"Creatomate render {render_id} {render_status['status']}: "
                    f"{render_status.get('error_message', 'unknown')}"
                )
        else:
            raise RuntimeError(f"Creatomate render {render_id} timed out")

        # Download output
        output_url = render_status.get("url")
        if not output_url:
            raise RuntimeError("Creatomate returned no output URL")

        video_resp = httpx.get(output_url, timeout=120, follow_redirects=True)
        video_resp.raise_for_status()
        video_bytes = video_resp.content

    except Exception as exc:
        error_str = str(exc)
        complete_step(sb, step_id, status="failed", error_message=error_str[:200])
        complete_run(sb, run_id, status="failed")
        return SkillResult.failed(
            reason=f"Assembly failed: {error_str[:200]}",
            attempted=1, succeeded=0, failed_count=1, error_class="vendor",
        )

    # ── Upload to R2 ────────────────────────────────────────────────────
    key = f"{property_id}/masters/{input_hash[:12]}.mp4"
    r2_url = skills_r2_upload(key, video_bytes, "video/mp4")

    # ── Write video_artifacts(kind='master') ────────────────────────────
    render_cost = round((total_duration / 60) * CREATOMATE_COST_PER_RENDER_MINUTE, 4)
    artifact_id = str(uuid.uuid4())

    # Supersede prior masters
    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()
    sb.table("video_artifacts").update(
        {"superseded_at": now_iso}
    ).eq("direction_id", direction_id).eq("kind", "master").is_(
        "superseded_at", "null"
    ).execute()

    direction_resp = sb.table("directions").select("concept_id").eq(
        "direction_id", direction_id
    ).limit(1).execute()
    concept_id = direction_resp.data[0]["concept_id"] if direction_resp.data else None

    sb.table("video_artifacts").insert({
        "artifact_id": artifact_id,
        "property_id": property_id,
        "kind": "master",
        "direction_id": direction_id,
        "concept_id": concept_id,
        "input_hash": input_hash,
        "storage_url": r2_url,
        "duration_seconds": round(total_duration, 2),
        "vendor": "creatomate",
        "renderer": "creatomate",
        "render_payload": renderscript,
        "clip_ids": [c["artifact_id"] for c in clips],
        "narration_artifact_id": narration["artifact_id"] if narration else None,
        "music_artifact_id": music["artifact_id"] if music else None,
        "has_audio_ducking": narration is not None and music is not None,
        "aspect_ratio": aspect_ratio,
        "resolution": f"{renderscript['width']}x{renderscript['height']}",
        "cost_estimate_usd": render_cost,
        "status": "ready",
        "created_by_agent": "skills/assemble",
    }).execute()

    emit_cost(sb, run_id, property_id,
              vendor="creatomate", service="render",
              units=1, unit_name="renders",
              unit_cost=render_cost, total_cost=render_cost,
              workflow_name="assemble", generation_reason="master_assembly")

    complete_step(sb, step_id, status="complete", metadata={
        "artifact_id": artifact_id,
        "duration_seconds": round(total_duration, 2),
        "clips_used": len(clips),
        "has_narration": narration is not None,
        "has_music": music is not None,
        "cost_usd": render_cost,
    })
    complete_run(sb, run_id, status="complete")

    return SkillResult.ok({
        "artifact_id": artifact_id,
        "duration_seconds": round(total_duration, 2),
        "clips_used": len(clips),
        "has_narration": narration is not None,
        "has_music": music is not None,
        "cost_usd": render_cost,
        "run_id": run_id,
    })
