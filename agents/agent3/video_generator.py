"""
TS-09 — 8-Video Launch Library Generator
Tools: ElevenLabs (voice) + Runway ML Gen-4 (cinematic motion)
       + Creatomate (assembly + multi-format rendering)

Generates all 8 property videos at intake. One-time cost ~$6.42/property.
Total ~20 rendered files per property stored permanently in R2.

Voice architecture (from TS-09 notes):
  VIBE VOICES: 6 Staylio-owned voices created via ElevenLabs Voice Design.
               No notice period risk. Permanent assets.
  REVIEW VOICES: Pool of ElevenLabs premade voices (no removal risk).
                 Rotated across the 3 guest review videos.

Video generation triggers after TS-07b photo tagging completes —
category-winner photos and vibe metadata are available as inputs.

Minimum 3 guest book entries required for all 3 review videos.
If fewer exist at intake, review videos queue until the 3rd entry
is submitted — property goes live without them.
"""

import os
import time
import logging
import asyncio
from typing import Optional

import httpx
import runwayml

from agents.agent3.models import (
    VideoAsset,
    VideoFormat,
    VideoType,
    VIDEO_FORMAT_MATRIX,
    VIDEO_TARGET_DURATION,
)
from agents.agent3.r2_storage import upload_video

logger = logging.getLogger(__name__)

# ── API Configuration ─────────────────────────────────────────────────────
ELEVENLABS_API_KEY  = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"

RUNWAYML_API_SECRET = os.environ.get("RUNWAYML_API_SECRET", "")
RUNWAY_MODEL        = "gen4_turbo"
RUNWAY_CREDITS_PER_SECOND = 5   # Gen-4 Turbo: 5 credits/second

CREATOMATE_API_KEY  = os.environ.get("CREATOMATE_API_KEY", "")
CREATOMATE_API_BASE = "https://api.creatomate.com/v2"

# ── Voice Configuration ───────────────────────────────────────────────────
# These voice IDs must be configured in the Staylio ElevenLabs account.
# Vibe voices are created via Voice Design — Staylio-owned, no removal risk.
# Review voices are ElevenLabs premade defaults — no removal risk.

VIBE_VOICE_IDS: dict[str, str] = {
    # Populated with actual Voice Design IDs before production build
    # Format: "vibe_profile": "elevenlabs_voice_id"
    "romantic_escape":       os.environ.get("VOICE_ROMANTIC_ESCAPE", ""),
    "family_adventure":      os.environ.get("VOICE_FAMILY_ADVENTURE", ""),
    "multigenerational_retreat": os.environ.get("VOICE_MULTIGENERATIONAL", ""),
    "wellness_retreat":      os.environ.get("VOICE_WELLNESS", ""),
    "adventure_base_camp":   os.environ.get("VOICE_ADVENTURE", ""),
    "social_celebrations":   os.environ.get("VOICE_SOCIAL", ""),
}

# Pool of ElevenLabs premade voices for guest review narration
# Rotated: video 3 → pool[0], video 4 → pool[1], video 8 → pool[2]
REVIEW_VOICE_POOL: list[str] = [
    os.environ.get("VOICE_REVIEW_1", ""),   # e.g. "Rachel" — premade
    os.environ.get("VOICE_REVIEW_2", ""),   # e.g. "Adam" — premade
    os.environ.get("VOICE_REVIEW_3", ""),   # e.g. "Bella" — premade
]

# Creatomate template IDs (built in Creatomate dashboard before Agent 3 build)
CREATOMATE_TEMPLATES: dict[str, dict[VideoFormat, str]] = {
    "vibe": {
        VideoFormat.VERTICAL:  os.environ.get("CREATOMATE_TEMPLATE_VIBE_MATCH", ""),
        VideoFormat.LANDSCAPE: os.environ.get("CREATOMATE_TEMPLATE_VIBE_MATCH_16X9", ""),
    },
    "review": {
        VideoFormat.VERTICAL:  os.environ.get("CREATOMATE_TEMPLATE_GUEST_REVIEW", ""),
    },
    "walkthrough": {
        VideoFormat.VERTICAL:  os.environ.get("CREATOMATE_TEMPLATE_WALKTHROUGH", ""),
    },
    "local": {
        VideoFormat.VERTICAL:  os.environ.get("CREATOMATE_LOCAL_9_16", ""),
        VideoFormat.SQUARE:    os.environ.get("CREATOMATE_LOCAL_1_1", ""),
    },
    "feature": {
        VideoFormat.VERTICAL:  os.environ.get("CREATOMATE_FEATURE_9_16", ""),
    },
    "seasonal": {
        VideoFormat.VERTICAL:  os.environ.get("CREATOMATE_SEASONAL_9_16", ""),
        VideoFormat.SQUARE:    os.environ.get("CREATOMATE_SEASONAL_1_1", ""),
    },
}

# Runway polling config
RUNWAY_POLL_INTERVAL = 5    # seconds
RUNWAY_MAX_POLLS     = 120  # 10 min max per video generation


async def generate_all_videos(
    property_id: str,
    vibe_profile: str,
    category_winners: dict[str, str],    # {category: enhanced_photo_url}
    hero_photo_url: Optional[str],
    guest_reviews: list[dict],            # List of review dicts from KB
    content_package: dict,               # Agent 2 output dict
    seasonal_notes: Optional[str],
    location_highlight: Optional[str],   # Top local recommendation name
) -> list[VideoAsset]:
    """
    Generate all 8 property videos.
    Returns list of VideoAsset records (with R2 URLs) for storage.
    """
    videos: list[VideoAsset] = []
    logger.info(f"[DIAG-REVIEWS] property_id={property_id} total_reviews={len(guest_reviews)} preview={[{'keys': list(r.keys()), 'is_guest_book': r.get('is_guest_book'), 'text_present': bool(r.get('text'))} for r in guest_reviews[:3]]}")
    review_count = len([r for r in guest_reviews if r.get("is_guest_book")])
    has_enough_reviews = review_count >= 3

    # Video generation tasks — run sequentially to control API rate limits
    # In production with higher volume, these can be parallelised with semaphores

    # VIDEO 1 — Vibe Match / Hero (narrated, all 3 formats)
    v1 = await _generate_vibe_match_video(
        property_id, vibe_profile, category_winners, hero_photo_url,
        content_package.get("hero_headline"),
        content_package.get("vibe_tagline"),
    )
    videos.extend(v1)

    # VIDEO 2 — Silent Walk-Through (9:16 only, no narration)
    v2 = await _generate_walkthrough_video(property_id, category_winners)
    videos.extend(v2)

    # VIDEOS 3, 4, 8 — Guest Review videos (need 3 guest book entries)
    if has_enough_reviews:
        book_reviews = [r for r in guest_reviews if r.get("is_guest_book")][:3]
        for i, (review, review_idx) in enumerate(zip(book_reviews, [0, 1, 2])):
            video_type = [VideoType.GUEST_REVIEW_1, VideoType.GUEST_REVIEW_2, VideoType.GUEST_REVIEW_3][i]
            voice_id = REVIEW_VOICE_POOL[review_idx] if review_idx < len(REVIEW_VOICE_POOL) else ""
            logger.info(f"[DIAG-TEXT] video_type={video_type.value} len={len(review.get('text',''))} first_200={repr(review.get('text','')[:200])} last_200={repr(review.get('text','')[-200:])}")
            v = await _generate_review_video(
                property_id, video_type,
                review.get("text", ""), voice_id,
                category_winners, hero_photo_url,
            )
            videos.extend(v)
    else:
        logger.info(
            f"[TS-09] Property {property_id}: only {review_count} guest book entries — "
            f"review videos queued for later generation (need 3)"
        )

    # VIDEO 5 — Local Area Highlight (9:16 + 1:1, no narration)
    if location_highlight:
        v5 = await _generate_local_highlight_video(
            property_id, location_highlight, category_winners
        )
        videos.extend(v5)

    # VIDEO 6 — Feature Close-Up (9:16 only, no narration, 12-15s)
    v6 = await _generate_feature_closeup_video(property_id, category_winners)
    videos.extend(v6)

    # VIDEO 7 — Seasonal/Occasion (narrated, 9:16 + 1:1)
    v7 = await _generate_seasonal_video(
        property_id, vibe_profile, seasonal_notes or "", category_winners
    )
    videos.extend(v7)

    logger.info(
        f"[TS-09] Video generation complete for property {property_id}. "
        f"{len(videos)} rendered files. Review videos pending: {not has_enough_reviews}"
    )
    return videos


# ── Individual video generators ───────────────────────────────────────────

async def _generate_vibe_match_video(
    property_id: str,
    vibe_profile: str,
    category_winners: dict[str, str],
    hero_photo_url: Optional[str],
    hero_headline: Optional[str],
    vibe_tagline: Optional[str],
) -> list[VideoAsset]:
    """Video 1 — 45-60s hero video with vibe narration."""
    voice_id = VIBE_VOICE_IDS.get(vibe_profile, "")
    if not voice_id:
        logger.warning(f"[TS-09] No vibe voice configured for {vibe_profile}")
        return []

    # Generate narration script from headline and tagline
    script = _build_vibe_script(hero_headline, vibe_tagline, vibe_profile)

    # Generate narration audio
    audio_url = await _generate_elevenlabs_audio(script, voice_id, property_id, "vibe")
    if not audio_url:
        return []

    # Select photo sequence (vibe priority order from category winners)
    photos = _select_photo_sequence(category_winners, vibe_profile, count=6)
    if not photos:
        return []

    # Generate Runway ML cinematic motion clips for each photo
    motion_clips = await _generate_runway_clips(photos, property_id, "vibe")

    # Assemble with Creatomate
    template_map = CREATOMATE_TEMPLATES.get("vibe", {})
    return await _assemble_with_creatomate(
        property_id=property_id,
        video_type=VideoType.VIBE_MATCH,
        template_map=template_map,
        motion_clips=motion_clips,
        audio_url=audio_url,
        script=script,
        voice_id=voice_id,
        duration=VIDEO_TARGET_DURATION[VideoType.VIBE_MATCH],
    )


async def _generate_walkthrough_video(
    property_id: str,
    category_winners: dict[str, str],
) -> list[VideoAsset]:
    """Video 2 — 15-20s silent walk-through, 9:16 only."""
    photos = list(category_winners.values())[:5]
    if not photos:
        return []

    motion_clips = await _generate_runway_clips(photos, property_id, "walkthrough")
    template_map = CREATOMATE_TEMPLATES.get("walkthrough", {})
    return await _assemble_with_creatomate(
        property_id=property_id,
        video_type=VideoType.WALK_THROUGH,
        template_map=template_map,
        motion_clips=motion_clips,
        audio_url=None,
        script=None,
        voice_id=None,
        duration=VIDEO_TARGET_DURATION[VideoType.WALK_THROUGH],
    )


async def _generate_review_video(
    property_id: str,
    video_type: VideoType,
    review_text: str,
    voice_id: str,
    category_winners: dict[str, str],
    hero_photo_url: Optional[str],
) -> list[VideoAsset]:
    """Videos 3, 4, 8 — Guest review narrated over animated photo."""
    if not voice_id or not review_text:
        return []

    # MP3 guest book narration uses full review text — no truncation.
    # Note: if this function is extended to support video-constrained assembly,
    # apply duration-based trimming only in that path, not here.
    script = review_text.strip()

    audio_url = await _generate_elevenlabs_audio(
        script, voice_id, property_id, video_type.value
    )
    if not audio_url:
        return []

    # Persist MP3 immediately — independent of Creatomate
    try:
        from core.supabase_store import get_supabase
        get_supabase().table("video_assets").upsert(
            {
                "property_id": property_id,
                "video_type": f"audio_{video_type.value}",
                "format": "mp3",
                "r2_url": audio_url,
                "has_narration": True,
                "voice_id": voice_id,
            },
            on_conflict="property_id,video_type,format",
        ).execute()
        logger.info(f"[TS-09] Persisted MP3 audio row: audio_{video_type.value}")
    except Exception as exc:
        logger.warning(f"[TS-09] Could not persist audio row for {video_type.value}: {exc}")

    # Use hero photo or a random category winner as background
    background_url = hero_photo_url or (list(category_winners.values())[0] if category_winners else None)
    if not background_url:
        return []

    motion_clips = await _generate_runway_clips([background_url], property_id, video_type.value)
    template_map = CREATOMATE_TEMPLATES.get("review", {})
    return await _assemble_with_creatomate(
        property_id=property_id,
        video_type=video_type,
        template_map=template_map,
        motion_clips=motion_clips,
        audio_url=audio_url,
        script=script,
        voice_id=voice_id,
        duration=VIDEO_TARGET_DURATION[video_type],
    )


async def _generate_local_highlight_video(
    property_id: str,
    location_name: str,
    category_winners: dict[str, str],
) -> list[VideoAsset]:
    """Video 5 — Local area highlight, no narration."""
    photos = list(category_winners.values())[:3]
    if not photos:
        return []
    motion_clips = await _generate_runway_clips(photos, property_id, "local")
    template_map = CREATOMATE_TEMPLATES.get("local", {})
    return await _assemble_with_creatomate(
        property_id=property_id,
        video_type=VideoType.LOCAL_HIGHLIGHT,
        template_map=template_map,
        motion_clips=motion_clips,
        audio_url=None,
        script=location_name,  # Used as text overlay in Creatomate template
        voice_id=None,
        duration=VIDEO_TARGET_DURATION[VideoType.LOCAL_HIGHLIGHT],
    )


async def _generate_feature_closeup_video(
    property_id: str,
    category_winners: dict[str, str],
) -> list[VideoAsset]:
    """Video 6 — 12-15s feature close-up of the most compelling single feature."""
    # Prefer view, pool, or outdoor entertaining for the feature close-up
    preferred = ["view", "pool_hot_tub", "outdoor_entertaining", "master_bedroom"]
    photo_url = None
    for cat in preferred:
        if cat in category_winners:
            photo_url = category_winners[cat]
            break
    if not photo_url and category_winners:
        photo_url = list(category_winners.values())[0]
    if not photo_url:
        return []

    motion_clips = await _generate_runway_clips([photo_url], property_id, "feature")
    template_map = CREATOMATE_TEMPLATES.get("feature", {})
    return await _assemble_with_creatomate(
        property_id=property_id,
        video_type=VideoType.FEATURE_CLOSEUP,
        template_map=template_map,
        motion_clips=motion_clips,
        audio_url=None,
        script=None,
        voice_id=None,
        duration=VIDEO_TARGET_DURATION[VideoType.FEATURE_CLOSEUP],
    )


async def _generate_seasonal_video(
    property_id: str,
    vibe_profile: str,
    seasonal_notes: str,
    category_winners: dict[str, str],
) -> list[VideoAsset]:
    """Video 7 — Seasonal/occasion video with narration."""
    voice_id = VIBE_VOICE_IDS.get(vibe_profile, "")
    if not voice_id:
        return []

    script = _build_seasonal_script(seasonal_notes, vibe_profile)
    audio_url = await _generate_elevenlabs_audio(script, voice_id, property_id, "seasonal")
    if not audio_url:
        return []

    photos = list(category_winners.values())[:4]
    motion_clips = await _generate_runway_clips(photos, property_id, "seasonal")
    template_map = CREATOMATE_TEMPLATES.get("seasonal", {})
    return await _assemble_with_creatomate(
        property_id=property_id,
        video_type=VideoType.SEASONAL,
        template_map=template_map,
        motion_clips=motion_clips,
        audio_url=audio_url,
        script=script,
        voice_id=voice_id,
        duration=VIDEO_TARGET_DURATION[VideoType.SEASONAL],
    )


# ── ElevenLabs TTS ─────────────────────────────────────────────────────────

async def _generate_elevenlabs_audio(
    script: str,
    voice_id: str,
    property_id: str,
    video_label: str,
) -> Optional[str]:
    """
    Generate TTS audio via ElevenLabs API.
    Returns a temporary audio URL (or R2 URL after upload).
    """
    if not ELEVENLABS_API_KEY or not voice_id:
        logger.warning(f"[TS-09] ElevenLabs not configured for property {property_id}")
        return None

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{ELEVENLABS_API_BASE}/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": ELEVENLABS_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "text": script,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                        "style": 0.3,
                    },
                },
            )
            resp.raise_for_status()
            audio_bytes = resp.content

        # Upload to R2
        r2_url = upload_video(
            property_id=property_id,
            video_bytes=audio_bytes,
            video_type=f"audio_{video_label}",
            format_label="mp3",
            extension="mp3",
        )
        return r2_url

    except Exception as exc:
        logger.error(f"[TS-09] ElevenLabs TTS failed for property {property_id}: {exc}")
        return None


# ── Runway ML ─────────────────────────────────────────────────────────────

async def _generate_runway_clips(
    photo_urls: list[str],
    property_id: str,
    video_label: str,
) -> list[str]:
    """
    Generate cinematic motion clips from still photos via Runway ML Gen-4.
    Uses the official runwayml SDK; wait_for_task_output() handles polling.
    Caches generated clips in Supabase video_assets — skips Runway on re-run.
    Returns list of R2 URLs for the generated video clips.
    """
    if not RUNWAYML_API_SECRET:
        logger.warning(f"[TS-09] Runway ML not configured (RUNWAYML_API_SECRET missing) for property {property_id}")
        return []

    clip_urls = []
    async with runwayml.AsyncRunwayML(api_key=RUNWAYML_API_SECRET) as runway_client:
        for i, photo_url in enumerate(photo_urls):
            clip_video_type = f"clip_{video_label}_{i:02d}"

            # Check Supabase cache — skip Runway if this clip was already generated
            cached_url = _fetch_cached_clip(property_id, clip_video_type)
            if cached_url:
                logger.info(f"[TS-09] Reusing cached clip — skipping Runway: {clip_video_type}")
                clip_urls.append(cached_url)
                continue

            try:
                task = await runway_client.image_to_video.create(
                    model=RUNWAY_MODEL,
                    prompt_image=photo_url,
                    prompt_text="Slow cinematic pan, professional real estate video style, smooth motion",
                    duration=5,    # 5-second clips, assembled by Creatomate
                    ratio="1280:720",
                )

                # SDK blocks until SUCCEEDED, raises TaskFailedError on failure
                result = await task.wait_for_task_output(timeout=600)

                output_urls = getattr(result, "output", None) or []
                if not output_urls:
                    logger.warning(f"[TS-09] Runway returned no output for clip {i} of {property_id}")
                    continue

                # Download and upload to R2
                async with httpx.AsyncClient(timeout=120) as dl_client:
                    dl = await dl_client.get(output_urls[0])
                    dl.raise_for_status()

                r2_url = upload_video(
                    property_id=property_id,
                    video_bytes=dl.content,
                    video_type=clip_video_type,
                    format_label="mp4",
                )

                # Cache the clip so reruns skip this Runway call
                _save_clip_to_cache(property_id, clip_video_type, r2_url, i)
                clip_urls.append(r2_url)

            except Exception as exc:
                try:
                    body = exc.response.text
                except Exception:
                    body = ""
                logger.error(f"[TS-09] Runway clip generation failed for photo {i}: {exc} body={body}")

    return clip_urls


def _fetch_cached_clip(property_id: str, clip_video_type: str) -> Optional[str]:
    """
    Check Supabase video_assets for a previously generated clip.
    Returns the R2 URL if found, None otherwise.
    """
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("video_assets")
            .select("r2_url")
            .eq("property_id", property_id)
            .eq("video_type", clip_video_type)
            .not_.is_("r2_url", "null")
            .limit(1)
            .execute()
        )
        rows = result.data or []
        return rows[0]["r2_url"] if rows else None
    except Exception as exc:
        logger.warning(f"[TS-09] Could not check clip cache for {clip_video_type}: {exc}")
        return None


def _save_clip_to_cache(
    property_id: str,
    clip_video_type: str,
    r2_url: str,
    clip_index: int,
) -> None:
    """Save a generated clip to Supabase video_assets for future cache hits."""
    try:
        from core.supabase_store import get_supabase
        get_supabase().table("video_assets").upsert(
            {
                "property_id": property_id,
                "video_type": clip_video_type,
                "format": "mp4",
                "r2_url": r2_url,
                "duration_seconds": 5.0,
                "has_narration": False,
                "queued_for_social": False,
                "script_text": f"clip_{clip_index}",
            },
            on_conflict="property_id,video_type,format",
        ).execute()
    except Exception as exc:
        logger.warning(f"[TS-09] Could not cache clip {clip_video_type}: {exc}")


# ── Creatomate Assembly ────────────────────────────────────────────────────

async def _assemble_with_creatomate(
    property_id: str,
    video_type: VideoType,
    template_map: dict[VideoFormat, str],
    motion_clips: list[str],
    audio_url: Optional[str],
    script: Optional[str],
    voice_id: Optional[str],
    duration: float,
) -> list[VideoAsset]:
    """
    Assemble motion clips + audio into final video using Creatomate templates.
    Renders all formats specified in template_map.
    Returns list of VideoAsset records.
    """
    if not CREATOMATE_API_KEY or not motion_clips:
        return []

    video_assets = []
    async with httpx.AsyncClient(timeout=300) as client:
        for fmt, template_id in template_map.items():
            if not template_id or "FILL_IN_AFTER_BUILDING_TEMPLATE" in template_id:
                continue
            try:
                render_resp = await client.post(
                    f"{CREATOMATE_API_BASE}/renders",
                    headers={
                        "Authorization": f"Bearer {CREATOMATE_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "template_id": template_id,
                        "modifications": _build_creatomate_modifications(
                            motion_clips, audio_url, script, fmt
                        ),
                    },
                )
                render_resp.raise_for_status()
                renders = render_resp.json()
                render = renders[0] if isinstance(renders, list) else renders
                render_id = render.get("id")
                if not render_id:
                    continue

                # Poll for completion
                output_url = await _poll_creatomate_render(client, render_id)
                if not output_url:
                    continue

                # Download rendered video and upload to R2
                dl = await client.get(output_url, timeout=120)
                r2_url = upload_video(
                    property_id=property_id,
                    video_bytes=dl.content,
                    video_type=video_type.value,
                    format_label=fmt.value,
                )

                video_assets.append(VideoAsset(
                    property_id=property_id,
                    video_type=video_type,
                    format=fmt,
                    r2_url=r2_url,
                    duration_seconds=duration,
                    has_narration=bool(audio_url),
                    voice_id=voice_id,
                    script_text=script,
                    queued_for_social=True,
                ))

            except Exception as exc:
                try:
                    body = exc.response.text
                except Exception:
                    body = ""
                logger.error(
                    f"[TS-09] Creatomate assembly failed for {video_type.value}/{fmt.value}: {exc} body={body}"
                )

    return video_assets


async def _poll_creatomate_render(
    client: httpx.AsyncClient,
    render_id: str,
    max_polls: int = 60,
    poll_interval: int = 5,
) -> Optional[str]:
    """Poll a Creatomate render until complete. Returns output URL."""
    for _ in range(max_polls):
        await asyncio.sleep(poll_interval)
        try:
            resp = await client.get(
                f"{CREATOMATE_API_BASE}/renders/{render_id}",
                headers={"Authorization": f"Bearer {CREATOMATE_API_KEY}"},
            )
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status")
            if status == "succeeded":
                return data.get("url")
            elif status == "failed":
                logger.error(f"Creatomate render {render_id} failed: {data.get('error_message')}")
                return None
        except Exception as exc:
            logger.warning(f"Creatomate poll error: {exc}")
    logger.error(f"Creatomate render {render_id} timed out")
    return None


def _build_creatomate_modifications(
    clip_urls: list[str],
    audio_url: Optional[str],
    text_overlay: Optional[str],
    fmt: VideoFormat,
) -> dict:
    """Build Creatomate v2 modifications dict using flat dot-notation keys."""
    mods: dict = {}
    for i, url in enumerate(clip_urls[:6]):   # Templates support up to 6 clips
        mods[f"clip_{i+1}.source"] = url
    if audio_url:
        mods["Audio.source"] = audio_url   # Capital A — Creatomate auto-capitalises
    if text_overlay:
        mods["text_overlay.text"] = text_overlay
    return mods


# ── Script builders ───────────────────────────────────────────────────────

def _build_vibe_script(
    hero_headline: Optional[str],
    vibe_tagline: Optional[str],
    vibe_profile: str,
) -> str:
    """Build narration script for Video 1 from Agent 2 content."""
    parts = []
    if hero_headline:
        parts.append(hero_headline)
    if vibe_tagline:
        parts.append(vibe_tagline)
    if not parts:
        # Fallback by vibe
        fallbacks = {
            "romantic_escape": "A private retreat made for two.",
            "family_adventure": "Where families make memories that last forever.",
            "multigenerational_retreat": "Every generation deserves this.",
            "wellness_retreat": "Come as you are. Leave restored.",
            "adventure_base_camp": "Your next adventure starts here.",
            "social_celebrations": "Your group. Your moment. Your house.",
        }
        parts.append(fallbacks.get(vibe_profile, "Experience it for yourself."))
    return "  ".join(parts)


def _build_seasonal_script(seasonal_notes: str, vibe_profile: str) -> str:
    """Build narration script for Video 7 seasonal content."""
    if seasonal_notes and len(seasonal_notes) > 20:
        # Trim to TTS-appropriate length
        return seasonal_notes[:280].strip()
    # Fallback by vibe
    fallbacks = {
        "romantic_escape": "Peak season is here. The perfect moment for two.",
        "family_adventure": "Summer is waiting. Bring the whole family.",
        "multigenerational_retreat": "The season is right for a family reunion.",
        "wellness_retreat": "Step away from it all. The timing is perfect.",
        "adventure_base_camp": "Prime season for adventure. Your base camp is ready.",
        "social_celebrations": "The best season to celebrate. Bring your crew.",
    }
    return fallbacks.get(vibe_profile, "The season is right.")


def _select_photo_sequence(
    category_winners: dict[str, str],
    vibe_profile: str,
    count: int = 6,
) -> list[str]:
    """
    Select an ordered sequence of photos for video assembly,
    prioritised by vibe profile.
    """
    from agents.agent3.vision_tagger import VIBE_HERO_PRIORITY
    from agents.agent3.models import SubjectCategory

    priority = VIBE_HERO_PRIORITY.get(vibe_profile, [])
    ordered = []
    for cat in priority:
        url = category_winners.get(cat.value)
        if url and url not in ordered:
            ordered.append(url)
    # Fill remaining slots with other category winners
    for url in category_winners.values():
        if url not in ordered:
            ordered.append(url)
    return ordered[:count]
