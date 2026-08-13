"""
Voice resolution — LIVE from ElevenLabs collections.

No snapshot table. Erick's ElevenLabs curation IS the source of truth.
Add or remove a voice in the UI → takes effect on the next narration.

The only stored data is `vibe_collections`: 7 rows mapping vibe_profile
to ElevenLabs collection_id (because collection names are not exposed
in the public API).

Trade-off: one extra API call per narration. If ElevenLabs is unreachable,
narration fails — acceptable since narration needs ElevenLabs anyway.

Usage:
    from skills.voice_buckets import fetch_vibe_voices, resolve_guest_voice
"""

import hashlib
import logging
import os

import httpx

logger = logging.getLogger(__name__)

ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v2"

# Gender-neutral patterns — voice choice is unconstrained
_NEUTRAL_PATTERNS = ["family", "class of", "reunion", "group", "team", "club", "the "]


def _is_gender_neutral(reviewer_name: str) -> bool:
    """Check if the reviewer name is gender-neutral."""
    name_lower = (reviewer_name or "").lower().strip()
    return any(p in name_lower for p in _NEUTRAL_PATTERNS)


def _get_collection_id(sb, vibe_profile: str) -> str:
    """Look up the ElevenLabs collection_id for a vibe.

    Reads from vibe_collections table (7 rows, essentially static).
    Raises ValueError if no mapping exists.
    """
    resp = sb.table("vibe_collections").select("collection_id").eq(
        "vibe_profile", vibe_profile
    ).eq("active", True).limit(1).execute()

    if not resp.data:
        raise ValueError(
            f"No collection mapping for vibe '{vibe_profile}'. "
            f"Add a row to vibe_collections."
        )
    return resp.data[0]["collection_id"]


def fetch_vibe_voices(sb, vibe_profile: str) -> list:
    """Fetch all voices in a vibe's collection LIVE from ElevenLabs.

    Returns list of {"voice_id", "name", "gender"} dicts.
    Gender comes from each voice's labels. None if not set.
    Raises ValueError if the collection is empty or unreachable.
    """
    el_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not el_key:
        raise ValueError("ELEVENLABS_API_KEY not set.")

    collection_id = _get_collection_id(sb, vibe_profile)

    try:
        resp = httpx.get(
            f"{ELEVENLABS_API_BASE}/voices",
            headers={"xi-api-key": el_key},
            params={"collection_id": collection_id, "page_size": 100},
            timeout=15,
        )
        resp.raise_for_status()
    except Exception as exc:
        raise ValueError(
            f"Failed to fetch voices for vibe '{vibe_profile}' "
            f"(collection {collection_id}): {exc}"
        )

    voices = resp.json().get("voices", [])
    if not voices:
        raise ValueError(
            f"Collection for vibe '{vibe_profile}' ({collection_id}) is empty. "
            f"Add voices in the ElevenLabs dashboard."
        )

    return [
        {
            "voice_id": v["voice_id"],
            "name": v.get("name", ""),
            "gender": (v.get("labels") or {}).get("gender"),
        }
        for v in voices
    ]


def resolve_guest_voice(sb, vibe_profile: str, reviewer_name: str,
                        exclude_voice_id: str = None,
                        reviewer_gender: str = None) -> dict:
    """Get a guest voice for a reviewer, gender-matched and varied.

    Fetches the vibe's collection LIVE from ElevenLabs.

    exclude_voice_id: the hero narrator's voice_id — excluded so
    narrator and guest never sound identical.

    A voice with a MISSING gender label is eligible only for
    unconstrained slots — never guess gender from a name.

    Returns {"voice_id", "name", "gender", "reason"}
    or raises on empty collection/no match.
    """
    pool = fetch_vibe_voices(sb, vibe_profile)

    # Exclude the hero narrator's voice
    if exclude_voice_id:
        pool = [v for v in pool if v["voice_id"] != exclude_voice_id]

    is_neutral = _is_gender_neutral(reviewer_name)

    if is_neutral:
        # All voices eligible (including those with no gender label)
        candidates = pool
        reason = "gender-neutral attribution — unconstrained"
    elif reviewer_gender:
        # Only voices with matching gender label
        # Voices with NO gender label are NOT eligible for gendered slots
        candidates = [v for v in pool if v.get("gender") == reviewer_gender]
        reason = f"gender match: {reviewer_gender}"
    else:
        # No gender info — all voices eligible
        candidates = pool
        reason = "no gender info — unconstrained"

    if not candidates:
        gender_info = f"gender={reviewer_gender}" if reviewer_gender else "any gender"
        raise ValueError(
            f"No voice for vibe '{vibe_profile}' ({gender_info}) after exclusions. "
            f"Add more voices with gender labels in ElevenLabs."
        )

    # Deterministic selection: hash reviewer name into candidate index
    name_hash = int(hashlib.sha256(reviewer_name.strip().encode()).hexdigest(), 16)
    idx = name_hash % len(candidates)
    chosen = candidates[idx]

    return {
        "voice_id": chosen["voice_id"],
        "name": chosen["name"],
        "gender": chosen.get("gender"),
        "reason": reason,
    }
