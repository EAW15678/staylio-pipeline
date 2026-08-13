"""
Voice bucket resolution — reads voice_buckets table (one pool per vibe).

No role column. The staylio_* voices and guest voices are all ordinary
members of the same pool. Selection rules:

  hero narration → director picks any voice from the vibe's pool
  guest card → gender matches the writer; neutral unconstrained
  exclude the hero narrator's voice from that property's guest pool
  vary across cards — hash reviewer name for deterministic assignment
  fail loudly on missing bucket or no gender match

Usage:
    from skills.voice_buckets import get_vibe_pool, resolve_guest_voice
"""

import hashlib
import logging

logger = logging.getLogger(__name__)

# Gender-neutral patterns — voice choice is unconstrained
_NEUTRAL_PATTERNS = ["family", "class of", "reunion", "group", "team", "club", "the "]


def _is_gender_neutral(reviewer_name: str) -> bool:
    """Check if the reviewer name is gender-neutral."""
    name_lower = (reviewer_name or "").lower().strip()
    return any(p in name_lower for p in _NEUTRAL_PATTERNS)


def get_vibe_pool(sb, vibe_profile: str) -> list:
    """Get all active voices for a vibe.

    Returns list of {"voice_id", "voice_name", "gender"} dicts.
    Raises ValueError if the pool is empty.
    """
    resp = sb.table("voice_buckets").select("voice_id, voice_name, gender").eq(
        "vibe_profile", vibe_profile
    ).eq("active", True).execute()

    pool = resp.data or []
    if not pool:
        raise ValueError(
            f"No voices configured for vibe '{vibe_profile}'. "
            f"Add rows to voice_buckets."
        )
    return pool


def resolve_guest_voice(sb, vibe_profile: str, reviewer_name: str,
                        exclude_voice_id: str = None,
                        reviewer_gender: str = None) -> dict:
    """Get a guest voice for a reviewer, gender-matched and varied.

    exclude_voice_id: the hero narrator's voice_id — excluded from the
    guest pool so narrator and guest never sound identical.

    Returns {"voice_id", "voice_name", "gender", "reason"}
    or raises on missing bucket/no match.
    """
    pool = get_vibe_pool(sb, vibe_profile)

    # Exclude the hero narrator's voice
    if exclude_voice_id:
        pool = [v for v in pool if v["voice_id"] != exclude_voice_id]

    is_neutral = _is_gender_neutral(reviewer_name)

    if is_neutral:
        candidates = pool  # all genders eligible
        reason = "gender-neutral attribution — unconstrained"
    elif reviewer_gender:
        candidates = [v for v in pool if v.get("gender") == reviewer_gender]
        reason = f"gender match: {reviewer_gender}"
    else:
        candidates = pool  # no gender info — unconstrained
        reason = "no gender info — unconstrained"

    if not candidates:
        gender_info = f"gender={reviewer_gender}" if reviewer_gender else "any gender"
        raise ValueError(
            f"No voice for vibe '{vibe_profile}' ({gender_info}) after excluding "
            f"hero voice. Add more voices to voice_buckets."
        )

    # Deterministic selection: hash reviewer name into candidate index
    name_hash = int(hashlib.sha256(reviewer_name.strip().encode()).hexdigest(), 16)
    idx = name_hash % len(candidates)
    chosen = candidates[idx]

    return {
        "voice_id": chosen["voice_id"],
        "voice_name": chosen["voice_name"],
        "gender": chosen.get("gender"),
        "reason": reason,
    }
