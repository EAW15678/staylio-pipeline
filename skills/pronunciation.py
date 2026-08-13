"""
Pronunciation dictionary management for ElevenLabs TTS.

Builds per-property dictionaries from global + per-property entries in
the pronunciation_entries table. Caches dictionary_id/version_id so
rebuilds only happen when entries change.

Attaches via pronunciation_dictionary_locators on TTS calls.
Max 3 locators per call (ElevenLabs limit).

Usage:
    from skills.pronunciation import get_pronunciation_locators
    locators = get_pronunciation_locators(sb, property_id)
    # Pass to ElevenLabs TTS as pronunciation_dictionary_locators
"""

import hashlib
import json
import logging
import os

import httpx

logger = logging.getLogger(__name__)

ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"


def _compute_entries_hash(entries: list) -> str:
    """Hash sorted entries for change detection."""
    blob = json.dumps(
        sorted(entries, key=lambda e: (e["term"], e["pronunciation"])),
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _create_dictionary(entries: list, name: str) -> dict:
    """Create an ElevenLabs pronunciation dictionary from entries.

    Returns {"dictionary_id": str, "version_id": str} or raises.
    """
    el_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not el_key:
        raise ValueError("ELEVENLABS_API_KEY not set")

    rules = []
    for e in entries:
        if e["rule_type"] == "alias":
            rules.append({
                "string_to_replace": e["term"],
                "type": "alias",
                "alias": e["pronunciation"],
            })
        elif e["rule_type"] == "phoneme":
            rules.append({
                "string_to_replace": e["term"],
                "type": "phoneme",
                "phoneme": e["pronunciation"],
                "alphabet": e.get("alphabet") or "ipa",
            })

    if not rules:
        raise ValueError("No pronunciation rules to create dictionary from")

    resp = httpx.post(
        f"{ELEVENLABS_API_BASE}/pronunciation-dictionaries/add-from-rules",
        headers={"xi-api-key": el_key, "Content-Type": "application/json"},
        json={"name": name, "rules": rules},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    return {
        "dictionary_id": data["id"],
        "version_id": data["version_id"],
    }


def _update_dictionary(dictionary_id: str, entries: list) -> str:
    """Replace all rules on an existing dictionary. Returns new version_id."""
    el_key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not el_key:
        raise ValueError("ELEVENLABS_API_KEY not set")

    rules = []
    for e in entries:
        if e["rule_type"] == "alias":
            rules.append({
                "string_to_replace": e["term"],
                "type": "alias",
                "alias": e["pronunciation"],
            })
        elif e["rule_type"] == "phoneme":
            rules.append({
                "string_to_replace": e["term"],
                "type": "phoneme",
                "phoneme": e["pronunciation"],
                "alphabet": e.get("alphabet") or "ipa",
            })

    resp = httpx.post(
        f"{ELEVENLABS_API_BASE}/pronunciation-dictionaries/{dictionary_id}/set-rules",
        headers={"xi-api-key": el_key, "Content-Type": "application/json"},
        json={"rules": rules},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["version_id"]


def get_pronunciation_locators(sb, property_id: str) -> list:
    """Get pronunciation_dictionary_locators for a TTS call.

    Merges global entries (property_id IS NULL) with per-property entries.
    Creates/updates the ElevenLabs dictionary only when entries change.
    Returns a list of locator dicts (max 1 — merged into one dictionary).

    If no entries exist for this property, returns [] (no dictionary needed).
    If the dictionary cannot be created, raises ValueError (Ruling 6).
    """
    # Load global + per-property entries
    global_resp = sb.table("pronunciation_entries").select(
        "term, pronunciation, rule_type, alphabet"
    ).is_("property_id", "null").eq("active", True).execute()
    global_entries = global_resp.data or []

    prop_resp = sb.table("pronunciation_entries").select(
        "term, pronunciation, rule_type, alphabet"
    ).eq("property_id", property_id).eq("active", True).execute()
    prop_entries = prop_resp.data or []

    # Merge: per-property overrides global for same term
    merged = {}
    for e in global_entries:
        merged[e["term"]] = e
    for e in prop_entries:
        merged[e["term"]] = e  # override

    all_entries = list(merged.values())

    if not all_entries:
        return []  # No pronunciation entries — TTS runs without dictionary

    entries_hash = _compute_entries_hash(all_entries)

    # Check cache
    cache_resp = sb.table("pronunciation_dict_cache").select(
        "id, dictionary_id, version_id, entries_hash"
    ).eq("property_id", property_id).limit(1).execute()

    if cache_resp.data:
        cached = cache_resp.data[0]
        if cached["entries_hash"] == entries_hash:
            # Cache hit — entries unchanged, reuse
            return [{
                "pronunciation_dictionary_id": cached["dictionary_id"],
                "version_id": cached["version_id"],
            }]
        else:
            # Entries changed — update the existing dictionary
            try:
                new_version = _update_dictionary(cached["dictionary_id"], all_entries)
                sb.table("pronunciation_dict_cache").update({
                    "version_id": new_version,
                    "entries_hash": entries_hash,
                }).eq("id", cached["id"]).execute()
                logger.info("[pronunciation] Updated dictionary for %s (version=%s)", property_id[:12], new_version[:12])
                return [{
                    "pronunciation_dictionary_id": cached["dictionary_id"],
                    "version_id": new_version,
                }]
            except Exception as exc:
                raise ValueError(f"Failed to update pronunciation dictionary: {exc}")
    else:
        # No cache — create new dictionary
        try:
            name = f"staylio_{property_id[:12]}"
            result = _create_dictionary(all_entries, name)
            sb.table("pronunciation_dict_cache").insert({
                "property_id": property_id,
                "dictionary_id": result["dictionary_id"],
                "version_id": result["version_id"],
                "entries_hash": entries_hash,
            }).execute()
            logger.info("[pronunciation] Created dictionary for %s (id=%s)", property_id[:12], result["dictionary_id"][:12])
            return [{
                "pronunciation_dictionary_id": result["dictionary_id"],
                "version_id": result["version_id"],
            }]
        except Exception as exc:
            raise ValueError(f"Failed to create pronunciation dictionary: {exc}")
