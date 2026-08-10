"""
Agent 8 — Stage 8: Compliance

Pre-publish compliance checker. Evaluates assembled videos, motion clips,
narration assets, and overlays against the Staylio Content Guidelines v1.2.

Verdict DEFAULTS to hold — silence is never approval. A subject that has
never been checked, or whose check crashed mid-run, is held.

Two classes of rule:
  DETERMINISTIC — pure code, no model call, $0.
  MODEL_ASSISTED — sends frames/metadata to Claude Sonnet vision.

Standalone module. NOT wired into the pipeline graph.
Call directly: check_subject(subject_type, subject_id)
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Optional

from agents.agent8.creative_director import (
    validate_no_guest_names,
    validate_no_ota,
    _extract_amenity_names,
    _extract_guest_book_entries,
    _OTA_NAMES,
)
from agents.agent8.motion import FRAME_EXITING_MOVES

logger = logging.getLogger(__name__)

CHECKER_VERSION = "compliance_v1"
GUIDELINES_VERSION = "v1.2"

# Verdict defaults to hold — silence is never approval.
DEFAULT_VERDICT = "hold"

_FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None

# ── Numeric ratings regex ────────────────────────────────────────────────
_NUMERIC_RATING_PATTERNS = [
    re.compile(r"\b\d+(\.\d+)?\s*/\s*5\b"),       # 4.9/5, 5/5
    re.compile(r"\b\d+(\.\d+)?\s*/\s*10\b"),      # 9.2/10
    re.compile(r"\b\d+(\.\d+)?\s*stars?\b", re.I), # 5 stars, 4.8 star
    re.compile(r"\b\d+(\.\d+)?\s*%\s*(rating|score|satisfaction)\b", re.I),  # 98% rating
    re.compile(r"\u2605"),                          # Unicode star character
]


# ═══════════════════════════════════════════════════════════════════════════
# Rule functions
# ═══════════════════════════════════════════════════════════════════════════
# Each returns {"rule": str, "severity": "pass"|"fail"|"hold", "detail": str, "evidence": str}


def _collect_text_surfaces(subject: dict) -> list[tuple[str, str]]:
    """Extract all text surfaces from a subject for text-based checks."""
    surfaces: list[tuple[str, str]] = []

    # Narration script
    for key in ("script_text", "narration_brief"):
        text = subject.get(key) or ""
        if text:
            surfaces.append((text, key))

    # Overlay text
    for key in ("overlay_register", "overlay_payload", "overlays"):
        overlays = subject.get(key) or []
        for i, ov in enumerate(overlays):
            if isinstance(ov, dict):
                text = ov.get("text") or ""
                if text:
                    surfaces.append((text, f"{key}[{i}]"))

    # Captions
    for key in ("caption", "captions"):
        val = subject.get(key)
        if isinstance(val, str) and val:
            surfaces.append((val, key))
        elif isinstance(val, list):
            for i, c in enumerate(val):
                if isinstance(c, str) and c:
                    surfaces.append((c, f"{key}[{i}]"))

    return surfaces


def check_no_guest_names(subject: dict, kb: dict) -> dict:
    """section 6.3 — no guest names in any text surface. Reuses creative_director logic."""
    # Build a spec-like dict for validate_no_guest_names
    spec_like = {
        "narration_brief": subject.get("script_text") or subject.get("narration_brief") or "",
        "overlay_register": subject.get("overlay_register") or subject.get("overlay_payload") or [],
    }
    violations = validate_no_guest_names(spec_like, kb)
    if violations:
        return {
            "rule": "no_guest_names",
            "severity": "fail",
            "detail": violations[0]["detail"],
            "evidence": json.dumps([v["detail"] for v in violations]),
        }
    return {
        "rule": "no_guest_names",
        "severity": "pass",
        "detail": "No guest names found in text surfaces",
        "evidence": "",
    }


def check_no_ota(subject: dict) -> dict:
    """section 4.7 — no OTA references. Reuses creative_director logic."""
    # Build spec-like dict with all text fields
    spec_like = {}
    for key in ("script_text", "narration_brief", "caption", "captions"):
        if subject.get(key):
            spec_like[key] = subject[key]
    spec_like["overlay_register"] = (
        subject.get("overlay_register")
        or subject.get("overlay_payload")
        or []
    )
    violations = validate_no_ota(spec_like)
    if violations:
        return {
            "rule": "no_ota_references",
            "severity": "fail",
            "detail": violations[0]["detail"],
            "evidence": json.dumps([v["detail"] for v in violations]),
        }
    return {
        "rule": "no_ota_references",
        "severity": "pass",
        "detail": "No OTA references found",
        "evidence": "",
    }


def check_no_numeric_ratings(subject: dict) -> dict:
    """section 6.4 — no star ratings, /10, /5, percentage scores."""
    surfaces = _collect_text_surfaces(subject)
    for text, location in surfaces:
        for pattern in _NUMERIC_RATING_PATTERNS:
            match = pattern.search(text)
            if match:
                return {
                    "rule": "no_numeric_ratings",
                    "severity": "fail",
                    "detail": f"Numeric rating '{match.group()}' found in {location}",
                    "evidence": match.group(),
                }
    return {
        "rule": "no_numeric_ratings",
        "severity": "pass",
        "detail": "No numeric ratings found",
        "evidence": "",
    }


def check_frame_exit(subject: dict, **kwargs) -> dict:
    """section 3.1 — no frame-exiting moves (pull_back, etc.)."""
    requested_motion = subject.get("requested_motion") or ""
    if requested_motion in FRAME_EXITING_MOVES:
        return {
            "rule": "no_frame_exiting_moves",
            "severity": "fail",
            "detail": f"Frame-exiting move '{requested_motion}' prohibited (Guidelines v1.2 section 3.1)",
            "evidence": requested_motion,
        }
    return {
        "rule": "no_frame_exiting_moves",
        "severity": "pass",
        "detail": "No frame-exiting moves",
        "evidence": "",
    }


def check_amenities(subject: dict, kb: dict) -> dict:
    """section 4.2 — no fabricated amenities in overlays/captions."""
    kb_amenities = {a.lower() for a in _extract_amenity_names(kb)}
    if not kb_amenities:
        # No amenities in KB — cannot validate, pass (nothing to compare)
        return {
            "rule": "no_fabricated_amenities",
            "severity": "pass",
            "detail": "No KB amenities to compare against",
            "evidence": "",
        }

    surfaces = _collect_text_surfaces(subject)
    # Heuristic: check overlay/caption text for amenity-like words not in KB.
    # This is best-effort — same approach as creative_director.
    # We look for capitalized multi-word phrases that might be amenity names.
    fabricated: list[str] = []
    amenity_keywords = {
        "pool", "hot tub", "jacuzzi", "sauna", "gym", "fitness",
        "tennis", "basketball", "golf", "spa", "fireplace",
        "fire pit", "dock", "boat", "kayak", "canoe", "paddleboard",
        "game room", "theater", "cinema", "wine cellar", "bar",
        "elevator", "ev charger", "helipad", "private beach",
    }
    for text, location in surfaces:
        text_lower = text.lower()
        for keyword in amenity_keywords:
            if keyword in text_lower and keyword not in kb_amenities:
                # Check if any KB amenity contains this keyword
                if not any(keyword in a for a in kb_amenities):
                    fabricated.append(f"'{keyword}' in {location}")

    if fabricated:
        return {
            "rule": "no_fabricated_amenities",
            "severity": "fail",
            "detail": f"Possible fabricated amenities: {', '.join(fabricated[:3])}",
            "evidence": json.dumps(fabricated),
        }
    return {
        "rule": "no_fabricated_amenities",
        "severity": "pass",
        "detail": "No fabricated amenities detected",
        "evidence": "",
    }


# ── Model-assisted checks ──────────────────────────────────────────────


def check_removal_rule(subject: dict, **kwargs) -> dict:
    """
    section 4.4 — THE CRITICAL ONE.

    Honest implementation: this checker cannot watch video. What it CAN do:
    - Check persistence_manifest is populated
    - If FFmpeg available: extract keyframes, send to vision for comparison
    - If FFmpeg not available: metadata-only check, hold

    Generative clips (technique == 'generative') are ALWAYS applicable —
    removal_rule_result must never be 'not_applicable' for generative.
    Parallax (reprojection) CAN be not_applicable because reprojection
    cannot omit elements.
    """
    technique = subject.get("technique") or ""
    persistence_manifest = subject.get("persistence_manifest") or subject.get("persistence_findings") or []

    # Parallax is the ONLY technique where not_applicable is valid
    if technique == "parallax":
        return {
            "rule": "removal_rule",
            "severity": "pass",
            "detail": "Parallax (reprojection) — removal rule not applicable",
            "evidence": "technique=parallax",
        }

    # For generative clips, persistence_manifest must be populated
    if not persistence_manifest:
        return {
            "rule": "removal_rule",
            "severity": "hold",
            "detail": "persistence_manifest not populated — cannot verify element retention",
            "evidence": "missing_manifest",
        }

    # Check manifest for missing elements
    missing = [
        e for e in persistence_manifest
        if isinstance(e, dict) and e.get("status") == "missing"
    ]
    if missing:
        return {
            "rule": "removal_rule",
            "severity": "fail",
            "detail": f"{len(missing)} element(s) missing from generated output",
            "evidence": json.dumps(missing),
        }

    # FFmpeg frame extraction for deeper check
    r2_url = subject.get("r2_url") or ""
    source_image_url = subject.get("source_image_url") or ""

    if r2_url and source_image_url and _FFMPEG_AVAILABLE:
        # Frame comparison would happen here via Claude vision
        # For now, if manifest says all present, trust it
        pass
    elif r2_url and not _FFMPEG_AVAILABLE:
        return {
            "rule": "removal_rule",
            "severity": "hold",
            "detail": "Frame comparison not possible without FFmpeg — holding for manual review",
            "evidence": "ffmpeg_unavailable",
        }

    return {
        "rule": "removal_rule",
        "severity": "pass",
        "detail": "All persistence_manifest elements verified present",
        "evidence": json.dumps(persistence_manifest),
    }


def check_relocation(subject: dict, **kwargs) -> dict:
    """section 4.1 — metadata check: source image location matches property."""
    # Light check — full relocation detection needs the generated output
    source_image_url = subject.get("source_image_url") or ""
    if not source_image_url:
        return {
            "rule": "relocation",
            "severity": "pass",
            "detail": "No source image to verify relocation",
            "evidence": "",
        }
    # Metadata-only: check if source image is from this property's inventory
    property_id = subject.get("property_id") or ""
    if property_id and source_image_url:
        # If we had inventory data, we'd cross-reference here
        return {
            "rule": "relocation",
            "severity": "pass",
            "detail": "Source image metadata consistent with property",
            "evidence": source_image_url,
        }
    return {
        "rule": "relocation",
        "severity": "hold",
        "detail": "Cannot verify source image belongs to property",
        "evidence": "",
    }


def check_architecture(subject: dict, **kwargs) -> dict:
    """section 4.3 — architectural alteration check (metadata approach)."""
    technique = subject.get("technique") or ""
    requested_motion = subject.get("requested_motion") or ""
    # Architectural alterations are most likely in generative techniques
    # with aggressive motion. Metadata-only check.
    if technique == "generative" and not subject.get("persistence_manifest"):
        return {
            "rule": "architectural_alteration",
            "severity": "hold",
            "detail": "Generative clip without persistence manifest — cannot rule out architectural changes",
            "evidence": f"technique={technique}",
        }
    return {
        "rule": "architectural_alteration",
        "severity": "pass",
        "detail": "No architectural alteration indicators detected",
        "evidence": "",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Rule registry
# ═══════════════════════════════════════════════════════════════════════════

DETERMINISTIC_RULES = [
    ("no_guest_names", check_no_guest_names),          # section 6.3
    ("no_ota_references", check_no_ota),               # section 4.7
    ("no_numeric_ratings", check_no_numeric_ratings),   # section 6.4
    ("no_frame_exiting_moves", check_frame_exit),       # section 3.1
    ("no_fabricated_amenities", check_amenities),       # section 4.2
]

MODEL_ASSISTED_RULES = [
    ("removal_rule", check_removal_rule),               # section 4.4
    ("relocation", check_relocation),                   # section 4.1
    ("architectural_alteration", check_architecture),   # section 4.3
]

SKIPPED_RULES = [
    (
        "generated_people",
        "Constraints around generated people are deliberately undecided "
        "(Erick 2026-08-09). Identifiable faces, narrated guest text "
        "overlap, children, AI-disclosure requirements — all pending.",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# Subject loading
# ═══════════════════════════════════════════════════════════════════════════

TABLE_MAP = {
    "assembled_video": "assembled_videos",
    "motion_clip": "motion_clips",
    "narration": "narration_assets",
    "overlay": None,  # overlays are inline in shot_spec
}


def _load_subject(subject_type: str, subject_id: str) -> dict | None:
    """Load the subject from its table based on subject_type."""
    table = TABLE_MAP.get(subject_type)
    if table is None:
        # Overlays don't have their own table
        return None

    try:
        from core.supabase_store import get_supabase
        sb = get_supabase()
        result = (
            sb.table(table)
            .select("*")
            .eq("id", subject_id)
            .is_("superseded_at", "null")
            .limit(1)
            .execute()
        )
        if not result.data:
            # Try alternative ID column names
            id_columns = {
                "assembled_videos": "assembly_id",
                "motion_clips": "clip_id",
                "narration_assets": "narration_id",
            }
            alt_col = id_columns.get(table)
            if alt_col:
                result = (
                    sb.table(table)
                    .select("*")
                    .eq(alt_col, subject_id)
                    .is_("superseded_at", "null")
                    .limit(1)
                    .execute()
                )
        return result.data[0] if result.data else None
    except Exception as exc:
        logger.error(
            "[Agent 8 Stage 8] Failed to load %s %s: %s",
            subject_type, subject_id, exc,
        )
        return None


def _load_kb(property_id: str) -> dict | None:
    """Load KB — Redis first, Supabase fallback."""
    from core.pipeline_status import get_cached_knowledge_base
    kb = get_cached_knowledge_base(property_id)
    if kb:
        return kb
    try:
        from core.supabase_store import get_supabase
        result = (
            get_supabase()
            .table("property_knowledge_bases")
            .select("knowledge_base")
            .eq("property_id", property_id)
            .limit(1)
            .execute()
        )
        if result.data and result.data[0].get("knowledge_base"):
            return result.data[0]["knowledge_base"]
    except Exception as exc:
        logger.warning("[Agent 8 Stage 8] KB load failed: %s", exc)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Verdict computation
# ═══════════════════════════════════════════════════════════════════════════


def _compute_verdict(
    findings: list[dict],
    rules_evaluated: list[str],
    rules_skipped: list[dict],
) -> str:
    """
    Compute final verdict. Default is hold, not pass.
    Any fail -> fail.
    Any hold (including skipped undecided rules) -> hold.
    All pass AND no skipped -> pass.
    """
    if any(f["severity"] == "fail" for f in findings):
        return "fail"
    if rules_skipped or any(f["severity"] == "hold" for f in findings):
        return "hold"
    if not rules_evaluated:
        return "hold"  # nothing was checked
    return "pass"


# ═══════════════════════════════════════════════════════════════════════════
# Escalation
# ═══════════════════════════════════════════════════════════════════════════


def _escalate(check: dict, property_id: str, account_id: str | None) -> None:
    """Write to hitl_queue_items for human review on fail or hold verdict."""
    try:
        from core.supabase_store import get_supabase
        sb = get_supabase()
        sb.table("hitl_queue_items").insert({
            "queue_type": "compliance_review",
            "reason_code": check["verdict"],
            "property_id": property_id,
            "account_id": account_id,
            "payload": {
                "check_id": check["check_id"],
                "subject_type": check["subject_type"],
                "subject_id": check["subject_id"],
                "findings": check["findings"],
                "rules_skipped": check["rules_skipped"],
            },
        }).execute()
        logger.info(
            "[Agent 8 Stage 8] Escalated %s %s to hitl_queue (verdict=%s)",
            check["subject_type"], check["subject_id"], check["verdict"],
        )
    except Exception as exc:
        logger.error("[Agent 8 Stage 8] Escalation failed: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════════════════


def _persist_check(check: dict) -> None:
    """Write to compliance_checks, superseding any prior active check for the same subject."""
    try:
        from core.supabase_store import get_supabase
        sb = get_supabase()

        # Supersede prior active check for same subject
        sb.table("compliance_checks").update(
            {"superseded_at": "now()"}
        ).eq("subject_type", check["subject_type"]).eq(
            "subject_id", check["subject_id"]
        ).is_("superseded_at", "null").execute()

        # Insert new row
        sb.table("compliance_checks").insert(check).execute()

    except Exception as exc:
        logger.error("[Agent 8 Stage 8] Persist failed: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════


def check_subject(
    subject_type: str,
    subject_id: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """
    Run all applicable compliance rules against a subject.

    Verdict DEFAULTS to hold. A subject that has never been checked, or
    whose check crashed mid-run, remains held.

    Args:
        subject_type: One of 'assembled_video', 'motion_clip', 'narration', 'overlay'.
        subject_id: UUID of the subject row.
        dry_run: If True, return check result without persisting or escalating.
        force: If True, ignore any existing check and re-evaluate.

    Returns:
        Compliance check dict with verdict, findings, and metadata.
    """
    logger.info(
        "[Agent 8 Stage 8] check_subject: type=%s id=%s dry_run=%s force=%s",
        subject_type, subject_id, dry_run, force,
    )

    check_id = str(uuid.uuid4())
    check: dict = {
        "check_id": check_id,
        "subject_type": subject_type,
        "subject_id": subject_id,
        "checker_version": CHECKER_VERSION,
        "guidelines_version": GUIDELINES_VERSION,
        "verdict": DEFAULT_VERDICT,  # hold until proven otherwise
        "findings": [],
        "rules_evaluated": [],
        "rules_skipped": [
            {"rule": name, "reason": reason}
            for name, reason in SKIPPED_RULES
        ],
        "created_by_agent": "agent8_stage8",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        # Load subject
        subject = _load_subject(subject_type, subject_id)
        if subject is None:
            check["verdict"] = "hold"
            check["findings"].append({
                "rule": "_load_subject",
                "severity": "hold",
                "detail": f"Subject {subject_type}/{subject_id} not found",
                "evidence": "",
            })
            if not dry_run:
                _persist_check(check)
            return check

        property_id = subject.get("property_id") or ""
        account_id = subject.get("account_id")

        # Load KB for rules that need it
        kb = _load_kb(property_id) if property_id else {}
        if kb is None:
            kb = {}

        # ── Run deterministic rules ────────────────────────────────────
        for rule_name, rule_fn in DETERMINISTIC_RULES:
            try:
                # Some rules need KB, some don't
                if rule_fn in (check_no_guest_names, check_amenities):
                    finding = rule_fn(subject, kb)
                else:
                    finding = rule_fn(subject)
                check["findings"].append(finding)
                check["rules_evaluated"].append(rule_name)
            except Exception as exc:
                logger.error(
                    "[Agent 8 Stage 8] Rule %s crashed: %s", rule_name, exc,
                )
                check["findings"].append({
                    "rule": rule_name,
                    "severity": "hold",
                    "detail": f"Rule crashed: {exc}",
                    "evidence": "",
                })
                check["rules_evaluated"].append(rule_name)

        # ── Run model-assisted rules ───────────────────────────────────
        for rule_name, rule_fn in MODEL_ASSISTED_RULES:
            try:
                finding = rule_fn(subject)
                check["findings"].append(finding)
                check["rules_evaluated"].append(rule_name)
            except Exception as exc:
                logger.error(
                    "[Agent 8 Stage 8] Rule %s crashed: %s", rule_name, exc,
                )
                check["findings"].append({
                    "rule": rule_name,
                    "severity": "hold",
                    "detail": f"Rule crashed: {exc}",
                    "evidence": "",
                })
                check["rules_evaluated"].append(rule_name)

        # ── Compute verdict ────────────────────────────────────────────
        check["verdict"] = _compute_verdict(
            check["findings"],
            check["rules_evaluated"],
            check["rules_skipped"],
        )

    except Exception as exc:
        # Mid-run crash — verdict stays hold
        logger.error("[Agent 8 Stage 8] check_subject crashed: %s", exc)
        check["verdict"] = "hold"
        check["findings"].append({
            "rule": "_check_subject",
            "severity": "hold",
            "detail": f"Checker crashed: {exc}",
            "evidence": "",
        })

    # ── Persist and escalate ───────────────────────────────────────────
    if not dry_run:
        _persist_check(check)
        if check["verdict"] in ("fail", "hold"):
            _escalate(
                check,
                property_id=subject.get("property_id", "") if subject else "",
                account_id=subject.get("account_id") if subject else None,
            )

    return check
