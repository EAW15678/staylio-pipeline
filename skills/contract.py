"""
Skill contract — the typed result and execution wrapper for all skills.

Every skill returns SkillResult: ok(data) | held(reason) | failed(reason).
No silent degradation, ever.

Skills:
  - Read/write from staging Supabase ONLY (no Redis)
  - Declare vendor requirements, checked at CALL time (not import time)
  - Emit run_step + cost_event for every vendor call
  - Are deterministic where identity allows (photo_id, input_hash)
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Result types ─────────────────────────────────────────────────────────

@dataclass
class SkillResult:
    """Typed result from any skill. Never None, never silent.

    Ruling 6 semantics (target-architecture.md, Erick 2026-08-11):
      ok       = whole job done, or nothing-to-do (noop)
      failed   = partial or total vendor failure — includes
                 {attempted, succeeded, failed, error_class, human_required}
      held     = judgment gates only (compliance, quality review)

    billing/auth error_class → no retry, write hitl_queue_items, stop.
    """
    status: str  # "ok", "held", "failed"
    data: Any = None
    reason: Optional[str] = None
    cost_events: list = field(default_factory=list)
    steps: list = field(default_factory=list)

    @staticmethod
    def ok(data: Any, cost_events: list = None, steps: list = None) -> "SkillResult":
        return SkillResult(status="ok", data=data,
                           cost_events=cost_events or [], steps=steps or [])

    @staticmethod
    def noop(reason: str, data: Any = None) -> "SkillResult":
        """Nothing to do — idempotent converge, already done."""
        d = {"noop": True, "reason": reason}
        if isinstance(data, dict):
            d.update(data)
        return SkillResult(status="ok", data=d)

    @staticmethod
    def held(reason: str, data: Any = None) -> "SkillResult":
        return SkillResult(status="held", reason=reason, data=data)

    @staticmethod
    def failed(reason: str, attempted: int = 0, succeeded: int = 0,
               failed_count: int = 0, error_class: str = None,
               human_required: bool = False) -> "SkillResult":
        return SkillResult(
            status="failed",
            reason=reason,
            data={
                "attempted": attempted,
                "succeeded": succeeded,
                "failed": failed_count,
                "error_class": error_class,
                "human_required": human_required,
            },
        )

    @property
    def is_ok(self) -> bool:
        return self.status == "ok"

    @property
    def is_billing_error(self) -> bool:
        return (self.data or {}).get("error_class") == "billing"


# ── Substrate connection ─────────────────────────────────────────────────

_staging_client = None


def get_substrate():
    """Get the staging Supabase client. Fails loudly if not configured."""
    global _staging_client
    if _staging_client is not None:
        return _staging_client

    url = os.environ.get("STAGING_SUPABASE_URL")
    key = os.environ.get("STAGING_SUPABASE_KEY")
    if not url or not key:
        raise EnvironmentError(
            "STAGING_SUPABASE_URL and STAGING_SUPABASE_KEY must be set. "
            "Skills read/write ONLY from the substrate (staging Supabase). "
            "No Redis, no production database."
        )
    # Force HTTP/1.1 globally — Railway's Python 3.13 + httpx HTTP/2
    # causes StreamReset errors against Supabase PostgREST.
    import httpx
    _orig_client_init = httpx.Client.__init__
    def _patched_client_init(self, *args, **kwargs):
        kwargs['http2'] = False
        _orig_client_init(self, *args, **kwargs)
    httpx.Client.__init__ = _patched_client_init

    from supabase import create_client
    _staging_client = create_client(url, key)
    return _staging_client


def reset_substrate():
    """Reset the cached client (for testing)."""
    global _staging_client
    _staging_client = None


# ── Vendor requirement checking ──────────────────────────────────────────

def require_env(var_name: str, description: str) -> str:
    """Check an env var at CALL time, not import time. Fails loudly."""
    val = os.environ.get(var_name, "")
    if not val:
        raise EnvironmentError(
            f"Required env var {var_name} is not set. "
            f"Needed for: {description}. "
            f"Set it before calling this skill."
        )
    return val


# ── Run/step/cost recording ──────────────────────────────────────────────

def record_run(sb, property_id: str, workflow: str) -> str:
    """Create a run record and return the run_id."""
    run_id = str(uuid.uuid4())
    sb.table("runs").insert({
        "run_id": run_id,
        "property_id": property_id,
        "workflow": workflow,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }).execute()
    return run_id


def record_step(sb, run_id: str, step_name: str, status: str = "running") -> str:
    """Create a run_step record and return the step_id."""
    step_id = str(uuid.uuid4())
    sb.table("run_steps").insert({
        "step_id": step_id,
        "run_id": run_id,
        "step_name": step_name,
        "status": status,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }).execute()
    return step_id


def complete_step(sb, step_id: str, status: str = "complete",
                  error_message: str = None, metadata: dict = None):
    """Mark a run_step as complete or failed."""
    import json
    update = {
        "status": status,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    if error_message:
        update["error_message"] = error_message
    if metadata:
        update["metadata"] = metadata
    sb.table("run_steps").update(update).eq("step_id", step_id).execute()


def complete_run(sb, run_id: str, status: str = "complete", error_summary: str = None):
    """Mark a run as complete or failed."""
    update = {
        "status": status,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    if error_summary:
        update["error_summary"] = error_summary
    sb.table("runs").update(update).eq("run_id", run_id).execute()


def emit_cost(sb, run_id: str, property_id: str, vendor: str, service: str,
              units: float, unit_name: str, unit_cost: float = None,
              total_cost: float = None, workflow_name: str = None,
              generation_reason: str = None, discriminator: str = None):
    """Record a cost event in the substrate."""
    sb.table("cost_events").insert({
        "run_id": run_id,
        "property_id": property_id,
        "vendor": vendor,
        "service": service,
        "units": units,
        "unit_name": unit_name,
        "unit_cost": unit_cost,
        "total_cost": total_cost or (units * unit_cost if unit_cost else None),
        "workflow_name": workflow_name,
        "generation_reason": generation_reason,
        "discriminator": discriminator,
    }).execute()


# ── Skills R2 storage ────────────────────────────────────────────────────
# Skills use env-driven bucket + public URL — no hardcoded bucket names.
# STAGING_R2_BUCKET and STAGING_R2_PUBLIC_URL are set by Erick.

_skills_r2_client = None


def _get_skills_r2_client():
    """Get an S3-compatible client for the skills R2 bucket."""
    global _skills_r2_client
    if _skills_r2_client is not None:
        return _skills_r2_client
    import boto3
    from botocore.config import Config
    _skills_r2_client = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
    return _skills_r2_client


def skills_r2_upload(key: str, data: bytes, content_type: str = "image/jpeg") -> str:
    """Upload to the skills R2 bucket. Returns the public CDN URL.

    Bucket and public base are env-driven (STAGING_R2_BUCKET, STAGING_R2_PUBLIC_URL).
    No hardcoded bucket name reachable from any skill.
    """
    bucket = require_env("STAGING_R2_BUCKET", "R2 bucket for skill outputs")
    public_base = require_env("STAGING_R2_PUBLIC_URL", "R2 public URL for skill outputs")
    client = _get_skills_r2_client()
    client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    url = f"{public_base.rstrip('/')}/{key}"
    logger.info("[r2] Uploaded %s → %s (%d bytes)", key, url[:60], len(data))
    return url


def escalate_halt(
    sb,
    property_id: str,
    queue_type: str,
    reason_code: str,
    title: str,
    detail: str,
    *,
    property_name: str = None,
    run_id: str = None,
    step_name: str = None,
    extra_payload: dict = None,
):
    """Write a halt-class hitl_queue_items row AND send an email alert.

    Ruling 3-R: halt class = billing/auth, slug collision, zero-photo,
    unresolvable, cross-property contamination, legal-risk compliance.
    Quality annotations (needs_review) do NOT trigger alerts.
    """
    payload = extra_payload or {}
    payload["detail"] = detail[:500]

    # Look up account_id from the property
    account_id = None
    try:
        prop = sb.table("properties").select("account_id").eq("id", property_id).limit(1).execute()
        if prop.data:
            account_id = prop.data[0].get("account_id")
    except Exception:
        pass

    result = sb.table("hitl_queue_items").insert({
        "property_id": property_id,
        "account_id": account_id,
        "queue_type": queue_type,
        "reason_code": reason_code,
        "title": title,
        "description": detail[:500],
        "priority": "urgent",
        "status": "pending",
        "payload": payload,
    }).execute()

    hitl_row_id = result.data[0]["id"] if result.data else None
    logger.warning("[contract] HALT escalation: %s — %s (hitl=%s)",
                   queue_type, title[:60], hitl_row_id[:12] if hitl_row_id else "?")

    # Send email alert
    if hitl_row_id:
        try:
            from skills.notify import send_halt_alert
            send_halt_alert(
                sb, hitl_row_id, property_id,
                property_name=property_name or property_id[:12],
                error_class=reason_code,
                detail=detail,
                run_id=run_id,
                step_name=step_name,
            )
        except Exception as exc:
            logger.warning("[contract] Alert send failed: %s", str(exc)[:80])

    return hitl_row_id


def escalate_billing(sb, property_id: str, vendor: str, error_message: str,
                     run_id: str = None, step_name: str = None):
    """Convenience wrapper for billing/auth halts."""
    return escalate_halt(
        sb, property_id,
        queue_type="billing_error",
        reason_code=f"{vendor}_billing",
        title=f"{vendor} billing error — credits depleted or auth invalid",
        detail=error_message[:500],
        extra_payload={"vendor": vendor},
        run_id=run_id,
        step_name=step_name,
    )
