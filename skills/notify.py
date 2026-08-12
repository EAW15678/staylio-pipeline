"""
Skill: notify — alert a human when a halt-class failure occurs.

Transport: Email via Resend (resend.com). Free tier: 100 emails/day,
no domain verification needed for v1 (sends from onboarding@resend.dev).

Trigger: called by the contract's escalate path when severity=urgent.
NOT a DB trigger — the skill layer owns it so the alert carries context.

Ruling 3-R: ONLY halt-class failures alert. Quality annotations (needs_review)
stay silent.

Failure honesty: if the alert fails to send, the halt still halts.
The send-failure is recorded on the hitl row (alert_sent=false, alert_error).
No retry storms: one retry, then record.

Idempotent: alert_sent flag prevents double-send on re-runs.

Usage:
    from skills.notify import send_halt_alert
    send_halt_alert(sb, hitl_row_id, property_name, error_class, detail)
"""

import json
import logging
import os

import httpx

logger = logging.getLogger(__name__)

RESEND_API_URL = "https://api.resend.com/emails"


def send_halt_alert(
    sb,
    hitl_row_id: str,
    property_id: str,
    property_name: str,
    error_class: str,
    detail: str,
    run_id: str = None,
    step_name: str = None,
) -> bool:
    """Send a halt alert email via Resend.

    Returns True if sent, False if failed (recorded on the hitl row).
    Never raises — alert failure must not mask the underlying halt.
    """
    resend_key = os.environ.get("RESEND_API_KEY", "")
    alert_to = os.environ.get("ALERT_EMAIL_TO", "erick@staylio.ai")

    if not resend_key:
        logger.warning("[notify] RESEND_API_KEY not set — alert not sent")
        _mark_alert_failed(sb, hitl_row_id, "RESEND_API_KEY not set")
        return False

    # ── Idempotency: check if alert already sent for this hitl row ───────
    try:
        row = sb.table("hitl_queue_items").select("id, payload").eq("id", hitl_row_id).limit(1).execute()
        if row.data:
            payload = row.data[0].get("payload")
            if isinstance(payload, str):
                payload = json.loads(payload)
            if isinstance(payload, dict) and payload.get("alert_sent"):
                logger.info("[notify] Alert already sent for hitl %s — skipping", hitl_row_id[:12])
                return True
    except Exception:
        pass  # Proceed with send attempt

    # ── Build the email ──────────────────────────────────────────────────
    subject = f"[UpliftStays HALT] {property_name} — {error_class}"

    run_link = ""
    if run_id:
        run_link = f"<p><strong>Run:</strong> {run_id}</p>"
    step_info = ""
    if step_name:
        step_info = f"<p><strong>Failed step:</strong> {step_name}</p>"

    html_body = f"""
    <div style="font-family: -apple-system, sans-serif; max-width: 600px; margin: 0 auto;">
        <h2 style="color: #c0392b;">⚠️ Pipeline Halt</h2>
        <p><strong>Property:</strong> {property_name}</p>
        <p><strong>Property ID:</strong> <code>{property_id}</code></p>
        <p><strong>Error class:</strong> {error_class}</p>
        {step_info}
        {run_link}
        <hr style="border: none; border-top: 1px solid #eee; margin: 1.5rem 0;">
        <p><strong>Detail:</strong></p>
        <pre style="background: #f8f8f8; padding: 1rem; border-radius: 4px; font-size: 0.9rem; overflow-x: auto;">{detail[:1000]}</pre>
        <hr style="border: none; border-top: 1px solid #eee; margin: 1.5rem 0;">
        <p style="color: #888; font-size: 0.85rem;">
            This alert was sent because the pipeline encountered a halt-class failure
            that requires human intervention. The pipeline has stopped and will not
            proceed until the issue is resolved.
        </p>
        <p style="color: #888; font-size: 0.85rem;">
            HITL queue item: <code>{hitl_row_id}</code>
        </p>
    </div>
    """

    # ── Send via Resend ──────────────────────────────────────────────────
    # From: onboarding@resend.dev (Resend's default sender, no domain verification needed)
    email_payload = {
        "from": "UpliftStays Alerts <onboarding@resend.dev>",
        "to": [alert_to],
        "subject": subject,
        "html": html_body,
    }

    for attempt in range(2):  # One retry
        try:
            resp = httpx.post(
                RESEND_API_URL,
                headers={
                    "Authorization": f"Bearer {resend_key}",
                    "Content-Type": "application/json",
                },
                json=email_payload,
                timeout=15,
            )

            if resp.status_code in (200, 201):
                data = resp.json()
                message_id = data.get("id", "unknown")
                logger.info("[notify] Alert sent: message_id=%s to=%s subject=%s",
                           message_id, alert_to, subject)
                _mark_alert_sent(sb, hitl_row_id, message_id)
                return True
            else:
                error_text = resp.text[:200]
                logger.warning("[notify] Resend returned %d: %s (attempt %d)",
                              resp.status_code, error_text, attempt + 1)
                if attempt == 0:
                    continue  # One retry
                _mark_alert_failed(sb, hitl_row_id, f"Resend {resp.status_code}: {error_text}")
                return False

        except Exception as exc:
            logger.warning("[notify] Send failed: %s (attempt %d)", str(exc)[:80], attempt + 1)
            if attempt == 0:
                continue
            _mark_alert_failed(sb, hitl_row_id, str(exc)[:200])
            return False

    return False


def _mark_alert_sent(sb, hitl_row_id: str, message_id: str):
    """Update the hitl row's payload with alert_sent=true."""
    try:
        row = sb.table("hitl_queue_items").select("payload").eq("id", hitl_row_id).limit(1).execute()
        payload = {}
        if row.data:
            p = row.data[0].get("payload")
            if isinstance(p, str):
                payload = json.loads(p)
            elif isinstance(p, dict):
                payload = p
        payload["alert_sent"] = True
        payload["alert_message_id"] = message_id
        sb.table("hitl_queue_items").update(
            {"payload": json.dumps(payload)}
        ).eq("id", hitl_row_id).execute()
    except Exception as exc:
        logger.warning("[notify] Could not mark alert_sent on hitl row: %s", str(exc)[:80])


def _mark_alert_failed(sb, hitl_row_id: str, error: str):
    """Update the hitl row's payload with alert_sent=false, alert_error."""
    try:
        row = sb.table("hitl_queue_items").select("payload").eq("id", hitl_row_id).limit(1).execute()
        payload = {}
        if row.data:
            p = row.data[0].get("payload")
            if isinstance(p, str):
                payload = json.loads(p)
            elif isinstance(p, dict):
                payload = p
        payload["alert_sent"] = False
        payload["alert_error"] = error[:200]
        sb.table("hitl_queue_items").update(
            {"payload": json.dumps(payload)}
        ).eq("id", hitl_row_id).execute()
    except Exception as exc:
        logger.warning("[notify] Could not mark alert_error on hitl row: %s", str(exc)[:80])
