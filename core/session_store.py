"""
Visitor session create/resume logic for Staylio public endpoints.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import uuid

from core.supabase_store import get_supabase


@dataclass
class SessionResult:
    visitor_session_id: str
    property_id: str
    account_id: str
    started_at: str
    session_status: str  # "created" | "resumed"


def create_or_resume_session(
    property_id: str,
    account_id: str,
    session_key: str,
    visitor_key: Optional[str] = None,
    landing_url: Optional[str] = None,
    referrer_url: Optional[str] = None,
    device_type: Optional[str] = None,
    utm_source: Optional[str] = None,
    utm_medium: Optional[str] = None,
    utm_campaign: Optional[str] = None,
    utm_term: Optional[str] = None,
    utm_content: Optional[str] = None,
) -> SessionResult:
    """
    Create a new visitor session or resume an existing one.
    Uses (property_id, session_key) as the unique key.
    Returns session_status "created" or "resumed".
    """
    supabase = get_supabase()
    now = datetime.now(timezone.utc).isoformat()

    existing = (
        supabase
        .table("visitor_sessions")
        .select("id, first_seen_at")
        .eq("property_id", property_id)
        .eq("session_key", session_key)
        .limit(1)
        .execute()
    )

    if existing.data:
        row = existing.data[0]
        supabase.table("visitor_sessions").update(
            {"last_seen_at": now}
        ).eq("id", row["id"]).execute()

        return SessionResult(
            visitor_session_id=row["id"],
            property_id=property_id,
            account_id=account_id,
            started_at=row["first_seen_at"],
            session_status="resumed",
        )

    new_id = str(uuid.uuid4())
    supabase.table("visitor_sessions").insert({
        "id": new_id,
        "property_id": property_id,
        "account_id": account_id,
        "session_key": session_key,
        "visitor_key": visitor_key,
        "landing_url": landing_url,
        "referrer_url": referrer_url,
        "device_type": device_type,
        "utm_source": utm_source,
        "utm_medium": utm_medium,
        "utm_campaign": utm_campaign,
        "utm_term": utm_term,
        "utm_content": utm_content,
        "first_seen_at": now,
        "last_seen_at": now,
    }).execute()

    return SessionResult(
        visitor_session_id=new_id,
        property_id=property_id,
        account_id=account_id,
        started_at=now,
        session_status="created",
    )
