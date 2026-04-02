"""
Pipeline Status Writer
Writes pipeline step status to Supabase (permanent record)
and Redis (live dashboard polling cache).

Every agent step calls this to update status.
The PMC dashboard polls the /api/pipeline-status endpoint every 30 seconds,
which reads from Supabase to show progress.
"""

import os
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from upstash_redis import Redis                  # ← replaces standard redis
from supabase import create_client, Client

logger = logging.getLogger(__name__)

# ── Clients ───────────────────────────────────────────────────────────────
_supabase: Optional[Client] = None
_redis: Optional[Redis] = None                  # ← type updated


def _get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        _supabase = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_ROLE_KEY"],  # Service role for server-side writes
        )
    return _supabase


def _get_redis() -> Redis:                       # ← return type updated
    global _redis
    if _redis is None:
        # upstash-redis authenticates via URL + token, not the standard Redis
        # wire protocol. REDIS_URL and REDIS_TOKEN are both set in .env.
        _redis = Redis(
            url=os.environ["REDIS_URL"],
            token=os.environ["REDIS_TOKEN"],
        )
    return _redis


# ── Status Enum ───────────────────────────────────────────────────────────

class PipelineStepStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETE  = "complete"
    FAILED    = "failed"
    SKIPPED   = "skipped"   # Optional step not applicable for this property


AGENT_LABELS = {
    1: "Content Ingestion",
    2: "Content Enhancement",
    3: "Visual Media",
    4: "Local Discovery",
    5: "Website Builder",
    6: "Social Media",
    7: "Analytics & Reporting",
}


def update_pipeline_status(
    property_id: str,
    agent_number: int,
    status: PipelineStepStatus,
    error_message: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    """
    Write pipeline step status to both Supabase and Redis.

    Supabase: permanent record, read by dashboard via API
    Redis: fast cache, TTL 24h, allows instant polling without DB hit
    """
    now = datetime.now(timezone.utc).isoformat()
    step_key = f"agent_{agent_number}"
    agent_label = AGENT_LABELS.get(agent_number, f"Agent {agent_number}")

    # ── Supabase upsert ───────────────────────────────────────────────────
    try:
        record = {
            "property_id": property_id,
            "agent_id": agent_number,
            "agent_label": agent_label,
            "status": status,
            "updated_at": now,
            "error_message": error_message,
            "metadata": metadata or {},
        }
        if status == PipelineStepStatus.RUNNING:
            record["started_at"] = now
        elif status in (PipelineStepStatus.COMPLETE, PipelineStepStatus.FAILED):
            record["completed_at"] = now

        _get_supabase().table("pipeline_status").upsert(
            record,
            on_conflict="property_id,agent_id",
        ).execute()

    except Exception as exc:
        # Pipeline status write failure must not crash the agent
        logger.error(f"Supabase pipeline status write failed: {exc}")

    # ── Redis cache ───────────────────────────────────────────────────────
    try:
        redis_key = f"pipeline:{property_id}"
        r = _get_redis()

        # Get or init pipeline dict in Redis
        existing = r.get(redis_key)
        pipeline_state = json.loads(existing) if existing else {"steps": {}}
        pipeline_state["steps"][step_key] = {
            "agent": agent_number,
            "label": agent_label,
            "status": status,
            "updated_at": now,
            "error": error_message,
        }
        pipeline_state["last_updated"] = now

        r.setex(redis_key, 86400, json.dumps(pipeline_state))  # 24h TTL

    except Exception as exc:
        logger.error(f"Redis pipeline status write failed: {exc}")


def cache_knowledge_base(
    property_id: str,
    kb_dict: dict,
    ttl_seconds: int = 86400,
) -> None:
    """
    Cache the property knowledge base in Redis for parallel agent access.
    Key: property:{id}:knowledge_base
    TTL: 24h (enough for the full pipeline to complete)
    """
    try:
        r = _get_redis()
        r.setex(
            f"property:{property_id}:knowledge_base",
            ttl_seconds,
            json.dumps(kb_dict),
        )
        logger.debug(f"Knowledge base cached in Redis for property {property_id}")
    except Exception as exc:
        logger.error(f"Redis KB cache write failed for {property_id}: {exc}")


def get_cached_knowledge_base(property_id: str) -> Optional[dict]:
    """
    Retrieve cached knowledge base from Redis.
    Returns None if not found or expired.
    """
    try:
        r = _get_redis()
        raw = r.get(f"property:{property_id}:knowledge_base")
        return json.loads(raw) if raw else None
    except Exception as exc:
        logger.error(f"Redis KB cache read failed for {property_id}: {exc}")
        return None


def get_pipeline_status(property_id: str) -> Optional[dict]:
    """
    Retrieve pipeline status for a property from Supabase.
    Called by the /status/{property_id} endpoint.
    """
    try:
        result = (
            _get_supabase()
            .table("pipeline_status")
            .select("*")
            .eq("property_id", property_id)
            .execute()
        )
        return {"property_id": property_id, "steps": result.data}
    except Exception as exc:
        logger.error(f"Failed to fetch pipeline status for {property_id}: {exc}")
        return None

