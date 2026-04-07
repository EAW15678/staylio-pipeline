"""
Staylio.ai — Pipeline Entry Point
Railway deployment entry point for the LangGraph agent pipeline.
Exposes a FastAPI HTTP interface so Railway can detect and serve it.
"""

import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Staylio.ai pipeline starting up...")
    yield
    logger.info("Staylio.ai pipeline shutting down...")


# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Staylio.ai Pipeline",
    description="7-agent LangGraph marketing pipeline for STR properties",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://staylio.ai",
        "https://intake.staylio.ai",
        "null",                    # covers file:// origin during local testing
        "http://localhost:8080",
        "http://localhost:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request models ─────────────────────────────────────────────────────────
class PipelineRunRequest(BaseModel):
    property_id: str
    client_id: str | None = None
    listing_urls: list[str] = []
    intake_data: dict = {}


class SessionStartRequest(BaseModel):
    subdomain: str | None = None
    property_id: str | None = None
    session_key: str
    visitor_key: str | None = None
    landing_url: str | None = None
    referrer_url: str | None = None
    device_type: str | None = None
    utm_source: str | None = None
    utm_medium: str | None = None
    utm_campaign: str | None = None
    utm_term: str | None = None
    utm_content: str | None = None


class SessionStartResponse(BaseModel):
    visitor_session_id: str
    property_id: str
    account_id: str
    started_at: str
    session_status: str  # "created" | "resumed"


class IntakeSubmissionRequest(BaseModel):
    """
    Full intake portal submission — received from the browser portal.
    Railway writes all records to Supabase server-side using SERVICE_ROLE_KEY,
    so no database credentials are ever exposed in the portal HTML.
    """
    # Owner / PMC identity
    owner_name: str
    owner_email: str
    property_name: str

    # Property profile
    vibe_profile: str           # full enum value e.g. 'multigenerational_retreat'
    booking_url: str
    ical_url: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    pms_type: Optional[str] = None

    # Listing URLs
    listing_urls: list[str] = []
    airbnb_url: Optional[str] = None
    vrbo_url: Optional[str] = None
    pmc_website_url: Optional[str] = None

    # Rich intake content (stored in raw_data JSONB)
    owner_story: Optional[str] = None
    wow_factor: Optional[str] = None
    hidden_gems: Optional[str] = None
    arrival_info: Optional[str] = None
    guest_love: Optional[str] = None
    dont_miss: Optional[str] = None
    surround_areas: Optional[str] = None
    area_vibe: Optional[str] = None
    vibe_notes: Optional[str] = None
    extra_notes: Optional[str] = None
    seasonal_notes: Optional[str] = None
    local_events: Optional[str] = None
    off_season_notes: Optional[str] = None
    group_size: Optional[str] = None
    stay_length: Optional[str] = None
    owner_years: Optional[str] = None
    host_style: Optional[str] = None
    best_seasons: Optional[str] = None

    # Arrays
    guest_types: list[str] = []
    amenities: list[str] = []
    guest_book_entries: list[dict] = []


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Health check endpoint — Railway uses this to verify the service is running."""
    return {"status": "ok", "service": "staylio-pipeline", "version": "2.0.0"}


@app.get("/")
def root():
    return {
        "service": "Staylio.ai Pipeline",
        "version": "2.0.0",
        "status": "running",
        "endpoints": ["/health", "/intake", "/run", "/status/{property_id}"],
    }


@app.post("/intake")
async def intake_submission(
    request: IntakeSubmissionRequest,
    background_tasks: BackgroundTasks,
):
    """
    Receives the full intake portal submission from the browser.
    Writes pmc_clients, properties, and intake_submissions to Supabase
    using the SERVICE_ROLE_KEY (never exposed to the browser).
    Then triggers the pipeline in the background.

    Returns property_id and client_id so the portal can poll /status.
    """
    from core.supabase_store import get_supabase
    from pipeline.graph import run_intake_pipeline as execute_pipeline

    try:
        sb = get_supabase()

        client_id   = str(uuid4())
        property_id = str(uuid4())

        # ── Generate slug ──────────────────────────────────────────────
        slug = _make_slug(request.property_name)

        # ── Step 1: Insert PMC client ──────────────────────────────────
        sb.table("pmc_clients").insert({
            "client_id":    client_id,
            "company_name": request.property_name,
            "contact_name": request.owner_name,
            "contact_email": request.owner_email,
            "pms_type":     request.pms_type,
            "created_at":   datetime.now(timezone.utc).isoformat(),
        }).execute()

        # ── Step 2: Insert property ────────────────────────────────────
        sb.table("properties").insert({
            "property_id":  property_id,
            "client_id":    client_id,
            "client_type":  "pmc",
            "name":         request.property_name,
            "slug":         slug,
            "city":         request.city,
            "state":        request.state,
            "zip_code":     request.zip_code,
            "vibe_profile": request.vibe_profile,
        }).execute()

        # ── Step 3: Insert intake submission ───────────────────────────
        sb.table("intake_submissions").insert({
            "property_id":       property_id,
            "client_id":         client_id,
            "client_channel":    "pmc",
            "property_name":     request.property_name,
            "vibe_profile":      request.vibe_profile,
            "airbnb_url":        request.airbnb_url,
            "vrbo_url":          request.vrbo_url,
            "pmc_website_url":   request.pmc_website_url,
            "booking_url":       request.booking_url,
            "ical_url":          request.ical_url,
            "city":              request.city,
            "state":             request.state,
            "zip_code":          request.zip_code,
            "guest_book_entries": request.guest_book_entries or [],
            "submitted_at":      datetime.now(timezone.utc).isoformat(),
            "raw_data": {
                "owner_name":       request.owner_name,
                "owner_email":      request.owner_email,
                "owner_story":      request.owner_story,
                "wow_factor":       request.wow_factor,
                "hidden_gems":      request.hidden_gems,
                "arrival_info":     request.arrival_info,
                "guest_love":       request.guest_love,
                "dont_miss":        request.dont_miss,
                "surround_areas":   request.surround_areas,
                "area_vibe":        request.area_vibe,
                "vibe_notes":       request.vibe_notes,
                "extra_notes":      request.extra_notes,
                "seasonal_notes":   request.seasonal_notes,
                "local_events":     request.local_events,
                "off_season_notes": request.off_season_notes,
                "group_size":       request.group_size,
                "stay_length":      request.stay_length,
                "owner_years":      request.owner_years,
                "host_style":       request.host_style,
                "best_seasons":     request.best_seasons,
                "guest_types":      request.guest_types,
                "amenities":        request.amenities,
                "listing_urls":     request.listing_urls,
                "city":             request.city,
                "state":            request.state,
                "zip_code":         request.zip_code,
                "vibe_profile":     request.vibe_profile,
            },
        }).execute()

        # ── Step 4: Trigger pipeline in background ─────────────────────
        background_tasks.add_task(
            execute_pipeline,
            property_id=property_id,
            client_id=client_id,
        )

        logger.info(
            f"Intake received for '{request.property_name}' "
            f"(property_id={property_id}, client_id={client_id}). "
            f"Pipeline triggered."
        )

        return {
            "status":      "started",
            "property_id": property_id,
            "client_id":   client_id,
            "message":     "Intake saved. Pipeline is running. Poll /status/{property_id} for updates.",
        }

    except Exception as exc:
        logger.error(f"Intake submission failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/run")
async def run_pipeline(request: PipelineRunRequest, background_tasks: BackgroundTasks):
    """
    Direct pipeline trigger — used for testing via curl.
    Requires property and intake records to already exist in Supabase.
    """
    try:
        from pipeline.graph import run_intake_pipeline as execute_pipeline
        from core.property_context import resolve_property_context
        ctx = resolve_property_context(property_id=request.property_id)
        account_id = ctx.account_id

        background_tasks.add_task(
            execute_pipeline,
            property_id=request.property_id,
            client_id=account_id,
        )

        logger.info(f"Pipeline triggered for property {request.property_id}")
        return {
            "status": "started",
            "property_id": request.property_id,
            "message": "Pipeline running in background. Check /status/{property_id} for updates.",
        }

    except Exception as exc:
        logger.error(f"Pipeline trigger failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/public/sessions/start", response_model=SessionStartResponse)
async def session_start(request: SessionStartRequest):
    """
    Start or resume a visitor session on a property page.

    Resolves property context from subdomain (primary) or property_id (fallback).
    Never trusts caller-supplied account_id — always derives from DB.
    Returns visitor_session_id and session_status (created | resumed).
    """
    try:
        from core.property_context import resolve_property_context
        from core.session_store import create_or_resume_session

        ctx = resolve_property_context(
            property_id=request.property_id,
            subdomain=request.subdomain,
        )

        result = create_or_resume_session(
            property_id=ctx.property_id,
            account_id=ctx.account_id,
            session_key=request.session_key,
            visitor_key=request.visitor_key,
            landing_url=request.landing_url,
            referrer_url=request.referrer_url,
            device_type=request.device_type,
            utm_source=request.utm_source,
            utm_medium=request.utm_medium,
            utm_campaign=request.utm_campaign,
            utm_term=request.utm_term,
            utm_content=request.utm_content,
        )

        return SessionStartResponse(
            visitor_session_id=result.visitor_session_id,
            property_id=result.property_id,
            account_id=result.account_id,
            started_at=result.started_at,
            session_status=result.session_status,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Session start failed: {exc}")
        raise HTTPException(status_code=500, detail="Session start failed.")


@app.get("/status/{property_id}")
def get_pipeline_status(property_id: str):
    """
    Get the current pipeline status for a property.
    Reads from Supabase pipeline_status table.
    """
    try:
        from core.pipeline_status import get_pipeline_status as fetch_status
        status = fetch_status(property_id)
        return status or {"property_id": property_id, "status": "not_found"}
    except Exception as exc:
        logger.error(f"Status fetch failed for {property_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Utilities ──────────────────────────────────────────────────────────────
def _make_slug(name: str) -> str:
    """Generate a URL-safe slug from a property name."""
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:50]


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
