"""
Staylio.ai — Pipeline Entry Point
Railway deployment entry point for the LangGraph agent pipeline.
Exposes a FastAPI HTTP interface so Railway can detect and serve it.
"""

import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORTAL_API_SECRET = os.environ.get("PORTAL_API_SECRET", "")


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
        "https://vista-azule.staylio.ai",
        "null",
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


class PublicEventRequest(BaseModel):
    visitor_session_id: str
    event_name: str
    event_payload: dict | None = None
    occurred_at: datetime | None = None


class PublicCtaClickRequest(BaseModel):
    visitor_session_id: str
    cta_type: str
    cta_location: str | None = None
    destination_url: str | None = None
    clicked_at: datetime | None = None


class PublicLeadRequest(BaseModel):
    visitor_session_id: str
    primary_email: EmailStr
    cta_click_id: str | None = None
    full_name: str | None = None
    phone: str | None = None
    requested_dates: dict | None = None
    party_size: int | None = None
    trip_intent: str | None = None
    consent_status: str | None = None


class BookingReportRequest(BaseModel):
    property_id: str
    booking_value: float
    check_in_date: date
    check_out_date: date
    currency: str | None = None
    booking_date: date | None = None
    notes: str | None = None
    visitor_session_id: str | None = None
    lead_id: str | None = None


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


ALLOWED_EVENT_NAMES = {
    "page_viewed",
    "scroll_depth_reached",
    "gallery_opened",
    "video_played",
    "video_completed",
    "faq_expanded",
    "map_interacted",
    "owner_story_expanded",
}

@app.post("/public/events", status_code=202)
async def public_event(request: PublicEventRequest):
    """
    Record a page engagement event for a visitor session.

    Derives property_id and account_id from the session record.
    Never trusts caller-supplied context.
    Event name must be in the allowlist.
    Fails open on non-critical errors — tracking must never block the guest.
    """
    try:
        from core.supabase_store import get_supabase
        from datetime import timezone

        if request.event_name not in ALLOWED_EVENT_NAMES:
            raise HTTPException(
                status_code=400,
                detail=f"event_name '{request.event_name}' is not allowed.",
            )

        supabase = get_supabase()

        session = (
            supabase
            .table("visitor_sessions")
            .select("id, property_id, account_id")
            .eq("id", request.visitor_session_id)
            .limit(1)
            .execute()
        )

        if not session.data:
            raise HTTPException(
                status_code=404,
                detail=f"visitor_session_id '{request.visitor_session_id}' not found.",
            )

        row = session.data[0]
        occurred_at = (
            request.occurred_at.astimezone(timezone.utc).isoformat()
            if request.occurred_at
            else datetime.now(timezone.utc).isoformat()
        )

        supabase.table("page_events").insert({
            "property_id": row["property_id"],
            "account_id": row["account_id"],
            "visitor_session_id": row["id"],
            "event_name": request.event_name,
            "event_payload": request.event_payload,
            "occurred_at": occurred_at,
        }).execute()

        logger.info(
            f"[events] {request.event_name} recorded — "
            f"session={request.visitor_session_id} "
            f"property={row['property_id']}"
        )
        return {"status": "accepted"}

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            f"[events] write failed — "
            f"session={request.visitor_session_id} "
            f"event={request.event_name} "
            f"error={exc}",
            exc_info=True,
        )
        return {"status": "accepted"}


ALLOWED_CTA_TYPES = {"book_now", "check_availability", "save_property", "get_alerts"}

@app.post("/public/cta-clicks", status_code=202)
async def public_cta_click(request: PublicCtaClickRequest):
    """
    Record a CTA click before redirecting the guest.

    Derives property_id and account_id from the session record.
    Never trusts caller-supplied context.
    Returns cta_click_id so lead capture can link back to this click.
    Fails open — must never delay or block the booking redirect.
    """
    try:
        from core.supabase_store import get_supabase
        from datetime import timezone
        import uuid

        if request.cta_type not in ALLOWED_CTA_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"cta_type '{request.cta_type}' is not allowed.",
            )

        supabase = get_supabase()

        session = (
            supabase
            .table("visitor_sessions")
            .select("id, property_id, account_id")
            .eq("id", request.visitor_session_id)
            .limit(1)
            .execute()
        )

        if not session.data:
            raise HTTPException(
                status_code=404,
                detail=f"visitor_session_id '{request.visitor_session_id}' not found.",
            )

        row = session.data[0]
        clicked_at = (
            request.clicked_at.astimezone(timezone.utc).isoformat()
            if request.clicked_at
            else datetime.now(timezone.utc).isoformat()
        )
        click_id = str(uuid.uuid4())

        supabase.table("cta_clicks").insert({
            "id": click_id,
            "property_id": row["property_id"],
            "account_id": row["account_id"],
            "visitor_session_id": row["id"],
            "cta_type": request.cta_type,
            "cta_location": request.cta_location,
            "destination_url": request.destination_url,
            "clicked_at": clicked_at,
            "redirected": True,
        }).execute()

        logger.info(
            "[cta-clicks] %s recorded — session=%s property=%s click_id=%s",
            request.cta_type,
            request.visitor_session_id,
            row["property_id"],
            click_id,
        )
        return {"status": "accepted", "cta_click_id": click_id}

    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "[cta-clicks] write failed — session=%s cta_type=%s",
            request.visitor_session_id,
            request.cta_type,
        )
        return {"status": "accepted", "cta_click_id": None}


ALLOWED_CONSENT_STATUSES = {
    "unknown", "marketing_opt_in", "transactional_only", "opted_out"
}

@app.post("/public/leads")
async def public_lead_capture(request: PublicLeadRequest):
    """
    Capture a first-party lead from a property page.

    Derives property_id and account_id from the session record.
    Never trusts caller-supplied context.
    Verifies cta_click_id belongs to the same session/property before linking.
    Creates attribution_links row on successful CTA linkage.
    Returns 201 on success, 202 on fail-open.
    """
    try:
        from core.supabase_store import get_supabase
        import uuid

        if request.consent_status and request.consent_status not in ALLOWED_CONSENT_STATUSES:
            raise HTTPException(
                status_code=400,
                detail=f"consent_status '{request.consent_status}' is not allowed.",
            )

        supabase = get_supabase()

        session = (
            supabase
            .table("visitor_sessions")
            .select("id, property_id, account_id")
            .eq("id", request.visitor_session_id)
            .limit(1)
            .execute()
        )

        if not session.data:
            raise HTTPException(
                status_code=404,
                detail=f"visitor_session_id '{request.visitor_session_id}' not found.",
            )

        row = session.data[0]
        lead_id = str(uuid.uuid4())

        supabase.table("leads").insert({
            "id": lead_id,
            "account_id": row["account_id"],
            "property_id": row["property_id"],
            "visitor_session_id": row["id"],
            "primary_email": str(request.primary_email),
            "full_name": request.full_name,
            "phone": request.phone,
            "lead_source": "property_page",
            "consent_status": request.consent_status or "unknown",
            "requested_dates": request.requested_dates,
            "party_size": request.party_size,
            "trip_intent": request.trip_intent,
        }).execute()

        if request.cta_click_id:
            click_check = (
                supabase
                .table("cta_clicks")
                .select("id")
                .eq("id", request.cta_click_id)
                .eq("visitor_session_id", row["id"])
                .eq("property_id", row["property_id"])
                .limit(1)
                .execute()
            )
            if click_check.data:
                supabase.table("cta_clicks").update(
                    {"lead_id": lead_id}
                ).eq("id", request.cta_click_id).execute()

                supabase.table("attribution_links").insert({
                    "property_id": row["property_id"],
                    "account_id": row["account_id"],
                    "visitor_session_id": row["id"],
                    "lead_id": lead_id,
                    "cta_click_id": request.cta_click_id,
                    "evidence_type": "cta_match",
                    "confidence_level": "medium",
                }).execute()
            else:
                logger.warning(
                    "[leads] cta_click_id=%s does not belong to session=%s — skipping linkage",
                    request.cta_click_id,
                    request.visitor_session_id,
                )

        supabase.table("visitor_sessions").update(
            {"lead_id": lead_id}
        ).eq("id", row["id"]).execute()

        logger.info(
            "[leads] captured — session=%s property=%s lead_id=%s",
            request.visitor_session_id,
            row["property_id"],
            lead_id,
        )
        return JSONResponse(
            status_code=201,
            content={"status": "captured", "lead_id": lead_id},
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "[leads] capture failed — session=%s email=%s",
            request.visitor_session_id,
            str(request.primary_email),
        )
        return JSONResponse(
            status_code=202,
            content={"status": "accepted", "lead_id": None},
        )


@app.post("/portal/bookings/report", status_code=201)
async def report_booking(
    request: BookingReportRequest,
    x_portal_secret: str | None = Header(default=None),
):
    """
    Manually report a direct booking for a property.

    Writes to booking_reports (human input layer) and syncs a
    normalized row into bookings (system layer).
    Optionally creates an attribution_links row if session or lead provided.

    account_id is always derived from the property record — never trusted from caller.
    Property must belong to an account the caller has access to.
    """
    try:
        if not PORTAL_API_SECRET or x_portal_secret != PORTAL_API_SECRET:
            raise HTTPException(status_code=401, detail="Unauthorized.")

        from core.supabase_store import get_supabase
        from core.property_context import resolve_property_context
        import uuid

        if request.booking_value <= 0:
            raise HTTPException(
                status_code=400,
                detail="booking_value must be greater than 0.",
            )

        if request.check_out_date <= request.check_in_date:
            raise HTTPException(
                status_code=400,
                detail="check_out_date must be after check_in_date.",
            )

        # Resolve and validate property — derives account_id from DB
        ctx = resolve_property_context(property_id=request.property_id)

        supabase = get_supabase()

        # Step 1 — Write to booking_reports (human input layer)
        report_id = str(uuid.uuid4())
        supabase.table("booking_reports").insert({
            "id": report_id,
            "account_id": ctx.account_id,
            "property_id": ctx.property_id,
            "report_source": "operator_entry",
            "booking_date": str(request.booking_date) if request.booking_date else None,
            "check_in_date": request.check_in_date.isoformat(),
            "check_out_date": request.check_out_date.isoformat(),
            "booking_value": request.booking_value,
            "currency": request.currency or "USD",
            "notes": request.notes,
            "reported_at": datetime.now(timezone.utc).isoformat(),
        }).execute()

        # Step 2 — Sync normalized row into bookings (system layer)
        booking_id = str(uuid.uuid4())
        supabase.table("bookings").insert({
            "id": booking_id,
            "account_id": ctx.account_id,
            "property_id": ctx.property_id,
            "source_type": "reported",
            "source_system": "pmc_manual",
            "booking_status": "confirmed",
            "booking_value": request.booking_value,
            "currency": request.currency or "USD",
            "check_in_date": request.check_in_date.isoformat(),
            "check_out_date": request.check_out_date.isoformat(),
            "booking_created_at": str(request.booking_date) if request.booking_date else None,
            "attribution_state": "reported",
            "source_confidence": "low",
        }).execute()

        # Step 3 — Optional attribution linkage
        if request.visitor_session_id or request.lead_id:
            supabase.table("attribution_links").insert({
                "property_id": ctx.property_id,
                "account_id": ctx.account_id,
                "visitor_session_id": request.visitor_session_id,
                "lead_id": request.lead_id,
                "booking_id": booking_id,
                "evidence_type": "reported_match",
                "confidence_level": "low",
            }).execute()

        logger.info(
            "[bookings] reported — property=%s booking_id=%s value=%s",
            ctx.property_id,
            booking_id,
            request.booking_value,
        )
        return {
            "status": "reported",
            "booking_id": booking_id,
            "booking_report_id": report_id,
            "account_id": ctx.account_id,
            "property_id": ctx.property_id,
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "[bookings] report failed — property=%s value=%s",
            request.property_id,
            request.booking_value,
        )
        raise HTTPException(status_code=500, detail="Booking report failed.")


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
