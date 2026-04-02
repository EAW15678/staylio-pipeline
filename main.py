"""
Staylio.ai — Pipeline Entry Point
Railway deployment entry point for the LangGraph agent pipeline.
Exposes a FastAPI HTTP interface so Railway can detect and serve it.
"""

import logging
import os
from contextlib import asynccontextmanager

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
    client_id: str
    listing_urls: list[str] = []
    intake_data: dict = {}


class PipelineStatusRequest(BaseModel):
    property_id: str


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Health check endpoint — Railway uses this to verify the service is running."""
    return {"status": "ok", "service": "staylio-pipeline", "version": "2.0.0"}


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "service": "Staylio.ai Pipeline",
        "version": "2.0.0",
        "status": "running",
        "endpoints": ["/health", "/run", "/status/{property_id}"],
    }


@app.post("/run")
async def run_pipeline(request: PipelineRunRequest, background_tasks: BackgroundTasks):
    """
    Trigger the full 7-agent pipeline for a property.
    Runs asynchronously in the background — returns immediately with job ID.
    """
    try:
        from pipeline.graph import run_intake_pipeline as execute_pipeline

        background_tasks.add_task(
            execute_pipeline,
            property_id=request.property_id,
            client_id=request.client_id,
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


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
