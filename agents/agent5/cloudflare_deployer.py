"""
TS-12 — Cloudflare Pages Deployer
Tool: Cloudflare Pages API

Deploys built HTML landing pages to Cloudflare Pages.

Two deployment modes:
  MODE 1 — Staylio subdomain (default):
    {slug}.staylio.ai
    Wildcard SSL covers all properties automatically.
    No PMC DNS involvement required.

  MODE 2 — CNAME custom domain (Portfolio tier):
    {custom_subdomain}.pmcdomain.com
    PMC points their subdomain to Cloudflare Pages via CNAME.
    Cloudflare auto-provisions SSL for the custom domain.
    Five-minute PMC DNS change, no ongoing coordination.

Both modes use identical Cloudflare Pages infrastructure.
CNAME mode is a vanity layer over the same hosting.

Cloudflare Pages free tier: unlimited sites, unlimited bandwidth,
500 builds/month. Zero per-property hosting cost at any scale.

FIX NOTES (v6 → v7):
  Cloudflare Pages Direct Upload API requires a ZIP file, not tar.gz.
  Changed _build_deployment_bundle to produce a zip using zipfile module.
  Form field name must be 'file' with filename ending in .zip.
"""

import hashlib
import logging
import os
import zipfile
import io
from typing import Optional

import httpx

from agents.agent5.models import DeployMode

logger = logging.getLogger(__name__)

CLOUDFLARE_API_KEY    = os.environ.get("CLOUDFLARE_API_KEY", "")
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
CLOUDFLARE_PAGES_BASE = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/pages/projects"

# Staylio Pages project name (one project, many deployments)
PAGES_PROJECT_NAME = "staylio-properties"

# Public base URL
BOOKED_BASE_DOMAIN = "staylio.ai"


def deploy_property_page(
    property_id: str,
    slug: str,
    html_content: str,
    deploy_mode: DeployMode = DeployMode.STAYLIO_SUBDOMAIN,
    custom_domain: Optional[str] = None,
) -> tuple[bool, str, Optional[str]]:
    """
    Deploy a property landing page to Cloudflare Pages.

    Args:
        property_id:   Property UUID
        slug:          URL-safe property slug
        html_content:  Complete HTML string for the landing page
        deploy_mode:   STAYLIO_SUBDOMAIN or CNAME_CUSTOM
        custom_domain: Required for CNAME_CUSTOM mode

    Returns:
        (success: bool, page_url: str, deployment_id: Optional[str])
    """
    if not CLOUDFLARE_API_KEY or not CLOUDFLARE_ACCOUNT_ID:
        # Development mode — simulate deployment
        logger.warning("[TS-12] Cloudflare credentials not set — simulating deployment")
        page_url = _build_page_url(slug, deploy_mode, custom_domain)
        return True, page_url, f"sim-{hashlib.md5(property_id.encode()).hexdigest()[:8]}"

    page_url = _build_page_url(slug, deploy_mode, custom_domain)

    try:
        # Create deployment bundle (zip archive with index.html)
        bundle = _build_deployment_bundle(slug, html_content)

        # Upload to Cloudflare Pages Direct Upload API
        deployment_id = _upload_to_pages(slug, bundle)

        if not deployment_id:
            return False, page_url, None

        # Configure custom domain if CNAME mode
        if deploy_mode == DeployMode.CNAME_CUSTOM and custom_domain:
            _configure_custom_domain(custom_domain)

        logger.info(f"[TS-12] Deployed property {property_id} → {page_url} (id={deployment_id})")
        return True, page_url, deployment_id

    except Exception as exc:
        logger.error(f"[TS-12] Deployment failed for property {property_id}: {exc}")
        return False, page_url, None


def rebuild_page(
    property_id: str,
    slug: str,
    html_content: str,
    deploy_mode: DeployMode,
    custom_domain: Optional[str] = None,
) -> tuple[bool, str, Optional[str]]:
    """
    Rebuild and redeploy an existing property page.
    Called when content is refreshed, winner variant auto-deployed,
    or owner updates their page content.
    """
    logger.info(f"[TS-12] Rebuilding page for property {property_id}")
    return deploy_property_page(
        property_id=property_id,
        slug=slug,
        html_content=html_content,
        deploy_mode=deploy_mode,
        custom_domain=custom_domain,
    )


# ── Internal helpers ──────────────────────────────────────────────────────

def _build_page_url(
    slug: str,
    deploy_mode: DeployMode,
    custom_domain: Optional[str],
) -> str:
    """Construct the public URL for a property page."""
    if deploy_mode == DeployMode.CNAME_CUSTOM and custom_domain:
        return f"https://{custom_domain}"
    return f"https://{slug}.{BOOKED_BASE_DOMAIN}"


def _build_deployment_bundle(slug: str, html_content: str) -> bytes:
    """
    Create a zip archive for Cloudflare Pages Direct Upload API.
    Cloudflare Pages requires a ZIP file — tar.gz is not accepted.
    Contains: index.html at the root for the property slug route.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("index.html", html_content.encode("utf-8"))
    return buf.getvalue()


def _upload_to_pages(slug: str, bundle: bytes) -> Optional[str]:
    """
    Upload a deployment bundle to Cloudflare Pages via Direct Upload API.
    Sends a ZIP file as multipart/form-data.
    Returns the deployment ID or None on failure.
    """
    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{CLOUDFLARE_PAGES_BASE}/{PAGES_PROJECT_NAME}/deployments",
                headers={"Authorization": f"Bearer {CLOUDFLARE_API_KEY}"},
                files={"file": (f"{slug}.zip", bundle, "application/zip")},
            )
            resp.raise_for_status()
            return resp.json().get("result", {}).get("id")
    except Exception as exc:
        logger.error(f"[TS-12] Cloudflare Pages upload failed: {exc}")
        return None


def _configure_custom_domain(custom_domain: str) -> None:
    """
    Register a custom domain with the Cloudflare Pages project.
    Cloudflare auto-provisions SSL for the domain.
    PMC must have pointed their CNAME before this is called.
    """
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{CLOUDFLARE_PAGES_BASE}/{PAGES_PROJECT_NAME}/domains",
                headers={
                    "Authorization": f"Bearer {CLOUDFLARE_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={"name": custom_domain},
            )
            resp.raise_for_status()
            logger.info(f"[TS-12] Custom domain registered: {custom_domain}")
    except Exception as exc:
        logger.warning(f"[TS-12] Custom domain registration failed for {custom_domain}: {exc}")
