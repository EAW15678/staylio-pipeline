"""
TS-12 — Cloudflare R2 Deployer
Tool: Cloudflare R2 (via boto3 S3-compatible API)

Deploys built HTML landing pages to Cloudflare R2.
A Cloudflare Worker (staylio-router) intercepts *.staylio.ai traffic,
reads the subdomain, fetches {slug}/index.html from R2, and serves it.

FIX NOTES (v8 → v9):
  Replaced Cloudflare Pages Direct Upload API with R2 upload via boto3.
  Pages does not support wildcard custom domains (documented CF limitation).
  agent.py call signature is unchanged.
"""

import hashlib
import logging
import os
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from agents.agent5.models import DeployMode

logger = logging.getLogger(__name__)

CLOUDFLARE_ACCOUNT_ID  = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID       = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY   = os.environ.get("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME         = os.environ.get("R2_BUCKET_NAME", "staylio-pages")

BOOKED_BASE_DOMAIN = "staylio.ai"


def deploy_property_page(
    property_id: str,
    slug: str,
    html_content: str,
    deploy_mode: DeployMode = DeployMode.STAYLIO_SUBDOMAIN,
    custom_domain: Optional[str] = None,
) -> tuple[bool, str, Optional[str]]:
    page_url = _build_page_url(slug, deploy_mode, custom_domain)

    if not all([CLOUDFLARE_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
        logger.warning("[TS-12] R2 credentials not set — simulating deployment")
        sim_id = f"sim-{hashlib.md5(property_id.encode()).hexdigest()[:8]}"
        return True, page_url, sim_id

    r2_key = f"{slug}/index.html"

    try:
        _upload_to_r2(r2_key, html_content)
        logger.info(
            f"[TS-12] Deployed property {property_id} → {page_url} "
            f"(r2_key={r2_key}, bucket={R2_BUCKET_NAME})"
        )
        return True, page_url, r2_key

    except Exception as exc:
        logger.error(f"[TS-12] R2 deployment failed for property {property_id}: {exc}")
        return False, page_url, None


def rebuild_page(
    property_id: str,
    slug: str,
    html_content: str,
    deploy_mode: DeployMode,
    custom_domain: Optional[str] = None,
) -> tuple[bool, str, Optional[str]]:
    logger.info(f"[TS-12] Rebuilding page for property {property_id}")
    return deploy_property_page(
        property_id=property_id,
        slug=slug,
        html_content=html_content,
        deploy_mode=deploy_mode,
        custom_domain=custom_domain,
    )


def _build_page_url(
    slug: str,
    deploy_mode: DeployMode,
    custom_domain: Optional[str],
) -> str:
    if deploy_mode == DeployMode.CNAME_CUSTOM and custom_domain:
        return f"https://{custom_domain}"
    return f"https://{slug}.{BOOKED_BASE_DOMAIN}"


def _get_r2_client():
    endpoint_url = f"https://{CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",
    )


def _upload_to_r2(r2_key: str, html_content: str) -> None:
    client = _get_r2_client()
    try:
        client.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=r2_key,
            Body=html_content.encode("utf-8"),
            ContentType="text/html; charset=utf-8",
            CacheControl="public, max-age=300, s-maxage=3600",
        )
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "unknown")
        logger.error(f"[TS-12] R2 ClientError uploading {r2_key}: code={error_code} {exc}")
        raise
