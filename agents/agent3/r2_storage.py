"""
TS-19 — Cloudflare R2 Asset Storage

Handles upload and URL generation for all media assets.
R2 is S3-compatible — uses boto3 with a custom endpoint.

Four buckets:
  staylio-originals  — pre-enhancement source files (permanent)
  staylio-enhanced   — Claid.ai output (production assets)
  staylio-crops      — social format derivatives from category winners
  staylio-video      — Runway ML + Creatomate rendered video files

Zero egress fees — the defining advantage over AWS S3.
All Cloudflare Pages property pages serve assets directly
from R2 via the same CDN edge network.

Key naming convention: {property_id}/{asset_type}/{filename}
e.g. prop-abc123/enhanced/photo_001.jpg
     prop-abc123/video/vibe_match_9_16.mp4
"""

import os
import hashlib
import logging
import mimetypes
from pathlib import Path
from typing import Optional

import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)

# ── R2 Bucket Names ───────────────────────────────────────────────────────
BUCKET_ORIGINALS = "staylio-originals"
BUCKET_ENHANCED  = "staylio-enhanced"
BUCKET_CROPS     = "staylio-crops"
BUCKET_VIDEO     = "staylio-video"

# Public CDN base URL for R2 assets (configured in Cloudflare dashboard)
R2_PUBLIC_BASE = os.environ.get("R2_PUBLIC_BASE", "https://assets.staylio.ai")
R2_ENHANCED_PUBLIC_URL = os.environ.get("R2_ENHANCED_PUBLIC_URL", R2_PUBLIC_BASE)
R2_VIDEO_PUBLIC_URL = os.environ.get("R2_VIDEO_PUBLIC_URL", R2_PUBLIC_BASE)


def _get_r2_client():
    """
    Create an S3-compatible boto3 client pointing at Cloudflare R2.
    R2 endpoint format: https://{account_id}.r2.cloudflarestorage.com
    """
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def upload_photo_original(
    property_id: str,
    photo_bytes: bytes,
    filename: str,
    content_type: str = "image/jpeg",
) -> str:
    """
    Upload a pre-enhancement original photo to staylio-originals.
    Returns the public CDN URL.
    """
    key = _photo_key(property_id, "original", filename)
    return _upload(BUCKET_ORIGINALS, key, photo_bytes, content_type)


def upload_photo_enhanced(
    property_id: str,
    photo_bytes: bytes,
    filename: str,
    content_type: str = "image/jpeg",
) -> str:
    """
    Upload a Claid.ai enhanced photo to staylio-enhanced.
    Returns the public CDN URL.
    """
    key = _photo_key(property_id, "enhanced", filename)
    return _upload(BUCKET_ENHANCED, key, photo_bytes, content_type)


def upload_social_crop(
    property_id: str,
    crop_bytes: bytes,
    filename: str,
    format_label: str,   # "1_1" | "9_16" | "16_9"
) -> str:
    """
    Upload a social format crop to staylio-crops.
    Returns the public CDN URL.
    """
    key = f"{property_id}/crops/{format_label}/{filename}"
    return _upload(BUCKET_CROPS, key, crop_bytes, "image/jpeg")


def upload_video(
    property_id: str,
    video_bytes: bytes,
    video_type: str,      # e.g. "vibe_match"
    format_label: str,    # "9_16" | "1_1" | "16_9"
    extension: str = "mp4",
) -> str:
    """
    Upload a rendered video file to staylio-video.
    Returns the public CDN URL.
    """
    filename = f"{video_type}_{format_label}.{extension}"
    key = f"{property_id}/video/{filename}"
    return _upload(BUCKET_VIDEO, key, video_bytes, f"video/{extension}")


def download_bytes(bucket: str, key: str) -> Optional[bytes]:
    """Download an asset from R2. Returns bytes or None on failure."""
    try:
        client = _get_r2_client()
        response = client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()
    except Exception as exc:
        logger.error(f"R2 download failed for {bucket}/{key}: {exc}")
        return None


def public_url(bucket: str, key: str) -> str:
    """Construct the public CDN URL for an R2 asset."""
    if bucket == BUCKET_ENHANCED:
        return f"{R2_ENHANCED_PUBLIC_URL}/{key}"
    if bucket == BUCKET_VIDEO:
        return f"{R2_VIDEO_PUBLIC_URL}/{key}"
    return f"{R2_PUBLIC_BASE}/{key}"


# ── Internal helpers ──────────────────────────────────────────────────────

def _upload(
    bucket: str,
    key: str,
    data: bytes,
    content_type: str,
) -> str:
    """Upload bytes to R2 and return the public CDN URL."""
    client = _get_r2_client()
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
        CacheControl="public, max-age=31536000, immutable",
    )
    url = public_url(bucket, key)
    logger.debug(f"R2 upload: {bucket}/{key} → {url}")
    return url


def _photo_key(property_id: str, asset_type: str, filename: str) -> str:
    """
    Build a consistent R2 key for a photo asset.
    Format: {property_id}/{original|enhanced}/{filename}
    """
    return f"{property_id}/{asset_type}/{filename}"


def _stable_filename(url: str, index: int) -> str:
    """
    Generate a stable filename from a URL + index.
    Used when downloading photos from scraped URLs.
    """
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"photo_{index:03d}_{url_hash}.jpg"
