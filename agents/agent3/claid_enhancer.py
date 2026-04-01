"""
TS-07 — Photo Enhancement Pipeline
Tool: Claid.ai

Enhances all property photos through Claid.ai's real estate
enhancement pipeline. 100-photo ceiling per property.
One-time cost at intake: ~$4.00/property.

GOVERNANCE ENFORCEMENT — THIS IS NON-NEGOTIABLE:
The permitted operations whitelist is enforced at the API call level.
Prohibited operations raise a hard error and are never submitted to Claid.ai.
This protects against regulatory violations (California AB 723, FTC,
NAR standards) and client ToS liability.

PERMITTED: upscaling, noise reduction, HDR, white balance,
           color grading, sharpness, panoramic stitching
PROHIBITED: background removal, generative fill, object add/remove,
            virtual staging — any operation that alters physical reality
"""

import os
import logging
import asyncio
import httpx
from typing import Optional

logger = logging.getLogger(__name__)

CLAID_API_BASE  = "https://api.claid.ai/v1-beta1"
CLAID_API_KEY   = os.environ.get("CLAID_API_KEY", "")

# Governance: the ONLY operations Agent 3 may request from Claid.ai
PERMITTED_OPERATIONS = frozenset({
    "upscale",
    "noise_reduction",
    "hdr",
    "white_balance",
    "color_grading",
    "sharpness",
    "panoramic_stitch",
    "auto_enhance",           # Claid's combined enhancement (safe — no generative elements)
    "light_correction",
    "exposure_correction",
})

# Governance: operations that are NEVER permitted regardless of request source
PROHIBITED_OPERATIONS = frozenset({
    "background_removal",
    "background_replacement",
    "inpainting",
    "generative_fill",
    "object_removal",
    "object_addition",
    "virtual_staging",
    "scene_generation",
    "ai_outpainting",
    "sky_replacement",
    "remove_objects",
})

# Standard enhancement preset applied to all property photos
STANDARD_ENHANCEMENT_PRESET = {
    "operations": [
        {"type": "upscale", "scale": 2},
        {"type": "light_correction"},
        {"type": "noise_reduction", "strength": "auto"},
        {"type": "color_grading", "preset": "real_estate"},
        {"type": "sharpness", "amount": 0.3},
    ],
    "output": {
        "format": "jpeg",
        "quality": 92,
    }
}

# Maximum photos per property (cost ceiling)
PHOTO_CEILING = 100
# Claid.ai batch concurrency per property
BATCH_CONCURRENCY = 10


def validate_operations(operations: list[dict]) -> None:
    """
    Hard governance check — raises ValueError if any prohibited
    operation is in the request. Called before EVERY Claid.ai API call.

    This is the engineering enforcement layer for the TS-07 whitelist.
    It must not be bypassed under any circumstances.
    """
    for op in operations:
        op_type = op.get("type", "").lower()
        if op_type in PROHIBITED_OPERATIONS:
            raise ValueError(
                f"GOVERNANCE VIOLATION: Operation '{op_type}' is prohibited. "
                f"Agent 3 may only invoke permitted operations: {sorted(PERMITTED_OPERATIONS)}. "
                f"Prohibited operations alter the physical reality of the property and violate "
                f"California AB 723, FTC deceptive advertising standards, and NAR guidelines."
            )
        if op_type not in PERMITTED_OPERATIONS:
            raise ValueError(
                f"GOVERNANCE VIOLATION: Operation '{op_type}' is not on the permitted "
                f"whitelist. Only these operations are allowed: {sorted(PERMITTED_OPERATIONS)}"
            )


async def enhance_photo_async(
    session: httpx.AsyncClient,
    original_url: str,
    photo_index: int,
) -> Optional[bytes]:
    """
    Enhance a single photo via Claid.ai async API.
    Returns enhanced image bytes or None on failure.
    """
    # Governance check — always runs before API call
    validate_operations(STANDARD_ENHANCEMENT_PRESET["operations"])

    payload = {
        "input": original_url,
        "operations": STANDARD_ENHANCEMENT_PRESET["operations"],
        "output": STANDARD_ENHANCEMENT_PRESET["output"],
    }

    try:
        resp = await session.post(
            f"{CLAID_API_BASE}/image/edit",
            headers={
                "Authorization": f"Bearer {CLAID_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()

        data = resp.json()
        output_url = (
            data.get("output", {}).get("tmp_url")
            or data.get("data", {}).get("output", {}).get("tmp_url")
        )
        if not output_url:
            logger.warning(f"Claid.ai returned no output URL for photo {photo_index}")
            return None

        # Download the enhanced image bytes
        dl_resp = await session.get(output_url, timeout=30.0)
        dl_resp.raise_for_status()
        return dl_resp.content

    except Exception as exc:
        logger.error(f"Claid.ai enhancement failed for photo {photo_index}: {exc}")
        return None


async def enhance_photo_batch(
    photo_urls: list[str],
    property_id: str,
) -> list[tuple[str, Optional[bytes]]]:
    """
    Enhance a batch of photos concurrently.
    Returns list of (original_url, enhanced_bytes_or_None) tuples.

    Caps at PHOTO_CEILING (100) photos per property.
    Runs BATCH_CONCURRENCY requests in parallel to balance
    speed and Claid.ai rate limits.
    """
    capped_urls = photo_urls[:PHOTO_CEILING]
    if len(photo_urls) > PHOTO_CEILING:
        logger.info(
            f"[TS-07] Property {property_id}: capped at {PHOTO_CEILING} photos "
            f"(had {len(photo_urls)}, dropped {len(photo_urls) - PHOTO_CEILING})"
        )

    results: list[tuple[str, Optional[bytes]]] = []

    # Process in batches of BATCH_CONCURRENCY
    async with httpx.AsyncClient() as session:
        for batch_start in range(0, len(capped_urls), BATCH_CONCURRENCY):
            batch = capped_urls[batch_start:batch_start + BATCH_CONCURRENCY]
            tasks = [
                enhance_photo_async(session, url, batch_start + i)
                for i, url in enumerate(batch)
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for url, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Enhancement task raised exception for {url}: {result}")
                    results.append((url, None))
                else:
                    results.append((url, result))

    success_count = sum(1 for _, b in results if b is not None)
    logger.info(
        f"[TS-07] Claid.ai enhancement complete for property {property_id}: "
        f"{success_count}/{len(capped_urls)} photos enhanced successfully"
    )
    return results


def enhance_photo_batch_sync(
    photo_urls: list[str],
    property_id: str,
) -> list[tuple[str, Optional[bytes]]]:
    """Synchronous wrapper for use in non-async LangGraph nodes."""
    return asyncio.run(enhance_photo_batch(photo_urls, property_id))
