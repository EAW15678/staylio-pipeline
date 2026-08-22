"""
OBSERVE-D: Generate and cache depth maps for canonical photographs.

Depth maps are deterministic derivatives of the enhanced rendition.
They are computed once on Modal (GPU) and stored as renditions in R2.
Re-running is a noop when the source hasn't changed.

This skill is called after observe completes. Depth estimation failing
does NOT fail the overall workflow — it logs and continues.
"""

import logging
from skills.contract import SkillResult, get_substrate

logger = logging.getLogger(__name__)


def generate_depth_maps(
    property_id: str,
    *,
    force: bool = False,
) -> SkillResult:
    """Generate depth maps for all canonical photographs of a property.

    For each canonical photograph:
    1. Find the enhanced rendition (or original if no enhanced).
    2. Check if a depth_map rendition already exists for the same source.
    3. If missing or stale, call the Modal depth service.
    4. Store the result as a rendition with kind='depth_map'.

    Args:
        property_id: The property UUID.
        force: If True, regenerate even if depth maps exist.

    Returns:
        SkillResult with counts of generated, skipped, failed.
    """
    sb = get_substrate()

    # Load canonical photographs with their content hashes
    photos_resp = sb.table("photographs").select(
        "photo_id, content_hash, image_width, image_height"
    ).eq("property_id", property_id).eq("is_canonical", True).execute()
    photos = photos_resp.data or []

    if not photos:
        return SkillResult.noop(
            f"No canonical photographs for property {property_id}",
            {"property_id": property_id},
        )

    generated = 0
    skipped = 0
    failed = 0
    failures = []

    for photo in photos:
        pid = photo["photo_id"]
        content_hash = photo["content_hash"]

        try:
            # Get renditions for this photo
            rends_resp = sb.table("renditions").select(
                "rendition_id, kind, storage_url, source_content_hash"
            ).eq("photo_id", pid).execute()
            rends = {r["kind"]: r for r in (rends_resp.data or [])}

            # Find source: prefer enhanced, fall back to original
            source_rend = rends.get("enhanced") or rends.get("original")
            if not source_rend:
                logger.debug("[depth] %s: no source rendition, skipping", pid)
                skipped += 1
                continue

            source_url = source_rend["storage_url"]

            # Check if depth map exists and is current
            existing_depth = rends.get("depth_map")
            if existing_depth and not force:
                # Cache identity: does the depth map match the current source?
                if existing_depth.get("source_content_hash") == content_hash:
                    skipped += 1
                    continue
                else:
                    logger.info(
                        "[depth] %s: source changed (hash %s → %s), regenerating",
                        pid,
                        existing_depth.get("source_content_hash", "null"),
                        content_hash,
                    )

            # Call Modal depth service
            r2_key = f"{property_id}/depth/{pid}.png"
            result = _call_modal_depth(source_url, r2_key)

            # Upsert rendition
            sb.table("renditions").upsert(
                {
                    "photo_id": pid,
                    "kind": "depth_map",
                    "storage_url": result["depth_map_url"],
                    "format": "png",
                    "width": result["width"],
                    "height": result["height"],
                    "byte_size": result["byte_size"],
                    "source_content_hash": content_hash,
                },
                on_conflict="photo_id,kind,format",
            ).execute()

            generated += 1
            logger.info(
                "[depth] %s: generated in %.1fs (%d bytes)",
                pid, result["elapsed_seconds"], result["byte_size"],
            )

        except Exception as exc:
            error_msg = str(exc)[:300]
            logger.warning("[depth] %s: failed — %s", pid, error_msg)
            failures.append({"photo_id": pid, "error": error_msg})
            failed += 1

    total = generated + skipped + failed
    logger.info(
        "[depth] property=%s total=%d generated=%d skipped=%d failed=%d",
        property_id, total, generated, skipped, failed,
    )

    if generated == 0 and failed == 0:
        return SkillResult.noop(
            f"All {skipped} depth maps current",
            {"skipped": skipped},
        )

    return SkillResult.ok({
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
        "failures": failures[:5],  # cap for readability
    })


def _call_modal_depth(image_url: str, r2_key: str) -> dict:
    """Call the Modal depth-service function.

    Uses the Modal Python SDK to call the deployed function directly.
    """
    import modal

    depth_fn = modal.Function.from_name("depth-service", "generate_depth_map")
    result = depth_fn.remote(image_url=image_url, r2_key=r2_key)
    return result
