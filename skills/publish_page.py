"""
Skill: publish_page — render and deploy a property landing page.

Owns THE slug: reads properties.slug, generates once if null (the single
generator), writes to properties.slug. Never regenerates.

Renders via render_page, uploads HTML via skills_r2_upload to staging
bucket under pages/{slug}/index.html.

Ruling 3-R halt conditions:
  - Slug collision with a DIFFERENT property → failed + urgent hitl
  - Zero photos rendered → failed + urgent hitl
  - Unresolvable > 0 → failed + urgent hitl

Usage:
    from skills.publish_page import publish_page
    result = publish_page("82cb9d7e-...")
"""

import json
import logging
import re

from skills.contract import (
    SkillResult, get_substrate, require_env,
    record_run, record_step, complete_step, complete_run, emit_cost,
    skills_r2_upload,
)

logger = logging.getLogger(__name__)


def _make_slug(name: str) -> str:
    """Generate a URL-safe slug from a property name.

    The single slug generator. Kept from main.py:1304 (_make_slug, 50 chars)
    because it is the simplest and matches the intake path. The other two
    (agent1:_generate_slug at 60 chars, agent5:_fallback_slug at 50 chars)
    are redundant — same regex, different lengths.

    Max 50 chars for subdomain safety.
    """
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")
    return slug[:50]


def publish_page(
    property_id: str,
    *,
    force: bool = False,
) -> SkillResult:
    """Render and deploy a property landing page to staging R2.

    Returns SkillResult.ok({slug, page_url, html_size, ...})
    """
    try:
        sb = get_substrate()
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    try:
        require_env("R2_ENDPOINT_URL", "R2 for page upload")
        require_env("R2_ACCESS_KEY_ID", "R2 for page upload")
        require_env("R2_SECRET_ACCESS_KEY", "R2 for page upload")
        require_env("STAGING_R2_BUCKET", "R2 bucket for page upload")
        require_env("STAGING_R2_PUBLIC_URL", "R2 public URL for page")
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    # ── Load property ────────────────────────────────────────────────────
    prop_resp = sb.table("properties").select("*").eq("id", property_id).limit(1).execute()
    if not prop_resp.data:
        return SkillResult.failed(
            reason=f"Property {property_id} not found",
            attempted=0, succeeded=0, failed_count=0,
        )
    prop = prop_resp.data[0]

    # ── Resolve slug ─────────────────────────────────────────────────────
    slug = prop.get("slug")
    if not slug:
        name = prop.get("name") or "property"
        slug = _make_slug(name)
        sb.table("properties").update({"slug": slug}).eq("id", property_id).execute()
        logger.info("[publish_page] Generated slug '%s' from name '%s'", slug, name)

    # ── Slug collision check (Ruling 3-R) ────────────────────────────────
    collision = sb.table("properties").select("id").eq("slug", slug).neq("id", property_id).execute()
    if collision.data:
        from skills.contract import escalate_halt
        escalate_halt(
            sb, property_id,
            queue_type="publish_halt",
            reason_code="slug_collision",
            title=f"Slug '{slug}' collides with property {collision.data[0]['id'][:12]}",
            detail=f"Cannot publish — slug '{slug}' is already owned by another property.",
            property_name=prop.get("name"),
            extra_payload={"slug": slug, "colliding_property": collision.data[0]["id"]},
        )
        return SkillResult.failed(
            reason=f"Slug collision: '{slug}' owned by {collision.data[0]['id'][:12]}",
            attempted=1, succeeded=0, failed_count=1,
            error_class="halt", human_required=True,
        )

    # ── Check existing publication — noop if not forcing ─────────────────
    if not force:
        existing = sb.table("landing_pages").select("page_id", count="exact").eq(
            "property_id", property_id
        ).limit(0).execute()
        if existing.count > 0:
            return SkillResult.noop(
                f"Page already published for this property. Use force=True to republish.",
                {"existing_pages": existing.count},
            )

    # ── Record run ───────────────────────────────────────────────────────
    run_id = record_run(sb, property_id, "onboard")
    step_id = record_step(sb, run_id, "publish_page")

    # ── Render the page ──────────────────────────────────────────────────
    from skills.render_page import render_page
    render_result = render_page(property_id)

    if not render_result.is_ok:
        complete_step(sb, step_id, status="failed", error_message=f"render_page failed: {render_result.reason}")
        complete_run(sb, run_id, status="failed")
        return SkillResult.failed(
            reason=f"render_page failed: {render_result.reason}",
            attempted=1, succeeded=0, failed_count=1,
        )

    html = render_result.data["html"]
    photo_count = render_result.data["photo_count"]
    unresolvable = render_result.data["unresolvable"]
    gallery_count = render_result.data["gallery_count"]

    # ── Ruling 3-R halt checks ───────────────────────────────────────────
    if photo_count == 0:
        from skills.contract import escalate_halt
        escalate_halt(sb, property_id,
            queue_type="publish_halt", reason_code="zero_photos",
            title="Cannot publish — property has 0 photos",
            detail="The property has no photographs. Cannot build a landing page.",
            property_name=prop.get("name"), run_id=run_id, step_name="publish_page")
        complete_step(sb, step_id, status="failed", error_message="Zero photos")
        complete_run(sb, run_id, status="failed")
        return SkillResult.failed(
            reason="Zero photos — cannot publish",
            attempted=1, succeeded=0, failed_count=1,
            error_class="halt", human_required=True,
        )

    if unresolvable > 0:
        from skills.contract import escalate_halt
        escalate_halt(sb, property_id,
            queue_type="publish_halt", reason_code="unresolvable_photos",
            title=f"{unresolvable} photos could not be resolved",
            detail=f"{unresolvable} photographs have no rendition URL and cannot be displayed.",
            property_name=prop.get("name"), run_id=run_id, step_name="publish_page",
            extra_payload={"unresolvable": unresolvable})
        complete_step(sb, step_id, status="failed", error_message=f"{unresolvable} unresolvable photos")
        complete_run(sb, run_id, status="failed")
        return SkillResult.failed(
            reason=f"{unresolvable} unresolvable photos",
            attempted=1, succeeded=0, failed_count=1,
            error_class="halt", human_required=True,
        )

    # ── Upload to R2 ───────────────────────────────────────────────────
    # PUBLISH_R2_BUCKET overrides STAGING_R2_BUCKET for production deploy.
    # Production Worker expects key = "{slug}/index.html" (no prefix).
    # Staging uses "pages/{slug}/index.html" for isolation.
    import os as _os
    publish_bucket = _os.environ.get("PUBLISH_R2_BUCKET", "")
    if publish_bucket:
        # Production path — write directly to the Worker's bucket
        html_bytes = html.encode("utf-8")
        key = f"{slug}/index.html"
        from skills.contract import _get_skills_r2_client
        client = _get_skills_r2_client()
        client.put_object(Bucket=publish_bucket, Key=key, Body=html_bytes,
                          ContentType="text/html; charset=utf-8")
        page_url = f"https://{slug}.upliftstays.com"
        logger.info("[publish_page] Production upload %d bytes → %s (bucket=%s)",
                    len(html_bytes), page_url, publish_bucket)
    else:
        # Staging path — skills_r2_upload with pages/ prefix
        html_bytes = html.encode("utf-8")
        key = f"pages/{slug}/index.html"
        page_url = skills_r2_upload(key, html_bytes, "text/html; charset=utf-8")

    logger.info("[publish_page] Uploaded %d bytes → %s", len(html_bytes), page_url)

    # ── Write landing_pages ──────────────────────────────────────────────
    sb.table("landing_pages").upsert({
        "property_id": property_id,
        "slug": slug,
        "page_url": page_url,
        "deploy_mode": "subdomain",
        "status": "deployed",
        "schema_generated": True,
        "last_built_at": "now()",
    }, on_conflict="property_id").execute()

    # ── Complete ─────────────────────────────────────────────────────────
    complete_step(sb, step_id, status="complete", metadata={
        "slug": slug,
        "page_url": page_url,
        "html_size": len(html_bytes),
        "photo_count": photo_count,
        "gallery_count": gallery_count,
        "published_to": "staging",
    })
    complete_run(sb, run_id, status="complete")

    return SkillResult.ok({
        "slug": slug,
        "page_url": page_url,
        "html_size": len(html_bytes),
        "photo_count": photo_count,
        "gallery_count": gallery_count,
        "published_to": "staging",
        "run_id": run_id,
    })
