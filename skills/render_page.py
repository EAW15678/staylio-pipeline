"""
Skill: render_page — thin wrapper over core.page_builder.orchestrate.build_page.

Rewritten 2026-08-19 (PAGE-4). The previous version (pre-561dc5f) read the
substrate directly, then translated curated_section back to old GCV names
via a section_to_cat dict at lines 150-158, then passed the result to
agents.agent5.page_builder.build_landing_page_html — which never actually
used the curation data because its _curation_active check required a "status"
key that this file never set.

This version calls build_page() from core.page_builder.orchestrate, which
reads the substrate directly and uses curated_section natively. No translation
dict. No agents/ import. Single data path.

Usage:
    from skills.render_page import render_page
    result = render_page("a1b2c3d4-...")
    if result.is_ok:
        html = result.data["html"]
"""

import logging

from skills.contract import SkillResult, get_substrate

logger = logging.getLogger(__name__)


def render_page(property_id: str) -> SkillResult:
    """Build a landing page HTML from substrate data.

    Returns:
      SkillResult.ok({html, photo_count, gallery_count, ...}) on success
      SkillResult.held(reason, {hold_code}) when data is insufficient
      SkillResult.failed(reason) on error
    """
    try:
        sb = get_substrate()
    except EnvironmentError as e:
        return SkillResult.failed(str(e))

    from core.page_builder.orchestrate import build_page
    result = build_page(sb, property_id)

    if result.get("hold"):
        logger.warning(
            "[render_page] Hold for %s: %s (%s)",
            property_id[:12], result["reason"], result.get("hold_code"),
        )
        return SkillResult.held(
            reason=result["reason"],
            data={"hold_code": result.get("hold_code")},
        )

    if not result["ok"]:
        return SkillResult.failed(reason=result["reason"])

    logger.info(
        "[render_page] Property %s: %d photos, %d sections, gallery=%d, unresolvable=%d",
        property_id[:12],
        result["photo_count"],
        len(result["sections_built"]),
        result["gallery_count"],
        result["unresolvable"],
    )

    return SkillResult.ok({
        "html": result["html"],
        "sections_built": result["sections_built"],
        "gallery_count": result["gallery_count"],
        "photo_count": result["photo_count"],
        "lightbox_present": result["lightbox_present"],
        "unresolvable": result["unresolvable"],
    })
