"""
Substrate-native page builder — portable HTML/CSS/JS rendering functions.

Created 2026-08-19 (PAGE-2). Depends on nothing under agents/.
This package replaces the embedded vocabulary from agents/agent5/page_builder.py
with Erick's real seven-section taxonomy (Exterior, Pool, Living Areas, Kitchen,
Bedrooms, Bathrooms, Extras) and moves all portable rendering functions into
substrate-owned code.

The four database-reaching fallback functions (_load_media_assets_from_supabase,
_load_curation_from_supabase, _get_hero_video_url, _get_review_audio_urls) are
NOT included here. They remain in agents/agent5/page_builder.py, serving only
the old agent system until its retirement.

The orchestrator (step 3) is also not included — no render_page replacement,
no amenity photo-proof, no native video lookup, no missing-data handling.
"""

from core.page_builder.schema_markup import build_schema_from_inputs, generate_schema_jsonld
from core.page_builder.ab_testing import generate_growthbook_snippet
