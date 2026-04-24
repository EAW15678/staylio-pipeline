-- Migration: property_image_curations
-- Stores the one-time LLM vision curation result for each property's image set.
-- Keyed on (property_id, image_set_hash) — a new hash triggers a new row
-- (image set changed) while an existing hash is returned from cache immediately.
--
-- Apply via: Supabase Dashboard → SQL Editor, or supabase db push

CREATE TABLE IF NOT EXISTS property_image_curations (
    id                       UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    property_id              TEXT        NOT NULL,
    image_set_hash           TEXT        NOT NULL,
    curation_model           TEXT        NOT NULL DEFAULT 'claude-sonnet-4-6',
    status                   TEXT        NOT NULL DEFAULT 'pending',
    -- 'pending' | 'complete' | 'failed'
    asset_count              INT,
    per_image_results        JSONB,
    -- Array of per-image curation objects (see llm_curator.py for schema)
    property_recommendations JSONB,
    -- {page_hero, photo_tour_sections, category_order, vibe, merchandising_strategy}
    created_at               TIMESTAMPTZ DEFAULT NOW(),
    completed_at             TIMESTAMPTZ,

    UNIQUE (property_id, image_set_hash)
);

CREATE INDEX IF NOT EXISTS idx_pic_property_id
    ON property_image_curations (property_id);

CREATE INDEX IF NOT EXISTS idx_pic_status
    ON property_image_curations (status);

-- RLS: enable but allow service role full access (same pattern as other tables)
ALTER TABLE property_image_curations ENABLE ROW LEVEL SECURITY;

-- =============================================================================
-- Column notes:
--
--   per_image_results — jsonb array, one object per retained photo:
--     asset_id              text     (R2 original URL — stable unique key)
--     llm_category          text     (corrected room category)
--     room_subtype          text?    (e.g. 'island_kitchen', 'soaking_tub')
--     duplicate_group       text?    (shared string for same-room images)
--     is_primary_in_group   bool     (true = best angle in this group)
--     rank_within_category  int      (1 = best in its category)
--     role                  text     (hero|supporting|gallery_only|exclude)
--     reason                text?    (non-null only when role = exclude)
--     alt                   text     (descriptive alt text for SEO/a11y)
--
--   property_recommendations — jsonb object:
--     page_hero             text     (asset_id of selected page hero)
--     photo_tour_sections   jsonb[]  ({section, hero, supporting:[...]})
--     category_order        text[]   (ordered section names)
--     vibe                  text     (one-sentence property vibe)
--     merchandising_strategy jsonb   ({prioritize:[], deprioritize:[]})
-- =============================================================================
