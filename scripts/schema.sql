-- =============================================================================
-- Staylio Database Schema
-- Run this in the Supabase SQL Editor to create all required tables.
-- Execute once when setting up a new Staylio project.
-- =============================================================================

-- ── Client accounts ───────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS pmc_clients (
    client_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_name      TEXT NOT NULL,
    contact_name      TEXT,
    contact_email     TEXT,
    tier              TEXT DEFAULT 'base' CHECK (tier IN ('base', 'portfolio')),
    ayrshare_profile_key TEXT,
    pms_type          TEXT,
    pms_api_connected BOOLEAN DEFAULT FALSE,
    custom_domain     TEXT,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS io_clients (
    client_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_name        TEXT NOT NULL,
    owner_email       TEXT,
    ayrshare_profile_key TEXT,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);

-- ── Properties ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS properties (
    property_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_id         UUID NOT NULL,
    client_type       TEXT NOT NULL CHECK (client_type IN ('pmc', 'io')),
    slug              TEXT UNIQUE,
    name              TEXT,
    city              TEXT,
    state             TEXT,
    zip_code          TEXT,
    latitude          NUMERIC(10, 7),
    longitude         NUMERIC(10, 7),
    vibe_profile      TEXT,
    booking_url       TEXT,
    ical_url          TEXT,
    status            TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'paused', 'offboarded')),
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS intake_submissions (
    submission_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    property_id       UUID REFERENCES properties(property_id),
    client_id         UUID NOT NULL,
    raw_data          JSONB,
    submitted_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS property_knowledge_bases (
    property_id       UUID PRIMARY KEY REFERENCES properties(property_id),
    data              JSONB NOT NULL,
    version           INTEGER DEFAULT 1,
    last_scraped_at   TIMESTAMPTZ,
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);

-- ── Pipeline ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS pipeline_status (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    property_id       UUID NOT NULL REFERENCES properties(property_id),
    agent_id          INTEGER NOT NULL CHECK (agent_id BETWEEN 1 AND 7),
    status            TEXT NOT NULL CHECK (status IN ('pending', 'running', 'complete', 'failed')),
    started_at        TIMESTAMPTZ,
    completed_at      TIMESTAMPTZ,
    error_message     TEXT,
    retry_count       INTEGER DEFAULT 0,
    metadata          JSONB,
    UNIQUE (property_id, agent_id)
);

-- ── Content ───────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS content_versions (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    property_id       UUID NOT NULL REFERENCES properties(property_id),
    version           INTEGER NOT NULL DEFAULT 1,
    content_package   JSONB,
    quality_score     NUMERIC(3,2),
    quality_status    TEXT CHECK (quality_status IN ('approved', 'flagged', 'needs_review')),
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS media_assets (
    asset_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    property_id       UUID NOT NULL REFERENCES properties(property_id),
    asset_type        TEXT NOT NULL CHECK (asset_type IN ('original', 'enhanced', 'crop', 'video')),
    r2_url            TEXT NOT NULL,
    subject_category  TEXT,
    composition_score NUMERIC(4,2),
    is_hero           BOOLEAN DEFAULT FALSE,
    vision_tags       JSONB,
    format            TEXT,
    video_type        TEXT,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

-- ── Landing pages ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS landing_pages (
    property_id       UUID PRIMARY KEY REFERENCES properties(property_id),
    slug              TEXT UNIQUE NOT NULL,
    page_url          TEXT NOT NULL,
    deploy_mode       TEXT DEFAULT 'booked_subdomain',
    status            TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'building', 'deployed', 'failed', 'rebuilding')),
    content_version   INTEGER DEFAULT 1,
    cloudflare_deployment_id TEXT,
    schema_generated  BOOLEAN DEFAULT FALSE,
    last_built_at     TIMESTAMPTZ,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

-- ── Social media ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS social_posts (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    property_id       UUID NOT NULL REFERENCES properties(property_id),
    platform          TEXT NOT NULL CHECK (platform IN ('instagram', 'tiktok', 'facebook', 'pinterest')),
    content_type      TEXT,
    caption           TEXT,
    hashtags          JSONB,
    media_url         TEXT,
    video_type        TEXT,
    page_url          TEXT,
    utm_link          TEXT,
    scheduled_at      TIMESTAMPTZ,
    published_at      TIMESTAMPTZ,
    status            TEXT DEFAULT 'scheduled' CHECK (status IN ('scheduled', 'published', 'failed', 'cancelled')),
    ayrshare_post_id  TEXT,
    platform_post_id  TEXT,
    views             INTEGER DEFAULT 0,
    likes             INTEGER DEFAULT 0,
    shares            INTEGER DEFAULT 0,
    comments          INTEGER DEFAULT 0,
    completion_rate   NUMERIC(5,4) DEFAULT 0,
    link_clicks       INTEGER DEFAULT 0,
    nominated_for_spark BOOLEAN DEFAULT FALSE,
    spark_cluster_id  TEXT,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS content_calendars (
    property_id       UUID PRIMARY KEY REFERENCES properties(property_id),
    launch_date       DATE,
    total_scheduled   INTEGER DEFAULT 0,
    sprint_complete   BOOLEAN DEFAULT FALSE,
    summary           JSONB,
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS meta_campaigns (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    property_id       UUID NOT NULL REFERENCES properties(property_id),
    phase             TEXT NOT NULL,
    status            TEXT DEFAULT 'pending',
    meta_campaign_id  TEXT,
    meta_adset_id     TEXT,
    meta_ad_id        TEXT,
    budget_usd        NUMERIC(8,2),
    spend_to_date     NUMERIC(8,2) DEFAULT 0,
    impressions       INTEGER DEFAULT 0,
    clicks            INTEGER DEFAULT 0,
    start_date        DATE,
    end_date          DATE,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (property_id, phase)
);

CREATE TABLE IF NOT EXISTS spark_clusters (
    cluster_id        TEXT PRIMARY KEY,
    region_name       TEXT,
    monthly_budget_usd NUMERIC(8,2) DEFAULT 750,
    tiktok_campaign_id TEXT,
    active_spark_post_id TEXT,
    active_property_id UUID,
    is_active         BOOLEAN DEFAULT FALSE,
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS spark_cluster_members (
    cluster_id        TEXT NOT NULL REFERENCES spark_clusters(cluster_id),
    property_id       UUID NOT NULL REFERENCES properties(property_id),
    joined_at         TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (cluster_id, property_id)
);

-- ── Analytics ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS property_analytics_config (
    property_id       UUID PRIMARY KEY REFERENCES properties(property_id),
    client_id         UUID NOT NULL,
    slug              TEXT,
    page_url          TEXT,
    attribution_tier  TEXT DEFAULT 'tier_1_utm',
    pixel_snippet     TEXT,
    pixel_installed   BOOLEAN DEFAULT FALSE,
    tracking_active   BOOLEAN DEFAULT TRUE,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS analytics_snapshots (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    property_id       UUID NOT NULL REFERENCES properties(property_id),
    report_month      TEXT NOT NULL,    -- format: "2026-03"
    data              JSONB,
    updated_at        TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (property_id, report_month)
);

CREATE TABLE IF NOT EXISTS attribution_events (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    property_id       UUID NOT NULL,
    booking_date      DATE,
    stay_start        DATE,
    stay_end          DATE,
    stay_nights       INTEGER,
    booking_value     NUMERIC(10,2),
    source            TEXT,            -- 'pixel' | 'pms_api'
    utm_source        TEXT,
    utm_campaign      TEXT,
    utm_content       TEXT,
    pms_reservation_id TEXT,
    cancellation_status TEXT,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS monthly_reports (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    property_id       UUID NOT NULL REFERENCES properties(property_id),
    client_id         UUID NOT NULL,
    report_month      TEXT NOT NULL,
    analytics         JSONB,
    pdf_r2_url        TEXT,
    email_delivered   BOOLEAN DEFAULT FALSE,
    generated_at      TIMESTAMPTZ,
    UNIQUE (property_id, report_month)
);

-- ── PMS credentials (encrypted at rest via Supabase Vault in production) ──

CREATE TABLE IF NOT EXISTS pms_credentials (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    property_id       UUID NOT NULL REFERENCES properties(property_id),
    pms_type          TEXT NOT NULL,
    credentials       JSONB,           -- OAuth tokens, API keys
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (property_id, pms_type)
);

-- ── Row Level Security ────────────────────────────────────────────────────
-- Enable RLS on all tables. Policies are added separately per user role.

ALTER TABLE pmc_clients              ENABLE ROW LEVEL SECURITY;
ALTER TABLE io_clients               ENABLE ROW LEVEL SECURITY;
ALTER TABLE properties               ENABLE ROW LEVEL SECURITY;
ALTER TABLE intake_submissions       ENABLE ROW LEVEL SECURITY;
ALTER TABLE property_knowledge_bases ENABLE ROW LEVEL SECURITY;
ALTER TABLE pipeline_status          ENABLE ROW LEVEL SECURITY;
ALTER TABLE content_versions         ENABLE ROW LEVEL SECURITY;
ALTER TABLE media_assets             ENABLE ROW LEVEL SECURITY;
ALTER TABLE landing_pages            ENABLE ROW LEVEL SECURITY;
ALTER TABLE social_posts             ENABLE ROW LEVEL SECURITY;
ALTER TABLE content_calendars        ENABLE ROW LEVEL SECURITY;
ALTER TABLE meta_campaigns           ENABLE ROW LEVEL SECURITY;
ALTER TABLE spark_clusters           ENABLE ROW LEVEL SECURITY;
ALTER TABLE spark_cluster_members    ENABLE ROW LEVEL SECURITY;
ALTER TABLE property_analytics_config ENABLE ROW LEVEL SECURITY;
ALTER TABLE analytics_snapshots      ENABLE ROW LEVEL SECURITY;
ALTER TABLE attribution_events       ENABLE ROW LEVEL SECURITY;
ALTER TABLE monthly_reports          ENABLE ROW LEVEL SECURITY;
ALTER TABLE pms_credentials          ENABLE ROW LEVEL SECURITY;

-- =============================================================================
-- Schema created successfully.
-- You should now see 19 tables in the Supabase Table Editor.
-- =============================================================================
