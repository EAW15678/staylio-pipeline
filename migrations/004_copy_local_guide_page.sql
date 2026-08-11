-- 004: Copy versions, local guides, and landing pages.
-- Applied to staging ypvylzrotmiyypapowaa on 2026-08-11

CREATE TABLE copy_versions (
  copy_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  property_id uuid NOT NULL REFERENCES properties(id),
  version integer NOT NULL DEFAULT 1,
  content jsonb NOT NULL DEFAULT '{}',
  quality_score real, quality_result text CHECK (quality_result IN ('pass', 'needs_review', 'fail')),
  generated_by_model text, seo_target_keywords text[],
  status text NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'approved', 'published', 'superseded')),
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (property_id, version)
);

CREATE TABLE local_guides (
  guide_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  property_id uuid NOT NULL REFERENCES properties(id),
  area_introduction text, dont_miss_picks jsonb DEFAULT '[]',
  primary_recommendations jsonb DEFAULT '[]', places_by_category jsonb DEFAULT '{}',
  total_places integer DEFAULT 0, location_name text,
  created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (property_id)
);

CREATE TABLE landing_pages (
  page_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  property_id uuid NOT NULL REFERENCES properties(id),
  slug text NOT NULL, page_url text,
  deploy_mode text DEFAULT 'subdomain' CHECK (deploy_mode IN ('subdomain', 'cname', 'simulated')),
  status text NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'building', 'deployed', 'failed')),
  content_version integer, schema_generated boolean DEFAULT false,
  cloudflare_deployment_id text, last_built_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (property_id)
);
