-- 003: Observations (visual metadata + curation) and guest evidence.
-- Applied to staging ypvylzrotmiyypapowaa on 2026-08-11

CREATE TABLE observations (
  observation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  property_id uuid NOT NULL REFERENCES properties(id),
  photo_id uuid NOT NULL REFERENCES photographs(photo_id),
  observation_version integer NOT NULL DEFAULT 1,
  model text,
  read_basis text CHECK (read_basis IN ('original', 'enhanced')),
  depth_structure text, depth_tier text, space_direction text,
  light_direction text, light_quality text, time_of_day_read text,
  negative_space jsonb DEFAULT '[]', foreground_elements jsonb DEFAULT '[]',
  frame_element text, beyond_frame_element text,
  subject_singularity text, focal_point text, tonal_signature text,
  located_amenities jsonb DEFAULT '[]',
  motion_affordance jsonb DEFAULT '[]', motion_risk jsonb DEFAULT '[]',
  persistence_manifest jsonb DEFAULT '[]',
  role text CHECK (role IN ('hero', 'supporting', 'gallery_only', 'exclude')),
  curated_section text, quality_score real, rank_within_section integer,
  alt_text text, duplicate_group text, is_best_in_group boolean DEFAULT true,
  gallery_visible boolean DEFAULT true, tour_eligible boolean DEFAULT true,
  superseded_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (property_id, photo_id, superseded_at)
);

CREATE INDEX idx_observations_property_active ON observations(property_id) WHERE superseded_at IS NULL;
CREATE INDEX idx_observations_photo ON observations(photo_id);

CREATE TABLE guest_evidence (
  evidence_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  property_id uuid NOT NULL REFERENCES properties(id),
  written_text text DEFAULT '', verbal_text text DEFAULT '',
  reviewer_name text, stay_date text,
  source text NOT NULL DEFAULT 'intake_portal',
  is_guest_book boolean NOT NULL DEFAULT true,
  star_rating real, host_response text,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_guest_evidence_property ON guest_evidence(property_id);
