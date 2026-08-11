-- 002: Physical spaces, photographs (content-hash identity), renditions.
-- Applied to staging ypvylzrotmiyypapowaa on 2026-08-11
-- See 001 for accounts/properties prerequisite.

CREATE TABLE physical_spaces (
  space_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  property_id uuid NOT NULL REFERENCES properties(id),
  label text NOT NULL,
  room_type text,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (property_id, label)
);

CREATE TABLE photographs (
  photo_id uuid PRIMARY KEY,
  property_id uuid NOT NULL REFERENCES properties(id),
  content_hash text UNIQUE NOT NULL,
  phash text,
  canonical_photo_id uuid REFERENCES photographs(photo_id),
  space_id uuid REFERENCES physical_spaces(space_id),
  source_urls text[] NOT NULL DEFAULT '{}',
  source_systems text[] NOT NULL DEFAULT '{}',
  is_canonical boolean NOT NULL DEFAULT true,
  image_width integer,
  image_height integer,
  quality_tier text CHECK (quality_tier IN ('high', 'medium', 'low')),
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_photographs_property ON photographs(property_id);
CREATE INDEX idx_photographs_phash ON photographs(phash) WHERE phash IS NOT NULL;

CREATE TYPE rendition_kind AS ENUM ('original', 'enhanced', 'social_crop');

CREATE TABLE renditions (
  rendition_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  photo_id uuid NOT NULL REFERENCES photographs(photo_id),
  kind rendition_kind NOT NULL,
  storage_url text NOT NULL,
  format text,
  width integer,
  height integer,
  byte_size bigint,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (photo_id, kind, format)
);

CREATE INDEX idx_renditions_photo ON renditions(photo_id);
