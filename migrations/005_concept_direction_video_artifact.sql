-- 005: Agent 8 content engine — concepts, directions, video artifacts.
-- Applied to staging ypvylzrotmiyypapowaa on 2026-08-11

CREATE TABLE concepts (
  concept_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  property_id uuid NOT NULL REFERENCES properties(id),
  cycle_month date NOT NULL, concept_number integer NOT NULL,
  concept_version integer NOT NULL DEFAULT 1,
  title text NOT NULL, premise text, vibe_prior text,
  evidence_basis jsonb DEFAULT '[]', source_material jsonb DEFAULT '[]',
  target_platforms text[] DEFAULT '{}', utm_content_slug text,
  status text NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'active', 'superseded')),
  review_note text, reviewed_at timestamptz, reviewed_by text,
  superseded_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(), created_by_agent text
);

CREATE INDEX idx_concepts_property_month ON concepts(property_id, cycle_month) WHERE superseded_at IS NULL;

CREATE TABLE directions (
  direction_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  property_id uuid NOT NULL REFERENCES properties(id),
  concept_id uuid NOT NULL REFERENCES concepts(concept_id),
  direction_version integer NOT NULL DEFAULT 1,
  beats jsonb NOT NULL DEFAULT '[]', beat_count integer DEFAULT 0,
  target_duration_sec real DEFAULT 45, narrative_order text,
  continuity_notes jsonb DEFAULT '[]',
  narration_brief text, narration_provenance text,
  music_brief jsonb DEFAULT '{}', overlay_register jsonb DEFAULT '[]',
  director_rationale text, director_model text,
  evidence_used jsonb DEFAULT '[]', vibe_drift text,
  input_hash text, direction_checksum text,
  rejection_reasons jsonb,
  status text NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'approved', 'rejected', 'superseded')),
  review_note text, reviewed_at timestamptz, reviewed_by text,
  superseded_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(), created_by_agent text
);

CREATE INDEX idx_directions_concept ON directions(concept_id) WHERE superseded_at IS NULL;

CREATE TYPE video_artifact_kind AS ENUM ('clip', 'narration', 'music', 'production', 'variant');

CREATE TABLE video_artifacts (
  artifact_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  property_id uuid NOT NULL REFERENCES properties(id),
  kind video_artifact_kind NOT NULL,
  input_hash text UNIQUE,
  direction_id uuid REFERENCES directions(direction_id),
  concept_id uuid REFERENCES concepts(concept_id),
  photo_id uuid REFERENCES photographs(photo_id),
  storage_url text, duration_seconds real, model text, vendor text,
  beat_ordinal integer, requested_motion text,
  technique text DEFAULT 'generative', motion_params jsonb DEFAULT '{}',
  script_text text, voice_id text, voice_label text,
  alignment jsonb, language text DEFAULT 'en',
  provenance text, source_reference jsonb, contains_verbatim_source boolean,
  prompt_text text, composition_plan jsonb, brief jsonb, sample_rate integer,
  clip_ids uuid[] DEFAULT '{}',
  narration_artifact_id uuid REFERENCES video_artifacts(artifact_id),
  music_artifact_id uuid REFERENCES video_artifacts(artifact_id),
  overlay_payload jsonb, render_payload jsonb,
  renderer text, template_id_used text, template_checksum text,
  has_audio_ducking boolean DEFAULT false, post_process jsonb DEFAULT '[]',
  platform text, aspect_ratio text, resolution text,
  caption text, hashtags text[], utm_content text, link_url text,
  persistence_check text DEFAULT 'unchecked', persistence_findings jsonb,
  compliance_check_id uuid,
  cost_estimate_usd real, attempt_count integer DEFAULT 0,
  status text NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'queued', 'rendering', 'ready', 'failed', 'rejected', 'held', 'draft')),
  failure_reason text, superseded_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(), created_by_agent text
);

CREATE INDEX idx_video_artifacts_property ON video_artifacts(property_id);
CREATE INDEX idx_video_artifacts_direction ON video_artifacts(direction_id) WHERE superseded_at IS NULL;
CREATE INDEX idx_video_artifacts_kind ON video_artifacts(kind, status) WHERE superseded_at IS NULL;
