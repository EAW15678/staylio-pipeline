-- 006: Publications, workflow tracking, cost events, visitor/booking (carried as-is).
-- Applied to staging ypvylzrotmiyypapowaa on 2026-08-11

CREATE TABLE publications (
  publication_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  property_id uuid NOT NULL REFERENCES properties(id),
  artifact_id uuid REFERENCES video_artifacts(artifact_id),
  platform text NOT NULL, caption text, hashtags text[],
  media_url text, link_url text,
  utm_source text, utm_medium text, utm_campaign text, utm_content text,
  scheduled_at timestamptz, published_at timestamptz,
  external_post_id text, provider text,
  status text NOT NULL DEFAULT 'draft'
    CHECK (status IN ('draft', 'scheduled', 'published', 'failed', 'cancelled')),
  engagement jsonb DEFAULT '{}',
  nominated_for_spark boolean DEFAULT false, spark_cluster_id text,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX idx_publications_property ON publications(property_id);

CREATE TABLE runs (
  run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  property_id uuid NOT NULL REFERENCES properties(id),
  workflow text NOT NULL CHECK (workflow IN ('onboard', 'monthly_cycle', 'refresh', 'report')),
  status text NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'complete', 'failed')),
  started_at timestamptz NOT NULL DEFAULT now(), completed_at timestamptz,
  error_summary text, metadata jsonb DEFAULT '{}'
);

CREATE TABLE run_steps (
  step_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id uuid NOT NULL REFERENCES runs(run_id),
  step_name text NOT NULL,
  status text NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'complete', 'failed', 'skipped')),
  started_at timestamptz NOT NULL DEFAULT now(), completed_at timestamptz,
  error_message text, metadata jsonb DEFAULT '{}'
);
CREATE INDEX idx_run_steps_run ON run_steps(run_id);

CREATE TABLE cost_events (
  id bigserial PRIMARY KEY,
  run_id uuid REFERENCES runs(run_id),
  property_id uuid REFERENCES properties(id),
  vendor text NOT NULL, service text NOT NULL,
  units real NOT NULL, unit_name text NOT NULL,
  unit_cost real, total_cost real,
  workflow_name text, generation_reason text, discriminator text,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX idx_cost_events_property ON cost_events(property_id);
CREATE INDEX idx_cost_events_run ON cost_events(run_id);

-- Visitor/booking/attribution — carried as-is from production
CREATE TABLE visitor_sessions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  property_id uuid REFERENCES properties(id), account_id uuid REFERENCES accounts(id),
  session_key text NOT NULL, visitor_key text NOT NULL,
  device_type text, landing_url text, referrer_url text,
  city text, region text, country text,
  utm_source text, utm_medium text, utm_campaign text, utm_term text, utm_content text,
  lead_id uuid,
  first_seen_at timestamptz DEFAULT now(), last_seen_at timestamptz DEFAULT now(),
  created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE page_events (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  property_id uuid REFERENCES properties(id), account_id uuid REFERENCES accounts(id),
  visitor_session_id uuid REFERENCES visitor_sessions(id),
  event_name text NOT NULL, event_payload jsonb DEFAULT '{}',
  occurred_at timestamptz NOT NULL DEFAULT now(), created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE cta_clicks (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  property_id uuid REFERENCES properties(id), account_id uuid REFERENCES accounts(id),
  visitor_session_id uuid REFERENCES visitor_sessions(id),
  cta_type text, cta_location text, destination_url text, lead_id uuid,
  redirected boolean DEFAULT false,
  clicked_at timestamptz NOT NULL DEFAULT now(), created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE bookings (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  property_id uuid REFERENCES properties(id), account_id uuid REFERENCES accounts(id),
  external_booking_id text, source_system text, source_type text, source_text text,
  source_confidence text, booking_value real, currency text DEFAULT 'USD',
  check_in_date date, check_out_date date, guest_identifier text,
  booking_status text DEFAULT 'confirmed',
  booking_created_at timestamptz, attribution_state text DEFAULT 'unattributed',
  reported_by_user_id uuid, raw_payload jsonb,
  created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE booking_reports (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  property_id uuid REFERENCES properties(id), account_id uuid REFERENCES accounts(id),
  booking_date date, booking_value real, currency text DEFAULT 'USD',
  booking_reference text, check_in_date date, check_out_date date,
  report_source text, reported_at timestamptz, reported_by_user_id uuid, notes text,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE attribution_links (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  property_id uuid REFERENCES properties(id), account_id uuid REFERENCES accounts(id),
  visitor_session_id uuid REFERENCES visitor_sessions(id),
  cta_click_id uuid REFERENCES cta_clicks(id),
  lead_id uuid, booking_id uuid REFERENCES bookings(id),
  evidence_type text, confidence_level text, created_by_user_id uuid, notes text,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE intake_sessions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  account_id uuid REFERENCES accounts(id), property_id uuid REFERENCES properties(id),
  started_by_user_id uuid, current_step text DEFAULT 'step_1',
  status text NOT NULL DEFAULT 'in_progress'
    CHECK (status IN ('not_started', 'in_progress', 'submitted', 'complete')),
  started_at timestamptz DEFAULT now(), submitted_at timestamptz, completed_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE intake_answers (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  intake_session_id uuid NOT NULL REFERENCES intake_sessions(id),
  property_id uuid REFERENCES properties(id),
  question_key text NOT NULL, answer_json jsonb,
  answer_source text DEFAULT 'portal', answered_by_user_id uuid,
  answered_at timestamptz DEFAULT now(), created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE hitl_queue_items (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  property_id uuid REFERENCES properties(id), account_id uuid REFERENCES accounts(id),
  queue_type text NOT NULL, reason_code text, title text, description text,
  priority text DEFAULT 'normal',
  status text NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'assigned', 'resolved', 'dismissed')),
  payload jsonb DEFAULT '{}', assigned_to_user_id uuid,
  due_at timestamptz, resolved_at timestamptz,
  created_by_type text, created_by_user_id uuid,
  created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE compliance_checks (
  check_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  property_id uuid REFERENCES properties(id),
  concept_id uuid REFERENCES concepts(concept_id),
  direction_id uuid REFERENCES directions(direction_id),
  subject_type text NOT NULL, subject_id uuid NOT NULL,
  checker_version text, guidelines_version text,
  verdict text NOT NULL DEFAULT 'hold' CHECK (verdict IN ('pass', 'hold', 'fail')),
  findings jsonb DEFAULT '[]', rules_evaluated jsonb DEFAULT '[]', rules_skipped jsonb DEFAULT '[]',
  removal_rule_result text, missing_elements jsonb, model text,
  escalated boolean DEFAULT false,
  hitl_queue_item_id uuid REFERENCES hitl_queue_items(id),
  superseded_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(), created_by_agent text
);
