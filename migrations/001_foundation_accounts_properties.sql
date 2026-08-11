-- 001: Foundation — accounts, users, account_users, properties
-- Applied to staging ypvylzrotmiyypapowaa on 2026-08-11

CREATE TABLE accounts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  legal_name text,
  billing_email text,
  primary_contact_name text,
  primary_contact_email text,
  status text NOT NULL DEFAULT 'active'
    CHECK (status IN ('active', 'suspended', 'archived')),
  subscription_status text DEFAULT 'trial'
    CHECK (subscription_status IN ('trial', 'active', 'past_due', 'cancelled')),
  trial_started_at timestamptz,
  trial_ends_at timestamptz,
  contract_start_at timestamptz,
  contract_end_at timestamptz,
  timezone text DEFAULT 'America/New_York',
  stripe_customer_id text,
  notes text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  archived_at timestamptz
);

COMMENT ON TABLE accounts IS 'Billing entity that owns one or more properties.';
COMMENT ON COLUMN accounts.id IS 'Surrogate UUID primary key.';
COMMENT ON COLUMN accounts.status IS 'Account lifecycle: active, suspended, archived.';
COMMENT ON COLUMN accounts.subscription_status IS 'Billing state: trial, active, past_due, cancelled.';

CREATE TABLE users (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  email text UNIQUE NOT NULL,
  full_name text,
  role text NOT NULL DEFAULT 'owner'
    CHECK (role IN ('owner', 'manager', 'viewer', 'admin')),
  user_type text DEFAULT 'owner',
  status text NOT NULL DEFAULT 'active'
    CHECK (status IN ('active', 'disabled')),
  last_login_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  disabled_at timestamptz
);

COMMENT ON TABLE users IS 'Authenticated user accounts (Supabase Auth managed).';

CREATE TABLE account_users (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  account_id uuid NOT NULL REFERENCES accounts(id),
  user_id uuid NOT NULL REFERENCES users(id),
  relationship_role text NOT NULL DEFAULT 'owner'
    CHECK (relationship_role IN ('owner', 'manager', 'viewer')),
  status text NOT NULL DEFAULT 'active',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (account_id, user_id)
);

COMMENT ON TABLE account_users IS 'Many-to-many: which users belong to which accounts.';

CREATE TABLE properties (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  account_id uuid NOT NULL REFERENCES accounts(id),
  name text NOT NULL,
  slug text UNIQUE,
  vibe_profile text,
  property_type text,
  address_line_1 text,
  address_line_2 text,
  city text,
  state_region text,
  postal_code text,
  country text DEFAULT 'US',
  latitude double precision,
  longitude double precision,
  market text,
  booking_url text,
  ical_url text,
  airbnb_url text,
  vrbo_url text,
  booking_com_url text,
  google_vacation_rentals_url text,
  primary_listing_url text,
  subdomain text,
  status text NOT NULL DEFAULT 'draft'
    CHECK (status IN ('draft', 'onboarding', 'active', 'paused', 'archived')),
  go_live_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  archived_at timestamptz
);

COMMENT ON TABLE properties IS 'A vacation rental that UpliftStays markets. Root entity.';
COMMENT ON COLUMN properties.slug IS 'URL-safe identifier used as subdomain ({slug}.upliftstays.com). THE authoritative slug.';
COMMENT ON COLUMN properties.vibe_profile IS 'One of 7 archetypes: romantic_escape, family_adventure, multigenerational_retreat, wellness_retreat, adventure_base_camp, social_celebrations, creative_remote_work.';
COMMENT ON COLUMN properties.status IS 'Property lifecycle: draft → onboarding → active → paused → archived.';
