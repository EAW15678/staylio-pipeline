-- M-TRACK-0.2: Staylio Marketing sentinel account + property
-- (replay-safe revision)
--
-- Purpose: Bootstrap a non-customer "Staylio Marketing" account and
-- a sentinel "Staylio Marketing Site" property that the staylio.ai
-- marketing site can use as its property_id when calling
-- /public/sessions/start. This identity is what marketing-site
-- analytics events will be attached to in page_events.
--
-- Replay safety:
--   First run:  creates one account + one property, returns both UUIDs.
--   Second run: returns the SAME UUIDs from the first run. Non-NULL.
--   No duplicate accounts. No duplicate properties.
--   No mutation of existing rows on replay (no ON CONFLICT DO UPDATE).
--
-- Detect-or-create pattern:
--   Account:  keyed on exact name match ('Staylio Marketing').
--             accounts has no natural unique key beyond id, so a
--             WHERE NOT EXISTS guard is used instead of ON CONFLICT.
--   Property: keyed on subdomain ('marketing-site'), which carries a
--             UNIQUE constraint. Same WHERE NOT EXISTS guard used for
--             consistency; UNIQUE constraint is a backstop.
--
-- Constraints honored:
--   - No schema changes (no DDL)
--   - No CHECK constraint changes
--   - No RLS changes
--   - No rollup view changes
--   - Uses only existing allowed enum values:
--       accounts.account_type  = 'owner'
--       accounts.status        = 'active'
--       properties.client_type = 'owner'
--       properties.status      = 'active'
--   - No account_users row created (this account is not customer-
--     accessible; pipeline service-role inserts bypass RLS)
--
-- Run with: psql or Supabase SQL editor (service role)
-- DO NOT EXECUTE WITHOUT REVIEW -- one-time production bootstrap.

BEGIN;

WITH

-- Step 1: detect existing account by exact name
existing_account AS (
  SELECT id
  FROM   accounts
  WHERE  name = 'Staylio Marketing'
  LIMIT  1
),

-- Step 2: insert account only when none exists
inserted_account AS (
  INSERT INTO accounts (account_type, name, status)
  SELECT 'owner', 'Staylio Marketing', 'active'
  WHERE  NOT EXISTS (SELECT 1 FROM existing_account)
  RETURNING id
),

-- Step 3: resolve canonical account id (existing wins; insert fills on first run)
account AS (
  SELECT id FROM existing_account
  UNION ALL
  SELECT id FROM inserted_account
),

-- Step 4: detect existing property by subdomain (unique constraint)
existing_property AS (
  SELECT id
  FROM   properties
  WHERE  subdomain = 'marketing-site'
  LIMIT  1
),

-- Step 5: insert property only when none exists
inserted_property AS (
  INSERT INTO properties (
    name,
    slug,
    subdomain,
    status,
    account_id,
    client_id,
    client_type
  )
  SELECT
    'Staylio Marketing Site',
    'marketing-site',
    'marketing-site',
    'active',
    account.id,
    account.id,
    'owner'
  FROM account
  WHERE NOT EXISTS (SELECT 1 FROM existing_property)
  RETURNING id
),

-- Step 6: resolve canonical property id (existing wins; insert fills on first run)
property AS (
  SELECT id FROM existing_property
  UNION ALL
  SELECT id FROM inserted_property
)

SELECT
  account.id  AS staylio_marketing_account_id,
  property.id AS staylio_marketing_property_id
FROM account, property;

COMMIT;

-- After running, capture both UUIDs from the SELECT output:
--   staylio_marketing_account_id  -> store as STAYLIO_MARKETING_ACCOUNT_ID
--   staylio_marketing_property_id -> store as STAYLIO_MARKETING_PROPERTY_ID
--
-- The property UUID becomes the propertyId prop passed to
-- AnalyticsProvider on staylio.ai marketing pages (M-TRACK-1 slice).
