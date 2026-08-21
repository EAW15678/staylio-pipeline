-- DIRECTOR-3: Add quality self-assessment fields to directions.

ALTER TABLE public.directions ADD COLUMN IF NOT EXISTS quality_self_score jsonb;
COMMENT ON COLUMN public.directions.quality_self_score IS
  'Nine-dimension quality self-assessment by the director. JSONB object with score (1-10) and why per dimension. Added 2026-08-21 (DIRECTOR-3). Self-scoring is a soft signal — not a quality guarantee.';

ALTER TABLE public.directions ADD COLUMN IF NOT EXISTS quality_shortfalls jsonb;
COMMENT ON COLUMN public.directions.quality_shortfalls IS
  'Dimensions below threshold at final attempt, null if none. Added 2026-08-21 (DIRECTOR-3). Thresholds are first guesses and need tuning against real runs.';
