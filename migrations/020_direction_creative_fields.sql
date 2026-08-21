-- DIRECTOR-1: Add creative reasoning fields to directions.
-- These record the director's pre-beat reasoning so it's inspectable.

ALTER TABLE public.directions ADD COLUMN IF NOT EXISTS hero_proposition text;
COMMENT ON COLUMN public.directions.hero_proposition IS
  'The single strongest booking driver identified by the director. Added 2026-08-21 (DIRECTOR-1).';

ALTER TABLE public.directions ADD COLUMN IF NOT EXISTS booking_drivers jsonb;
COMMENT ON COLUMN public.directions.booking_drivers IS
  'Top 3 booking drivers ranked by the director. JSONB array of strings. Added 2026-08-21 (DIRECTOR-1).';

ALTER TABLE public.directions ADD COLUMN IF NOT EXISTS directing_mode text;
COMMENT ON COLUMN public.directions.directing_mode IS
  'Inferred directing mode (The Reveal, The Weekend, etc.). Added 2026-08-21 (DIRECTOR-1).';

ALTER TABLE public.directions ADD COLUMN IF NOT EXISTS mode_rationale text;
COMMENT ON COLUMN public.directions.mode_rationale IS
  'One-line reason the director chose this mode. Added 2026-08-21 (DIRECTOR-1).';
