-- 018_vendor_rates.sql
-- ENHANCE-3: Vendor rates in data, not code.
--
-- The credit price changes with volume/enterprise pricing. Typing it
-- into code means overwriting a constant silently restates every
-- historical cost at the new rate. Dated rows let past runs stay true
-- at the price actually paid while a future rate applies going forward.

CREATE TABLE public.vendor_rates (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  vendor text NOT NULL,
  unit_name text NOT NULL,
  unit_cost numeric NOT NULL,
  effective_from timestamptz NOT NULL,
  effective_to timestamptz,  -- NULL = current
  note text,
  created_at timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE public.vendor_rates IS
  'Vendor pricing rates dated by effective period. Past runs stay true '
  'at the price actually paid; future rate changes apply going forward '
  'without rewriting history. Created ENHANCE-3 (2026-08-19).';

COMMENT ON COLUMN public.vendor_rates.vendor IS
  'Vendor identifier matching cost_events.vendor (e.g. claid, runway, elevenlabs).';

COMMENT ON COLUMN public.vendor_rates.unit_name IS
  'Unit of billing matching cost_events.unit_name (e.g. credits, seconds, characters).';

COMMENT ON COLUMN public.vendor_rates.unit_cost IS
  'Cost per unit in USD. For Claid: $0.059 per credit ($59 per 1,000 self-serve).';

COMMENT ON COLUMN public.vendor_rates.effective_from IS
  'Start of the period this rate applies. Inclusive.';

COMMENT ON COLUMN public.vendor_rates.effective_to IS
  'End of the period this rate applies. NULL means the rate is current. '
  'Set this when a new rate takes over — never delete the old row.';

COMMENT ON COLUMN public.vendor_rates.note IS
  'Human-readable context for the rate — bundle name, contract terms, etc.';

-- Seed the current Claid rate
INSERT INTO public.vendor_rates (vendor, unit_name, unit_cost, effective_from, note)
VALUES ('claid', 'credits', 0.059, '2026-08-01', '$59 per 1,000 self-serve bundle; falls with volume pricing.');
