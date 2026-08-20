-- AMENITY-2: Document the structured amenities shape change.
-- properties.amenities changes from a flat string array to
-- [{name: str, category: str}, ...]. No DDL change needed — the column
-- is already unconstrained jsonb.

COMMENT ON COLUMN public.properties.amenities IS
  'Structured amenities: [{"name": str, "category": str}, ...]. Shape changed
  2026-08-20 (AMENITY-2) from a flat string array. agents/agent5 and
  agents/agent8 read this column expecting flat strings and will break if
  invoked after this change — accepted, the old system retires today.';
