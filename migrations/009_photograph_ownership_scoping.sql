-- 009: Photograph ownership — property-scoped identity.
-- Ruling: "photograph belongs to property" (Erick, 2026-08-11).
-- Applied to staging ypvylzrotmiyypapowaa on 2026-08-11.
--
-- photo_id = uuid5(NAMESPACE, property_id || ':' || content_hash)
-- Same bytes in two properties = two photographs with distinct photo_ids.

-- Must truncate dependent tables first (FK ordering)
-- TRUNCATE observations, renditions, guest_evidence, photographs CASCADE;

-- Drop the global unique, add property-scoped unique
ALTER TABLE photographs DROP CONSTRAINT photographs_content_hash_key;
ALTER TABLE photographs ADD CONSTRAINT photographs_property_content_hash_key
  UNIQUE (property_id, content_hash);

COMMENT ON COLUMN photographs.photo_id IS 'Deterministic UUID: uuid5(NAMESPACE, property_id || ":" || content_hash). Property-scoped — same bytes in two properties = two photo_ids. Ruling: Erick 2026-08-11.';
COMMENT ON COLUMN photographs.content_hash IS 'SHA-256 hex digest of the original image bytes. Unique WITHIN a property (not globally). Same image onboarded to two properties is two photographs.';
