-- 022: Add depth_map to rendition_kind enum.
-- Depth maps are deterministic derivatives of the photograph,
-- stored as renditions alongside original/enhanced/social_crop.

ALTER TYPE rendition_kind ADD VALUE IF NOT EXISTS 'depth_map';

-- source_content_hash records which version of the photograph
-- produced this rendition. If the source is re-enhanced or replaced,
-- a stale depth map can be detected and regenerated.
ALTER TABLE renditions ADD COLUMN IF NOT EXISTS source_content_hash text;

COMMENT ON COLUMN renditions.source_content_hash IS
  'Content hash of the source rendition that produced this derived rendition. '
  'Used for cache invalidation — if the source changes, derivatives are stale.';
