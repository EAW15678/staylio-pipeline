"""
Agent 3 Test Suite
Run with: pytest booked/agents/agent3/tests/ -v

Tests cover:
  - Governance hard block (prohibited operations must raise)
  - Permitted operations whitelist passes without error
  - Label-to-category mapping
  - Composition score calculation
  - Provenance consistency check (Jaccard similarity)
  - Category winner selection logic
  - Vibe hero priority selection
  - Video format matrix completeness
  - R2 key naming conventions
  - Agent node output contract
"""

import pytest
from unittest.mock import MagicMock, patch

from agents.agent3.claid_enhancer import (
    validate_operations,
    PERMITTED_OPERATIONS,
    PROHIBITED_OPERATIONS,
    STANDARD_ENHANCEMENT_PRESET,
    PHOTO_CEILING,
)
from agents.agent3.models import (
    MediaAsset,
    SubjectCategory,
    VideoFormat,
    VideoType,
    VIDEO_FORMAT_MATRIX,
    VIDEO_TARGET_DURATION,
    VisualMediaPackage,
)
from agents.agent3.r2_storage import (
    BUCKET_ORIGINALS,
    BUCKET_ENHANCED,
    BUCKET_CROPS,
    BUCKET_VIDEO,
    _photo_key,
    _stable_filename,
)
from agents.agent3.vision_tagger import (
    LABEL_TO_CATEGORY,
    VIBE_HERO_PRIORITY,
    _classify_category,
    _compute_composition_score,
    _run_provenance_checks,
    _select_category_winners,
    _select_hero,
)
from models.property import VibeProfile


# ── Governance Tests ──────────────────────────────────────────────────────

class TestGovernanceHardBlock:
    """
    These tests enforce the TS-07 operation whitelist.
    If any prohibited operation passes validation, it is a CRITICAL failure.
    """

    def test_permitted_operations_pass(self):
        """Standard enhancement preset must pass governance check."""
        # Should not raise
        validate_operations(STANDARD_ENHANCEMENT_PRESET["operations"])

    def test_background_removal_is_blocked(self):
        with pytest.raises(ValueError, match="GOVERNANCE VIOLATION"):
            validate_operations([{"type": "background_removal"}])

    def test_virtual_staging_is_blocked(self):
        with pytest.raises(ValueError, match="GOVERNANCE VIOLATION"):
            validate_operations([{"type": "virtual_staging"}])

    def test_generative_fill_is_blocked(self):
        with pytest.raises(ValueError, match="GOVERNANCE VIOLATION"):
            validate_operations([{"type": "generative_fill"}])

    def test_object_removal_is_blocked(self):
        with pytest.raises(ValueError, match="GOVERNANCE VIOLATION"):
            validate_operations([{"type": "object_removal"}])

    def test_object_addition_is_blocked(self):
        with pytest.raises(ValueError, match="GOVERNANCE VIOLATION"):
            validate_operations([{"type": "object_addition"}])

    def test_background_replacement_is_blocked(self):
        with pytest.raises(ValueError, match="GOVERNANCE VIOLATION"):
            validate_operations([{"type": "background_replacement"}])

    def test_scene_generation_is_blocked(self):
        with pytest.raises(ValueError, match="GOVERNANCE VIOLATION"):
            validate_operations([{"type": "scene_generation"}])

    def test_all_prohibited_operations_blocked(self):
        """Every prohibited operation must be blocked."""
        for op_type in PROHIBITED_OPERATIONS:
            with pytest.raises(ValueError, match="GOVERNANCE VIOLATION"):
                validate_operations([{"type": op_type}])

    def test_unknown_operation_blocked(self):
        """Operations not on the whitelist must also be blocked."""
        with pytest.raises(ValueError, match="GOVERNANCE VIOLATION"):
            validate_operations([{"type": "some_new_undocumented_feature"}])

    def test_mixed_permitted_and_prohibited_blocked(self):
        """Even if most operations are permitted, one prohibited must fail."""
        operations = [
            {"type": "upscale", "scale": 2},
            {"type": "noise_reduction"},
            {"type": "background_removal"},   # This one must trigger the block
        ]
        with pytest.raises(ValueError, match="GOVERNANCE VIOLATION"):
            validate_operations(operations)

    def test_all_permitted_operations_whitelisted(self):
        """Every permitted operation must pass individually."""
        for op_type in PERMITTED_OPERATIONS:
            # Should not raise
            validate_operations([{"type": op_type}])

    def test_photo_ceiling_is_100(self):
        """The photo ceiling must be 100 per TS-07."""
        assert PHOTO_CEILING == 100


# ── Category Classification Tests ─────────────────────────────────────────

class TestCategoryClassification:
    def test_pool_labels_classified_correctly(self):
        assert _classify_category(["swimming pool", "water", "resort"]) == SubjectCategory.POOL_HOT_TUB

    def test_ocean_view_classified_correctly(self):
        assert _classify_category(["ocean", "horizon", "sky"]) == SubjectCategory.VIEW

    def test_bedroom_classified_correctly(self):
        assert _classify_category(["bedroom", "bed", "pillow"]) in (
            SubjectCategory.MASTER_BEDROOM,
            SubjectCategory.STANDARD_BEDROOM,
        )

    def test_kitchen_classified_correctly(self):
        assert _classify_category(["kitchen", "countertop", "cooking"]) == SubjectCategory.KITCHEN

    def test_empty_labels_returns_uncategorised(self):
        assert _classify_category([]) == SubjectCategory.UNCATEGORISED

    def test_unknown_labels_returns_uncategorised(self):
        assert _classify_category(["abstract", "xyz_unknown_label"]) == SubjectCategory.UNCATEGORISED

    def test_most_matched_category_wins(self):
        """With mixed labels, the category with most matches should win."""
        # 3 pool labels vs 1 bedroom label — pool should win
        labels = ["swimming pool", "hot tub", "water", "bed"]
        result = _classify_category(labels)
        assert result == SubjectCategory.POOL_HOT_TUB


# ── Composition Score Tests ───────────────────────────────────────────────

class TestCompositionScore:
    def test_well_lit_sharp_photo_scores_high(self):
        score = _compute_composition_score(
            brightness=0.55,   # Ideal brightness
            sharpness=0.90,
            label_count=12,
        )
        assert score > 0.70

    def test_overexposed_photo_scores_lower(self):
        normal = _compute_composition_score(brightness=0.55, sharpness=0.80, label_count=10)
        overexposed = _compute_composition_score(brightness=0.95, sharpness=0.80, label_count=10)
        assert normal > overexposed

    def test_dark_photo_scores_lower(self):
        normal = _compute_composition_score(brightness=0.55, sharpness=0.80, label_count=10)
        dark = _compute_composition_score(brightness=0.10, sharpness=0.80, label_count=10)
        assert normal > dark

    def test_score_range_is_0_to_1(self):
        for brightness in [0.0, 0.3, 0.55, 0.8, 1.0]:
            for sharpness in [0.0, 0.5, 1.0]:
                score = _compute_composition_score(brightness, sharpness, label_count=5)
                assert 0.0 <= score <= 1.0, f"Score {score} out of range for b={brightness} s={sharpness}"


# ── Provenance Check Tests ─────────────────────────────────────────────────

class TestProvenanceCheck:
    def _make_asset(self, orig_labels, enh_labels):
        asset = MediaAsset(
            property_id="p1",
            asset_url_original="https://r2.example.com/originals/photo.jpg",
            asset_url_enhanced="https://r2.example.com/enhanced/photo.jpg",
        )
        asset.labels_original = orig_labels
        asset.labels_enhanced = enh_labels
        return asset

    def test_identical_labels_no_flag(self):
        asset = self._make_asset(
            ["pool", "water", "outdoor"],
            ["pool", "water", "outdoor"],
        )
        _run_provenance_checks([asset])
        assert asset.provenance_flag is False

    def test_similar_labels_no_flag(self):
        """Small differences in labels are OK — noise reduction changes some labels."""
        asset = self._make_asset(
            ["pool", "water", "outdoor", "sky"],
            ["pool", "water", "outdoor", "blue sky"],   # Slight variation
        )
        _run_provenance_checks([asset])
        assert asset.provenance_flag is False

    def test_completely_different_labels_flagged(self):
        """If labels are completely different, the photo must be flagged."""
        asset = self._make_asset(
            ["bedroom", "bed", "pillow", "furniture"],
            ["ocean", "beach", "sky", "horizon", "water"],  # Background was replaced
        )
        _run_provenance_checks([asset])
        assert asset.provenance_flag is True

    def test_empty_original_labels_no_flag(self):
        """If no original labels (baseline failed), don't falsely flag."""
        asset = self._make_asset([], ["pool", "water"])
        _run_provenance_checks([asset])
        assert asset.provenance_flag is False

    def test_new_view_label_on_interior_photo_flagged(self):
        """
        A photo that was an interior shot gaining 'ocean' or 'view' labels
        after enhancement suggests possible background replacement.
        """
        asset = self._make_asset(
            ["bedroom", "bed", "furniture", "interior", "wood floor"],
            ["ocean", "horizon", "water", "sky", "sunset"],  # Completely different
        )
        _run_provenance_checks([asset])
        assert asset.provenance_flag is True


# ── Hero Selection Tests ──────────────────────────────────────────────────

class TestHeroSelection:
    def test_vibe_hero_priorities_defined_for_all_vibes(self):
        for vibe in VibeProfile:
            assert vibe in VIBE_HERO_PRIORITY, f"Missing hero priority for {vibe}"
            assert len(VIBE_HERO_PRIORITY[vibe]) >= 3

    def test_romantic_escape_prefers_view_over_pool(self):
        category_winners = {
            SubjectCategory.VIEW: "https://r2.example.com/enhanced/view.jpg",
            SubjectCategory.POOL_HOT_TUB: "https://r2.example.com/enhanced/pool.jpg",
        }
        hero = _select_hero(category_winners, VibeProfile.ROMANTIC_ESCAPE)
        assert hero == "https://r2.example.com/enhanced/view.jpg"

    def test_family_adventure_prefers_pool_over_master_bedroom(self):
        category_winners = {
            SubjectCategory.POOL_HOT_TUB: "https://r2.example.com/enhanced/pool.jpg",
            SubjectCategory.MASTER_BEDROOM: "https://r2.example.com/enhanced/master.jpg",
        }
        hero = _select_hero(category_winners, VibeProfile.FAMILY_ADVENTURE)
        assert hero == "https://r2.example.com/enhanced/pool.jpg"

    def test_hero_selection_with_no_winners_returns_none(self):
        hero = _select_hero({}, VibeProfile.ROMANTIC_ESCAPE)
        assert hero is None

    def test_hero_falls_back_to_available_category(self):
        """If priority categories aren't available, use what's there."""
        category_winners = {
            SubjectCategory.BATHROOM: "https://r2.example.com/enhanced/bath.jpg",
        }
        hero = _select_hero(category_winners, VibeProfile.ROMANTIC_ESCAPE)
        assert hero is not None  # Should fall back rather than return None


# ── Category Winner Selection ─────────────────────────────────────────────

class TestCategoryWinnerSelection:
    def _make_assets(self, specs: list[dict]) -> list[MediaAsset]:
        assets = []
        for i, spec in enumerate(specs):
            a = MediaAsset(
                property_id="p1",
                asset_url_original=f"https://orig/{i}.jpg",
                asset_url_enhanced=f"https://enh/{i}.jpg",
            )
            a.subject_category = spec["category"]
            a.composition_score = spec["score"]
            a.safe_search_pass = spec.get("safe_search", True)
            a.provenance_flag = spec.get("flagged", False)
            assets.append(a)
        return assets

    def test_highest_scoring_photo_wins_per_category(self):
        assets = self._make_assets([
            {"category": SubjectCategory.POOL_HOT_TUB, "score": 0.6},
            {"category": SubjectCategory.POOL_HOT_TUB, "score": 0.9},  # Winner
            {"category": SubjectCategory.POOL_HOT_TUB, "score": 0.4},
        ])
        winners = _select_category_winners(assets)
        assert SubjectCategory.POOL_HOT_TUB in winners
        assert winners[SubjectCategory.POOL_HOT_TUB] == "https://enh/1.jpg"

    def test_failed_safe_search_excluded(self):
        assets = self._make_assets([
            {"category": SubjectCategory.VIEW, "score": 0.9, "safe_search": False},
        ])
        winners = _select_category_winners(assets)
        assert SubjectCategory.VIEW not in winners

    def test_provenance_flagged_excluded(self):
        assets = self._make_assets([
            {"category": SubjectCategory.EXTERIOR, "score": 0.85, "flagged": True},
        ])
        winners = _select_category_winners(assets)
        assert SubjectCategory.EXTERIOR not in winners

    def test_uncategorised_excluded_from_winners(self):
        assets = self._make_assets([
            {"category": SubjectCategory.UNCATEGORISED, "score": 0.99},
        ])
        winners = _select_category_winners(assets)
        assert SubjectCategory.UNCATEGORISED not in winners

    def test_category_rank_assigned(self):
        assets = self._make_assets([
            {"category": SubjectCategory.KITCHEN, "score": 0.7},
            {"category": SubjectCategory.KITCHEN, "score": 0.9},
            {"category": SubjectCategory.KITCHEN, "score": 0.5},
        ])
        _select_category_winners(assets)
        ranks = sorted([a.category_rank for a in assets])
        assert ranks == [1, 2, 3]

    def test_social_crop_queued_for_winners(self):
        assets = self._make_assets([
            {"category": SubjectCategory.VIEW, "score": 0.8},
            {"category": SubjectCategory.POOL_HOT_TUB, "score": 0.75},
        ])
        _select_category_winners(assets)
        queued = [a for a in assets if a.social_crop_queued]
        assert len(queued) == 2


# ── Video Format Matrix Tests ─────────────────────────────────────────────

class TestVideoFormatMatrix:
    def test_all_8_video_types_in_format_matrix(self):
        for vt in VideoType:
            assert vt in VIDEO_FORMAT_MATRIX, f"Missing format matrix entry for {vt}"

    def test_all_8_video_types_in_duration_map(self):
        for vt in VideoType:
            assert vt in VIDEO_TARGET_DURATION, f"Missing duration for {vt}"

    def test_vibe_match_has_all_three_formats(self):
        formats = VIDEO_FORMAT_MATRIX[VideoType.VIBE_MATCH]
        assert VideoFormat.VERTICAL in formats
        assert VideoFormat.SQUARE in formats
        assert VideoFormat.LANDSCAPE in formats

    def test_walk_through_is_vertical_only(self):
        """Video 2 is 9:16 only per TS-09 spec."""
        formats = VIDEO_FORMAT_MATRIX[VideoType.WALK_THROUGH]
        assert formats == [VideoFormat.VERTICAL]

    def test_feature_closeup_is_vertical_only(self):
        """Video 6 is 9:16 only per TS-09 spec."""
        formats = VIDEO_FORMAT_MATRIX[VideoType.FEATURE_CLOSEUP]
        assert formats == [VideoFormat.VERTICAL]

    def test_total_rendered_files_approximately_20(self):
        """Per TS-09: ~20 rendered files per property (8 videos × avg 2.5 formats)."""
        total = sum(len(fmts) for fmts in VIDEO_FORMAT_MATRIX.values())
        assert 18 <= total <= 22, f"Expected ~20 rendered files, got {total}"

    def test_durations_within_spec(self):
        """Durations must fall within specified ranges per TS-09."""
        spec_ranges = {
            VideoType.VIBE_MATCH:        (45, 60),
            VideoType.WALK_THROUGH:      (15, 20),
            VideoType.GUEST_REVIEW_1:    (30, 45),
            VideoType.GUEST_REVIEW_2:    (30, 45),
            VideoType.GUEST_REVIEW_3:    (30, 45),
            VideoType.LOCAL_HIGHLIGHT:   (20, 30),
            VideoType.FEATURE_CLOSEUP:   (12, 15),
            VideoType.SEASONAL:          (20, 30),
        }
        for vt, (min_s, max_s) in spec_ranges.items():
            dur = VIDEO_TARGET_DURATION[vt]
            assert min_s <= dur <= max_s, f"{vt}: duration {dur}s outside spec [{min_s}, {max_s}]"


# ── R2 Storage Tests ─────────────────────────────────────────────────────

class TestR2Storage:
    def test_photo_key_format(self):
        key = _photo_key("prop-001", "enhanced", "photo_001_abc123.jpg")
        assert key == "prop-001/enhanced/photo_001_abc123.jpg"

    def test_stable_filename_is_deterministic(self):
        """Same URL always produces same filename."""
        url = "https://a0.muscache.com/photos/12345.jpg"
        f1 = _stable_filename(url, 5)
        f2 = _stable_filename(url, 5)
        assert f1 == f2

    def test_stable_filename_is_different_for_different_urls(self):
        f1 = _stable_filename("https://example.com/photo1.jpg", 0)
        f2 = _stable_filename("https://example.com/photo2.jpg", 0)
        assert f1 != f2

    def test_four_buckets_are_distinct(self):
        buckets = {BUCKET_ORIGINALS, BUCKET_ENHANCED, BUCKET_CROPS, BUCKET_VIDEO}
        assert len(buckets) == 4


# ── Agent Node Contract Tests ─────────────────────────────────────────────

class TestAgent3NodeContract:
    def test_no_photos_completes_without_error(self):
        """A property with no photos should complete gracefully."""
        from agents.agent3.agent import agent3_node

        kb = {
            "property_id": "p1",
            "vibe_profile": "romantic_escape",
            "photos": [],
            "guest_reviews": [],
        }
        state = {
            "property_id": "p1",
            "knowledge_base": kb,
            "errors": [],
        }

        with patch("booked.agents.agent3.agent.get_cached_knowledge_base", return_value=kb), \
             patch("booked.agents.agent3.agent.update_pipeline_status"):
            result = agent3_node(state)

        assert result["agent3_complete"] is True

    def test_successful_processing_sets_flag(self):
        """Full processing pipeline should set agent3_complete=True."""
        from agents.agent3.agent import agent3_node

        kb = {
            "property_id": "p1",
            "vibe_profile": "family_adventure",
            "photos": [{"url": "https://example.com/photo1.jpg", "source": "airbnb"}],
            "guest_reviews": [],
            "dont_miss_picks": [],
        }
        state = {
            "property_id": "p1",
            "knowledge_base": kb,
            "content_package": {},
            "errors": [],
        }

        mock_pkg = VisualMediaPackage(property_id="p1")
        mock_pkg.hero_photo_url = "https://r2.example.com/enhanced/hero.jpg"
        mock_pkg.category_winners = {"view": "https://r2.example.com/enhanced/view.jpg"}

        with patch("booked.agents.agent3.agent.get_cached_knowledge_base", return_value=kb), \
             patch("booked.agents.agent3.agent.asyncio") as mock_asyncio, \
             patch("booked.agents.agent3.agent.enhance_photo_batch_sync",
                   return_value=[("https://example.com/photo1.jpg", b"fake_bytes")]), \
             patch("booked.agents.agent3.agent.upload_photo_original",
                   return_value="https://r2.example.com/originals/photo.jpg"), \
             patch("booked.agents.agent3.agent.upload_photo_enhanced",
                   return_value="https://r2.example.com/enhanced/photo.jpg"), \
             patch("booked.agents.agent3.agent.tag_original_for_provenance",
                   side_effect=lambda a: a), \
             patch("booked.agents.agent3.agent.tag_and_score_photos",
                   return_value=([], "https://r2.example.com/enhanced/hero.jpg")), \
             patch("booked.agents.agent3.agent.generate_social_crops", return_value=[]), \
             patch("booked.agents.agent3.agent._save_media_assets"), \
             patch("booked.agents.agent3.agent._save_video_assets"), \
             patch("booked.agents.agent3.agent._save_social_crops"), \
             patch("booked.agents.agent3.agent._flag_provenance_violations"), \
             patch("booked.agents.agent3.agent.cache_knowledge_base"), \
             patch("booked.agents.agent3.agent.update_pipeline_status"):

            # Mock asyncio.run to return empty values
            mock_asyncio.run.side_effect = [
                [("https://example.com/photo1.jpg", b"fake_bytes")],  # _download_photos
                [],  # generate_all_videos
            ]
            mock_asyncio.Semaphore = lambda n: MagicMock()

            result = agent3_node(state)

        assert result["agent3_complete"] is True
        assert "visual_media_package" in result
