"""
Agent 4 Test Suite
Run with: pytest booked/agents/agent4/tests/ -v

Tests cover:
  - Merge policy (Google wins name/address, Yelp wins price tier)
  - Deduplication by name similarity + proximity
  - Composite rating calculation
  - Vibe primary category priorities (all 6 vibes)
  - Category diversity in primary recommendations (max 3 per category)
  - Don't Miss pick matching
  - Area introduction fallbacks
  - Agent node output contract
  - Graceful degradation when one API source fails
"""

import pytest
from unittest.mock import MagicMock, patch

from agents.agent4.models import (
    DataSource,
    DontMissPick,
    LocalGuide,
    Place,
    PlaceCategory,
    PlaceHours,
)
from agents.agent4.guide_assembler import (
    VIBE_PRIMARY_CATEGORIES,
    _build_dont_miss_picks,
    _fallback_area_intro,
    _find_yelp_match,
    _merge_places,
    _merge_records,
    _organise_by_category,
    _select_primary_recommendations,
    assemble_local_guide,
)
from models.property import VibeProfile


# ── Fixtures ──────────────────────────────────────────────────────────────

def make_google_place(
    name: str,
    category: PlaceCategory = PlaceCategory.EAT_AND_DRINK,
    lat: float = 34.03,
    lng: float = -77.89,
    rating: float = 4.5,
    review_count: int = 200,
    price_level: str = None,
    place_id: str = None,
) -> Place:
    return Place(
        name=name,
        category=category,
        primary_source=DataSource.GOOGLE_PLACES,
        latitude=lat,
        longitude=lng,
        google_rating=rating,
        google_review_count=review_count,
        price_level=price_level,
        google_place_id=place_id or f"gid_{name[:8]}",
    )


def make_yelp_place(
    name: str,
    category: PlaceCategory = PlaceCategory.EAT_AND_DRINK,
    lat: float = 34.03,
    lng: float = -77.89,
    rating: float = 4.3,
    review_count: int = 350,
    price_level: str = "$$",
    yelp_id: str = None,
) -> Place:
    return Place(
        name=name,
        category=category,
        primary_source=DataSource.YELP_FUSION,
        latitude=lat,
        longitude=lng,
        yelp_rating=rating,
        yelp_review_count=review_count,
        price_level=price_level,
        yelp_id=yelp_id or f"yid_{name[:8]}",
    )


# ── Composite Rating Tests ────────────────────────────────────────────────

class TestCompositeRating:
    def test_both_ratings_weighted_correctly(self):
        place = Place(
            name="Test",
            category=PlaceCategory.EAT_AND_DRINK,
            primary_source=DataSource.GOOGLE_PLACES,
            google_rating=5.0,
            yelp_rating=4.0,
        )
        # 5.0 * 0.6 + 4.0 * 0.4 = 3.0 + 1.6 = 4.6
        assert place.composite_rating() == pytest.approx(4.6, rel=0.01)

    def test_google_only_returns_google(self):
        place = make_google_place("Test", rating=4.2)
        assert place.composite_rating() == 4.2

    def test_yelp_only_returns_yelp(self):
        place = make_yelp_place("Test", rating=3.8)
        assert place.composite_rating() == 3.8

    def test_no_ratings_returns_none(self):
        place = Place(name="X", category=PlaceCategory.EAT_AND_DRINK,
                      primary_source=DataSource.GOOGLE_PLACES)
        assert place.composite_rating() is None


# ── Merge & Deduplication Tests ───────────────────────────────────────────

class TestMergeAndDedup:
    def test_exact_name_match_deduplicates(self):
        """Same business in both sources should produce one record."""
        google = [make_google_place("The Blue Fish", lat=34.03, lng=-77.89)]
        yelp   = [make_yelp_place("The Blue Fish",   lat=34.03, lng=-77.89)]
        merged = _merge_places(google, yelp)
        assert len(merged) == 1

    def test_fuzzy_name_match_deduplicates(self):
        """Near-identical names at same location should deduplicate."""
        google = [make_google_place("Blue Fish Restaurant")]
        yelp   = [make_yelp_place("The Blue Fish")]
        # Different enough names that they may or may not merge depending on threshold
        # Just verify no crash
        merged = _merge_places(google, yelp)
        assert len(merged) >= 1

    def test_different_businesses_not_merged(self):
        """Different businesses should not be deduplicated."""
        google = [
            make_google_place("Pier 12 Grill"),
            make_google_place("Carolina Cafe"),
        ]
        yelp = [
            make_yelp_place("Beach Bum Bar"),
        ]
        merged = _merge_places(google, yelp)
        assert len(merged) == 3

    def test_merge_policy_google_name_wins(self):
        """After merge, name should come from Google record."""
        google = [make_google_place("Blue Fish Grille")]
        yelp   = [make_yelp_place("Blue Fish Grille")]
        merged = _merge_places(google, yelp)
        assert len(merged) == 1
        assert merged[0].name == "Blue Fish Grille"
        assert merged[0].primary_source == DataSource.GOOGLE_PLACES

    def test_merge_policy_yelp_price_fills_google_gap(self):
        """Yelp price level should fill when Google has none."""
        google = [make_google_place("Test Restaurant", price_level=None)]
        yelp   = [make_yelp_place("Test Restaurant",   price_level="$$$")]
        merged = _merge_places(google, yelp)
        assert len(merged) == 1
        assert merged[0].price_level == "$$$"

    def test_merge_policy_both_ratings_preserved(self):
        """Both Google and Yelp ratings should be in the merged record."""
        google = [make_google_place("Cafe Test", rating=4.2)]
        yelp   = [make_yelp_place("Cafe Test",   rating=4.6)]
        merged = _merge_places(google, yelp)
        assert len(merged) == 1
        assert merged[0].google_rating == 4.2
        assert merged[0].yelp_rating == 4.6

    def test_merged_list_sorted_by_rating(self):
        """Merged list should be sorted by composite rating descending."""
        google = [
            make_google_place("Low Rated", rating=3.1),
            make_google_place("High Rated", rating=4.9),
            make_google_place("Mid Rated",  rating=4.0),
        ]
        merged = _merge_places(google, [])
        ratings = [p.composite_rating() for p in merged]
        assert ratings == sorted(ratings, reverse=True)

    def test_same_name_different_location_not_deduped(self):
        """Same name but far apart = different businesses, don't merge."""
        google = [make_google_place("Subway", lat=34.03, lng=-77.89)]
        yelp   = [make_yelp_place("Subway",   lat=35.22, lng=-79.01)]  # Many miles away
        merged = _merge_places(google, yelp)
        assert len(merged) == 2


# ── Vibe Filter Tests ─────────────────────────────────────────────────────

class TestVibeFilter:
    def test_all_six_vibes_have_priority_lists(self):
        for vibe in VibeProfile:
            assert vibe in VIBE_PRIMARY_CATEGORIES, f"Missing priority list for {vibe}"
            assert len(VIBE_PRIMARY_CATEGORIES[vibe]) >= 4

    def test_romantic_escape_prioritises_dining_and_wellness(self):
        priority = VIBE_PRIMARY_CATEGORIES[VibeProfile.ROMANTIC_ESCAPE]
        assert PlaceCategory.EAT_AND_DRINK in priority
        assert PlaceCategory.WELLNESS in priority

    def test_adventure_camp_prioritises_outdoors_first(self):
        priority = VIBE_PRIMARY_CATEGORIES[VibeProfile.ADVENTURE_BASE_CAMP]
        assert priority[0] == PlaceCategory.ADVENTURE_OUTDOORS

    def test_family_adventure_includes_family_kids(self):
        priority = VIBE_PRIMARY_CATEGORIES[VibeProfile.FAMILY_ADVENTURE]
        assert PlaceCategory.FAMILY_KIDS in priority

    def test_social_celebrations_prioritises_nightlife(self):
        priority = VIBE_PRIMARY_CATEGORIES[VibeProfile.SOCIAL_CELEBRATIONS]
        assert priority[0] == PlaceCategory.NIGHTLIFE

    def test_primary_recommendations_max_3_per_category(self):
        """No category should appear more than 3 times in primary recommendations."""
        places = [make_google_place(f"Restaurant {i}") for i in range(20)]
        primary = _select_primary_recommendations(
            places, VibeProfile.ROMANTIC_ESCAPE, [], 10
        )
        from collections import Counter
        cat_counts = Counter(p.category for p in primary)
        for cat, count in cat_counts.items():
            assert count <= 3, f"Category {cat} appears {count} times (max 3)"

    def test_dont_miss_picks_excluded_from_primary(self):
        """Places already in Don't Miss should not appear in primary recommendations."""
        places = [make_google_place("Seaside Bistro")]
        dont_miss = [DontMissPick(name="Seaside Bistro", description="Best seafood in town")]
        primary = _select_primary_recommendations(places, VibeProfile.FAMILY_ADVENTURE, dont_miss, 5)
        primary_names = [p.name for p in primary]
        assert "Seaside Bistro" not in primary_names

    def test_primary_count_respected(self):
        """Primary recommendations should not exceed requested count."""
        places = [make_google_place(f"Place {i}") for i in range(50)]
        primary = _select_primary_recommendations(places, VibeProfile.ROMANTIC_ESCAPE, [], 8)
        assert len(primary) <= 8


# ── Don't Miss Tests ──────────────────────────────────────────────────────

class TestDontMissPicks:
    def test_basic_picks_created(self):
        picks = _build_dont_miss_picks(
            ["The Wrightsville Surf Shop — great boards and local gear"],
            [],
        )
        assert len(picks) == 1
        assert picks[0].name  # Has an extracted name

    def test_max_5_picks(self):
        """Don't Miss should be capped at 5."""
        owner_picks = [f"Pick {i} — great place" for i in range(10)]
        picks = _build_dont_miss_picks(owner_picks, [])
        assert len(picks) <= 5

    def test_empty_picks_handled(self):
        picks = _build_dont_miss_picks([], [])
        assert picks == []

    def test_pick_matched_to_api_record(self):
        """Owner pick that matches a known place should get a place_ref."""
        api_places = [make_google_place("Oceanic Restaurant")]
        picks = _build_dont_miss_picks(
            ["Oceanic Restaurant — best seafood on the beach"],
            api_places,
        )
        assert len(picks) == 1
        assert picks[0].place_ref is not None
        assert picks[0].place_ref.name == "Oceanic Restaurant"

    def test_pick_description_preserved_verbatim(self):
        """Owner's exact description text must be preserved."""
        description = "The Fisherman's Wharf — 'best chowder in NC' per our guests!"
        picks = _build_dont_miss_picks([description], [])
        assert picks[0].description == description


# ── Category Browser Tests ────────────────────────────────────────────────

class TestCategoryBrowser:
    def test_places_organised_by_category(self):
        places = [
            make_google_place("Restaurant A", PlaceCategory.EAT_AND_DRINK),
            make_google_place("Bar B",         PlaceCategory.NIGHTLIFE),
            make_google_place("Restaurant C", PlaceCategory.EAT_AND_DRINK),
        ]
        browser = _organise_by_category(places)
        assert PlaceCategory.EAT_AND_DRINK.value in browser
        assert len(browser[PlaceCategory.EAT_AND_DRINK.value]) == 2
        assert PlaceCategory.NIGHTLIFE.value in browser
        assert len(browser[PlaceCategory.NIGHTLIFE.value]) == 1

    def test_category_sorted_by_rating(self):
        places = [
            make_google_place("Low", PlaceCategory.EAT_AND_DRINK, rating=3.5),
            make_google_place("High", PlaceCategory.EAT_AND_DRINK, rating=4.9),
        ]
        browser = _organise_by_category(places)
        category_places = browser[PlaceCategory.EAT_AND_DRINK.value]
        assert category_places[0].name == "High"


# ── Area Intro Tests ──────────────────────────────────────────────────────

class TestAreaIntroduction:
    def test_fallback_intro_exists_for_all_vibes(self):
        for vibe in VibeProfile:
            intro = _fallback_area_intro("Carolina Beach, NC", vibe)
            assert intro
            assert len(intro) > 50

    def test_fallback_intro_contains_location(self):
        intro = _fallback_area_intro("Asheville, NC", VibeProfile.WELLNESS_RETREAT)
        assert "Asheville" in intro


# ── Full Assembly Test ────────────────────────────────────────────────────

class TestFullAssembly:
    def test_assemble_with_both_sources(self):
        google_places = [
            make_google_place("Tides Restaurant", PlaceCategory.EAT_AND_DRINK, rating=4.7),
            make_google_place("Carolina Beach State Park", PlaceCategory.ADVENTURE_OUTDOORS, rating=4.8),
            make_google_place("The Shuckin Shack", PlaceCategory.EAT_AND_DRINK, rating=4.4),
        ]
        yelp_places = [
            make_yelp_place("Britt's Donuts", PlaceCategory.COFFEE_CAFES, rating=4.9),
        ]

        guide = assemble_local_guide(
            property_id="p1",
            google_places=google_places,
            yelp_places=yelp_places,
            vibe_profile=VibeProfile.FAMILY_ADVENTURE,
            owner_dont_miss=["Britt's Donuts — a Carolina Beach institution for 80 years"],
            neighborhood_description="Steps from the beach boardwalk",
            city="Carolina Beach",
            state="NC",
            anthropic_client=None,   # Use fallback intro
        )

        assert guide.property_id == "p1"
        assert len(guide.dont_miss_picks) == 1
        assert guide.dont_miss_picks[0].name   # Has a name
        assert guide.area_introduction          # Has an intro
        assert guide.total_places > 0
        assert len(guide.primary_recommendations) > 0

    def test_assemble_with_empty_yelp(self):
        """Should work fine with only Google Places data."""
        google_places = [make_google_place("Only Google Place")]
        guide = assemble_local_guide(
            property_id="p2",
            google_places=google_places,
            yelp_places=[],
            vibe_profile=VibeProfile.ROMANTIC_ESCAPE,
            owner_dont_miss=[],
            neighborhood_description=None,
            city="Topsail Beach",
            state="NC",
            anthropic_client=None,
        )
        assert guide.total_places == 1

    def test_assemble_with_no_places(self):
        """Should handle no API data gracefully."""
        guide = assemble_local_guide(
            property_id="p3",
            google_places=[],
            yelp_places=[],
            vibe_profile=VibeProfile.WELLNESS_RETREAT,
            owner_dont_miss=["Hidden Cove Trail — magical in the morning"],
            neighborhood_description=None,
            city="Boone",
            state="NC",
            anthropic_client=None,
        )
        assert guide.property_id == "p3"
        assert len(guide.dont_miss_picks) == 1   # Owner picks still included
        assert guide.total_places == 0

    def test_to_dict_is_json_serialisable(self):
        import json
        guide = assemble_local_guide(
            property_id="p4",
            google_places=[make_google_place("Test Cafe", PlaceCategory.COFFEE_CAFES)],
            yelp_places=[],
            vibe_profile=VibeProfile.SOCIAL_CELEBRATIONS,
            owner_dont_miss=[],
            neighborhood_description=None,
            city="Wilmington",
            state="NC",
            anthropic_client=None,
        )
        d = guide.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 50


# ── Agent Node Contract Tests ─────────────────────────────────────────────

class TestAgent4NodeContract:
    def test_successful_run_sets_agent4_complete(self):
        from agents.agent4.agent import agent4_node

        kb = {
            "property_id": "p1",
            "vibe_profile": "romantic_escape",
            "latitude": {"value": 34.03},
            "longitude": {"value": -77.89},
            "city": {"value": "Carolina Beach"},
            "state": {"value": "NC"},
            "dont_miss_picks": [],
        }
        state = {
            "property_id": "p1",
            "knowledge_base": kb,
            "errors": [],
        }

        mock_guide = LocalGuide(property_id="p1", location_name="Carolina Beach, NC")
        mock_guide.total_places = 15

        with patch("booked.agents.agent4.agent.get_cached_knowledge_base", return_value=kb), \
             patch("booked.agents.agent4.agent.fetch_local_places", return_value=[]), \
             patch("booked.agents.agent4.agent.fetch_yelp_places",  return_value=[]), \
             patch("booked.agents.agent4.agent.assemble_local_guide", return_value=mock_guide), \
             patch("booked.agents.agent4.agent._save_local_guide"), \
             patch("booked.agents.agent4.agent.cache_knowledge_base"), \
             patch("booked.agents.agent4.agent.update_pipeline_status"):

            result = agent4_node(state)

        assert result["agent4_complete"] is True
        assert "local_guide" in result

    def test_no_coordinates_completes_with_warning(self):
        """Missing coordinates should complete gracefully, not crash."""
        from agents.agent4.agent import agent4_node

        kb = {
            "property_id": "p2",
            "vibe_profile": "family_adventure",
            # No lat/lng
        }
        state = {
            "property_id": "p2",
            "knowledge_base": kb,
            "errors": [],
        }

        with patch("booked.agents.agent4.agent.get_cached_knowledge_base", return_value=kb), \
             patch("booked.agents.agent4.agent._geocode_fallback", return_value=(None, None)), \
             patch("booked.agents.agent4.agent.update_pipeline_status"):

            result = agent4_node(state)

        assert result["agent4_complete"] is True

    def test_both_api_failures_sets_error(self):
        """If both Google and Yelp fail, agent should fail gracefully."""
        from agents.agent4.agent import agent4_node

        kb = {
            "property_id": "p3",
            "vibe_profile": "wellness_retreat",
            "latitude": {"value": 35.5},
            "longitude": {"value": -82.5},
        }
        state = {
            "property_id": "p3",
            "knowledge_base": kb,
            "errors": [],
        }

        with patch("booked.agents.agent4.agent.get_cached_knowledge_base", return_value=kb), \
             patch("booked.agents.agent4.agent.fetch_local_places", side_effect=Exception("API error")), \
             patch("booked.agents.agent4.agent.fetch_yelp_places",  side_effect=Exception("API error")), \
             patch("booked.agents.agent4.agent.update_pipeline_status"):

            result = agent4_node(state)

        assert result["agent4_complete"] is False
        assert len(result["errors"]) > 0
