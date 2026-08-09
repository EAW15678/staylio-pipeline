"""
Tests for Agent 8 Stage 2 concept generator.

Runs dry_run against Vista Azule (a1b2c3d4-0001-0001-0001-000000000001,
vibe multigenerational_retreat) with a mocked Claude response.
"""

import json
from datetime import date
from unittest.mock import patch, MagicMock

import pytest

from agents.agent8.concept_generator import (
    generate_concepts,
    _build_prompt,
    _build_utm_content_slug,
    _extract_amenity_names,
)

VISTA_AZULE_ID = "a1b2c3d4-0001-0001-0001-000000000001"

MOCK_KB = {
    "name": {"value": "Vista Azule", "source": "intake_portal", "confidence": 1.0},
    "slug": "vista-azule",
    "vibe_profile": "multigenerational_retreat",
    "city": {"value": "Carolina Beach", "source": "vrbo", "confidence": 0.8},
    "state": {"value": "NC", "source": "vrbo", "confidence": 0.8},
    "bedrooms": {"value": 5, "source": "airbnb", "confidence": 0.8},
    "bathrooms": {"value": 4.5, "source": "airbnb", "confidence": 0.8},
    "owner_story": "We bought this property to bring our extended family together.",
    "wow_factor": "The copper soaking tub on the deck with mountain views.",
    "hidden_gems": "A path behind the barn leads to a waterfall.",
    "amenities": [
        {"value": "Hot tub", "source": "airbnb", "confidence": 0.8},
        {"value": "Private pool", "source": "vrbo", "confidence": 0.8},
        {"value": "Fire pit", "source": "airbnb", "confidence": 0.8},
        {"value": "Game room", "source": "vrbo", "confidence": 0.8},
        {"value": "Chef's kitchen", "source": "vrbo", "confidence": 0.8},
        {"value": "Fast WiFi (200+ Mbps)", "source": "airbnb", "confidence": 0.8},
    ],
    "guest_reviews": [
        {
            "text": "My parents, my sister's family, and our kids — nobody felt like they got the bad room.",
            "source": "guest_book",
            "reviewer_name": "The Williamson Family",
            "stay_date": "August 2024",
            "is_guest_book": True,
        },
        {
            "text": "Best vacation we've ever had as a family. The pool was perfect.",
            "source": "airbnb",
            "reviewer_name": "Sarah",
            "stay_date": "July 2024",
            "is_guest_book": False,
        },
    ],
}

MOCK_CLAUDE_RESPONSE = {
    "concepts": [
        {
            "title": "Three generations, one sunset",
            "premise": "A multi-generational family watches the sunset from the deck together. The copper tub and fire pit create separate but connected moments for different age groups.",
            "source_material": ["wow_factor", "amenity: Fire pit"],
        },
        {
            "title": "The room equity promise",
            "premise": "Every family gets a wing they're proud of. No one draws the short straw on bedrooms — the layout was designed so each group feels the arrangement is fair.",
            "source_material": ["guest_book: The Williamson Family"],
        },
        {
            "title": "Chef's kitchen, family recipe",
            "premise": "Grandma's recipe, the chef's kitchen, and three generations crowded around the island. The kitchen is designed for group cooking, not just reheating.",
            "source_material": ["amenity: Chef's kitchen"],
        },
        {
            "title": "The waterfall discovery",
            "premise": "Around day two, someone always finds the path behind the barn. The waterfall becomes the highlight — a discovery that belongs to this trip alone.",
            "source_material": ["hidden_gems"],
        },
        {
            "title": "Game room after dark",
            "premise": "After the kids are down, the adults take over the game room. Pool, cards, conversation — the rare vacation where parents actually get an evening.",
            "source_material": ["amenity: Game room"],
        },
        {
            "title": "The pool that works for everyone",
            "premise": "Ages 5 to 75, everyone in the pool. The private pool is the great equalizer — no reservations, no time slots, no strangers.",
            "source_material": ["amenity: Private pool"],
        },
        {
            "title": "WiFi strong enough for remote work",
            "premise": "One parent works remotely for two days while the family vacations. The 200+ Mbps WiFi means they're productive without missing anything.",
            "source_material": ["amenity: Fast WiFi (200+ Mbps)", "owner_story"],
        },
        {
            "title": "Nobody wanted to leave",
            "premise": "The last morning is always the hardest. Three families pack up slowly, already talking about next year's dates.",
            "source_material": [],
        },
    ]
}

# Prohibited shot-spec language
SHOT_SPEC_TERMS = [
    "pan", "zoom", "dolly", "tracking shot", "close-up", "wide shot",
    "b-roll", "transition", "fade", "cut to", "montage", "drone shot",
    "timelapse", "slow motion", "clip", "footage", "music",
]


class TestConceptGenerator:
    """Test generate_concepts in dry_run mode with mocked KB and Claude."""

    def _mock_generate(self):
        """Run generate_concepts with mocked dependencies."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = json.dumps(MOCK_CLAUDE_RESPONSE)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch(
            "agents.agent8.concept_generator._load_kb",
            return_value=MOCK_KB,
        ), patch(
            "agents.agent8.concept_generator._load_kb_from_supabase",
            return_value=None,
        ), patch.dict(
            "os.environ", {"ANTHROPIC_API_KEY": "test-key"},
        ), patch(
            "agents.agent8.concept_generator.anthropic.Anthropic",
            return_value=mock_client,
        ):
            return generate_concepts(
                VISTA_AZULE_ID,
                date(2026, 9, 1),
                dry_run=True,
            )

    def test_returns_8_concepts(self):
        concepts = self._mock_generate()
        assert len(concepts) == 8

    def test_all_titles_distinct(self):
        concepts = self._mock_generate()
        titles = [c["title"] for c in concepts]
        assert len(set(titles)) == 8, f"Duplicate titles found: {titles}"

    def test_no_fabricated_amenities(self):
        """Verify source_material only references amenities from the KB."""
        kb_amenities = {a["value"].lower() for a in MOCK_KB["amenities"]}
        concepts = self._mock_generate()
        for c in concepts:
            for ref in c.get("source_material", []):
                if ref.startswith("amenity: "):
                    amenity_name = ref[len("amenity: "):]
                    assert amenity_name.lower() in kb_amenities, (
                        f"Fabricated amenity '{amenity_name}' in concept '{c['title']}'. "
                        f"KB amenities: {kb_amenities}"
                    )

    def test_no_shot_spec_language(self):
        """Verify concepts contain no camera/production terminology."""
        concepts = self._mock_generate()
        for c in concepts:
            combined = f"{c['title']} {c['premise']}".lower()
            for term in SHOT_SPEC_TERMS:
                assert term not in combined, (
                    f"Shot-spec term '{term}' found in concept '{c['title']}'"
                )

    def test_utm_content_slug_format(self):
        concepts = self._mock_generate()
        for c in concepts:
            slug = c["utm_content_slug"]
            assert slug.startswith("vista-azule_2026-09_c"), (
                f"Unexpected slug format: {slug}"
            )

    def test_utm_content_slug_deterministic(self):
        """Same inputs produce same slug."""
        s1 = _build_utm_content_slug("vista-azule", date(2026, 9, 1), 3)
        s2 = _build_utm_content_slug("vista-azule", date(2026, 9, 1), 3)
        assert s1 == s2 == "vista-azule_2026-09_c3"

    def test_vibe_prior_set(self):
        concepts = self._mock_generate()
        for c in concepts:
            assert c["vibe_prior"] == "multigenerational_retreat"

    def test_status_is_draft(self):
        concepts = self._mock_generate()
        for c in concepts:
            assert c["status"] == "draft"

    def test_created_by_agent(self):
        concepts = self._mock_generate()
        for c in concepts:
            assert c["created_by_agent"] == "agent8_stage2"

    def test_concept_numbers_1_through_8(self):
        concepts = self._mock_generate()
        numbers = sorted(c["concept_number"] for c in concepts)
        assert numbers == list(range(1, 9))


class TestAmenityExtraction:
    def test_extracts_from_property_field_dicts(self):
        names = _extract_amenity_names(MOCK_KB)
        assert "Hot tub" in names
        assert "Private pool" in names
        assert len(names) == 6

    def test_handles_plain_strings(self):
        kb = {"amenities": ["Pool", "WiFi"]}
        names = _extract_amenity_names(kb)
        assert names == ["Pool", "WiFi"]

    def test_handles_empty(self):
        assert _extract_amenity_names({}) == []


class TestPromptConstruction:
    def test_prompt_contains_amenities(self):
        prompt = _build_prompt(MOCK_KB, date(2026, 9, 1))
        assert "Hot tub" in prompt
        assert "Private pool" in prompt
        assert "Chef's kitchen" in prompt

    def test_prompt_contains_guest_book_verbatim(self):
        prompt = _build_prompt(MOCK_KB, date(2026, 9, 1))
        assert "nobody felt like they got the bad room" in prompt

    def test_prompt_contains_hard_rules(self):
        prompt = _build_prompt(MOCK_KB, date(2026, 9, 1))
        assert "GUEST-BOOK TEXT IS VERBATIM" in prompt
        assert "NO FABRICATED AMENITIES" in prompt
        assert "NO SHOT SPEC" in prompt
        assert "NO OTA LINKS" in prompt

    def test_prompt_contains_vibe_as_prior(self):
        prompt = _build_prompt(MOCK_KB, date(2026, 9, 1))
        assert "PRIOR, NOT A TEMPLATE" in prompt
