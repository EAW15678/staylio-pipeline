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
    _supersede_and_insert,
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


# ── Version-bump tests (exercises _supersede_and_insert with mock DB) ───

class MockSupabaseTable:
    """
    In-memory mock of a Supabase table for testing the supersede-and-insert
    path. No real DB-backed test infrastructure exists in this repo — this
    simulates the Supabase client's fluent API against an in-memory list.
    """

    def __init__(self):
        self._rows: list[dict] = []
        self._query_chain: dict = {}

    def table(self, name: str):
        self._query_chain = {}
        return self

    def select(self, cols: str, count: str = None):
        self._query_chain["select"] = cols
        return self

    def eq(self, col: str, val):
        self._query_chain.setdefault("filters", []).append((col, "eq", val))
        return self

    def is_(self, col: str, val: str):
        self._query_chain.setdefault("filters", []).append((col, "is", val))
        return self

    def in_(self, col: str, vals: list):
        self._query_chain.setdefault("filters", []).append((col, "in", vals))
        return self

    def insert(self, rows):
        if isinstance(rows, list):
            import uuid
            for r in rows:
                r.setdefault("concept_id", str(uuid.uuid4()))
                self._rows.append(dict(r))
        else:
            import uuid
            rows.setdefault("concept_id", str(uuid.uuid4()))
            self._rows.append(dict(rows))
        return self

    def update(self, data: dict):
        self._query_chain["update_data"] = data
        return self

    def execute(self):
        filters = self._query_chain.get("filters", [])
        update_data = self._query_chain.get("update_data")

        if update_data:
            # Apply update to matching rows
            for row in self._rows:
                if self._matches(row, filters):
                    row.update(update_data)
            result = MagicMock()
            result.data = []
            self._query_chain = {}
            return result

        # Select
        matched = [r for r in self._rows if self._matches(r, filters)]
        result = MagicMock()
        result.data = matched
        result.count = len(matched)
        self._query_chain = {}
        return result

    def _matches(self, row: dict, filters: list) -> bool:
        for col, op, val in filters:
            rv = row.get(col)
            if op == "eq" and rv != val:
                return False
            if op == "is" and val == "null" and rv is not None:
                return False
            if op == "in" and rv not in val:
                return False
        return True

    def get_active_rows(self, property_id: str, cycle_month: str) -> list[dict]:
        """Helper: get active (non-superseded) rows for a property+cycle."""
        return [
            r for r in self._rows
            if r.get("property_id") == property_id
            and r.get("cycle_month") == cycle_month
            and r.get("superseded_at") is None
        ]


class TestVersionBump:
    """
    Tests the supersede-and-insert path using an in-memory mock of the
    Supabase client. No real DB connection — the repo has no DB-backed
    test infrastructure.
    """

    def _make_rows(self, version: int = 1) -> list[dict]:
        """Build 8 concept rows for testing."""
        return [
            {
                "property_id": VISTA_AZULE_ID,
                "cycle_month": "2026-09-01",
                "concept_number": cn,
                "concept_version": version,
                "utm_content_slug": f"vista-azule_2026-09_c{cn}",
                "title": f"Concept {cn} v{version}",
                "premise": f"Premise for concept {cn}",
                "vibe_prior": "multigenerational_retreat",
                "evidence_basis": [],
                "source_material": [],
                "target_platforms": ["tiktok", "pinterest", "facebook", "instagram"],
                "status": "draft",
                "created_by_agent": "agent8_stage2",
                "superseded_at": None,
            }
            for cn in range(1, 9)
        ]

    def test_first_run_inserts_8_at_version_1(self):
        sb = MockSupabaseTable()
        rows = self._make_rows(version=1)
        result = _supersede_and_insert(sb, VISTA_AZULE_ID, date(2026, 9, 1), rows)
        assert len(result) == 8
        assert all(r["concept_version"] == 1 for r in result)
        active = sb.get_active_rows(VISTA_AZULE_ID, "2026-09-01")
        assert len(active) == 8

    def test_second_run_supersedes_v1_inserts_v2(self):
        sb = MockSupabaseTable()

        # First run
        rows_v1 = self._make_rows(version=1)
        _supersede_and_insert(sb, VISTA_AZULE_ID, date(2026, 9, 1), rows_v1)

        # Second run
        rows_v2 = self._make_rows(version=1)  # version will be overridden
        result = _supersede_and_insert(sb, VISTA_AZULE_ID, date(2026, 9, 1), rows_v2)

        # V2 rows inserted at version 2
        assert all(r["concept_version"] == 2 for r in result)

        # Exactly 8 active rows (v2), 8 superseded (v1)
        active = sb.get_active_rows(VISTA_AZULE_ID, "2026-09-01")
        assert len(active) == 8
        assert all(r["concept_version"] == 2 for r in active)

        superseded = [
            r for r in sb._rows
            if r["property_id"] == VISTA_AZULE_ID
            and r["superseded_at"] is not None
        ]
        assert len(superseded) == 8
        assert all(r["concept_version"] == 1 for r in superseded)

    def test_slug_identical_across_versions(self):
        sb = MockSupabaseTable()

        rows_v1 = self._make_rows(version=1)
        _supersede_and_insert(sb, VISTA_AZULE_ID, date(2026, 9, 1), rows_v1)

        rows_v2 = self._make_rows(version=1)
        result_v2 = _supersede_and_insert(sb, VISTA_AZULE_ID, date(2026, 9, 1), rows_v2)

        # Slugs must be identical between v1 and v2 for same concept_number
        for cn in range(1, 9):
            expected_slug = f"vista-azule_2026-09_c{cn}"
            v2_row = next(r for r in result_v2 if r["concept_number"] == cn)
            assert v2_row["utm_content_slug"] == expected_slug

    def test_third_run_produces_version_3(self):
        sb = MockSupabaseTable()

        # Three sequential runs
        for run in range(3):
            rows = self._make_rows(version=1)
            result = _supersede_and_insert(sb, VISTA_AZULE_ID, date(2026, 9, 1), rows)

        # Latest version is 3
        assert all(r["concept_version"] == 3 for r in result)

        # Exactly 8 active rows
        active = sb.get_active_rows(VISTA_AZULE_ID, "2026-09-01")
        assert len(active) == 8
        assert all(r["concept_version"] == 3 for r in active)

        # 16 superseded rows (v1 + v2)
        superseded = [
            r for r in sb._rows
            if r["property_id"] == VISTA_AZULE_ID
            and r["superseded_at"] is not None
        ]
        assert len(superseded) == 16

        # Slugs still stable
        for cn in range(1, 9):
            slug = f"vista-azule_2026-09_c{cn}"
            active_row = next(r for r in active if r["concept_number"] == cn)
            assert active_row["utm_content_slug"] == slug

    def test_partial_unique_one_active_per_slot(self):
        """concept_ledger_one_active: only one active row per (property, month, number)."""
        sb = MockSupabaseTable()

        _supersede_and_insert(sb, VISTA_AZULE_ID, date(2026, 9, 1), self._make_rows())
        _supersede_and_insert(sb, VISTA_AZULE_ID, date(2026, 9, 1), self._make_rows())

        active = sb.get_active_rows(VISTA_AZULE_ID, "2026-09-01")
        # Check uniqueness: no two active rows share the same concept_number
        active_numbers = [r["concept_number"] for r in active]
        assert len(active_numbers) == len(set(active_numbers)) == 8

    def test_partial_unique_slug_active(self):
        """concept_ledger_utm_slug_active: only one active row per slug."""
        sb = MockSupabaseTable()

        _supersede_and_insert(sb, VISTA_AZULE_ID, date(2026, 9, 1), self._make_rows())
        _supersede_and_insert(sb, VISTA_AZULE_ID, date(2026, 9, 1), self._make_rows())

        active = sb.get_active_rows(VISTA_AZULE_ID, "2026-09-01")
        active_slugs = [r["utm_content_slug"] for r in active]
        assert len(active_slugs) == len(set(active_slugs)) == 8
