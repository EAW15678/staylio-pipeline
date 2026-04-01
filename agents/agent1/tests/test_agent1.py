"""
Agent 1 Test Suite
Run with: pytest booked/agents/agent1/tests/ -v

Tests cover:
  - Knowledge base merge policy (intake data wins)
  - URL platform detection
  - Slug generation
  - Firecrawl response mapping
  - Apify response mapping (Airbnb, VRBO)
  - Scraper error handling (graceful degradation)
  - Agent node output contract
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from models.property import (
    ClientChannel,
    DataSource,
    GuestReview,
    PhotoAsset,
    PropertyField,
    PropertyKnowledgeBase,
    VibeProfile,
)
from agents.agent1.apify_scraper import detect_ota_platform
from agents.agent1.agent import _generate_slug


# ── Fixtures ──────────────────────────────────────────────────────────────

def make_kb(
    property_id: str = "prop-001",
    client_channel: ClientChannel = ClientChannel.PMC,
    with_intake_name: bool = False,
) -> PropertyKnowledgeBase:
    """Create a minimal knowledge base for testing."""
    kb = PropertyKnowledgeBase(
        property_id=property_id,
        client_id="client-001",
        client_channel=client_channel,
    )
    if with_intake_name:
        kb.name = PropertyField(
            value="Vista Azule",
            source=DataSource.INTAKE_PORTAL,
            confidence=1.0,
        )
        kb.vibe_profile = VibeProfile.FAMILY_ADVENTURE
        kb.booking_url = "https://pmc.example.com/book/vista-azule"
    return kb


# ── Merge Policy Tests ────────────────────────────────────────────────────

class TestMergePolicy:
    def test_intake_wins_over_scrape(self):
        """Intake portal data must always win regardless of confidence."""
        kb = make_kb()
        intake_field = PropertyField("Intake Name", DataSource.INTAKE_PORTAL, confidence=1.0)
        scrape_field = PropertyField("Scraped Name", DataSource.AIRBNB, confidence=0.99)

        kb.name = intake_field
        result = kb.merge_field(kb.name, scrape_field)
        assert result.value == "Intake Name"
        assert result.source == DataSource.INTAKE_PORTAL

    def test_scrape_fills_empty_field(self):
        """If no intake data, scraped data should fill the field."""
        kb = make_kb()
        assert kb.name is None

        scrape_field = PropertyField("Scraped Name", DataSource.AIRBNB, confidence=0.80)
        result = kb.merge_field(kb.name, scrape_field)
        assert result.value == "Scraped Name"

    def test_higher_confidence_scrape_wins(self):
        """Among scraped sources, higher confidence should win."""
        kb = make_kb()
        firecrawl_field = PropertyField("Firecrawl Name", DataSource.PMC_WEBSITE, confidence=0.85)
        claude_field = PropertyField("Claude Name", DataSource.CLAUDE_PARSED, confidence=0.70)

        kb.name = firecrawl_field
        result = kb.merge_field(kb.name, claude_field)
        assert result.value == "Firecrawl Name"   # firecrawl has higher confidence

    def test_intake_wins_over_high_confidence_scrape(self):
        """Even low-confidence intake beats high-confidence scrape."""
        kb = make_kb()
        intake_field = PropertyField("Intake Name", DataSource.INTAKE_PORTAL, confidence=0.5)
        scrape_field = PropertyField("Scrape Name", DataSource.AIRBNB, confidence=0.99)

        kb.name = intake_field
        result = kb.merge_field(kb.name, scrape_field)
        assert result.value == "Intake Name"

    def test_none_incoming_returns_current(self):
        """merge_field with None incoming should return current unchanged."""
        kb = make_kb(with_intake_name=True)
        current = kb.name
        result = kb.merge_field(current, None)
        assert result.value == "Vista Azule"

    def test_both_none_returns_none(self):
        """merge_field with both None should return None."""
        kb = make_kb()
        result = kb.merge_field(None, None)
        assert result is None


# ── Platform Detection Tests ─────────────────────────────────────────────

class TestPlatformDetection:
    def test_airbnb_detection(self):
        urls = [
            "https://www.airbnb.com/rooms/12345678",
            "https://airbnb.com/rooms/87654321",
            "http://www.airbnb.com/rooms/555",
        ]
        for url in urls:
            assert detect_ota_platform(url) == "airbnb", f"Failed for {url}"

    def test_vrbo_detection(self):
        urls = [
            "https://www.vrbo.com/123456ha",
            "https://vrbo.com/listings/123",
            "https://www.homeaway.com/vacation-rental/123",
        ]
        for url in urls:
            assert detect_ota_platform(url) in ("vrbo",), f"Failed for {url}"

    def test_booking_com_detection(self):
        url = "https://www.booking.com/hotel/gb/property.html"
        assert detect_ota_platform(url) == "booking_com"

    def test_unknown_platform(self):
        urls = [
            "https://pmc.example.com/property/vista-azule",
            "https://www.vacasa.com/property/123",
            "",
        ]
        for url in urls:
            result = detect_ota_platform(url)
            assert result is None or result == "unknown", f"Unexpected result for {url}: {result}"

    def test_none_url(self):
        assert detect_ota_platform(None) is None


# ── Slug Generation Tests ─────────────────────────────────────────────────

class TestSlugGeneration:
    def test_basic_slug(self):
        assert _generate_slug("Vista Azule") == "vista-azule"

    def test_slug_with_special_chars(self):
        assert _generate_slug("The Cabin @ Blue Ridge!") == "the-cabin--blue-ridge"

    def test_slug_max_length(self):
        long_name = "A Very Long Property Name That Exceeds The Maximum Slug Length For Subdomains"
        slug = _generate_slug(long_name)
        assert len(slug) <= 60

    def test_slug_lowercase(self):
        assert _generate_slug("OCEAN VIEW VILLA") == "ocean-view-villa"

    def test_slug_trims_hyphens(self):
        slug = _generate_slug("  ---Property Name---  ")
        assert not slug.startswith("-")
        assert not slug.endswith("-")


# ── Firecrawl Response Mapping Tests ─────────────────────────────────────

class TestFirecrawlMapping:
    def test_maps_basic_fields(self):
        """Firecrawl response fields should map to correct KB fields."""
        from agents.agent1.firecrawl_scraper import scrape_pmc_website

        mock_response = {
            "success": True,
            "data": {
                "property_name": "Sunrise Cottage",
                "headline": "Your perfect beach escape",
                "description": "A beautiful 3BR cottage on the shore",
                "bedrooms": 3,
                "bathrooms": 2.5,
                "max_occupancy": 8,
                "property_type": "cottage",
                "amenities": ["Pool", "WiFi", "Hot Tub"],
                "city": "Carolina Beach",
                "state": "NC",
                "photo_urls": [
                    "https://example.com/photos/exterior.jpg",
                    "https://example.com/photos/kitchen.jpg",
                ],
            }
        }

        kb = make_kb()

        with patch("booked.agents.agent1.firecrawl_scraper._firecrawl_extract",
                   return_value=mock_response["data"]):
            result = scrape_pmc_website("https://pmc.example.com/property", kb)

        assert result.name.value == "Sunrise Cottage"
        assert result.bedrooms.value == 3
        assert result.bathrooms.value == 2.5
        assert result.city.value == "Carolina Beach"
        assert result.state.value == "NC"
        assert len(result.amenities) == 3
        assert len(result.photos) == 2
        assert result.name.source == DataSource.PMC_WEBSITE

    def test_intake_data_not_overwritten(self):
        """Firecrawl must not overwrite intake portal data."""
        from agents.agent1.firecrawl_scraper import scrape_pmc_website

        mock_response = {
            "property_name": "Scraped Name — Not The Intake Name",
            "bedrooms": 99,  # Wrong — intake says 4
        }

        kb = make_kb(with_intake_name=True)
        kb.bedrooms = PropertyField(4, DataSource.INTAKE_PORTAL, confidence=1.0)

        with patch("booked.agents.agent1.firecrawl_scraper._firecrawl_extract",
                   return_value=mock_response):
            result = scrape_pmc_website("https://pmc.example.com/property", kb)

        # Intake data must survive
        assert result.name.value == "Vista Azule"
        assert result.bedrooms.value == 4

    def test_graceful_failure_on_scrape_error(self):
        """Firecrawl failure should add to errors, not raise."""
        from agents.agent1.firecrawl_scraper import scrape_pmc_website

        kb = make_kb()

        with patch("booked.agents.agent1.firecrawl_scraper._firecrawl_extract",
                   side_effect=Exception("Connection timeout")):
            result = scrape_pmc_website("https://pmc.example.com/property", kb)

        assert len(result.ingestion_errors) > 0
        assert "Connection timeout" in result.ingestion_errors[0]
        # KB should be unchanged otherwise
        assert result.name is None


# ── Airbnb Response Mapping Tests ─────────────────────────────────────────

class TestAirbnbMapping:
    MOCK_LISTING = {
        "name": "Beachfront Bungalow",
        "description": "Steps from the shore",
        "bedrooms": 2,
        "bathrooms": 1,
        "personCapacity": 4,
        "roomType": "Entire home/apt",
        "stars": 4.87,
        "location": {"city": "Topsail Beach", "state": "NC", "zipCode": "28445"},
        "lat": 34.4265, "lng": -77.5961,
        "amenities": ["WiFi", "Kitchen", "Air conditioning"],
        "photos": [
            {"url": "https://a0.muscache.com/im/pictures/photo1.jpg"},
            {"url": "https://a0.muscache.com/im/pictures/photo2.jpg"},
        ],
        "reviews": [
            {
                "comments": "Amazing place! Will return.",
                "reviewer": {"firstName": "Sarah"},
                "createdAt": "2024-08-15",
                "rating": 5,
            }
        ],
    }

    def test_maps_all_core_fields(self):
        from agents.agent1.apify_scraper import _scrape_airbnb

        kb = make_kb(client_channel=ClientChannel.IO)

        with patch("booked.agents.agent1.apify_scraper._run_apify_actor",
                   return_value=[self.MOCK_LISTING]):
            result = _scrape_airbnb("https://airbnb.com/rooms/123", kb, scrape_reviews=True)

        assert result.name.value == "Beachfront Bungalow"
        assert result.bedrooms.value == 2
        assert result.city.value == "Topsail Beach"
        assert result.state.value == "NC"
        assert result.latitude.value == 34.4265
        assert len(result.amenities) == 3
        assert len(result.photos) == 2
        assert len(result.guest_reviews) == 1
        assert result.guest_reviews[0].reviewer_name == "Sarah"
        assert result.guest_reviews[0].is_guest_book is False
        assert result.guest_reviews[0].source == DataSource.AIRBNB

    def test_reviews_not_scraped_when_flag_false(self):
        from agents.agent1.apify_scraper import _scrape_airbnb

        kb = make_kb(client_channel=ClientChannel.IO)

        with patch("booked.agents.agent1.apify_scraper._run_apify_actor",
                   return_value=[self.MOCK_LISTING]):
            result = _scrape_airbnb("https://airbnb.com/rooms/123", kb, scrape_reviews=False)

        assert len(result.guest_reviews) == 0

    def test_empty_actor_response(self):
        from agents.agent1.apify_scraper import _scrape_airbnb

        kb = make_kb(client_channel=ClientChannel.IO)

        with patch("booked.agents.agent1.apify_scraper._run_apify_actor",
                   return_value=None):
            result = _scrape_airbnb("https://airbnb.com/rooms/123", kb, scrape_reviews=True)

        assert len(result.ingestion_errors) > 0
        assert result.name is None   # KB unchanged


# ── Knowledge Base Serialisation ─────────────────────────────────────────

class TestKBSerialisation:
    def test_to_dict_roundtrip(self):
        """to_dict should produce JSON-serialisable output."""
        kb = make_kb(with_intake_name=True)
        kb.bedrooms = PropertyField(4, DataSource.INTAKE_PORTAL)
        kb.photos.append(PhotoAsset(url="https://example.com/photo.jpg", source=DataSource.INTAKE_PORTAL))
        kb.guest_reviews.append(GuestReview(
            text="Perfect family vacation!",
            source=DataSource.INTAKE_PORTAL,
            is_guest_book=True,
        ))

        d = kb.to_dict()
        # Must be JSON serialisable
        json_str = json.dumps(d)
        assert len(json_str) > 0

        # Spot-check key fields
        assert d["property_id"] == "prop-001"
        assert d["name"]["value"] == "Vista Azule"
        assert d["name"]["source"] == DataSource.INTAKE_PORTAL
        assert d["bedrooms"]["value"] == 4
        assert len(d["photos"]) == 1
        assert len(d["guest_reviews"]) == 1
        assert d["guest_reviews"][0]["is_guest_book"] is True


# ── Agent Node Contract Tests ─────────────────────────────────────────────

class TestAgentNodeContract:
    """
    Tests that the LangGraph node returns the correct state shape
    and sets the correct flags for downstream agents.
    """

    def test_successful_ingestion_sets_flags(self):
        from agents.agent1.agent import agent1_node

        mock_kb = make_kb(with_intake_name=True)
        mock_kb.ingestion_complete = True

        initial_state = {
            "property_id": "prop-001",
            "client_id": "client-001",
            "knowledge_base_ready": False,
            "knowledge_base": {},
            "errors": [],
            "agent1_complete": False,
            "agent2_ready": False,
            "agent3_ready": False,
            "agent4_ready": False,
        }

        with patch("booked.agents.agent1.agent.load_intake_submission", return_value=mock_kb), \
             patch("booked.agents.agent1.agent._run_scrapers", return_value=mock_kb), \
             patch("booked.agents.agent1.agent.normalise_and_fill_gaps", return_value=mock_kb), \
             patch("booked.agents.agent1.agent.save_knowledge_base", return_value=True), \
             patch("booked.agents.agent1.agent.cache_knowledge_base"), \
             patch("booked.agents.agent1.agent.update_pipeline_status"):

            result = agent1_node(initial_state)

        assert result["agent1_complete"] is True
        assert result["knowledge_base_ready"] is True
        assert result["agent2_ready"] is True
        assert result["agent3_ready"] is True
        assert result["agent4_ready"] is True
        assert isinstance(result["knowledge_base"], dict)

    def test_missing_intake_sets_failure(self):
        from agents.agent1.agent import agent1_node

        initial_state = {
            "property_id": "nonexistent-prop",
            "client_id": "client-001",
            "knowledge_base_ready": False,
            "knowledge_base": {},
            "errors": [],
            "agent1_complete": False,
            "agent2_ready": False,
            "agent3_ready": False,
            "agent4_ready": False,
        }

        with patch("booked.agents.agent1.agent.load_intake_submission", return_value=None), \
             patch("booked.agents.agent1.agent.update_pipeline_status"):

            result = agent1_node(initial_state)

        assert result["agent1_complete"] is False
        assert result["knowledge_base_ready"] is False
        assert len(result["errors"]) > 0
