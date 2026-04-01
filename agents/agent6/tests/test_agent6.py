"""
Agent 6 Test Suite
Run with: pytest booked/agents/agent6/tests/ -v

Tests cover:
  - UTM generation correctness (all required params present)
  - UTM validation — every post must have valid UTM
  - Content calendar structure (60 days, correct post counts)
  - Minimum 60-minute gap between same-platform posts
  - Correct video type sequencing per sprint phase
  - Spark nomination scoring (engagement_score formula)
  - Spark cluster eligibility threshold (5 properties)
  - Spark nomination skips when score below minimum
  - Meta campaign phase budget totals ($150 total)
  - Ayrshare pre-publish validation blocks invalid posts
  - Agent node output contract
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from agents.agent6.models import (
    CampaignPhase,
    ContentType,
    META_PHASE_BUDGETS,
    META_TOTAL_LAUNCH_BUDGET,
    Platform,
    PostRecord,
    PostStatus,
    SPARK_CLUSTER_MIN_PROPERTIES,
    SparkCluster,
    SPRINT_CADENCE,
    STEADY_STATE_CADENCE_PER_WEEK,
)
from agents.agent6.utm_generator import (
    build_utm_link,
    build_utm_link_for_post,
    validate_utm_link,
)
from agents.agent6.content_calendar import (
    TIKTOK_VIDEO_SEQUENCE,
    build_content_calendar,
)
from agents.agent6.spark_nominator import (
    MIN_SPARK_SCORE,
    evaluate_spark_nominations,
    get_cluster_for_property,
)
from agents.agent6.ayrshare_publisher import _validate_post


# ── Fixtures ──────────────────────────────────────────────────────────────

def make_post(
    property_id: str = "prop-001",
    platform: Platform = Platform.TIKTOK,
    video_type: str = "vibe_match",
    with_utm: bool = True,
    caption: str = "Amazing property — link in bio",
) -> PostRecord:
    page_url = "https://vista-azule.staylio.ai"
    utm_link = build_utm_link(page_url, platform, "vista-azule", f"{video_type}_w01_01") if with_utm else ""
    return PostRecord(
        property_id=property_id,
        platform=platform,
        content_type=ContentType.VIDEO_REEL,
        caption=caption,
        hashtags=["#vacation", "#beach"],
        media_url="https://r2.example.com/video/vibe_match_9_16.mp4",
        video_type=video_type,
        page_url=page_url,
        utm_link=utm_link,
        scheduled_at=datetime.now(timezone.utc).isoformat(),
    )


def make_cluster(property_count: int = 5) -> SparkCluster:
    return SparkCluster(
        cluster_id="carolina-beach-nc",
        region_name="Carolina Beach, NC",
        property_ids=[f"prop-{i:03d}" for i in range(property_count)],
        monthly_budget_usd=750.0,
    )


def make_video_assets() -> list[dict]:
    video_types = [
        "vibe_match", "walk_through", "guest_review_1", "guest_review_2",
        "local_highlight", "feature_closeup", "seasonal", "guest_review_3",
    ]
    assets = []
    for vt in video_types:
        for fmt in (["9_16", "1_1", "16_9"] if vt == "vibe_match" else ["9_16", "1_1"]):
            assets.append({
                "video_type": vt,
                "format": fmt,
                "r2_url": f"https://r2.example.com/video/{vt}_{fmt}.mp4",
                "duration_seconds": 30,
            })
    return assets


# ── UTM Generation Tests ──────────────────────────────────────────────────

class TestUTMGeneration:
    def test_all_required_params_present(self):
        url = build_utm_link(
            "https://vista-azule.staylio.ai",
            Platform.TIKTOK,
            "vista-azule",
            "vibe_match_w01_01",
        )
        valid, missing = validate_utm_link(url)
        assert valid, f"Missing params: {missing}"

    def test_utm_source_matches_platform(self):
        for platform in Platform:
            url = build_utm_link(
                "https://test.staylio.ai", platform, "test-slug", "test-content"
            )
            assert f"utm_source={platform.value}" in url

    def test_utm_medium_is_social(self):
        url = build_utm_link("https://test.staylio.ai", Platform.INSTAGRAM, "slug", "content")
        assert "utm_medium=social" in url

    def test_utm_campaign_is_slug(self):
        url = build_utm_link("https://test.staylio.ai", Platform.FACEBOOK, "my-cabin", "content")
        assert "utm_campaign=my-cabin" in url

    def test_utm_content_contains_video_type(self):
        url = build_utm_link_for_post(
            "https://test.staylio.ai", Platform.TIKTOK,
            "slug", "guest_review_1", week=2, sequence=3
        )
        assert "guest_review_1" in url
        assert "w02" in url

    def test_existing_query_string_stripped(self):
        """Base URL with existing params should have them replaced by UTM."""
        url = build_utm_link(
            "https://test.staylio.ai?old_param=val",
            Platform.TIKTOK, "slug", "content"
        )
        assert "old_param" not in url
        assert "utm_source=tiktok" in url

    def test_empty_url_returns_empty(self):
        url = build_utm_link("", Platform.TIKTOK, "slug", "content")
        assert url == ""

    def test_validate_missing_utm_content(self):
        url = "https://test.staylio.ai?utm_source=tiktok&utm_medium=social&utm_campaign=slug"
        valid, missing = validate_utm_link(url)
        assert not valid
        assert "utm_content" in missing

    def test_validate_empty_url_fails(self):
        valid, missing = validate_utm_link("")
        assert not valid


# ── Content Calendar Tests ────────────────────────────────────────────────

class TestContentCalendar:
    def test_calendar_spans_60_days(self):
        calendar = build_content_calendar(
            property_id="p1",
            page_url="https://test.staylio.ai",
            slug="test-slug",
            vibe_profile="romantic_escape",
            video_assets=make_video_assets(),
            social_captions=[],
            photo_urls=[{"url": "https://r2.example.com/photo.jpg"}],
        )
        # Verify posts span the expected date range
        assert calendar.total_scheduled > 0
        dates = set()
        for post in calendar.posts:
            if post.scheduled_at:
                dates.add(post.scheduled_at[:10])   # date part only
        assert len(dates) <= 60
        assert len(dates) >= 55   # Allow slight variation

    def test_tiktok_posts_twice_daily_in_weeks_1_2(self):
        calendar = build_content_calendar(
            property_id="p1",
            page_url="https://test.staylio.ai",
            slug="test-slug",
            vibe_profile="family_adventure",
            video_assets=make_video_assets(),
            social_captions=[],
            photo_urls=[{"url": "https://r2.example.com/photo.jpg"}],
        )
        # Count TikTok posts in first 14 days
        start = datetime.fromisoformat(calendar.launch_date)
        end_14 = start + timedelta(days=14)
        tiktok_early = [
            p for p in calendar.posts
            if p.platform == Platform.TIKTOK
            and p.scheduled_at
            and datetime.fromisoformat(p.scheduled_at[:19]) < end_14.replace(tzinfo=timezone.utc)
        ]
        # Should be ~28 posts (2x daily × 14 days)
        assert 20 <= len(tiktok_early) <= 30

    def test_all_posts_have_valid_utm_links(self):
        """Critical: every post must have a valid UTM link."""
        calendar = build_content_calendar(
            property_id="p1",
            page_url="https://test.staylio.ai",
            slug="test-slug",
            vibe_profile="wellness_retreat",
            video_assets=make_video_assets(),
            social_captions=[],
            photo_urls=[{"url": "https://r2.example.com/photo.jpg"}],
        )
        for post in calendar.posts:
            if post.media_url:   # Only check posts with media
                valid, missing = validate_utm_link(post.utm_link)
                assert valid, (
                    f"Post {post.platform}/{post.video_type} has invalid UTM: {missing}. "
                    f"UTM was: {post.utm_link}"
                )

    def test_minimum_60_minute_gap_enforced(self):
        """Posts on the same platform same day must be >= 60 mins apart."""
        calendar = build_content_calendar(
            property_id="p1",
            page_url="https://test.staylio.ai",
            slug="test-slug",
            vibe_profile="social_celebrations",
            video_assets=make_video_assets(),
            social_captions=[],
            photo_urls=[{"url": "https://r2.example.com/photo.jpg"}],
        )
        # Group posts by platform and date
        from collections import defaultdict
        platform_day_posts: dict = defaultdict(list)
        for post in calendar.posts:
            if post.scheduled_at:
                key = (post.platform, post.scheduled_at[:10])
                platform_day_posts[key].append(
                    datetime.fromisoformat(post.scheduled_at.replace("Z", "+00:00"))
                )

        violations = []
        for key, times in platform_day_posts.items():
            times.sort()
            for i in range(1, len(times)):
                gap_minutes = (times[i] - times[i-1]).total_seconds() / 60
                if gap_minutes < 59:   # Allow 1-minute tolerance
                    violations.append(
                        f"{key}: gap={gap_minutes:.0f}min between "
                        f"{times[i-1].strftime('%H:%M')} and {times[i].strftime('%H:%M')}"
                    )
        assert not violations, f"60-minute gap violations: {violations}"

    def test_all_platforms_get_posts(self):
        calendar = build_content_calendar(
            property_id="p1",
            page_url="https://test.staylio.ai",
            slug="test-slug",
            vibe_profile="adventure_base_camp",
            video_assets=make_video_assets(),
            social_captions=[],
            photo_urls=[{"url": "https://r2.example.com/photo.jpg"}],
        )
        platforms_with_posts = {p.platform for p in calendar.posts if p.media_url}
        assert Platform.TIKTOK in platforms_with_posts
        assert Platform.INSTAGRAM in platforms_with_posts
        assert Platform.PINTEREST in platforms_with_posts
        assert Platform.FACEBOOK in platforms_with_posts

    def test_calendar_approaches_260_total_posts(self):
        """Spec target: ~260 posts over 60 days."""
        calendar = build_content_calendar(
            property_id="p1",
            page_url="https://test.staylio.ai",
            slug="test-slug",
            vibe_profile="romantic_escape",
            video_assets=make_video_assets(),
            social_captions=[],
            photo_urls=[{"url": f"https://r2.example.com/photo_{i}.jpg"} for i in range(8)],
        )
        assert 200 <= calendar.total_scheduled <= 320

    def test_tiktok_video_sequence_defined_for_all_phases(self):
        for phase in ("weeks_1_2", "weeks_3_4", "weeks_5_8"):
            assert phase in TIKTOK_VIDEO_SEQUENCE
            assert len(TIKTOK_VIDEO_SEQUENCE[phase]) >= 3


# ── Ayrshare Pre-Publish Validation ──────────────────────────────────────

class TestAyrshareValidation:
    def test_valid_post_passes(self):
        post = make_post(with_utm=True)
        errors = _validate_post(post)
        assert errors == []

    def test_missing_utm_fails(self):
        """The most critical check — no post without UTM."""
        post = make_post(with_utm=False)
        post.utm_link = ""
        errors = _validate_post(post)
        assert any("UTM" in e for e in errors)

    def test_invalid_utm_fails(self):
        post = make_post()
        post.utm_link = "https://test.staylio.ai"  # No UTM params
        errors = _validate_post(post)
        assert any("UTM" in e for e in errors)

    def test_missing_media_url_fails(self):
        post = make_post()
        post.media_url = ""
        errors = _validate_post(post)
        assert any("media" in e.lower() for e in errors)

    def test_caption_too_long_fails(self):
        post = make_post(platform=Platform.PINTEREST)
        post.caption = "A" * 600   # Pinterest limit is 500
        errors = _validate_post(post)
        assert any("long" in e.lower() for e in errors)


# ── Spark Ad Nomination Tests ─────────────────────────────────────────────

class TestSparkNomination:
    def test_engagement_score_formula(self):
        """engagement_score = completion*0.4 + shares/100*0.3 + likes/500*0.2 + clicks/50*0.1"""
        post = make_post()
        post.completion_rate = 1.0
        post.shares = 100
        post.likes = 500
        post.link_clicks = 50
        expected = 0.4 * 1.0 + 0.3 * 1.0 + 0.2 * 1.0 + 0.1 * 1.0
        assert post.engagement_score() == pytest.approx(expected, rel=0.01)

    def test_zero_engagement_scores_zero(self):
        post = make_post()
        assert post.engagement_score() == 0.0

    def test_cluster_below_minimum_not_eligible(self):
        cluster = make_cluster(property_count=4)   # Needs 5
        assert not cluster.is_eligible

    def test_cluster_at_minimum_is_eligible(self):
        cluster = make_cluster(property_count=SPARK_CLUSTER_MIN_PROPERTIES)
        assert cluster.is_eligible

    def test_nomination_returns_best_post(self):
        cluster = make_cluster(property_count=5)
        posts = [
            make_post("prop-000"),
            make_post("prop-001"),
            make_post("prop-002"),
        ]
        posts[0].completion_rate = 0.3; posts[0].shares = 50
        posts[1].completion_rate = 0.8; posts[1].shares = 200   # Winner
        posts[2].completion_rate = 0.2; posts[2].shares = 10

        # Give all posts platform_post_ids (required for nomination)
        for i, p in enumerate(posts):
            p.platform_post_id = f"tiktok_post_{i}"
            p.property_id = f"prop-{i:03d}"

        nominated = evaluate_spark_nominations(cluster, posts)
        assert nominated is not None
        assert nominated.property_id == "prop-001"

    def test_nomination_skips_below_minimum_score(self):
        cluster = make_cluster(property_count=5)
        post = make_post("prop-000")
        post.completion_rate = 0.0
        post.platform_post_id = "tiktok_123"
        nominated = evaluate_spark_nominations(cluster, [post])
        assert nominated is None   # Score below MIN_SPARK_SCORE

    def test_nomination_skips_posts_outside_cluster(self):
        cluster = make_cluster(property_count=5)
        # Post from a property NOT in the cluster
        post = make_post("outside-property-999")
        post.completion_rate = 1.0
        post.shares = 1000
        post.platform_post_id = "tiktok_999"
        nominated = evaluate_spark_nominations(cluster, [post])
        assert nominated is None

    def test_nomination_requires_platform_post_id(self):
        """Posts without a TikTok post ID can't be Spark Ads."""
        cluster = make_cluster(property_count=5)
        post = make_post("prop-000")
        post.completion_rate = 1.0
        post.shares = 1000
        post.platform_post_id = None   # Not yet published on TikTok
        nominated = evaluate_spark_nominations(cluster, [post])
        assert nominated is None

    def test_cluster_id_from_location(self):
        cluster_id = get_cluster_for_property("p1", "Carolina Beach", "NC")
        assert cluster_id == "carolina-beach-nc"

    def test_cluster_id_none_without_location(self):
        assert get_cluster_for_property("p1", None, None) is None


# ── Meta Campaign Budget Tests ────────────────────────────────────────────

class TestMetaBudgets:
    def test_phase_budgets_sum_to_total(self):
        """The 3 phases must sum to $150 total per property."""
        total = sum(META_PHASE_BUDGETS.values())
        assert total == pytest.approx(META_TOTAL_LAUNCH_BUDGET, rel=0.01)

    def test_phase_a_budget(self):
        assert META_PHASE_BUDGETS[CampaignPhase.PHASE_A_AWARENESS] == pytest.approx(35.0)

    def test_phase_b_budget(self):
        assert META_PHASE_BUDGETS[CampaignPhase.PHASE_B_INFEED] == pytest.approx(70.0)

    def test_phase_c_budget(self):
        assert META_PHASE_BUDGETS[CampaignPhase.PHASE_C_RETARGETING] == pytest.approx(45.0)


# ── Agent Node Contract Tests ─────────────────────────────────────────────

class TestAgent6NodeContract:
    def test_successful_run_sets_flags(self):
        from agents.agent6.agent import agent6_node

        state = {
            "property_id": "p1",
            "knowledge_base": {
                "property_id": "p1",
                "client_id": "client-001",
                "vibe_profile": "romantic_escape",
                "slug": "vista-azule",
                "booking_url": "https://pmc.example.com/book",
                "city": {"value": "Carolina Beach"},
                "state": {"value": "NC"},
            },
            "content_package": {"social_captions": []},
            "visual_media_package": {
                "video_assets": make_video_assets(),
                "category_winners": {
                    "view": "https://r2.example.com/enhanced/view.jpg"
                },
            },
            "page_url": "https://vista-azule.staylio.ai",
            "errors": [],
        }

        mock_calendar = MagicMock()
        mock_calendar.posts = []
        mock_calendar.total_scheduled = 0
        mock_calendar.to_summary.return_value = {"total": 0}

        with patch("booked.agents.agent6.agent.get_cached_knowledge_base", return_value=None), \
             patch("booked.agents.agent6.agent._load_cache", return_value={}), \
             patch("booked.agents.agent6.agent.build_content_calendar", return_value=mock_calendar), \
             patch("booked.agents.agent6.agent.launch_meta_campaign", return_value=[]), \
             patch("booked.agents.agent6.agent.get_cluster_for_property", return_value="carolina-beach-nc"), \
             patch("booked.agents.agent6.agent._register_property_in_cluster"), \
             patch("booked.agents.agent6.agent._save_calendar"), \
             patch("booked.agents.agent6.agent._save_meta_campaigns"), \
             patch("booked.agents.agent6.agent._get_ayrshare_profile_key", return_value=None), \
             patch("booked.agents.agent6.agent.update_pipeline_status"):

            result = agent6_node(state)

        assert result["agent6_complete"] is True
        assert result["agent7_ready"] is True
        assert "social_calendar" in result

    def test_missing_page_url_returns_error(self):
        from agents.agent6.agent import agent6_node

        state = {
            "property_id": "p1",
            "knowledge_base": {"property_id": "p1", "vibe_profile": "romantic_escape"},
            "page_url": "",
            "errors": [],
        }

        with patch("booked.agents.agent6.agent.get_cached_knowledge_base", return_value=None), \
             patch("booked.agents.agent6.agent._load_cache", return_value={}), \
             patch("booked.agents.agent6.agent.update_pipeline_status"):

            result = agent6_node(state)

        assert result["agent6_complete"] is False
        assert len(result["errors"]) > 0
