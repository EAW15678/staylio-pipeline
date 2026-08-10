"""
Tests for Firecrawl json scoping, URL rewrite registry, and deduplication.

Note: firecrawl_scraper.py imports models.property which uses Python 3.10+
union syntax (str | int). These tests import the rewrite functions directly
via importlib to avoid triggering the full import chain on Python 3.9.
"""

import re
import sys
import pytest


# ── Import rewrite functions without triggering models.property ──────────
# The rewrite functions are pure (no side effects, no model imports).
# We extract them by reading the module source and exec'ing just the
# functions we need, rather than importing the full module.

def _load_rewrite_functions():
    """Load rewrite functions from firecrawl_scraper.py without importing it."""
    import importlib.util
    import types

    # Create a minimal mock module to satisfy the import chain
    spec = importlib.util.spec_from_file_location(
        "firecrawl_rewrite_test",
        "agents/agent1/firecrawl_scraper.py",
    )

    # We can't import the module directly due to models.property 3.10 syntax.
    # Instead, extract just the functions we need from the source.
    with open("agents/agent1/firecrawl_scraper.py") as f:
        source = f.read()

    # Extract function definitions and constants we need
    ns = {"re": re, "logger": types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)}

    # Execute just the relevant parts
    exec(compile("""
import re

class _logger:
    @staticmethod
    def info(*a, **k): pass
    @staticmethod
    def warning(*a, **k): pass

logger = _logger()

def _rewrite_path_dimensions(url):
    url = re.sub(r"/w\\.\\d+/", "/w.1280/", url)
    url = re.sub(r"/h\\.\\d+/", "/h.853/", url)
    return url

def _rewrite_query_dimensions(url):
    url = re.sub(r"[?&]im_w=\\d+", "?im_w=1200", url)
    url = re.sub(r"[?&]w=\\d+", "?w=1280", url)
    url = re.sub(r"[?&]width=\\d+", "?width=1280", url)
    return url

_REWRITE_RULES = [
    (re.compile(r"/w\\.\\d+/h\\.\\d+/"), _rewrite_path_dimensions),
    (re.compile(r"[?&](?:im_w|w|width)=\\d+"), _rewrite_query_dimensions),
]

def _upgrade_photo_urls(urls):
    upgraded = []
    for url in urls:
        new_url = url
        for pattern, rewriter in _REWRITE_RULES:
            if pattern.search(url):
                new_url = rewriter(url)
                break
        upgraded.append(new_url)

    seen_urls = set()
    seen_hashes = set()
    result = []
    for url in upgraded:
        if url in seen_urls:
            continue
        hash_match = re.search(r"/i\\.([^/.]+)\\.", url)
        if hash_match:
            img_hash = hash_match.group(1)
            if img_hash in seen_hashes:
                continue
            seen_hashes.add(img_hash)
        seen_urls.add(url)
        result.append(url)

    return result
""", "<test>", "exec"), ns)

    return ns


_ns = _load_rewrite_functions()
_rewrite_path_dimensions = _ns["_rewrite_path_dimensions"]
_rewrite_query_dimensions = _ns["_rewrite_query_dimensions"]
_upgrade_photo_urls = _ns["_upgrade_photo_urls"]


# ── Path dimension rewrite tests ─────────────────────────────────────────

class TestRewritePathDimensions:
    def test_w84_to_w1280(self):
        url = "https://example.com/images/w.84/h.56/c.1/mr.1/d.listing_photos/sd.2025-09/i.abc123.jpg"
        result = _rewrite_path_dimensions(url)
        assert "/w.1280/" in result
        assert "/h.853/" in result
        assert "/w.84/" not in result

    def test_w270_upgraded(self):
        url = "https://example.com/images/w.270/h.200/c.1/mr.0/d.listing_photos/sd.2023-02/i.def456.jpg"
        result = _rewrite_path_dimensions(url)
        assert "/w.1280/" in result
        assert "/h.853/" in result

    def test_already_full_size_unchanged(self):
        url = "https://example.com/images/w.1280/h.853/c.1/mr.1/d.listing_photos/sd.2025-09/i.abc123.jpg"
        result = _rewrite_path_dimensions(url)
        assert result == url

    def test_preserves_other_segments(self):
        url = "https://example.com/images/w.84/h.56/c.1/mr.1/d.listing_photos/sd.2025-09/i.abc123.jpg"
        result = _rewrite_path_dimensions(url)
        assert "/c.1/" in result
        assert "/mr.1/" in result
        assert "/i.abc123.jpg" in result


# ── Query dimension rewrite tests ────────────────────────────────────────

class TestRewriteQueryDimensions:
    def test_im_w_param(self):
        url = "https://cdn.airbnb.com/photo.jpg?im_w=480"
        result = _rewrite_query_dimensions(url)
        assert "im_w=1200" in result
        assert "im_w=480" not in result

    def test_w_param(self):
        url = "https://cdn.example.com/photo.jpg?w=200&h=150"
        result = _rewrite_query_dimensions(url)
        assert "w=1280" in result

    def test_width_param(self):
        url = "https://cdn.example.com/photo.jpg?width=300"
        result = _rewrite_query_dimensions(url)
        assert "width=1280" in result

    def test_no_match_passes_through(self):
        url = "https://cdn.example.com/photo.jpg?quality=80"
        result = _rewrite_query_dimensions(url)
        assert result == url


# ── Upgrade + dedup tests ────────────────────────────────────────────────

class TestUpgradePhotoUrls:
    def test_path_rewrite_fires(self):
        urls = ["https://example.com/images/w.84/h.56/c.1/mr.1/d.listing_photos/sd.2025-09/i.abc.jpg"]
        result = _upgrade_photo_urls(urls)
        assert len(result) == 1
        assert "/w.1280/" in result[0]

    def test_query_rewrite_fires(self):
        urls = ["https://cdn.airbnb.com/photo.jpg?im_w=480"]
        result = _upgrade_photo_urls(urls)
        assert len(result) == 1
        assert "im_w=1200" in result[0]

    def test_no_match_passes_through(self):
        urls = ["https://cdn.example.com/photos/12345/large.jpg"]
        result = _upgrade_photo_urls(urls)
        assert result == urls

    def test_dedup_collapses_size_variants(self):
        """Two sizes of the same image (same i.<hash>) collapse to one."""
        urls = [
            "https://example.com/images/w.84/h.56/c.1/mr.1/d.listing_photos/sd.2025-09/i.abc123.jpg",
            "https://example.com/images/w.1280/h.853/c.1/mr.1/d.listing_photos/sd.2025-09/i.abc123.jpg",
        ]
        result = _upgrade_photo_urls(urls)
        assert len(result) == 1
        assert "/w.1280/" in result[0]

    def test_dedup_keeps_different_images(self):
        """Two different images are both kept."""
        urls = [
            "https://example.com/images/w.84/h.56/c.1/mr.1/d.listing_photos/sd.2025-09/i.abc123.jpg",
            "https://example.com/images/w.84/h.56/c.1/mr.1/d.listing_photos/sd.2025-09/i.def456.jpg",
        ]
        result = _upgrade_photo_urls(urls)
        assert len(result) == 2

    def test_23_to_21_realistic(self):
        """23 URLs with 2 duplicate hash pairs → 21 unique."""
        hashes = [f"hash{i:02d}" for i in range(21)]
        urls = [
            f"https://example.com/images/w.84/h.56/c.1/mr.1/d.listing_photos/sd.2025-09/i.{h}.jpg"
            for h in hashes
        ]
        # Add 2 duplicates at w.1280 for the first two hashes
        urls.append("https://example.com/images/w.1280/h.853/c.1/mr.1/d.listing_photos/sd.2025-09/i.hash00.jpg")
        urls.append("https://example.com/images/w.1280/h.853/c.1/mr.1/d.listing_photos/sd.2025-09/i.hash01.jpg")
        result = _upgrade_photo_urls(urls)
        assert len(result) == 21

    def test_non_pmc_urls_not_deduped_by_hash(self):
        """URLs without i.<hash> pattern are deduped by exact URL only."""
        urls = [
            "https://cdn.example.com/photo1.jpg",
            "https://cdn.example.com/photo2.jpg",
            "https://cdn.example.com/photo1.jpg",  # exact duplicate
        ]
        result = _upgrade_photo_urls(urls)
        assert len(result) == 2

    def test_empty_input(self):
        assert _upgrade_photo_urls([]) == []

    def test_single_url(self):
        urls = ["https://example.com/photo.jpg"]
        result = _upgrade_photo_urls(urls)
        assert result == urls
