"""
Schema validation conftest — connects to STAGING Supabase and verifies that
every table+column the codebase references actually exists.

Two modes:
  pytest                          → mocked (fast, $0, existing behavior)
  pytest --staging                → schema-verified against staging DB
  pytest --staging --schema-only  → ONLY schema validation tests (skip all others)

Requires env vars: STAGING_SUPABASE_URL, STAGING_SUPABASE_KEY

This is the deliverable that kills the 15-mismatch bug class: code that
references a non-existent column fails HERE, not in production after
spending money on vendor calls.
"""

import os
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--staging",
        action="store_true",
        default=False,
        help="Run schema validation tests against staging Supabase",
    )
    parser.addoption(
        "--schema-only",
        action="store_true",
        default=False,
        help="Run ONLY schema validation tests (skip all others)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--schema-only"):
        # Keep only tests marked with @pytest.mark.schema
        schema_items = [item for item in items if "schema" in item.keywords]
        items[:] = schema_items
    elif not config.getoption("--staging"):
        # Skip schema tests unless --staging is passed
        skip_staging = pytest.mark.skip(reason="need --staging option to run schema tests")
        for item in items:
            if "schema" in item.keywords:
                item.add_marker(skip_staging)


@pytest.fixture(scope="session")
def staging_db():
    """Connect to staging Supabase for schema validation tests."""
    url = os.environ.get("STAGING_SUPABASE_URL")
    key = os.environ.get("STAGING_SUPABASE_KEY")
    if not url or not key:
        pytest.skip("STAGING_SUPABASE_URL / STAGING_SUPABASE_KEY not set")
    from supabase import create_client
    return create_client(url, key)
