"""
PHASH-1/PHASH-2: Behavioural tests for pHash at acquisition.

Tests 1-3 exercise acquire_owner_photos with mocked dependencies.
Test 4 exercises onboard call order with patched skills.
All mocked — no vendor calls, no database, $0.
"""

import hashlib
import io
from unittest.mock import MagicMock, patch, call

from PIL import Image


def _make_jpeg(width, height):
    """Create a minimal synthetic JPEG in memory."""
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _mock_supabase():
    """Build a MagicMock Supabase client that handles chained calls.

    Returns (sb_mock, inserts) where inserts is a list that captures
    every dict passed to .insert().
    """
    sb = MagicMock()
    inserts = []

    def capture_insert(data):
        inserts.append(data)
        chain = MagicMock()
        chain.execute = MagicMock(return_value=MagicMock(data=[data]))
        return chain

    # .table("photographs").select(...).eq(...).limit(1).execute() → empty (no existing)
    select_chain = MagicMock()
    select_chain.execute = MagicMock(return_value=MagicMock(data=[]))

    def table_handler(name):
        t = MagicMock()
        t.select = MagicMock(return_value=MagicMock(
            eq=MagicMock(return_value=MagicMock(
                limit=MagicMock(return_value=select_chain)
            ))
        ))
        t.insert = capture_insert
        return t

    sb.table = table_handler
    return sb, inserts


def _mock_httpx_response(img_bytes):
    """Build a mock httpx response that returns img_bytes."""
    resp = MagicMock()
    resp.content = img_bytes
    resp.raise_for_status = MagicMock()
    return resp


# ── Test 1: ≥ 2.0MP photograph gets a pHash ─────────────────────────

@patch("skills.acquire_listing.skills_r2_upload", return_value="https://r2.test/photo.jpg")
@patch("skills.acquire_listing.get_substrate")
def test_large_photo_gets_phash(mock_get_substrate, mock_r2):
    """A ≥ 2.0MP photograph acquired via acquire_owner_photos carries
    a non-null phash in the dict passed to photographs.insert().

    Baseline: fails on be46929 (acquire writes no phash field).
    """
    jpeg_bytes = _make_jpeg(2000, 1500)  # 3.0MP
    sb, inserts = _mock_supabase()
    mock_get_substrate.return_value = sb

    with patch("skills.acquire_listing.httpx.Client") as MockClient:
        mock_client = MagicMock()
        mock_client.get.return_value = _mock_httpx_response(jpeg_bytes)
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        MockClient.return_value = mock_client

        from skills.acquire_listing import acquire_owner_photos
        result = acquire_owner_photos("test-prop-id", ["https://example.com/photo.jpg"])

    # Find the photographs insert (not renditions)
    photo_inserts = [d for d in inserts if "phash" in d or "content_hash" in d]
    assert len(photo_inserts) >= 1, f"Expected a photographs insert, got {len(inserts)} inserts total"
    assert photo_inserts[0]["phash"] is not None, "phash must be non-null for a 3MP image"
    assert result["photos_new"] == 1


# ── Test 2: sub-200px photograph gets a pHash ────────────────────────

@patch("skills.acquire_listing.skills_r2_upload", return_value="https://r2.test/photo.jpg")
@patch("skills.acquire_listing.get_substrate")
def test_tiny_photo_gets_phash(mock_get_substrate, mock_r2):
    """A photograph below MIN_ENHANCE_DIMENSION (200px) acquired via
    acquire_owner_photos carries a non-null phash.

    Baseline: fails on be46929 (acquire writes no phash field).
    """
    jpeg_bytes = _make_jpeg(84, 56)  # Thumbnail-sized
    sb, inserts = _mock_supabase()
    mock_get_substrate.return_value = sb

    with patch("skills.acquire_listing.httpx.Client") as MockClient:
        mock_client = MagicMock()
        mock_client.get.return_value = _mock_httpx_response(jpeg_bytes)
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        MockClient.return_value = mock_client

        from skills.acquire_listing import acquire_owner_photos
        result = acquire_owner_photos("test-prop-id", ["https://example.com/tiny.jpg"])

    photo_inserts = [d for d in inserts if "content_hash" in d]
    assert len(photo_inserts) >= 1
    assert photo_inserts[0]["phash"] is not None, "phash must be non-null even for 84x56"


# ── Test 3: pHash failure → row still inserted with null hash ────────

@patch("skills.acquire_listing.skills_r2_upload", return_value="https://r2.test/photo.jpg")
@patch("skills.acquire_listing.get_substrate")
def test_phash_failure_still_inserts(mock_get_substrate, mock_r2):
    """When imagehash.phash raises, the photographs insert still happens
    with phash=None and acquisition continues to the next URL.

    Baseline: fails on be46929 (acquire writes no phash field).
    """
    jpeg_bytes = _make_jpeg(800, 600)
    sb, inserts = _mock_supabase()
    mock_get_substrate.return_value = sb

    with patch("skills.acquire_listing.httpx.Client") as MockClient:
        mock_client = MagicMock()
        mock_client.get.return_value = _mock_httpx_response(jpeg_bytes)
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        MockClient.return_value = mock_client

        with patch("imagehash.phash", side_effect=RuntimeError("hash exploded")):
            from skills.acquire_listing import acquire_owner_photos
            result = acquire_owner_photos("test-prop-id", [
                "https://example.com/photo1.jpg",
                "https://example.com/photo2.jpg",
            ])

    photo_inserts = [d for d in inserts if "content_hash" in d]
    assert len(photo_inserts) == 2, f"Both photos should be inserted despite pHash failure, got {len(photo_inserts)}"
    assert photo_inserts[0]["phash"] is None, "phash must be None when imagehash.phash raises"
    assert photo_inserts[1]["phash"] is None, "phash must be None for second photo too"
    assert result["photos_new"] == 2


# ── Test 4: onboard calls deduplicate before enhance ─────────────────

def test_onboard_deduplicate_before_enhance():
    """onboard invokes deduplicate before enhance, asserted on the
    position of _run_skill("deduplicate", ...) vs _run_skill("enhance", ...)
    calls in the actual source.

    This asserts on the function call pattern in the source rather than
    executing onboard, because onboard's late imports inside the function
    body (from skills.X import X) make full mock-driven execution
    impractical without coupling to every internal query shape.

    Structural assertion — behaviour unreachable without a full
    integration harness for onboard's 7 skill calls + database queries.

    Baseline: fails on be46929 (enhance at step 3, deduplicate at step 4).
    """
    import ast
    import inspect
    from workflows.onboard import onboard

    source = inspect.getsource(onboard)
    tree = ast.parse(source)

    # Find all calls to _run_skill and record their first argument (the name)
    skill_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "_run_skill":
                if node.args and isinstance(node.args[0], ast.Constant):
                    skill_calls.append((node.args[0].value, node.lineno))

    skill_names = [name for name, _ in skill_calls]
    assert "deduplicate" in skill_names, f"deduplicate not in _run_skill calls: {skill_names}"
    assert "enhance" in skill_names, f"enhance not in _run_skill calls: {skill_names}"

    dedupe_idx = skill_names.index("deduplicate")
    enhance_idx = skill_names.index("enhance")
    assert dedupe_idx < enhance_idx, (
        f"deduplicate (_run_skill call {dedupe_idx}) must come before "
        f"enhance (call {enhance_idx}). Actual order: {skill_names}"
    )
