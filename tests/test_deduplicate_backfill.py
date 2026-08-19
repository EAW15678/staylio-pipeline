"""
PHASH-3: Behavioural tests for the pHash backfill in deduplicate.

All tests exercise skills.deduplicate.deduplicate with mocked Supabase
and httpx. No database, no vendor calls, $0.
"""

import io
from unittest.mock import MagicMock, patch, call

from PIL import Image


def _make_jpeg(width, height):
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_photo(photo_id, phash=None, width=1200, height=800, source="vrbo"):
    return {
        "photo_id": photo_id,
        "phash": phash,
        "image_width": width,
        "image_height": height,
        "is_canonical": True,
        "source_image_id": None,
        "source_systems": [source],
    }


def _setup_deduplicate_mock(photos, rendition_urls=None, download_bytes=None,
                            download_fail_ids=None):
    """Build mocks for deduplicate.

    photos: list of photo dicts (as returned by the select query)
    rendition_urls: {photo_id: url} for backfill
    download_bytes: bytes to return from httpx download (same for all)
    download_fail_ids: set of photo_ids where download should raise

    Returns (sb_mock, updates, httpx_responses)
    where updates captures every .update() call as (photo_id, payload).
    """
    rendition_urls = rendition_urls or {}
    download_fail_ids = download_fail_ids or set()
    updates = []

    def mock_table(name):
        t = MagicMock()
        if name == "photographs":
            # .select(...).eq("property_id", ...).execute() → photos
            select_chain = MagicMock()
            select_chain.execute = MagicMock(return_value=MagicMock(data=photos))
            t.select = MagicMock(return_value=MagicMock(
                eq=MagicMock(return_value=select_chain)
            ))
            # .update({...}).eq("photo_id", pid).execute()
            def make_update(data):
                uc = MagicMock()
                def eq_fn(col, val):
                    updates.append((val, data))
                    ex = MagicMock()
                    ex.execute = MagicMock(return_value=MagicMock(data=[]))
                    return ex
                uc.eq = eq_fn
                return uc
            t.update = make_update
        elif name == "renditions":
            # .select("storage_url").eq("photo_id", pid).eq("kind", "original").limit(1).execute()
            def rend_select(cols):
                def eq1(col, val):
                    url = rendition_urls.get(val)
                    def eq2(col2, val2):
                        def limit(n):
                            ex = MagicMock()
                            ex.execute = MagicMock(return_value=MagicMock(
                                data=[{"storage_url": url}] if url else []
                            ))
                            return ex
                        lm = MagicMock()
                        lm.limit = limit
                        return lm
                    em = MagicMock()
                    em.eq = eq2
                    return em
                em = MagicMock()
                em.eq = eq1
                return em
            t.select = rend_select
        return t

    sb = MagicMock()
    sb.table = mock_table

    # Mock httpx client for backfill downloads
    mock_http_client = MagicMock()
    def http_get(url):
        # Find which photo this is for
        pid = None
        for p, u in rendition_urls.items():
            if u == url:
                pid = p
                break
        if pid and pid in download_fail_ids:
            raise ConnectionError(f"Download failed for {url}")
        resp = MagicMock()
        resp.content = download_bytes or _make_jpeg(100, 100)
        resp.raise_for_status = MagicMock()
        return resp
    mock_http_client.get = http_get
    mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
    mock_http_client.__exit__ = MagicMock(return_value=False)

    return sb, updates, mock_http_client


# ── Test 1: backfill computes hash and includes in clustering ────────

def test_backfill_computes_hash_and_clusters():
    """A photograph with phash=None gets a hash computed during deduplicate,
    and is included in clustering on the same pass.

    Must fail on b82febb — no backfill exists.
    """
    jpeg = _make_jpeg(100, 100)

    # Two photos: one hashed, one not. Same image → should cluster.
    import imagehash
    from PIL import Image as PILImage
    expected_hash = str(imagehash.phash(PILImage.open(io.BytesIO(jpeg))))

    photos = [
        _make_photo("photo-hashed", phash=expected_hash, width=3840, height=2567),
        _make_photo("photo-unhashed", phash=None, width=1200, height=800),
    ]

    sb, updates, mock_http = _setup_deduplicate_mock(
        photos,
        rendition_urls={"photo-unhashed": "https://r2.test/photo-unhashed.jpg"},
        download_bytes=jpeg,
    )

    with patch("skills.deduplicate.get_substrate", return_value=sb), \
         patch("skills.deduplicate.record_run", return_value="fake-run"), \
         patch("skills.deduplicate.record_step", return_value="fake-step"), \
         patch("skills.deduplicate.complete_run"), \
         patch("skills.deduplicate.complete_step") as mock_complete_step, \
         patch("httpx.Client", return_value=mock_http):

        from skills.deduplicate import deduplicate
        result = deduplicate("test-prop", force=True)

    # The backfill should have written phash for photo-unhashed
    phash_updates = [(pid, d) for pid, d in updates if "phash" in d and d["phash"] is not None]
    assert len(phash_updates) >= 1, f"Expected a phash backfill write, got {len(phash_updates)} updates"
    assert phash_updates[0][0] == "photo-unhashed"

    # The two should have clustered (same hash → same pHash cluster)
    # photo-hashed (3840×2567) wins canonical, photo-unhashed gets demoted
    canonical_updates = [(pid, d) for pid, d in updates if "is_canonical" in d and d["is_canonical"] is False]
    assert len(canonical_updates) >= 1, "photo-unhashed should be demoted after clustering"


# ── Test 2: backfill failure does not abort ──────────────────────────

def test_backfill_failure_does_not_abort():
    """A backfill failure (download raises) increments the failure counter
    and does not abort — the skill completes and clusters remaining photos.

    Must fail on b82febb — no backfill exists.
    """
    jpeg = _make_jpeg(100, 100)
    import imagehash
    from PIL import Image as PILImage
    h = str(imagehash.phash(PILImage.open(io.BytesIO(jpeg))))

    photos = [
        _make_photo("photo-ok", phash=h, width=1200, height=800),
        _make_photo("photo-fail", phash=None, width=1200, height=800),
        _make_photo("photo-ok2", phash=h, width=3840, height=2567),
    ]

    sb, updates, mock_http = _setup_deduplicate_mock(
        photos,
        rendition_urls={"photo-fail": "https://r2.test/photo-fail.jpg"},
        download_fail_ids={"photo-fail"},
        download_bytes=jpeg,
    )

    with patch("skills.deduplicate.get_substrate", return_value=sb), \
         patch("skills.deduplicate.record_run", return_value="fake-run"), \
         patch("skills.deduplicate.record_step", return_value="fake-step"), \
         patch("skills.deduplicate.complete_run"), \
         patch("skills.deduplicate.complete_step") as mock_cs, \
         patch("httpx.Client", return_value=mock_http):

        from skills.deduplicate import deduplicate
        result = deduplicate("test-prop", force=True)

    assert result.status == "ok", f"Skill should complete despite backfill failure, got {result.status}"
    metadata = mock_cs.call_args[1].get("metadata", {})
    assert metadata.get("phash_backfill_failed", 0) >= 1, "Should record at least 1 backfill failure"


# ── Test 3: backfill writes phash ONLY ───────────────────────────────

def test_backfill_writes_phash_only():
    """The backfill update payload contains only 'phash' — no image_width,
    image_height, or quality_tier.

    Must fail on b82febb — no backfill exists.
    """
    jpeg = _make_jpeg(200, 150)

    photos = [
        _make_photo("photo-needs-hash", phash=None, width=1200, height=800),
    ]

    sb, updates, mock_http = _setup_deduplicate_mock(
        photos,
        rendition_urls={"photo-needs-hash": "https://r2.test/photo.jpg"},
        download_bytes=jpeg,
    )

    with patch("skills.deduplicate.get_substrate", return_value=sb), \
         patch("skills.deduplicate.record_run", return_value="fake-run"), \
         patch("skills.deduplicate.record_step", return_value="fake-step"), \
         patch("skills.deduplicate.complete_run"), \
         patch("skills.deduplicate.complete_step"), \
         patch("httpx.Client", return_value=mock_http):

        from skills.deduplicate import deduplicate
        deduplicate("test-prop", force=True)

    phash_updates = [(pid, d) for pid, d in updates if "phash" in d and "is_canonical" not in d]
    assert len(phash_updates) >= 1, "Expected a phash backfill write"
    payload = phash_updates[0][1]
    assert "phash" in payload, "Update must contain phash"
    assert "image_width" not in payload, "Update must NOT contain image_width"
    assert "image_height" not in payload, "Update must NOT contain image_height"
    assert "quality_tier" not in payload, "Update must NOT contain quality_tier"


# ── Test 4: force guard respects missing hashes ──────────────────────

def test_force_guard_proceeds_when_hashes_missing():
    """With force=False and non-canonical rows present, the skill noops
    when every photo has a phash, and proceeds when any lacks one.

    Must fail on b82febb — guard noops unconditionally when non_canonical > 0.
    """
    # Case A: all hashed + non-canonical → should noop
    photos_all_hashed = [
        _make_photo("p1", phash="abcdef0123456789", width=3840, height=2567),
        {**_make_photo("p2", phash="abcdef0123456789", width=1200, height=800),
         "is_canonical": False},
    ]

    sb_a, _, _ = _setup_deduplicate_mock(photos_all_hashed)
    with patch("skills.deduplicate.get_substrate", return_value=sb_a):
        from skills.deduplicate import deduplicate
        result_a = deduplicate("test-prop", force=False)
    assert result_a.status == "ok" and result_a.data and result_a.data.get("noop"), \
        f"Should noop when all hashed + non-canonical, got status={result_a.status}"

    # Case B: missing hash + non-canonical → should proceed (not noop)
    jpeg = _make_jpeg(100, 100)
    photos_missing_hash = [
        _make_photo("p1", phash="abcdef0123456789", width=3840, height=2567),
        {**_make_photo("p2", phash=None, width=1200, height=800),
         "is_canonical": False},
    ]

    sb_b, _, mock_http = _setup_deduplicate_mock(
        photos_missing_hash,
        rendition_urls={"p2": "https://r2.test/p2.jpg"},
        download_bytes=jpeg,
    )
    with patch("skills.deduplicate.get_substrate", return_value=sb_b), \
         patch("skills.deduplicate.record_run", return_value="fake-run"), \
         patch("skills.deduplicate.record_step", return_value="fake-step"), \
         patch("skills.deduplicate.complete_run"), \
         patch("skills.deduplicate.complete_step"), \
         patch("httpx.Client", return_value=mock_http):
        result_b = deduplicate("test-prop", force=False)
    assert result_b.status == "ok" and (not result_b.data or not result_b.data.get("noop")), \
        f"Should proceed when hashes missing, got status={result_b.status}, data={result_b.data}"
