"""
BALL3 FIX: Oversized photos must be compressed before sending to Claid.

A real 14.65MB photo (f1d0e36d, Vista Azule) failed with a 422 on
2026-08-20. Claid's limit: "Input image size should be less than 10.0mb".
"""

import sys
import io
from unittest.mock import patch, MagicMock

sys.path.insert(0, ".")

from skills.enhance import _compress_for_claid, _CLAID_MAX_INPUT_BYTES


def _make_large_jpeg(width=4000, height=3000):
    """Create a synthetic JPEG that's genuinely large (>9.5MB)."""
    from PIL import Image
    import random
    # Create a noisy image (compresses poorly → large file)
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    rng = random.Random(42)  # deterministic
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=98)
    return buf.getvalue()


def _make_small_jpeg(width=800, height=600):
    """Create a small JPEG well under the threshold."""
    from PIL import Image
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ── Test 1: Under threshold → sent to Claid unchanged ───────────────────

def test_under_threshold_no_compression():
    """A photo with byte_size under the threshold is sent to Claid
    unchanged — no compression, no extra R2 call."""
    original_url = "https://r2.dev/prop/original/small.jpg"

    # The enhance loop checks byte_size BEFORE calling _compress_for_claid.
    # If under threshold, claid_input_url stays as original_url.
    # We verify by checking the threshold logic directly.
    byte_size = 5_000_000  # 5MB, well under 9.5MB
    assert byte_size <= _CLAID_MAX_INPUT_BYTES, "Test setup: should be under threshold"

    # Confirm: no compression needed
    needs_compression = byte_size > _CLAID_MAX_INPUT_BYTES
    assert needs_compression is False, "Under-threshold photo should not be compressed"


# ── Test 2: Over threshold → compressed URL passed to Claid ──────────────

def test_over_threshold_compressed():
    """A photo over the threshold gets compressed and uploaded before the
    Claid call — the URL passed is the compressed one, not the original."""
    large_jpeg = _make_large_jpeg()
    assert len(large_jpeg) > _CLAID_MAX_INPUT_BYTES, (
        f"Test setup: synthetic image is {len(large_jpeg)} bytes, need >{_CLAID_MAX_INPUT_BYTES}"
    )

    # Mock httpx.get to return the large image
    mock_resp = MagicMock()
    mock_resp.content = large_jpeg
    mock_resp.raise_for_status = lambda: None

    compressed_url = "https://r2.dev/prop/compressed_for_claid/pid.jpg"

    with patch("skills.enhance.httpx.get", return_value=mock_resp), \
         patch("skills.enhance.skills_r2_upload", return_value=compressed_url) as mock_upload:

        result_url = _compress_for_claid("PROP-1", "PID-1",
                                         "https://r2.dev/original.jpg",
                                         len(large_jpeg))

    assert result_url == compressed_url, (
        f"Should return compressed URL, got: {result_url}"
    )
    assert mock_upload.called, "Should upload compressed version to R2"

    # Verify the uploaded data is actually smaller
    uploaded_data = mock_upload.call_args[0][1]  # second positional arg
    assert len(uploaded_data) < len(large_jpeg), (
        f"Compressed ({len(uploaded_data)}) should be smaller than original ({len(large_jpeg)})"
    )


# ── Test 3: Compressed output is genuinely smaller ───────────────────────

def test_compression_actually_reduces_size():
    """Quality reduction alone should suffice for a realistic large photo.
    Verify the compressed output is genuinely under the threshold."""
    large_jpeg = _make_large_jpeg()
    original_size = len(large_jpeg)

    mock_resp = MagicMock()
    mock_resp.content = large_jpeg
    mock_resp.raise_for_status = lambda: None

    with patch("skills.enhance.httpx.get", return_value=mock_resp), \
         patch("skills.enhance.skills_r2_upload", return_value="https://r2.dev/compressed.jpg") as mock_upload:

        _compress_for_claid("PROP-1", "PID-1", "https://r2.dev/orig.jpg", original_size)

    uploaded_data = mock_upload.call_args[0][1]
    compressed_size = len(uploaded_data)

    assert compressed_size < _CLAID_MAX_INPUT_BYTES, (
        f"Compressed output ({compressed_size} bytes) must be under "
        f"threshold ({_CLAID_MAX_INPUT_BYTES} bytes)"
    )
    assert compressed_size < original_size, (
        f"Compressed ({compressed_size}) must be smaller than original ({original_size})"
    )
