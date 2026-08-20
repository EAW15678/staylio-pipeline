"""
BALL3 FIX: Oversized photos must be compressed before sending to Claid.

Tests import and call the real _resolve_claid_input_url and
_compress_for_claid from skills/enhance.py — not local reimplementations.
"""

import sys
import io
from unittest.mock import patch, MagicMock

sys.path.insert(0, ".")

from skills.enhance import (
    _resolve_claid_input_url,
    _compress_for_claid,
    _CLAID_MAX_INPUT_BYTES,
)


# ── Test 1: Under threshold → returns original, no compression called ────

def test_under_threshold_returns_original():
    """A photo with byte_size under the threshold is returned unchanged.
    _compress_for_claid is never called."""
    original_url = "https://r2.dev/prop/original/small.jpg"

    with patch("skills.enhance._compress_for_claid") as mock_compress:
        result = _resolve_claid_input_url("PROP-1", "PID-1", original_url, 5_000_000)

    assert result == original_url, f"Should return original URL, got: {result}"
    assert mock_compress.called is False, "_compress_for_claid should NOT be called"


# ── Test 2: Over threshold → returns compressed, compress called ─────────

def test_over_threshold_returns_compressed():
    """A photo over the threshold calls _compress_for_claid and returns
    its result — the compressed URL, not the original."""
    original_url = "https://r2.dev/prop/original/large.jpg"
    compressed_url = "https://r2.dev/prop/compressed_for_claid/PID-1.jpg"

    with patch("skills.enhance._compress_for_claid", return_value=compressed_url) as mock_compress:
        result = _resolve_claid_input_url("PROP-1", "PID-1", original_url, 15_000_000)

    assert result == compressed_url, f"Should return compressed URL, got: {result}"
    assert mock_compress.called is True, "_compress_for_claid should be called"
    mock_compress.assert_called_once_with("PROP-1", "PID-1", original_url, 15_000_000)


# ── Test 3: _compress_for_claid actually reduces size ────────────────────

def test_compression_actually_reduces_size():
    """Quality reduction alone should suffice for a realistic large photo.
    Verify the compressed output is genuinely under the threshold."""
    from PIL import Image
    import random

    # Create a synthetic large JPEG (noisy → compresses poorly → large)
    rng = random.Random(42)
    img = Image.new("RGB", (4000, 3000))
    pixels = img.load()
    for y in range(3000):
        for x in range(4000):
            pixels[x, y] = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=98)
    large_jpeg = buf.getvalue()
    original_size = len(large_jpeg)

    assert original_size > _CLAID_MAX_INPUT_BYTES, (
        f"Test setup: need >{_CLAID_MAX_INPUT_BYTES}, got {original_size}"
    )

    mock_resp = MagicMock()
    mock_resp.content = large_jpeg
    mock_resp.raise_for_status = lambda: None

    with patch("skills.enhance.httpx.get", return_value=mock_resp), \
         patch("skills.enhance.skills_r2_upload", return_value="https://r2.dev/compressed.jpg") as mock_upload:

        _compress_for_claid("PROP-1", "PID-1", "https://r2.dev/orig.jpg", original_size)

    uploaded_data = mock_upload.call_args[0][1]
    compressed_size = len(uploaded_data)

    assert compressed_size < _CLAID_MAX_INPUT_BYTES, (
        f"Compressed ({compressed_size}) must be under threshold ({_CLAID_MAX_INPUT_BYTES})"
    )
    assert compressed_size < original_size, (
        f"Compressed ({compressed_size}) must be smaller than original ({original_size})"
    )
