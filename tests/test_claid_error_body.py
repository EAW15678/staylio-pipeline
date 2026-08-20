"""
C2 FIX: _call_claid must capture Claid's response body on error,
not just the status code.

A real 422 on 2026-08-20 (photo f1d0e36d, Vista Azule) was logged with
no usable detail. This test proves the body is now in the exception.
"""

import sys
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

sys.path.insert(0, ".")


def test_claid_422_includes_response_body():
    """A 422 from Claid raises an exception whose string contains the
    response body text — not just the status code."""

    error_body = '{"errors": [{"message": "operations.hdr.value exceeds maximum"}]}'

    # Build a real httpx.Response for the 422
    mock_response = httpx.Response(
        status_code=422,
        request=httpx.Request("POST", "https://api.claid.ai/v1-beta1/image/edit"),
        text=error_body,
    )

    async def mock_post(*args, **kwargs):
        raise httpx.HTTPStatusError(
            "Client error '422 Unprocessable Entity'",
            request=mock_response.request,
            response=mock_response,
        )

    async def run():
        session = AsyncMock()
        session.post = mock_post

        with patch.dict("os.environ", {"CLAID_API_KEY": "test-key"}):
            from skills.enhance import _call_claid
            try:
                await _call_claid(session, "https://example.com/photo.jpg", {"upscale": {}})
                assert False, "_call_claid should have raised"
            except httpx.HTTPStatusError as e:
                error_str = str(e)
                # The response body must be in the exception message
                assert "operations.hdr.value exceeds maximum" in error_str, (
                    f"Exception should contain response body, got: {error_str[:200]}"
                )
                # The status code should still be there too
                assert "422" in error_str, f"Exception should mention 422, got: {error_str[:200]}"

    asyncio.run(run())
