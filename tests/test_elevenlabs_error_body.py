"""
BALL2 FIX: fetch_vibe_voices must capture ElevenLabs' response body on error.

A real 400 on 2026-08-20 (collection 6qynxss0wmnF5Wu051du) was logged with
no usable detail. Same bug class as C2 (Claid), fixed the same day.
"""

import sys
from unittest.mock import patch, MagicMock

import httpx

sys.path.insert(0, ".")


def test_elevenlabs_400_includes_response_body():
    """A 400 from ElevenLabs raises a ValueError whose message contains
    the response body text — not just '400 Bad Request'."""

    error_body = '{"detail": {"status": "collection_not_found", "message": "Collection does not exist or is not accessible"}}'

    mock_response = httpx.Response(
        status_code=400,
        request=httpx.Request("GET", "https://api.elevenlabs.io/v1/voices"),
        text=error_body,
    )

    def mock_get(*args, **kwargs):
        resp = mock_response
        resp.raise_for_status = lambda: (_ for _ in ()).throw(
            httpx.HTTPStatusError(
                "Client error '400 Bad Request'",
                request=resp.request,
                response=resp,
            )
        )
        return resp

    # Mock _get_collection_id to return a known collection
    with patch("skills.voice_buckets.httpx.get", side_effect=mock_get), \
         patch("skills.voice_buckets._get_collection_id", return_value="test-collection"), \
         patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"}):

        from skills.voice_buckets import fetch_vibe_voices

        try:
            fetch_vibe_voices(MagicMock(), "multigenerational_retreat")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            error_str = str(e)
            assert "collection_not_found" in error_str, (
                f"Error should contain response body, got: {error_str[:200]}"
            )
            assert "400" in error_str, (
                f"Error should mention status code, got: {error_str[:200]}"
            )
