"""
agents/agent8/retry.py — shared retry with exponential backoff.

Local to Agent 8. Not a repo-wide utility — the rest of the codebase
has no shared retry helper (llm_curator.py falls back to GCV on failure
with no retry; that gap is not reproduced here).

Usage:
    from agents.agent8.retry import retry_with_backoff

    result = retry_with_backoff(
        fn=lambda: client.messages.create(...),
        max_retries=3,
        backoff_base=2.0,
        retryable=(anthropic.RateLimitError, anthropic.APIError),
        label="Claude concept generation",
    )
"""

import logging
import time
from typing import Callable, Optional, Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_retries: int = 3,
    backoff_base: float = 2.0,
    retryable: Sequence[type] = (),
    label: str = "operation",
) -> T:
    """
    Call fn() with exponential backoff on failure.

    Args:
        fn: Zero-argument callable to attempt.
        max_retries: Maximum number of attempts (including the first).
        backoff_base: Base for exponential wait (seconds). Wait = base^attempt.
        retryable: Tuple of exception types to retry on. Any exception NOT in
                   this tuple is raised immediately.
        label: Human-readable name for log messages.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception if all retries are exhausted, or any non-retryable
        exception immediately.
    """
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            return fn()
        except tuple(retryable) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                wait = backoff_base ** (attempt + 1)
                logger.warning(
                    "[Agent 8] %s failed (attempt %d/%d), retrying in %.1fs: %s",
                    label, attempt + 1, max_retries, wait, exc,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "[Agent 8] %s failed after %d attempts: %s",
                    label, max_retries, exc,
                )

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{label} failed with no exception captured")
