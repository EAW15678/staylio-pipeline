"""
core/retry.py — shared retry with exponential backoff.

Repo-wide retry utility. Provides both sync and async variants.

Usage (sync):
    from core.retry import retry_with_backoff

    result = retry_with_backoff(
        fn=lambda: client.messages.create(...),
        max_retries=3,
        backoff_base=2.0,
        retryable=(anthropic.RateLimitError, anthropic.APIError),
        label="Claude concept generation",
    )

Usage (async):
    from core.retry import async_retry_with_backoff

    result = await async_retry_with_backoff(
        fn=lambda: client.post(...),
        max_retries=3,
        backoff_base=2.0,
        retryable=(httpx.HTTPStatusError, httpx.ConnectError),
        label="ElevenLabs TTS",
    )
"""

import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional, Sequence, TypeVar

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
                    "[%s] %s failed (attempt %d/%d), retrying in %.1fs: %s",
                    label, label, attempt + 1, max_retries, wait, exc,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "[%s] %s failed after %d attempts: %s",
                    label, label, max_retries, exc,
                )

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{label} failed with no exception captured")


async def async_retry_with_backoff(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    backoff_base: float = 2.0,
    retryable: Sequence[type] = (),
    label: str = "operation",
) -> T:
    """
    Await fn() with exponential backoff on failure.

    Same interface as retry_with_backoff but uses asyncio.sleep instead
    of time.sleep, and fn() must return an awaitable.

    Args:
        fn: Zero-argument callable returning an awaitable to attempt.
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
            return await fn()
        except tuple(retryable) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                wait = backoff_base ** (attempt + 1)
                logger.warning(
                    "[%s] %s failed (attempt %d/%d), retrying in %.1fs: %s",
                    label, label, attempt + 1, max_retries, wait, exc,
                )
                await asyncio.sleep(wait)
            else:
                logger.error(
                    "[%s] %s failed after %d attempts: %s",
                    label, label, max_retries, exc,
                )

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{label} failed with no exception captured")
