"""
agents/agent8/retry.py — thin re-export from core.retry.

Kept so existing Agent 8 imports continue to work unchanged:
    from agents.agent8.retry import retry_with_backoff
"""

from core.retry import retry_with_backoff  # noqa: F401
