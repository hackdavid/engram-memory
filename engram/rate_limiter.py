"""Token-bucket rate limiter for LLM API calls."""

from __future__ import annotations

import time

from engram.exceptions import RateLimitExceededError


class RateLimiter:
    """Token-bucket rate limiter.

    rpm: requests per minute
    burst: maximum burst capacity (bucket size)
    """

    def __init__(self, rpm: int = 60, burst: int = 10) -> None:
        self._rate = rpm / 60.0  # tokens per second
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

    async def acquire(self) -> None:
        """Consume one token. Raises RateLimitExceededError if bucket is empty."""
        self._refill()
        if self._tokens < 1.0:
            raise RateLimitExceededError(
                "Rate limit exceeded. Try again shortly."
            )
        self._tokens -= 1.0
