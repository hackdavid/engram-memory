"""Phase 4: Token-bucket rate limiter tests."""

import pytest
import asyncio

from engram.exceptions import RateLimitExceededError
from engram.rate_limiter import RateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_allows_within_limit():
    rl = RateLimiter(rpm=60, burst=5)
    for _ in range(5):
        await rl.acquire()


@pytest.mark.asyncio
async def test_rate_limiter_blocks_over_burst():
    rl = RateLimiter(rpm=60, burst=2)
    await rl.acquire()
    await rl.acquire()
    with pytest.raises(RateLimitExceededError):
        await rl.acquire()


@pytest.mark.asyncio
async def test_rate_limiter_refills_over_time():
    rl = RateLimiter(rpm=6000, burst=1)  # 100/sec
    await rl.acquire()
    await asyncio.sleep(0.02)
    await rl.acquire()  # should have refilled


@pytest.mark.asyncio
async def test_rate_limiter_burst_equals_initial_tokens():
    rl = RateLimiter(rpm=60, burst=3)
    await rl.acquire()
    await rl.acquire()
    await rl.acquire()
    with pytest.raises(RateLimitExceededError):
        await rl.acquire()
