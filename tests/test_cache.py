"""Phase 4: LRU cache tests."""

import pytest
import asyncio

from engram.cache.lru_cache import MemoryCache


@pytest.mark.asyncio
async def test_cache_miss_returns_none():
    cache = MemoryCache(max_size=10, ttl_seconds=60)
    result = await cache.get("user1", "query_hash_1")
    assert result is None


@pytest.mark.asyncio
async def test_cache_set_then_get():
    cache = MemoryCache(max_size=10, ttl_seconds=60)
    await cache.set("user1", "qh1", {"data": "value"})
    result = await cache.get("user1", "qh1")
    assert result == {"data": "value"}


@pytest.mark.asyncio
async def test_cache_ttl_expiry():
    cache = MemoryCache(max_size=10, ttl_seconds=0.1)
    await cache.set("user1", "qh1", {"data": "value"})
    await asyncio.sleep(0.2)
    result = await cache.get("user1", "qh1")
    assert result is None


@pytest.mark.asyncio
async def test_cache_invalidate_user():
    cache = MemoryCache(max_size=10, ttl_seconds=60)
    await cache.set("user1", "qh1", {"a": 1})
    await cache.set("user1", "qh2", {"b": 2})
    await cache.set("user2", "qh1", {"c": 3})
    await cache.invalidate_user("user1")
    assert await cache.get("user1", "qh1") is None
    assert await cache.get("user1", "qh2") is None
    assert await cache.get("user2", "qh1") == {"c": 3}


@pytest.mark.asyncio
async def test_cache_lru_eviction():
    cache = MemoryCache(max_size=2, ttl_seconds=60)
    await cache.set("u1", "q1", "first")
    await cache.set("u1", "q2", "second")
    await cache.set("u1", "q3", "third")  # evicts q1
    assert await cache.get("u1", "q1") is None
    assert await cache.get("u1", "q3") == "third"


@pytest.mark.asyncio
async def test_cache_concurrent_access():
    cache = MemoryCache(max_size=100, ttl_seconds=60)

    async def writer(i):
        await cache.set("u1", f"q{i}", f"val{i}")

    await asyncio.gather(*[writer(i) for i in range(50)])
    hits = 0
    for i in range(50):
        if await cache.get("u1", f"q{i}") is not None:
            hits += 1
    assert hits > 0


@pytest.mark.asyncio
async def test_cache_clear():
    cache = MemoryCache(max_size=10, ttl_seconds=60)
    await cache.set("u1", "q1", "data")
    await cache.clear()
    assert await cache.get("u1", "q1") is None


@pytest.mark.asyncio
async def test_cache_overwrite_same_key():
    cache = MemoryCache(max_size=10, ttl_seconds=60)
    await cache.set("u1", "q1", "original")
    await cache.set("u1", "q1", "updated")
    result = await cache.get("u1", "q1")
    assert result == "updated"
