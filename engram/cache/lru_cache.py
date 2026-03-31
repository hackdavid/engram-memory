"""Per-user in-memory LRU cache with TTL."""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from typing import Any


class MemoryCache:
    """Thread-safe per-user LRU cache with TTL expiry."""

    def __init__(self, max_size: int = 256, ttl_seconds: float = 300.0) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = asyncio.Lock()

    @staticmethod
    def _make_key(user_id: str, query_hash: str) -> str:
        return f"{user_id}::{query_hash}"

    async def get(self, user_id: str, query_hash: str) -> Any | None:
        key = self._make_key(user_id, query_hash)
        async with self._lock:
            if key not in self._store:
                return None
            ts, value = self._store[key]
            if time.monotonic() - ts > self._ttl:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    async def set(self, user_id: str, query_hash: str, value: Any) -> None:
        key = self._make_key(user_id, query_hash)
        async with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (time.monotonic(), value)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    async def invalidate_user(self, user_id: str) -> None:
        prefix = f"{user_id}::"
        async with self._lock:
            keys_to_remove = [k for k in self._store if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._store[k]

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()
