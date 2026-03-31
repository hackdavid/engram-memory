"""Lightweight asyncio-based background task scheduler."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class BackgroundRunner:
    """Registers and runs periodic async tasks."""

    def __init__(self) -> None:
        self._tasks: dict[str, dict[str, Any]] = {}
        self._running_handles: list[asyncio.Task] = []
        self._stop_event = asyncio.Event()

    def register(
        self,
        name: str,
        fn: Callable[..., Coroutine],
        interval_seconds: float,
    ) -> None:
        self._tasks[name] = {"fn": fn, "interval": interval_seconds}

    async def _loop(self, name: str, fn: Callable, interval: float) -> None:
        while not self._stop_event.is_set():
            try:
                await fn()
            except Exception:
                logger.exception("Background task '%s' failed.", name)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
                break  # stop_event was set
            except asyncio.TimeoutError:
                pass  # interval elapsed, loop again

    async def start(self) -> None:
        self._stop_event.clear()
        for name, spec in self._tasks.items():
            handle = asyncio.create_task(
                self._loop(name, spec["fn"], spec["interval"])
            )
            self._running_handles.append(handle)

    async def stop(self) -> None:
        self._stop_event.set()
        if self._running_handles:
            await asyncio.gather(*self._running_handles, return_exceptions=True)
        self._running_handles.clear()

    async def run_once(self, name: str) -> None:
        """Execute a registered task once (useful for testing)."""
        spec = self._tasks.get(name)
        if spec is None:
            raise KeyError(f"No task registered with name '{name}'")
        await spec["fn"]()
