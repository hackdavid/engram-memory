"""Lightweight in-process metrics (counters and histograms)."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any


class Counter:
    """Simple thread-safe counter."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._value = 0
        self._lock = threading.Lock()

    def inc(self, amount: int = 1) -> None:
        with self._lock:
            self._value += amount

    @property
    def value(self) -> int:
        return self._value


class Histogram:
    """Simple histogram that tracks min, max, sum, count."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._count = 0
        self._sum = 0.0
        self._min = float("inf")
        self._max = float("-inf")
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._count += 1
            self._sum += value
            self._min = min(self._min, value)
            self._max = max(self._max, value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "count": self._count,
                "sum": self._sum,
                "min": self._min if self._count > 0 else 0.0,
                "max": self._max if self._count > 0 else 0.0,
                "avg": self._sum / self._count if self._count > 0 else 0.0,
            }


class MetricsRegistry:
    """Central registry for all Engram metrics."""

    def __init__(self) -> None:
        self._counters: dict[str, Counter] = {}
        self._histograms: dict[str, Histogram] = {}

    def counter(self, name: str) -> Counter:
        if name not in self._counters:
            self._counters[name] = Counter(name)
        return self._counters[name]

    def histogram(self, name: str) -> Histogram:
        if name not in self._histograms:
            self._histograms[name] = Histogram(name)
        return self._histograms[name]

    def snapshot(self) -> dict[str, Any]:
        return {
            "counters": {n: c.value for n, c in self._counters.items()},
            "histograms": {n: h.snapshot() for n, h in self._histograms.items()},
        }


metrics = MetricsRegistry()
