"""Optional OpenTelemetry tracing integration."""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace

    _tracer = trace.get_tracer("engram_memory")
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False
    _tracer = None


def traced(name: str | None = None) -> Callable:
    """Decorator that wraps an async function in an OTEL span.

    If opentelemetry is not installed, the decorator is a no-op.
    """

    def decorator(fn: Callable) -> Callable:
        if not _HAS_OTEL:
            return fn

        span_name = name or fn.__qualname__

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            with _tracer.start_as_current_span(span_name):
                return await fn(*args, **kwargs)

        return wrapper

    return decorator
