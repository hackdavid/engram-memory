"""Phase 4: Observability tests (logging, metrics, tracing)."""

import json
import logging

from engram.observability.logging import JSONFormatter, setup_logging
from engram.observability.metrics import Counter, Histogram, MetricsRegistry, metrics
from engram.observability.tracing import traced


def test_json_formatter_produces_valid_json():
    fmt = JSONFormatter()
    record = logging.LogRecord(
        name="engram.test", level=logging.INFO, pathname="",
        lineno=0, msg="hello %s", args=("world",), exc_info=None,
    )
    output = fmt.format(record)
    parsed = json.loads(output)
    assert parsed["msg"] == "hello world"
    assert parsed["level"] == "INFO"


def test_setup_logging_idempotent():
    setup_logging(level="DEBUG", json_format=True)
    root = logging.getLogger("engram")
    handler_count = len(root.handlers)
    setup_logging(level="DEBUG", json_format=True)
    assert len(root.handlers) == handler_count


def test_counter_increments():
    c = Counter("test_counter")
    assert c.value == 0
    c.inc()
    assert c.value == 1
    c.inc(5)
    assert c.value == 6


def test_histogram_observes():
    h = Histogram("test_hist")
    h.observe(1.0)
    h.observe(3.0)
    h.observe(2.0)
    snap = h.snapshot()
    assert snap["count"] == 3
    assert snap["min"] == 1.0
    assert snap["max"] == 3.0
    assert abs(snap["avg"] - 2.0) < 1e-6


def test_histogram_empty_snapshot():
    h = Histogram("empty")
    snap = h.snapshot()
    assert snap["count"] == 0
    assert snap["avg"] == 0.0


def test_metrics_registry():
    registry = MetricsRegistry()
    c = registry.counter("ingest_count")
    c.inc()
    h = registry.histogram("latency_ms")
    h.observe(42.0)
    snap = registry.snapshot()
    assert snap["counters"]["ingest_count"] == 1
    assert snap["histograms"]["latency_ms"]["count"] == 1


def test_global_metrics_singleton():
    metrics.counter("global_test").inc()
    assert metrics.counter("global_test").value >= 1


def test_traced_decorator_is_noop_without_otel():
    @traced("test_span")
    async def dummy():
        return 42

    import asyncio
    result = asyncio.get_event_loop().run_until_complete(dummy())
    assert result == 42
