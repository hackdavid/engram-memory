"""Phase 7: Packaging validation tests."""

import subprocess
import sys


def test_package_installs_cleanly():
    """pip install -e . should succeed (dry-run)."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "--dry-run"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"pip install failed:\n{result.stderr}"


def test_imports_after_install():
    from engram_memory import AsyncMemoryClient, Config, MemoryClient  # noqa: F401
    from engram_memory.models import (  # noqa: F401
        GraphSnapshot,
        HealthStatus,
        IngestResult,
        NodeInstruction,
        RecallResult,
        RelInstruction,
        ScoredNode,
    )
    from engram_memory.exceptions import (  # noqa: F401
        CircuitOpenError,
        ConcurrentModificationError,
        ConfigurationError,
        EmbeddingDimensionMismatchError,
        EngramError,
        ExtractionError,
        HasRelationshipsError,
        InvalidUserIdError,
        MigrationError,
        RateLimitExceededError,
    )
    from engram_memory.constants import SDK_SCHEMA_VERSION

    assert SDK_SCHEMA_VERSION >= 1


def test_version_string():
    from engram_memory import __version__

    assert isinstance(__version__, str)
    parts = __version__.split(".")
    assert len(parts) >= 2


def test_subpackage_imports():
    from engram_memory.graph.driver import GraphDriver  # noqa: F401
    from engram_memory.graph.engine import CypherEngine  # noqa: F401
    from engram_memory.graph.indexes import IndexManager  # noqa: F401
    from engram_memory.graph.migrations import MigrationRunner  # noqa: F401
    from engram_memory.graph.traversal import TraversalEngine  # noqa: F401
    from engram_memory.graph.scorer import CompositeScorer  # noqa: F401
    from engram_memory.graph.hierarchy import HierarchyManager  # noqa: F401
    from engram_memory.graph.sanitise import sanitise_label, sanitise_rel_type, validate_user_id  # noqa: F401
    from engram_memory.embeddings.base import BaseEmbedding  # noqa: F401
    from engram_memory.embeddings.two_tier import TwoTierEmbedder  # noqa: F401
    from engram_memory.llm.base import BaseLLM  # noqa: F401
    from engram_memory.extractors.llm_extractor import LLMExtractor  # noqa: F401
    from engram_memory.extractors.trivial_filter import is_trivial  # noqa: F401
    from engram_memory.cache.lru_cache import MemoryCache  # noqa: F401
    from engram_memory.hooks.base import Hook  # noqa: F401
    from engram_memory.hooks.logger_hook import LoggerHook  # noqa: F401
    from engram_memory.health.checks import HealthChecker  # noqa: F401
    from engram_memory.background.runner import BackgroundRunner  # noqa: F401
    from engram_memory.background.decay_task import DecayTask  # noqa: F401
    from engram_memory.rate_limiter import RateLimiter  # noqa: F401
    from engram_memory.observability.logging import setup_logging  # noqa: F401
    from engram_memory.observability.metrics import metrics, MetricsRegistry  # noqa: F401
    from engram_memory.observability.tracing import traced  # noqa: F401


def test_py_typed_marker_exists():
    import importlib.resources
    import engram_memory

    assert (importlib.resources.files(engram_memory) / "py.typed").is_file()
