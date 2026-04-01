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
    from engram_memory import AsyncMemoryClient, Config, MemoryClient
    from engram_memory.models import (
        GraphSnapshot,
        HealthStatus,
        IngestResult,
        NodeInstruction,
        RecallResult,
        RelInstruction,
        ScoredNode,
    )
    from engram_memory.exceptions import (
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
    from engram_memory.graph.driver import GraphDriver
    from engram_memory.graph.engine import CypherEngine
    from engram_memory.graph.indexes import IndexManager
    from engram_memory.graph.migrations import MigrationRunner
    from engram_memory.graph.traversal import TraversalEngine
    from engram_memory.graph.scorer import CompositeScorer
    from engram_memory.graph.hierarchy import HierarchyManager
    from engram_memory.graph.sanitise import sanitise_label, sanitise_rel_type, validate_user_id
    from engram_memory.embeddings.base import BaseEmbedding
    from engram_memory.embeddings.two_tier import TwoTierEmbedder
    from engram_memory.llm.base import BaseLLM
    from engram_memory.extractors.llm_extractor import LLMExtractor
    from engram_memory.extractors.trivial_filter import is_trivial
    from engram_memory.cache.lru_cache import MemoryCache
    from engram_memory.hooks.base import Hook
    from engram_memory.hooks.logger_hook import LoggerHook
    from engram_memory.health.checks import HealthChecker
    from engram_memory.background.runner import BackgroundRunner
    from engram_memory.background.decay_task import DecayTask
    from engram_memory.rate_limiter import RateLimiter
    from engram_memory.observability.logging import setup_logging
    from engram_memory.observability.metrics import metrics, MetricsRegistry
    from engram_memory.observability.tracing import traced


def test_py_typed_marker_exists():
    import importlib.resources
    import engram_memory

    assert (importlib.resources.files(engram_memory) / "py.typed").is_file()
