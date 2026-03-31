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
    from engram import AsyncMemoryClient, Config, MemoryClient
    from engram.models import (
        GraphSnapshot,
        HealthStatus,
        IngestResult,
        NodeInstruction,
        RecallResult,
        RelInstruction,
        ScoredNode,
    )
    from engram.exceptions import (
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
    from engram.constants import SDK_SCHEMA_VERSION

    assert SDK_SCHEMA_VERSION >= 1


def test_version_string():
    from engram import __version__

    assert isinstance(__version__, str)
    parts = __version__.split(".")
    assert len(parts) >= 2


def test_subpackage_imports():
    from engram.graph.driver import GraphDriver
    from engram.graph.engine import CypherEngine
    from engram.graph.indexes import IndexManager
    from engram.graph.migrations import MigrationRunner
    from engram.graph.traversal import TraversalEngine
    from engram.graph.scorer import CompositeScorer
    from engram.graph.hierarchy import HierarchyManager
    from engram.graph.sanitise import sanitise_label, sanitise_rel_type, validate_user_id
    from engram.embeddings.base import BaseEmbedding
    from engram.embeddings.two_tier import TwoTierEmbedder
    from engram.llm.base import BaseLLM
    from engram.extractors.llm_extractor import LLMExtractor
    from engram.extractors.trivial_filter import is_trivial
    from engram.cache.lru_cache import MemoryCache
    from engram.hooks.base import Hook
    from engram.hooks.logger_hook import LoggerHook
    from engram.health.checks import HealthChecker
    from engram.background.runner import BackgroundRunner
    from engram.background.decay_task import DecayTask
    from engram.rate_limiter import RateLimiter
    from engram.observability.logging import setup_logging
    from engram.observability.metrics import metrics, MetricsRegistry
    from engram.observability.tracing import traced


def test_py_typed_marker_exists():
    import importlib.resources
    import engram

    assert (importlib.resources.files(engram) / "py.typed").is_file()
