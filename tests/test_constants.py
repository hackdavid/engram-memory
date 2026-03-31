"""Phase 1: Verify constants are defined with correct types and values."""

from engram.constants import (
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_SCORE_ALPHA,
    DEFAULT_SCORE_BETA,
    DEFAULT_SCORE_GAMMA,
    DEFAULT_TRAVERSAL_DECAY,
    DEFAULT_TRAVERSAL_MAX_DEPTH,
    SDK_SCHEMA_VERSION,
)


def test_schema_version_is_positive_int():
    assert isinstance(SDK_SCHEMA_VERSION, int)
    assert SDK_SCHEMA_VERSION >= 1


def test_scoring_weights_sum_to_one():
    total = DEFAULT_SCORE_ALPHA + DEFAULT_SCORE_BETA + DEFAULT_SCORE_GAMMA
    assert abs(total - 1.0) < 1e-9


def test_traversal_defaults_sensible():
    assert 0.0 < DEFAULT_TRAVERSAL_DECAY < 1.0
    assert DEFAULT_TRAVERSAL_MAX_DEPTH >= 1


def test_cache_ttl_positive():
    assert DEFAULT_CACHE_TTL_SECONDS > 0
