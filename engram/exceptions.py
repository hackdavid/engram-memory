"""Engram SDK exception hierarchy."""


class EngramError(Exception):
    """Base exception for all Engram SDK errors."""


class ConfigurationError(EngramError):
    """Raised when SDK configuration is invalid or incomplete."""


class ExtractionError(EngramError):
    """Raised when LLM extraction fails after all retries."""


class CircuitOpenError(EngramError):
    """Raised when the LLM circuit breaker is open due to repeated failures."""


class ConcurrentModificationError(EngramError):
    """Raised when optimistic locking detects a concurrent write conflict."""


class HasRelationshipsError(EngramError):
    """Raised when attempting non-cascade delete on a node with relationships."""


class InvalidUserIdError(EngramError):
    """Raised when a user_id fails pattern validation."""


class EmbeddingDimensionMismatchError(EngramError):
    """Raised when configured embedding dimensions differ from the existing vector index."""


class RateLimitExceededError(EngramError):
    """Raised when LLM rate limit is exhausted."""


class MigrationError(EngramError):
    """Raised when a schema migration fails."""
