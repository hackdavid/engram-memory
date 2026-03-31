"""Phase 1: Verify exception hierarchy and messages."""

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


def test_all_exceptions_inherit_from_base():
    for exc_cls in [
        ConfigurationError,
        ExtractionError,
        CircuitOpenError,
        ConcurrentModificationError,
        HasRelationshipsError,
        InvalidUserIdError,
        EmbeddingDimensionMismatchError,
        RateLimitExceededError,
        MigrationError,
    ]:
        assert issubclass(exc_cls, EngramError)


def test_exception_message():
    err = InvalidUserIdError("bad-id!@#")
    assert "bad-id!@#" in str(err)


def test_base_is_exception():
    assert issubclass(EngramError, Exception)


def test_can_catch_specific_as_base():
    try:
        raise ExtractionError("LLM failed")
    except EngramError as e:
        assert "LLM failed" in str(e)
