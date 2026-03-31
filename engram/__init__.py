"""Engram -- a standalone graph memory SDK for AI applications."""

from engram._version import __version__
from engram.config import Config

__all__ = [
    "__version__",
    "Config",
    "MemoryClient",
    "AsyncMemoryClient",
]


def __getattr__(name: str):
    """Lazy imports for client classes (not yet built in Phase 1)."""
    if name in ("MemoryClient", "AsyncMemoryClient"):
        from engram.client import AsyncMemoryClient, MemoryClient  # noqa: F811

        return {"MemoryClient": MemoryClient, "AsyncMemoryClient": AsyncMemoryClient}[name]
    raise AttributeError(f"module 'engram' has no attribute {name!r}")
