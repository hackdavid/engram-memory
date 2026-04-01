"""Protocol-based hook interface for Engram lifecycle events."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from engram_memory.models import IngestResult, RecallResult


@runtime_checkable
class Hook(Protocol):
    """Extension point for Engram lifecycle events.

    All methods have default no-op implementations so consumers
    only need to override the hooks they care about.
    """

    async def pre_ingest(self, user_id: str, text: str) -> str | None:
        """Called before ingestion. Return modified text or None to keep original."""
        ...

    async def post_ingest(self, user_id: str, result: IngestResult) -> None:
        """Called after successful ingestion."""
        ...

    async def pre_recall(self, user_id: str, query: str) -> str | None:
        """Called before recall. Return modified query or None to keep original."""
        ...

    async def post_recall(self, user_id: str, result: RecallResult) -> None:
        """Called after successful recall."""
        ...
