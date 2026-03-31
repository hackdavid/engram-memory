"""Abstract base class for entity/relationship extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from engram.models import NodeInstruction, RelInstruction


class BaseExtractor(ABC):
    """Interface for all Engram extraction backends."""

    @abstractmethod
    async def extract(
        self,
        user_id: str,
        text: str,
        neighbourhood: list[dict[str, Any]],
    ) -> tuple[list[NodeInstruction], list[RelInstruction]]:
        """Parse user text and return graph instructions."""
