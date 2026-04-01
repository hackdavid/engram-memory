"""Abstract base class for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """Interface that all Engram embedding providers must implement."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors."""

    @abstractmethod
    def encode(self, text: str) -> list[float]:
        """Encode a single text string into a float vector."""

    @abstractmethod
    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts in a single call."""
