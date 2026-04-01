"""Two-tier embedding: coarse (int8 quantized) + fine (float32)."""

from __future__ import annotations

from typing import Any


class TwoTierEmbedder:
    """Wraps any base embedder and provides both fine and coarse embeddings."""

    def __init__(self, base: Any) -> None:
        self._base = base

    @property
    def dimensions(self) -> int:
        return self._base.dimensions

    @staticmethod
    def quantize(fine: list[float]) -> list[int]:
        """Scalar-quantize float32 -> int8 (-128..127)."""
        if not fine:
            return []
        max_abs = max(abs(v) for v in fine) or 1.0
        return [max(-128, min(127, int(round(v / max_abs * 127)))) for v in fine]

    def encode(self, text: str) -> list[float]:
        return [float(v) for v in self._base.encode(text)]

    def encode_both(self, text: str) -> tuple[list[float], list[int]]:
        """Return (fine_embedding, coarse_embedding) for a text."""
        fine = self.encode(text)
        coarse = self.quantize(fine)
        return fine, coarse

    def encode_batch_both(
        self, texts: list[str]
    ) -> list[tuple[list[float], list[int]]]:
        results = []
        for text in texts:
            results.append(self.encode_both(text))
        return results
