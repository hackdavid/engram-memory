"""Composite scoring: alpha * vector_sim + beta * decay^hops + gamma * strength."""

from __future__ import annotations

from typing import Any


class CompositeScorer:
    """Rank graph nodes by a weighted composite score."""

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.35,
        gamma: float = 0.15,
    ) -> None:
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

    def score(
        self,
        vector_similarity: float,
        hops: int,
        strength: float,
        decay: float,
    ) -> float:
        """Compute the composite score for a single node."""
        return (
            self._alpha * vector_similarity
            + self._beta * (decay ** hops)
            + self._gamma * strength
        )

    def rank(
        self,
        nodes: list[dict[str, Any]],
        decay: float,
    ) -> list[dict[str, Any]]:
        """Score and sort nodes descending by final_score."""
        scored = []
        for node in nodes:
            s = self.score(
                vector_similarity=node.get("vector_similarity", 0.0),
                hops=node.get("hops", 0),
                strength=node.get("strength", 0.0),
                decay=decay,
            )
            scored.append({**node, "final_score": s})
        scored.sort(key=lambda n: n["final_score"], reverse=True)
        return scored
