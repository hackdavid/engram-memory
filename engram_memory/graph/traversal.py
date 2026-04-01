"""Decay-weighted graph traversal via a single variable-length Cypher query."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_TRAVERSE_QUERY = (
    "MATCH (seed) WHERE elementId(seed) IN $seedIds "
    "MATCH path = (seed)-[*1..{max_depth}]-(m) "
    "WHERE m.userId = $userId AND m.isCurrent = true "
    "WITH DISTINCT m, min(length(path)) AS hops "
    "RETURN elementId(m) AS elementId, hops, "
    "       m.strength AS strength, labels(m)[0] AS label"
)


class TraversalEngine:
    """Graph traversal with exponential score decay per hop.

    Uses a single variable-length Cypher query instead of per-node BFS,
    collapsing potentially hundreds of round-trips into one.
    """

    def __init__(
        self,
        driver,
        decay: float = 0.5,
        max_depth: int = 3,
        min_score: float = 0.1,
    ) -> None:
        self._driver = driver
        self._decay = decay
        self._max_depth = max_depth
        self._min_score = min_score

    async def traverse(
        self,
        seeds: list[dict[str, Any]],
        user_id: str,
    ) -> list[dict[str, Any]]:
        """Expand from seed nodes, decaying score each hop.

        Returns a list of dicts with elementId, score, hops, label.
        """
        if not seeds:
            return []

        seed_ids = [s["elementId"] for s in seeds]

        results: list[dict[str, Any]] = []
        for seed in seeds:
            results.append({
                "elementId": seed["elementId"],
                "score": seed.get("score", 1.0),
                "hops": 0,
            })

        query = _TRAVERSE_QUERY.format(max_depth=self._max_depth)
        rows = await self._driver.execute(
            query,
            seedIds=seed_ids,
            userId=user_id,
        )

        seed_set = set(seed_ids)
        for row in rows:
            eid = row["elementId"]
            if eid in seed_set:
                continue

            hops = row["hops"]
            score = self._decay ** hops
            if score < self._min_score:
                continue

            results.append({
                "elementId": eid,
                "score": score,
                "hops": hops,
                "label": row.get("label"),
            })

        return results
