"""Decay-weighted BFS graph traversal."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

EXPAND_QUERY = (
    "MATCH (n)-[r]-(m) "
    "WHERE elementId(n) = $nodeId AND m.userId = $userId "
    "RETURN elementId(m) AS elementId, "
    "       m.strength AS score, "
    "       labels(m)[0] AS label"
)


class TraversalEngine:
    """BFS traversal with exponential score decay per hop."""

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
        """Expand from seed nodes via BFS, decaying score each hop.

        Returns a list of dicts with elementId, score, hops, label.
        """
        visited: set[str] = set()
        results: list[dict[str, Any]] = []
        queue: deque[tuple[str, float, int]] = deque()

        for seed in seeds:
            eid = seed["elementId"]
            score = seed.get("score", 1.0)
            visited.add(eid)
            queue.append((eid, score, 0))
            results.append({
                "elementId": eid,
                "score": score,
                "hops": 0,
            })

        while queue:
            node_id, parent_score, depth = queue.popleft()

            if depth >= self._max_depth:
                continue

            child_score = parent_score * self._decay
            if child_score < self._min_score:
                continue

            neighbours = await self._driver.execute(
                EXPAND_QUERY,
                nodeId=node_id,
                userId=user_id,
            )

            for nbr in neighbours:
                eid = nbr["elementId"]
                if eid in visited:
                    continue
                visited.add(eid)
                entry = {
                    "elementId": eid,
                    "score": child_score,
                    "hops": depth + 1,
                    "label": nbr.get("label"),
                }
                results.append(entry)
                queue.append((eid, child_score, depth + 1))

        return results
