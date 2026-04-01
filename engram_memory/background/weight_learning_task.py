"""Background task: learn scoring weights from traversal feedback."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_TRAVERSAL_STATS_QUERY = (
    "MATCH ()-[r]->() "
    "WHERE r._traversalCount IS NOT NULL AND r._traversalCount > 0 "
    "RETURN avg(r._traversalCount) AS avgTraversals, count(r) AS total"
)


class WeightLearningTask:
    """Periodic task that adjusts scoring weights based on traversal patterns.

    Currently logs stats for future weight tuning. Full Bayesian/gradient
    optimisation can be layered in a later phase.
    """

    def __init__(self, driver) -> None:
        self._driver = driver

    async def run(self) -> None:
        result = await self._driver.execute(_TRAVERSAL_STATS_QUERY)
        if result and result[0].get("total", 0) > 0:
            avg = result[0]["avgTraversals"]
            total = result[0]["total"]
            logger.info(
                "Weight learning: %d traversed rels, avg traversal count=%.2f.",
                total, avg,
            )
        else:
            logger.debug("Weight learning: no traversal data yet.")
