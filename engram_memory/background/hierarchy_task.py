"""Background task: rebuild cluster summaries."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_GET_CLUSTERS = (
    "MATCH (c:ClusterSummary) RETURN elementId(c) AS clusterId"
)


class HierarchyRebuildTask:
    """Periodic task that refreshes all cluster summaries."""

    def __init__(self, driver, hierarchy_manager) -> None:
        self._driver = driver
        self._hierarchy = hierarchy_manager

    async def run(self) -> None:
        clusters = await self._driver.execute(_GET_CLUSTERS)
        for row in clusters:
            cluster_id = row["clusterId"]
            try:
                await self._hierarchy.rebuild_cluster_summary(cluster_id)
            except Exception:
                logger.exception("Failed to rebuild cluster %s.", cluster_id)
        logger.info("Rebuilt %d cluster summaries.", len(clusters))
