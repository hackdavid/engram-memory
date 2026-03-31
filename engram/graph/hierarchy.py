"""Hierarchical summary tree -- cluster assignment, rebuild, and querying."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_FIND_CLUSTER = (
    "MATCH (c:ClusterSummary) "
    "WHERE c.userId = $userId AND c.topic = $clusterHint "
    "RETURN elementId(c) AS elementId, c.topic AS topic, c._embedding AS embedding"
)

_CREATE_CLUSTER = (
    "CREATE (c:ClusterSummary {"
    "  userId: $userId, topic: $clusterHint, _level: 1, "
    "  _embedding: $embedding, summary: $summary"
    "}) "
    "RETURN elementId(c) AS elementId"
)

_LINK_TO_CLUSTER = (
    "MATCH (n) WHERE elementId(n) = $nodeId AND n.userId = $userId "
    "MATCH (c) WHERE elementId(c) = $clusterId "
    "MERGE (n)-[:BELONGS_TO]->(c)"
)

_GET_CLUSTER_MEMBERS = (
    "MATCH (n)-[:BELONGS_TO]->(c) WHERE elementId(c) = $clusterId "
    "RETURN n.summary AS summary"
)

_UPDATE_CLUSTER_EMBEDDING = (
    "MATCH (c) WHERE elementId(c) = $clusterId "
    "SET c._embedding = $embedding, c.summary = $summary"
)

_QUERY_BROAD = (
    "CALL db.index.vector.queryNodes($indexName, $k, $queryVector) "
    "YIELD node, score "
    "WHERE node.userId = $userId AND node._level <= 1 "
    "RETURN elementId(node) AS elementId, node._level AS _level, "
    "       node.summary AS summary, score"
)

_QUERY_DETAIL = (
    "CALL db.index.vector.queryNodes($indexName, $k, $queryVector) "
    "YIELD node, score "
    "WHERE node.userId = $userId "
    "RETURN elementId(node) AS elementId, node._level AS _level, "
    "       node.summary AS summary, score"
)

AUTO_SCORE_THRESHOLD = 0.5


class HierarchyManager:
    """Manages the hierarchical summary tree of clusters."""

    def __init__(
        self,
        driver,
        embedder,
        index_name: str = "engram_embedding_index",
    ) -> None:
        self._driver = driver
        self._embedder = embedder
        self._index_name = index_name

    async def assign_to_cluster(
        self,
        user_id: str,
        node_id: str,
        cluster_hint: str | None = None,
    ) -> str:
        """Assign a node to a cluster, creating one if needed. Returns cluster elementId."""
        if cluster_hint:
            existing = await self._driver.execute(
                _FIND_CLUSTER, userId=user_id, clusterHint=cluster_hint,
            )
            if existing:
                cluster_id = existing[0]["elementId"]
                await self._driver.execute(
                    _LINK_TO_CLUSTER,
                    nodeId=node_id, userId=user_id, clusterId=cluster_id,
                )
                return cluster_id

        embedding = self._embedder.encode(cluster_hint or "general")
        summary = cluster_hint or "General"
        result = await self._driver.execute(
            _CREATE_CLUSTER,
            userId=user_id,
            clusterHint=cluster_hint or "General",
            embedding=embedding,
            summary=summary,
        )
        cluster_id = result[0]["elementId"]
        await self._driver.execute(
            _LINK_TO_CLUSTER,
            nodeId=node_id, userId=user_id, clusterId=cluster_id,
        )
        return cluster_id

    async def rebuild_cluster_summary(self, cluster_id: str) -> None:
        """Re-compute and embed the summary for a cluster from its members."""
        members = await self._driver.execute(
            _GET_CLUSTER_MEMBERS, clusterId=cluster_id,
        )
        summaries = [m["summary"] for m in members if m.get("summary")]
        combined = " | ".join(summaries) if summaries else "Empty cluster"
        embedding = self._embedder.encode(combined)
        await self._driver.execute(
            _UPDATE_CLUSTER_EMBEDDING,
            clusterId=cluster_id, embedding=embedding, summary=combined,
        )

    async def query_hierarchy(
        self,
        user_id: str,
        query_vector: list[float],
        detail_level: str = "auto",
        k: int = 10,
    ) -> list[dict[str, Any]]:
        """Query the hierarchy at a given detail level.

        detail_level: 'broad' (level 0-1 only), 'detailed' (all), or 'auto'
        (starts broad, falls through to detailed if best score < threshold).
        """
        if detail_level == "broad":
            return await self._driver.execute(
                _QUERY_BROAD,
                indexName=self._index_name, k=k,
                queryVector=query_vector, userId=user_id,
            )

        if detail_level == "detailed":
            return await self._driver.execute(
                _QUERY_DETAIL,
                indexName=self._index_name, k=k,
                queryVector=query_vector, userId=user_id,
            )

        broad = await self._driver.execute(
            _QUERY_BROAD,
            indexName=self._index_name, k=k,
            queryVector=query_vector, userId=user_id,
        )
        best_score = max((r.get("score", 0) for r in broad), default=0)
        if best_score >= AUTO_SCORE_THRESHOLD:
            return broad

        detailed = await self._driver.execute(
            _QUERY_DETAIL,
            indexName=self._index_name, k=k,
            queryVector=query_vector, userId=user_id,
        )
        return broad + detailed
