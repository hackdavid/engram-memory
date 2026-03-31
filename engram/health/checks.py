"""Aggregated health checks for all Engram dependencies."""

from __future__ import annotations

import logging

from engram.models import HealthStatus

logger = logging.getLogger(__name__)


class HealthChecker:
    """Runs health checks against Neo4j, LLM, embedder, and vector index."""

    def __init__(
        self,
        driver,
        llm,
        embedder,
        schema_version: int = 1,
    ) -> None:
        self._driver = driver
        self._llm = llm
        self._embedder = embedder
        self._schema_version = schema_version

    async def check(self, *, ping_llm: bool = True) -> HealthStatus:
        status = HealthStatus()

        # Neo4j connectivity
        try:
            status.neo4j_connected = await self._driver.ping()
        except Exception:
            status.neo4j_connected = False

        # Vector index
        if status.neo4j_connected:
            try:
                result = await self._driver.execute(
                    "SHOW INDEXES YIELD name, type, options "
                    "WHERE type = 'VECTOR' "
                    "RETURN options.indexConfig.`vector.dimensions` AS dimensions"
                )
                if result:
                    db_dim = result[0].get("dimensions")
                    status.vector_index_exists = (
                        db_dim is not None
                        and int(db_dim) == self._embedder.dimensions
                    )
                else:
                    status.vector_index_exists = False
            except Exception:
                status.vector_index_exists = False

        # LLM reachability (optional: skip extra round-trip before first ingest)
        if ping_llm:
            try:
                status.llm_reachable = await self._llm.ping()
            except Exception:
                status.llm_reachable = False
        else:
            status.llm_reachable = False

        # Embedding model
        try:
            status.embedding_model_loaded = self._embedder.dimensions > 0
        except Exception:
            status.embedding_model_loaded = False

        # Schema version
        if status.neo4j_connected:
            try:
                result = await self._driver.execute(
                    "MATCH (n) WHERE n._schemaVersion IS NOT NULL "
                    "RETURN max(n._schemaVersion) AS v"
                )
                if result and result[0].get("v") is not None:
                    status.schema_version_current = int(result[0]["v"]) >= self._schema_version
                else:
                    status.schema_version_current = True  # empty graph is current
            except Exception:
                status.schema_version_current = False

        return status
