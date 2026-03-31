"""Vector index creation and embedding dimension guard."""

from __future__ import annotations

import logging

from engram.exceptions import EmbeddingDimensionMismatchError

logger = logging.getLogger(__name__)

INDEX_NAME = "engram_embedding_index"
# Every node with `_embedding` queried by vector search must carry this label.
VECTOR_INDEX_NODE_LABEL = "_EngramNode"


class IndexManager:
    """Manages Neo4j vector indexes for Engram."""

    def __init__(self, driver, embedding_dimensions: int) -> None:
        self._driver = driver
        self._dimensions = embedding_dimensions

    async def ensure_vector_index(self) -> None:
        """Create the vector index if missing, or verify dimensions match."""
        existing = await self._driver.execute(
            "SHOW INDEXES YIELD name, type, options "
            "WHERE name = $name "
            "RETURN options.indexConfig.`vector.dimensions` AS dimensions",
            {"name": INDEX_NAME},
        )

        if existing:
            db_dim = existing[0].get("dimensions")
            if db_dim is not None and int(db_dim) != self._dimensions:
                raise EmbeddingDimensionMismatchError(
                    f"Existing index '{INDEX_NAME}' has {db_dim} dimensions, "
                    f"but config expects {self._dimensions}. "
                    f"Drop the index and re-create, or change embedding_dimensions."
                )
            logger.info("Vector index '%s' already exists with matching dimensions.", INDEX_NAME)
            return

        await self._driver.execute(
            f"CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS "
            f"FOR (n:{VECTOR_INDEX_NODE_LABEL}) ON (n._embedding) "
            f"OPTIONS {{indexConfig: {{"
            f"  `vector.dimensions`: {self._dimensions},"
            f"  `vector.similarity_function`: 'cosine'"
            f"}}}}"
        )
        logger.info("Created vector index '%s' (%d dimensions).", INDEX_NAME, self._dimensions)
