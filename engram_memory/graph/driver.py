"""Async Neo4j driver wrapper with connection pooling and health ping."""

from __future__ import annotations

import logging
from typing import Any

from neo4j import AsyncGraphDatabase

logger = logging.getLogger(__name__)


class GraphDriver:
    """Thin async wrapper around the Neo4j Python driver."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        max_pool_size: int = 50,
    ) -> None:
        self._uri = uri
        self._database = database
        try:
            self._driver = AsyncGraphDatabase.driver(
                uri, auth=(user, password), max_connection_pool_size=max_pool_size
            )
        except Exception:
            self._driver = None
            logger.warning("Failed to create Neo4j driver for %s", uri)

    async def execute(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Run a Cypher query and return result records as dicts.

        Parameters may be passed as a single dict or as keyword arguments
        (e.g. ``userId=..., nodeId=...``), matching Neo4j driver's style.
        """
        merged: dict[str, Any] = {}
        if parameters:
            merged.update(parameters)
        merged.update(kwargs)
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, merged)
            return [record.data() async for record in result]

    async def execute_write(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Run a write transaction with the driver's built-in retry."""

        merged: dict[str, Any] = {}
        if parameters:
            merged.update(parameters)
        merged.update(kwargs)

        async def _tx_work(tx):
            result = await tx.run(query, merged)
            return [record.data() async for record in result]

        async with self._driver.session(database=self._database) as session:
            return await session.execute_write(_tx_work)

    async def ping(self) -> bool:
        """Return True if the database is reachable."""
        try:
            if self._driver is None:
                return False
            async with self._driver.session(database=self._database) as session:
                await session.run("RETURN 1")
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Shut down the driver and release connections."""
        if self._driver is not None:
            await self._driver.close()
