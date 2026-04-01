"""Schema versioning and migration runner."""

from __future__ import annotations

import logging
from typing import Any

from engram_memory.exceptions import MigrationError

logger = logging.getLogger(__name__)

MIGRATIONS: dict[int, str] = {
    1: (
        "MATCH (n) WHERE n._schemaVersion IS NULL "
        "SET n._schemaVersion = 1, n.isCurrent = COALESCE(n.isCurrent, true), "
        "n.strength = COALESCE(n.strength, 1.0) "
        "RETURN count(n) AS updated"
    ),
}


class MigrationRunner:
    """Checks the graph's schema version and runs upgrade Cypher sequentially."""

    def __init__(self, driver, target_version: int) -> None:
        self._driver = driver
        self._target = target_version

    async def _get_current_version(self) -> int:
        result = await self._driver.execute(
            "MATCH (n) WHERE n._schemaVersion IS NOT NULL "
            "RETURN max(n._schemaVersion) AS max_version"
        )
        if result and result[0].get("max_version") is not None:
            return int(result[0]["max_version"])
        return 0

    async def check_and_migrate(self) -> int:
        """Run all pending migrations and return the count of migrations applied."""
        current = await self._get_current_version()

        if current >= self._target:
            logger.info("Schema is up to date (v%d).", current)
            return 0

        applied = 0
        for version in range(current + 1, self._target + 1):
            cypher = MIGRATIONS.get(version)
            if cypher is None:
                logger.debug("No migration script for v%d, skipping.", version)
                continue
            try:
                await self._driver.execute(cypher)
                applied += 1
                logger.info("Applied migration v%d.", version)
            except Exception as exc:
                raise MigrationError(
                    f"Migration to v{version} failed: {exc}"
                ) from exc

        return applied
