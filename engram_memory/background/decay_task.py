"""Background task: apply strength decay and archive low-strength nodes."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_DECAY_QUERY = (
    "MATCH (n) WHERE n.strength IS NOT NULL AND n._archived IS NOT true "
    "SET n.strength = n.strength * $decayFactor "
    "RETURN count(n) AS updated"
)

_ARCHIVE_QUERY = (
    "MATCH (n) WHERE n.strength IS NOT NULL "
    "AND n.strength < $threshold AND n._archived IS NOT true "
    "SET n._archived = true, n.isCurrent = false "
    "RETURN count(n) AS archived"
)


class DecayTask:
    """Periodic task that decays node strengths and archives weak nodes."""

    def __init__(
        self,
        driver,
        decay_factor: float = 0.95,
        archive_threshold: float = 0.01,
    ) -> None:
        self._driver = driver
        self._decay_factor = decay_factor
        self._threshold = archive_threshold

    async def run(self) -> None:
        result = await self._driver.execute(
            _DECAY_QUERY, decayFactor=self._decay_factor,
        )
        updated = result[0].get("updated", 0) if result else 0
        logger.info("Decay applied to %d nodes (factor=%.3f).", updated, self._decay_factor)

        result = await self._driver.execute(
            _ARCHIVE_QUERY, threshold=self._threshold,
        )
        archived = result[0].get("archived", 0) if result else 0
        if archived:
            logger.info("Archived %d nodes below threshold %.4f.", archived, self._threshold)
