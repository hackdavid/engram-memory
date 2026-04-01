"""Built-in logging hook -- logs lifecycle events at INFO level."""

from __future__ import annotations

import logging
from typing import Any

from engram_memory.models import IngestResult, RecallResult

logger = logging.getLogger(__name__)


class LoggerHook:
    """Default hook that logs all lifecycle events."""

    async def pre_ingest(self, user_id: str, text: str) -> None:
        logger.info("[hook] pre_ingest  user=%s len=%d", user_id, len(text))
        return None

    async def post_ingest(self, user_id: str, result: IngestResult) -> None:
        logger.info(
            "[hook] post_ingest user=%s skipped=%s nodes_created=%d rels=%d",
            user_id, result.skipped, len(result.nodes_created), result.relationships_created,
        )

    async def pre_recall(self, user_id: str, query: str) -> None:
        logger.info("[hook] pre_recall  user=%s query=%s", user_id, query[:80])
        return None

    async def post_recall(self, user_id: str, result: RecallResult) -> None:
        logger.info(
            "[hook] post_recall user=%s nodes=%d cached=%s",
            user_id, len(result.nodes), result.from_cache,
        )
