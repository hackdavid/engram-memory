"""LLM-powered entity/relationship extractor."""

from __future__ import annotations

import logging
from typing import Any

from engram.exceptions import ExtractionError
from engram.extractors.base import BaseExtractor
from engram.extractors.prompts import SYSTEM_PROMPT, build_user_prompt
from engram.models import NodeInstruction, RelInstruction

logger = logging.getLogger(__name__)


class LLMExtractor(BaseExtractor):
    """Calls an LLM adapter to extract nodes and relationships from text."""

    def __init__(self, llm) -> None:
        self._llm = llm

    async def extract(
        self,
        user_id: str,
        text: str,
        neighbourhood: list[dict[str, Any]],
    ) -> tuple[list[NodeInstruction], list[RelInstruction]]:
        user_prompt = build_user_prompt(text, neighbourhood)
        raw = await self._llm.generate_json(
            system=SYSTEM_PROMPT,
            user=user_prompt,
        )

        if "nodes" not in raw or "relationships" not in raw:
            raise ExtractionError(
                f"LLM output missing required keys 'nodes'/'relationships': {list(raw.keys())}"
            )

        try:
            nodes = [NodeInstruction(**n) for n in raw["nodes"]]
            rels = [RelInstruction(**r) for r in raw["relationships"]]
        except Exception as exc:
            raise ExtractionError(
                f"Failed to validate LLM output: {exc}"
            ) from exc

        logger.info(
            "Extracted %d nodes, %d rels for user %s",
            len(nodes), len(rels), user_id,
        )
        return nodes, rels
