"""Abstract base class for LLM providers with retry and circuit breaker."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from engram_memory.exceptions import CircuitOpenError, ExtractionError

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Interface that all Engram LLM providers must implement.

    Subclasses implement _call(). This base class provides retry with
    exponential backoff and a circuit breaker.
    """

    def __init__(
        self,
        max_retries: int = 3,
        circuit_breaker_threshold: int = 10,
        base_delay: float = 1.0,
    ) -> None:
        self._max_retries = max_retries
        self._cb_threshold = circuit_breaker_threshold
        self._base_delay = base_delay
        self._consecutive_failures = 0
        self._circuit_open = False

    @abstractmethod
    async def _call(self, system: str, user: str) -> str:
        """Make the actual API call and return the raw text response."""

    async def generate_json(
        self, system: str, user: str
    ) -> dict[str, Any]:
        """Call the LLM with retry, parse the response as JSON.

        Raises ExtractionError after exhausting retries.
        Raises CircuitOpenError if the breaker has tripped.
        """
        if self._circuit_open:
            raise CircuitOpenError(
                f"Circuit breaker is open after {self._cb_threshold} consecutive failures. "
                "Wait before retrying."
            )

        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                raw = await self._call(system, user)
                parsed = self._parse_json(raw)
                self._consecutive_failures = 0
                return parsed
            except Exception as exc:
                last_error = exc
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._cb_threshold:
                    self._circuit_open = True
                if attempt < self._max_retries:
                    delay = self._base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "LLM attempt %d/%d failed: %s. Retrying in %.1fs.",
                        attempt, self._max_retries, exc, delay,
                    )
                    await asyncio.sleep(delay)

        raise ExtractionError(
            f"LLM failed after {self._max_retries} attempts: {last_error}"
        )

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """Extract and parse JSON from the LLM response text."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # drop opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return json.loads(text)
