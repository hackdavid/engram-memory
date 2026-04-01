"""Abstract base class for LLM providers with retry and circuit breaker."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from engram_memory.exceptions import CircuitOpenError, ExtractionError

logger = logging.getLogger(__name__)

_ZERO_USAGE: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class BaseLLM(ABC):
    """Interface that all Engram LLM providers must implement.

    Subclasses implement _call() returning (content, usage_dict). This base
    class provides retry with exponential backoff and a circuit breaker, and
    exposes last_usage so callers can track token consumption.
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
        self._last_usage: dict[str, int] = dict(_ZERO_USAGE)

    @property
    def last_usage(self) -> dict[str, int]:
        """Token counts from the most recent successful LLM call."""
        return self._last_usage

    @abstractmethod
    async def _call(self, system: str, user: str) -> tuple[str, dict[str, int]]:
        """Make the actual API call.

        Returns:
            (content, usage) where usage has keys prompt_tokens,
            completion_tokens, total_tokens.
        """

    async def generate_json(
        self, system: str, user: str
    ) -> dict[str, Any]:
        """Call the LLM with retry, parse the response as JSON.

        After a successful call, token counts are available via self.last_usage.

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
                raw, usage = await self._call(system, user)
                parsed = self._parse_json(raw)
                self._consecutive_failures = 0
                self._last_usage = usage
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
