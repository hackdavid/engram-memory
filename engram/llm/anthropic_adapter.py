"""Anthropic Claude LLM adapter."""

from __future__ import annotations

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None  # type: ignore[assignment,misc]

from engram.llm.base import BaseLLM


class AnthropicLLM(BaseLLM):
    """LLM adapter for Anthropic's Claude API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
        circuit_breaker_threshold: int = 10,
        base_url: str | None = None,
    ) -> None:
        super().__init__(
            max_retries=max_retries,
            circuit_breaker_threshold=circuit_breaker_threshold,
            base_delay=0.5,
        )
        if AsyncAnthropic is None:
            raise ImportError("anthropic is required: pip install engram[anthropic]")
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._model = model

    async def _call(self, system: str, user: str) -> str:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text
