"""OpenAI GPT LLM adapter."""

from __future__ import annotations

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment,misc]

from engram_memory.llm.base import BaseLLM


class OpenAILLM(BaseLLM):
    """LLM adapter for OpenAI's Chat Completions API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        circuit_breaker_threshold: int = 10,
    ) -> None:
        super().__init__(
            max_retries=max_retries,
            circuit_breaker_threshold=circuit_breaker_threshold,
            base_delay=0.5,
        )
        if AsyncOpenAI is None:
            raise ImportError("openai is required: pip install engram_memory[openai]")
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def _call(self, system: str, user: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
