"""Unified LLM adapter via LiteLLM -- supports 100+ providers through a single interface."""

from __future__ import annotations

import os
from typing import Any

try:
    import litellm
except ImportError:
    litellm = None  # type: ignore[assignment]

from engram_memory.llm.base import BaseLLM


class LiteLLMAdapter(BaseLLM):
    """Calls any LLM provider through LiteLLM's unified acompletion() API.

    Supports Azure OpenAI, OpenAI, Anthropic, Google, Mistral, Ollama,
    AWS Bedrock, and 100+ other providers.

    For Azure:
        model="azure/<deployment-name>"
        api_key="...", api_base="https://xxx.openai.azure.com/", api_version="2024-02-01"

    For non-Azure:
        model="gpt-4o-mini" or "anthropic/claude-sonnet-4-20250514" or "gemini/gemini-2.5-pro"
        api_key="..." (or set the provider env var, e.g. OPENAI_API_KEY)
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        max_retries: int = 3,
        circuit_breaker_threshold: int = 10,
        max_tokens: int = 4096,
        extra_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            max_retries=max_retries,
            circuit_breaker_threshold=circuit_breaker_threshold,
            base_delay=0.5,
        )
        if litellm is None:
            raise ImportError("litellm is required: pip install litellm")

        self._litellm = litellm
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._api_version = api_version
        self._max_tokens = max_tokens
        self._extra_params = extra_params or {}

        self._litellm.drop_params = True

    async def _call(self, system: str, user: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self._max_tokens,
        }

        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._api_version:
            kwargs["api_version"] = self._api_version

        kwargs.update(self._extra_params)

        response = await self._litellm.acompletion(**kwargs)
        return response.choices[0].message.content

    async def ping(self) -> bool:
        """Quick health check -- send a trivial prompt."""
        try:
            await self._call(system="Reply OK.", user="ping")
            return True
        except Exception:
            return False
