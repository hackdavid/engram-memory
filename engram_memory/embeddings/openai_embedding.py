"""OpenAI API-based embedding provider (async)."""

from __future__ import annotations

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment,misc]

from engram_memory.embeddings.base import BaseEmbedding


class OpenAIEmbedding(BaseEmbedding):
    """Async embedding via OpenAI's text-embedding API.

    The sync encode()/encode_batch() raise NotImplementedError since
    OpenAI requires async I/O. Use encode_async()/encode_batch_async().
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
    ) -> None:
        if AsyncOpenAI is None:
            raise ImportError("openai is required: pip install engram_memory[openai]")
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def encode(self, text: str) -> list[float]:
        raise NotImplementedError("Use encode_async() -- OpenAIEmbedding is async-only")

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Use encode_batch_async() -- OpenAIEmbedding is async-only")

    async def encode_async(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            input=[text], model=self._model
        )
        return response.data[0].embedding

    async def encode_batch_async(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            input=texts, model=self._model
        )
        return [item.embedding for item in response.data]
