"""Local embedding using SentenceTransformers."""

from __future__ import annotations

import logging
import os

try:
    import sentence_transformers
except ImportError:
    sentence_transformers = None  # type: ignore[assignment,misc]

from engram.embeddings.base import BaseEmbedding


def _silence_transformers_load_noise() -> None:
    """Reduce Hugging Face / BertModel weight-map advisory noise on stderr (e.g. position_ids UNEXPECTED)."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    for name in (
        "transformers",
        "transformers.modeling_utils",
        "sentence_transformers",
        "sentence_transformers.SentenceTransformer",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


class LocalEmbedding(BaseEmbedding):
    """Wraps a SentenceTransformer model for local embedding generation."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        if sentence_transformers is None:
            raise ImportError(
                "sentence-transformers is required: pip install engram[local-embed]"
            )
        _silence_transformers_load_noise()
        self._model = sentence_transformers.SentenceTransformer(
            model_name,
            show_progress_bar=False,
        )

    @property
    def dimensions(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def encode(self, text: str) -> list[float]:
        vector = self._model.encode([text])[0]
        return [float(v) for v in vector]

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts)
        return [[float(v) for v in vec] for vec in vectors]
