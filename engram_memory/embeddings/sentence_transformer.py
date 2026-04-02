"""Local embedding using SentenceTransformers."""

from __future__ import annotations

import inspect
import logging
import os
import sys

try:
    import sentence_transformers
except ImportError:
    sentence_transformers = None  # type: ignore[assignment,misc]

from engram_memory.embeddings.base import BaseEmbedding


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
                "sentence-transformers is required: pip install engram-memory-sdk"
            )
        _silence_transformers_load_noise()
        print(
            "[engram_memory] Loading SentenceTransformer model "
            f"{model_name!r} (first run may download weights from Hugging Face; "
            "this can take a minute)...",
            file=sys.stderr,
            flush=True,
        )
        # show_progress_bar was removed from SentenceTransformer.__init__ in newer sentence-transformers.
        _st_kw: dict = {}
        if "show_progress_bar" in inspect.signature(
            sentence_transformers.SentenceTransformer
        ).parameters:
            _st_kw["show_progress_bar"] = False
        self._model = sentence_transformers.SentenceTransformer(model_name, **_st_kw)
        dim = int(self._model.get_sentence_embedding_dimension())
        print(
            f"[engram_memory] SentenceTransformer ready: {model_name!r}, {dim} dimensions.",
            file=sys.stderr,
            flush=True,
        )

    @property
    def dimensions(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def _encode_arrays(self, texts: list[str]):
        if "show_progress_bar" in inspect.signature(self._model.encode).parameters:
            return self._model.encode(texts, show_progress_bar=False)
        return self._model.encode(texts)

    def encode(self, text: str) -> list[float]:
        vector = self._encode_arrays([text])[0]
        return [float(v) for v in vector]

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = self._encode_arrays(texts)
        return [[float(v) for v in vec] for vec in vectors]
