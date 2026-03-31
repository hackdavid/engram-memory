"""Phase 2: Embedding provider tests (all mocked, no model downloads)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from engram.embeddings.base import BaseEmbedding


def test_base_embedding_is_abstract():
    with pytest.raises(TypeError):
        BaseEmbedding()


# ── SentenceTransformer (mocked) ────────────────────────────────────


def test_sentence_transformer_encode():
    import numpy as np

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

    import engram.embeddings.sentence_transformer as st_mod

    mock_pkg = MagicMock()
    mock_pkg.SentenceTransformer = MagicMock(return_value=mock_model)
    original = st_mod.sentence_transformers
    st_mod.sentence_transformers = mock_pkg
    try:
        emb = st_mod.LocalEmbedding(model_name="all-MiniLM-L6-v2")
        result = emb.encode("hello world")
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)
    finally:
        st_mod.sentence_transformers = original


def test_sentence_transformer_batch_encode():
    import numpy as np

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

    import engram.embeddings.sentence_transformer as st_mod

    mock_pkg = MagicMock()
    mock_pkg.SentenceTransformer = MagicMock(return_value=mock_model)
    original = st_mod.sentence_transformers
    st_mod.sentence_transformers = mock_pkg
    try:
        emb = st_mod.LocalEmbedding(model_name="all-MiniLM-L6-v2")
        results = emb.encode_batch(["hello", "world"])
        assert len(results) == 2
        assert len(results[0]) == 2
    finally:
        st_mod.sentence_transformers = original


def test_sentence_transformer_dimensions():
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384

    import engram.embeddings.sentence_transformer as st_mod

    mock_pkg = MagicMock()
    mock_pkg.SentenceTransformer = MagicMock(return_value=mock_model)
    original = st_mod.sentence_transformers
    st_mod.sentence_transformers = mock_pkg
    try:
        emb = st_mod.LocalEmbedding(model_name="all-MiniLM-L6-v2")
        assert emb.dimensions == 384
    finally:
        st_mod.sentence_transformers = original


# ── OpenAI Embedding (mocked) ───────────────────────────────────────


@pytest.mark.asyncio
async def test_openai_embedding_encode():
    mock_client = AsyncMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1, 0.2, 0.3])]
    )

    import engram.embeddings.openai_embedding as oai_mod

    original = oai_mod.AsyncOpenAI
    oai_mod.AsyncOpenAI = lambda **kw: mock_client
    try:
        emb = oai_mod.OpenAIEmbedding(api_key="sk-test", model="text-embedding-3-small")
        result = await emb.encode_async("hello")
        assert len(result) == 3
    finally:
        oai_mod.AsyncOpenAI = original


def test_openai_sync_encode_raises():
    mock_client = MagicMock()

    import engram.embeddings.openai_embedding as oai_mod

    original = oai_mod.AsyncOpenAI
    oai_mod.AsyncOpenAI = lambda **kw: mock_client
    try:
        emb = oai_mod.OpenAIEmbedding(api_key="sk-test")
        with pytest.raises(NotImplementedError):
            emb.encode("test")
    finally:
        oai_mod.AsyncOpenAI = original


# ── Two-Tier Embedding ──────────────────────────────────────────────


def test_two_tier_quantize():
    from engram.embeddings.two_tier import TwoTierEmbedder

    fine = [0.123, -0.456, 0.789]
    coarse = TwoTierEmbedder.quantize(fine)
    assert all(isinstance(v, int) for v in coarse)
    assert all(-128 <= v <= 127 for v in coarse)


def test_two_tier_quantize_empty():
    from engram.embeddings.two_tier import TwoTierEmbedder

    assert TwoTierEmbedder.quantize([]) == []


def test_two_tier_wraps_base_embedder():
    mock_base = MagicMock()
    mock_base.encode.return_value = [0.5, -0.3, 0.8]
    mock_base.dimensions = 3

    from engram.embeddings.two_tier import TwoTierEmbedder

    two_tier = TwoTierEmbedder(base=mock_base)
    fine, coarse = two_tier.encode_both("test text")
    assert len(fine) == 3
    assert len(coarse) == 3
    assert all(isinstance(v, float) for v in fine)
    assert all(isinstance(v, int) for v in coarse)


def test_two_tier_dimensions():
    mock_base = MagicMock()
    mock_base.dimensions = 384

    from engram.embeddings.two_tier import TwoTierEmbedder

    two_tier = TwoTierEmbedder(base=mock_base)
    assert two_tier.dimensions == 384
