"""Phase 2: Vector index manager tests."""

import pytest
from unittest.mock import AsyncMock

from engram_memory.exceptions import EmbeddingDimensionMismatchError
from engram_memory.graph.indexes import IndexManager


@pytest.mark.asyncio
async def test_ensure_index_creates_when_missing():
    driver = AsyncMock()
    driver.execute.return_value = []
    mgr = IndexManager(driver, embedding_dimensions=384)
    await mgr.ensure_vector_index()
    assert driver.execute.call_count == 2  # one SHOW, one CREATE
    create_call = driver.execute.call_args_list[-1]
    assert "VECTOR INDEX" in create_call.args[0]


@pytest.mark.asyncio
async def test_dimension_mismatch_raises():
    driver = AsyncMock()
    driver.execute.return_value = [{"dimensions": 768}]
    mgr = IndexManager(driver, embedding_dimensions=384)
    with pytest.raises(EmbeddingDimensionMismatchError):
        await mgr.ensure_vector_index()


@pytest.mark.asyncio
async def test_dimension_match_passes():
    driver = AsyncMock()
    driver.execute.return_value = [{"dimensions": 384}]
    mgr = IndexManager(driver, embedding_dimensions=384)
    await mgr.ensure_vector_index()


@pytest.mark.asyncio
async def test_dimension_none_creates_index():
    driver = AsyncMock()
    driver.execute.return_value = [{"dimensions": None}]
    mgr = IndexManager(driver, embedding_dimensions=384)
    await mgr.ensure_vector_index()
