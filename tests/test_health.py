"""Phase 4: Health checker tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from engram_memory.health.checks import HealthChecker


@pytest.mark.asyncio
async def test_skip_llm_ping():
    driver = AsyncMock()
    driver.ping.return_value = True
    driver.execute.return_value = [{"dimensions": 384}]
    llm = AsyncMock()
    llm.ping.return_value = True
    embedder = MagicMock()
    embedder.dimensions = 384

    checker = HealthChecker(driver=driver, llm=llm, embedder=embedder, schema_version=1)
    status = await checker.check(ping_llm=False)
    llm.ping.assert_not_awaited()
    assert status.llm_reachable is False
    assert status.neo4j_connected is True


@pytest.mark.asyncio
async def test_all_healthy():
    driver = AsyncMock()
    driver.ping.return_value = True
    driver.execute.return_value = [{"dimensions": 384}]
    llm = AsyncMock()
    llm.ping.return_value = True
    embedder = MagicMock()
    embedder.dimensions = 384

    checker = HealthChecker(driver=driver, llm=llm, embedder=embedder, schema_version=1)
    status = await checker.check()
    assert status.neo4j_connected is True
    assert status.llm_reachable is True
    assert status.embedding_model_loaded is True
    assert status.vector_index_exists is True
    assert status.is_healthy() is True


@pytest.mark.asyncio
async def test_neo4j_down():
    driver = AsyncMock()
    driver.ping.return_value = False
    llm = AsyncMock()
    llm.ping.return_value = True
    embedder = MagicMock()
    embedder.dimensions = 384

    checker = HealthChecker(driver=driver, llm=llm, embedder=embedder, schema_version=1)
    status = await checker.check()
    assert status.neo4j_connected is False
    assert status.is_healthy() is False


@pytest.mark.asyncio
async def test_llm_unreachable():
    driver = AsyncMock()
    driver.ping.return_value = True
    driver.execute.return_value = [{"dimensions": 384}]
    llm = AsyncMock()
    llm.ping.side_effect = Exception("timeout")
    embedder = MagicMock()
    embedder.dimensions = 384

    checker = HealthChecker(driver=driver, llm=llm, embedder=embedder, schema_version=1)
    status = await checker.check()
    assert status.llm_reachable is False


@pytest.mark.asyncio
async def test_no_vector_index():
    driver = AsyncMock()
    driver.ping.return_value = True
    driver.execute.return_value = []
    llm = AsyncMock()
    llm.ping.return_value = True
    embedder = MagicMock()
    embedder.dimensions = 384

    checker = HealthChecker(driver=driver, llm=llm, embedder=embedder, schema_version=1)
    status = await checker.check()
    assert status.vector_index_exists is False


@pytest.mark.asyncio
async def test_embedding_model_not_loaded():
    driver = AsyncMock()
    driver.ping.return_value = True
    driver.execute.return_value = [{"dimensions": 384}]
    llm = AsyncMock()
    llm.ping.return_value = True
    embedder = MagicMock()
    embedder.dimensions = 0

    checker = HealthChecker(driver=driver, llm=llm, embedder=embedder, schema_version=1)
    status = await checker.check()
    assert status.embedding_model_loaded is False
