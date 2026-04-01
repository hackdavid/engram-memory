"""Phase 2: Neo4j async driver wrapper tests (all mocked)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_neo4j_driver():
    """Build a mock that properly simulates AsyncDriver.session() as async CM."""
    driver = MagicMock()
    session = AsyncMock()

    cm = AsyncMock()
    cm.__aenter__.return_value = session
    cm.__aexit__.return_value = False
    driver.session.return_value = cm
    driver.close = AsyncMock()

    return driver, session


@pytest.mark.asyncio
async def test_driver_executes_query(mock_neo4j_driver):
    from engram_memory.graph.driver import GraphDriver

    driver_mock, session_mock = mock_neo4j_driver

    mock_record = MagicMock()
    mock_record.data.return_value = {"n": 1}

    mock_result = MagicMock()
    mock_result.__aiter__ = lambda self: iter([mock_record]).__aiter__() if False else self
    records = [mock_record]

    async def async_iter(self):
        for r in records:
            yield r

    mock_result.__aiter__ = async_iter
    session_mock.run.return_value = mock_result

    with patch("engram_memory.graph.driver.AsyncGraphDatabase") as mock_agd:
        mock_agd.driver.return_value = driver_mock
        gd = GraphDriver(uri="bolt://localhost:7687", user="neo4j", password="test")
        result = await gd.execute("RETURN 1 AS n")
        session_mock.run.assert_called_once()
        assert result == [{"n": 1}]


@pytest.mark.asyncio
async def test_driver_ping_success(mock_neo4j_driver):
    from engram_memory.graph.driver import GraphDriver

    driver_mock, session_mock = mock_neo4j_driver
    session_mock.run.return_value = AsyncMock()

    with patch("engram_memory.graph.driver.AsyncGraphDatabase") as mock_agd:
        mock_agd.driver.return_value = driver_mock
        gd = GraphDriver(uri="bolt://localhost:7687", user="neo4j", password="test")
        assert await gd.ping() is True


@pytest.mark.asyncio
async def test_driver_ping_failure():
    from engram_memory.graph.driver import GraphDriver

    with patch("engram_memory.graph.driver.AsyncGraphDatabase") as mock_agd:
        mock_agd.driver.side_effect = Exception("conn refused")
        gd = GraphDriver(uri="bolt://bad:7687", user="neo4j", password="test")
        assert await gd.ping() is False


@pytest.mark.asyncio
async def test_driver_close(mock_neo4j_driver):
    from engram_memory.graph.driver import GraphDriver

    driver_mock, _ = mock_neo4j_driver
    with patch("engram_memory.graph.driver.AsyncGraphDatabase") as mock_agd:
        mock_agd.driver.return_value = driver_mock
        gd = GraphDriver(uri="bolt://localhost:7687", user="neo4j", password="test")
        await gd.close()
        driver_mock.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_driver_default_database():
    from engram_memory.graph.driver import GraphDriver

    with patch("engram_memory.graph.driver.AsyncGraphDatabase") as mock_agd:
        mock_agd.driver.return_value = MagicMock()
        gd = GraphDriver(uri="bolt://localhost:7687", user="neo4j", password="test")
        assert gd._database == "neo4j"
