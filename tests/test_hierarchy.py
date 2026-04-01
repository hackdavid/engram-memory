"""Phase 3: Hierarchy manager tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from engram_memory.graph.hierarchy import HierarchyManager


@pytest.fixture
def mock_driver():
    return AsyncMock()


@pytest.fixture
def mock_embedder():
    emb = MagicMock()
    emb.encode.return_value = [0.1, 0.2, 0.3]
    return emb


@pytest.mark.asyncio
async def test_assign_to_existing_cluster(mock_driver, mock_embedder):
    mock_driver.execute.return_value = [
        {"elementId": "cluster_work", "topic": "Work", "embedding": [0.1, 0.2, 0.3]}
    ]
    mgr = HierarchyManager(driver=mock_driver, embedder=mock_embedder)
    cluster_id = await mgr.assign_to_cluster(
        user_id="u1", node_id="n1", cluster_hint="Work"
    )
    assert cluster_id == "cluster_work"


@pytest.mark.asyncio
async def test_assign_creates_new_cluster(mock_driver, mock_embedder):
    mock_driver.execute.side_effect = [
        [],  # no existing clusters
        [{"elementId": "new_cluster"}],  # create returns id
        None,  # link to cluster
    ]
    mgr = HierarchyManager(driver=mock_driver, embedder=mock_embedder)
    cluster_id = await mgr.assign_to_cluster(
        user_id="u1", node_id="n1", cluster_hint="Hobbies"
    )
    assert cluster_id == "new_cluster"


@pytest.mark.asyncio
async def test_assign_no_hint_creates_general(mock_driver, mock_embedder):
    mock_driver.execute.side_effect = [
        [{"elementId": "general_cluster"}],
        None,
    ]
    mgr = HierarchyManager(driver=mock_driver, embedder=mock_embedder)
    cluster_id = await mgr.assign_to_cluster(
        user_id="u1", node_id="n1", cluster_hint=None
    )
    assert cluster_id == "general_cluster"
    mock_embedder.encode.assert_called_once_with("general")


@pytest.mark.asyncio
async def test_rebuild_cluster_summary(mock_driver, mock_embedder):
    mock_driver.execute.side_effect = [
        [{"summary": "Alice works at Google"}, {"summary": "Bob is a PM at Google"}],
        None,  # update call
    ]
    mgr = HierarchyManager(driver=mock_driver, embedder=mock_embedder)
    await mgr.rebuild_cluster_summary(cluster_id="c1")
    mock_embedder.encode.assert_called()


@pytest.mark.asyncio
async def test_query_hierarchy_broad(mock_driver, mock_embedder):
    mock_driver.execute.return_value = [
        {"elementId": "c1", "_level": 1, "summary": "Work stuff", "score": 0.9}
    ]
    mgr = HierarchyManager(driver=mock_driver, embedder=mock_embedder)
    results = await mgr.query_hierarchy(
        user_id="u1", query_vector=[0.1, 0.2, 0.3], detail_level="broad"
    )
    for r in results:
        assert r.get("_level", 0) <= 1


@pytest.mark.asyncio
async def test_query_hierarchy_detailed(mock_driver, mock_embedder):
    mock_driver.execute.return_value = [
        {"elementId": "n1", "_level": 2, "summary": "Google job", "score": 0.9}
    ]
    mgr = HierarchyManager(driver=mock_driver, embedder=mock_embedder)
    results = await mgr.query_hierarchy(
        user_id="u1", query_vector=[0.1, 0.2, 0.3], detail_level="detailed"
    )
    assert len(results) == 1


@pytest.mark.asyncio
async def test_query_hierarchy_auto_falls_through(mock_driver, mock_embedder):
    mock_driver.execute.side_effect = [
        [{"elementId": "c1", "_level": 1, "summary": "Work", "score": 0.3}],
        [{"elementId": "n1", "_level": 2, "summary": "Google job", "score": 0.9}],
    ]
    mgr = HierarchyManager(driver=mock_driver, embedder=mock_embedder)
    results = await mgr.query_hierarchy(
        user_id="u1", query_vector=[0.1, 0.2, 0.3], detail_level="auto"
    )
    assert any(r.get("_level", 0) >= 2 for r in results)


@pytest.mark.asyncio
async def test_query_hierarchy_auto_stays_broad(mock_driver, mock_embedder):
    mock_driver.execute.return_value = [
        {"elementId": "c1", "_level": 1, "summary": "Work", "score": 0.85}
    ]
    mgr = HierarchyManager(driver=mock_driver, embedder=mock_embedder)
    results = await mgr.query_hierarchy(
        user_id="u1", query_vector=[0.1, 0.2, 0.3], detail_level="auto"
    )
    assert len(results) == 1
    assert results[0]["_level"] == 1
