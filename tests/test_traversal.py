"""Phase 3: Decay-weighted graph traversal tests (single-query implementation)."""

import pytest
from unittest.mock import AsyncMock

from engram_memory.graph.traversal import TraversalEngine


@pytest.fixture
def mock_driver():
    return AsyncMock()


@pytest.mark.asyncio
async def test_traversal_from_single_seed(mock_driver):
    mock_driver.execute.return_value = [
        {"elementId": "n1", "hops": 1, "strength": 0.8, "label": "Person"},
        {"elementId": "n2", "hops": 1, "strength": 0.7, "label": "Company"},
        {"elementId": "n3", "hops": 2, "strength": 0.5, "label": "Skill"},
    ]
    engine = TraversalEngine(driver=mock_driver, decay=0.5, max_depth=3, min_score=0.1)
    results = await engine.traverse(
        seeds=[{"elementId": "s1", "score": 1.0}],
        user_id="u1",
    )
    eids = {r["elementId"] for r in results}
    assert "s1" in eids
    assert "n1" in eids
    assert "n2" in eids
    assert "n3" in eids
    assert len(results) == 4  # seed + n1 + n2 + n3


@pytest.mark.asyncio
async def test_traversal_respects_max_depth(mock_driver):
    mock_driver.execute.return_value = [
        {"elementId": "n1", "hops": 1, "strength": 0.9, "label": "Node"},
        {"elementId": "n2", "hops": 1, "strength": 0.9, "label": "Node"},
    ]
    engine = TraversalEngine(driver=mock_driver, decay=0.5, max_depth=1, min_score=0.01)
    results = await engine.traverse(
        seeds=[{"elementId": "s1", "score": 1.0}],
        user_id="u1",
    )
    for r in results:
        assert r.get("hops", 0) <= 1


@pytest.mark.asyncio
async def test_traversal_stops_below_min_score(mock_driver):
    # decay=0.5 at hops=4 -> 0.5^4 = 0.0625 < min_score=0.1 -> filtered out
    mock_driver.execute.return_value = [
        {"elementId": "n1", "hops": 4, "strength": 0.05, "label": "Node"},
    ]
    engine = TraversalEngine(driver=mock_driver, decay=0.5, max_depth=5, min_score=0.1)
    results = await engine.traverse(
        seeds=[{"elementId": "s1", "score": 0.15}],
        user_id="u1",
    )
    deep_nodes = [r for r in results if r.get("hops", 0) > 0]
    assert len(deep_nodes) == 0


@pytest.mark.asyncio
async def test_traversal_no_duplicate_seeds(mock_driver):
    """Seed nodes returned by the Cypher query should not be duplicated."""
    mock_driver.execute.return_value = [
        {"elementId": "s1", "hops": 1, "strength": 0.9, "label": "Node"},
        {"elementId": "n1", "hops": 1, "strength": 0.8, "label": "Node"},
    ]
    engine = TraversalEngine(driver=mock_driver, decay=0.5, max_depth=5, min_score=0.1)
    results = await engine.traverse(
        seeds=[{"elementId": "s1", "score": 1.0}], user_id="u1"
    )
    s1_entries = [r for r in results if r["elementId"] == "s1"]
    assert len(s1_entries) == 1


@pytest.mark.asyncio
async def test_traversal_empty_seeds(mock_driver):
    engine = TraversalEngine(driver=mock_driver, decay=0.5, max_depth=3, min_score=0.1)
    results = await engine.traverse(seeds=[], user_id="u1")
    assert results == []
    mock_driver.execute.assert_not_called()


@pytest.mark.asyncio
async def test_traversal_includes_seed_in_results(mock_driver):
    mock_driver.execute.return_value = []
    engine = TraversalEngine(driver=mock_driver, decay=0.5, max_depth=3, min_score=0.1)
    results = await engine.traverse(
        seeds=[{"elementId": "seed1", "score": 1.0}], user_id="u1"
    )
    assert len(results) == 1
    assert results[0]["elementId"] == "seed1"
    assert results[0]["hops"] == 0
