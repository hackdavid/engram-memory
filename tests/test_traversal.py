"""Phase 3: Decay-weighted BFS traversal tests."""

import pytest
from unittest.mock import AsyncMock

from engram.graph.traversal import TraversalEngine


@pytest.fixture
def mock_driver():
    return AsyncMock()


@pytest.mark.asyncio
async def test_traversal_from_single_seed(mock_driver):
    mock_driver.execute.side_effect = [
        [{"elementId": "n1", "score": 0.8, "label": "Person"},
         {"elementId": "n2", "score": 0.7, "label": "Company"}],
        [{"elementId": "n3", "score": 0.5, "label": "Skill"}],
        [],
        [],  # expansion of n3 at depth 2
    ]
    engine = TraversalEngine(driver=mock_driver, decay=0.5, max_depth=3, min_score=0.1)
    results = await engine.traverse(
        seeds=[{"elementId": "s1", "score": 1.0}],
        user_id="u1",
    )
    eids = {r["elementId"] for r in results}
    assert "n1" in eids
    assert "n2" in eids
    assert "n3" in eids
    assert len(results) == 4  # seed + n1 + n2 + n3


@pytest.mark.asyncio
async def test_traversal_respects_max_depth(mock_driver):
    mock_driver.execute.return_value = [
        {"elementId": f"n{i}", "score": 0.9, "label": "Node"} for i in range(10)
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
    mock_driver.execute.return_value = [
        {"elementId": "n1", "score": 0.05, "label": "Node"}
    ]
    engine = TraversalEngine(driver=mock_driver, decay=0.5, max_depth=5, min_score=0.1)
    results = await engine.traverse(
        seeds=[{"elementId": "s1", "score": 0.15}],
        user_id="u1",
    )
    deep_nodes = [r for r in results if r.get("hops", 0) > 0]
    assert len(deep_nodes) == 0


@pytest.mark.asyncio
async def test_traversal_no_cycles(mock_driver):
    call_count = 0

    async def side_effect(query, **params):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return [{"elementId": "s1", "score": 0.9, "label": "Node"}]
        return []

    mock_driver.execute.side_effect = side_effect
    engine = TraversalEngine(driver=mock_driver, decay=0.5, max_depth=5, min_score=0.1)
    results = await engine.traverse(
        seeds=[{"elementId": "s1", "score": 1.0}], user_id="u1"
    )
    assert call_count <= 5


@pytest.mark.asyncio
async def test_traversal_empty_seeds(mock_driver):
    engine = TraversalEngine(driver=mock_driver, decay=0.5, max_depth=3, min_score=0.1)
    results = await engine.traverse(seeds=[], user_id="u1")
    assert results == []


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
