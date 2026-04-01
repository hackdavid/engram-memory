"""Phase 7: Full integration test -- end-to-end lifecycle with mocked components."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from engram_memory.client import AsyncMemoryClient
from engram_memory.models import HealthStatus, IngestResult, RecallResult


@pytest.fixture
def mock_components():
    embedder = MagicMock()
    del embedder.encode_async

    llm = AsyncMock()
    llm.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    return {
        "driver": AsyncMock(),
        "llm": llm,
        "embedder": embedder,
        "extractor": AsyncMock(),
        "engine": MagicMock(),
        "traversal": AsyncMock(),
        "scorer": MagicMock(),
        "hierarchy": AsyncMock(),
        "cache": AsyncMock(),
        "health_checker": AsyncMock(),
    }


def _make_client(mocks):
    client = AsyncMemoryClient.__new__(AsyncMemoryClient)
    client._init_from_mocks(**mocks)
    return client


@pytest.mark.asyncio
async def test_full_lifecycle(mock_components):
    """
    End-to-end:
      1. Health check
      2. Ingest a factual message
      3. Recall the fact
      4. Get graph snapshot
      5. Search (hierarchical)
      6. Delete a node (cascade)
      7. Verify cache was invalidated after both ingest and delete
    """
    mc = mock_components
    client = _make_client(mc)

    # -- 1. Health check --
    mc["health_checker"].check.return_value = HealthStatus(
        neo4j_connected=True,
        vector_index_exists=True,
        llm_reachable=True,
        embedding_model_loaded=True,
        schema_version_current=True,
    )
    health = await client.health_check()
    assert health.is_healthy()

    # -- 2. Ingest --
    mc["embedder"].encode.return_value = [0.1, 0.2, 0.3]
    mc["driver"].execute.side_effect = [
        [],  # context query
        [{"elementId": "eid-alice"}],  # batch node upsert
        [{"type": "WORKS_AT"}],  # batch rel upsert
    ]
    mc["extractor"].extract.return_value = (
        [MagicMock(
            operation="create", label="Person", merge_keys={"name": "Alice"},
            properties={"occupation": "engineer"}, summary="Alice is an engineer",
            cluster_hint="Work", supersedes_ref=None,
        )],
        [MagicMock(
            from_ref="temp_0", to_ref="eid-google", type="WORKS_AT",
            properties={"since": 2020},
        )],
    )
    mc["engine"].build_grouped_batch_upsert.return_value = [
        ("MERGE ...", {"batch": [], "userId": "u1", "schemaVersion": 1})
    ]
    mc["engine"].build_batch_relationships.return_value = [
        ("MATCH ...", {"batch": [], "userId": "u1"})
    ]

    with patch("engram_memory.client.is_trivial", return_value=False):
        ingest_result = await client.ingest(
            user_id="u1",
            text="Alice is an engineer at Google since 2020",
            reference_id="msg-001",
        )

    assert not ingest_result.skipped
    assert len(ingest_result.nodes_created) == 1
    assert ingest_result.relationships_created == 1
    mc["cache"].invalidate_user.assert_awaited_with("u1")
    mc["driver"].execute.side_effect = None

    # -- 3. Recall --
    mc["cache"].get.return_value = None  # cache miss
    mc["driver"].execute.return_value = [
        {
            "elementId": "eid-alice", "similarity": 0.92,
            "summary": "Alice is an engineer", "label": "Person",
            "properties": {"occupation": "engineer"},
            "referenceId": "msg-001", "strength": 0.95,
        }
    ]
    mc["traversal"].traverse.return_value = [
        {"elementId": "eid-google", "score": 0.4, "hops": 1, "label": "Company"}
    ]
    mc["scorer"].rank.return_value = [
        {
            "elementId": "eid-alice", "final_score": 0.87, "hops": 0,
            "summary": "Alice is an engineer", "label": "Person",
            "properties": {"occupation": "engineer"},
            "referenceId": "msg-001", "is_current": True,
        },
        {
            "elementId": "eid-google", "final_score": 0.52, "hops": 1,
            "summary": "", "label": "Company",
            "properties": {}, "referenceId": None, "is_current": True,
        },
    ]

    recall_result = await client.recall(user_id="u1", query="work experience", top_k=5)
    assert not recall_result.from_cache
    assert len(recall_result.nodes) == 2
    assert recall_result.nodes[0].score == 0.87
    mc["cache"].set.assert_awaited()

    # -- 4. Get graph --
    mc["driver"].execute.side_effect = [
        [{"elementId": "eid-alice", "labels": ["Person"], "props": {}}],
        [{"total": 1}],
        [{"from_id": "eid-alice", "type": "WORKS_AT", "to_id": "eid-google", "props": {}}],
        [{"total": 1}],
    ]
    graph = await client.get_graph(user_id="u1", page=1, page_size=10)
    assert graph.total_nodes == 1
    assert graph.total_relationships == 1
    mc["driver"].execute.side_effect = None

    # -- 5. Search (hierarchical) --
    mc["hierarchy"].query_hierarchy.return_value = [
        {"elementId": "cluster-work", "summary": "Work-related memories", "score": 0.88}
    ]
    search_result = await client.search(
        user_id="u1", query="career", top_k=5, detail_level="auto"
    )
    assert len(search_result.nodes) == 1

    # -- 6. Delete --
    mc["driver"].execute.return_value = [{"deleted": 1}]
    mc["cache"].invalidate_user.reset_mock()
    await client.delete_memory(user_id="u1", node_id="eid-alice", cascade=True)
    cypher = mc["driver"].execute.call_args.args[0]
    assert "DETACH DELETE" in cypher
    mc["cache"].invalidate_user.assert_awaited_with("u1")


@pytest.mark.asyncio
async def test_batch_ingest_lifecycle(mock_components):
    """Batch ingest filters trivial and processes factual messages."""
    mc = mock_components
    client = _make_client(mc)

    mc["embedder"].encode.return_value = [0.1]
    mc["driver"].execute.return_value = []
    mc["extractor"].extract.return_value = (
        [MagicMock(
            operation="create", label="Fact", merge_keys={"k": "v"},
            properties={}, summary="fact", cluster_hint=None, supersedes_ref=None,
        )],
        [],
    )
    mc["engine"].build_grouped_batch_upsert.return_value = [
        ("MERGE ...", {"batch": [], "userId": "u1", "schemaVersion": 1})
    ]
    mc["engine"].build_batch_relationships.return_value = None

    with patch("engram_memory.client.is_trivial", side_effect=[True, False, True, False]):
        results = await client.ingest_batch(
            user_id="u1",
            items=[
                {"text": "hi"},
                {"text": "I study at MIT"},
                {"text": "ok"},
                {"text": "My GPA is 3.9"},
            ],
        )

    assert len(results) == 4
    assert results[0].skipped is True
    assert results[1].skipped is False
    assert results[2].skipped is True
    assert results[3].skipped is False
    assert mc["extractor"].extract.await_count == 2


@pytest.mark.asyncio
async def test_recall_cache_hit_lifecycle(mock_components):
    """When cache has a result, recall returns immediately without graph queries."""
    mc = mock_components
    client = _make_client(mc)

    cached = RecallResult(
        nodes=[], total_candidates=3, from_cache=True,
    )
    mc["cache"].get.return_value = cached

    result = await client.recall(user_id="u1", query="anything")
    assert result.from_cache is True
    mc["driver"].execute.assert_not_called()
    mc["traversal"].traverse.assert_not_called()
    mc["scorer"].rank.assert_not_called()


@pytest.mark.asyncio
async def test_delete_safe_mode_lifecycle(mock_components):
    """Non-cascade delete checks relationships first."""
    mc = mock_components
    client = _make_client(mc)

    mc["driver"].execute.side_effect = [
        [{"rel_count": 0}],  # check: no rels
        [{"deleted": 1}],    # safe delete
    ]
    await client.delete_memory(user_id="u1", node_id="eid1", cascade=False)
    assert mc["driver"].execute.call_count == 2
