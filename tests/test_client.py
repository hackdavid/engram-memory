"""Phase 5: AsyncMemoryClient and MemoryClient tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from engram_memory.exceptions import HasRelationshipsError, InvalidUserIdError
from engram_memory.models import IngestResult, RecallResult


@pytest.fixture
def mock_components():
    """Create mocked versions of all SDK components."""
    return {
        "driver": AsyncMock(),
        "llm": AsyncMock(),
        "embedder": MagicMock(),
        "extractor": AsyncMock(),
        "engine": MagicMock(),
        "traversal": AsyncMock(),
        "scorer": MagicMock(),
        "hierarchy": AsyncMock(),
        "cache": AsyncMock(),
        "health_checker": AsyncMock(),
    }


def _make_client(mock_components):
    from engram_memory.client import AsyncMemoryClient

    client = AsyncMemoryClient.__new__(AsyncMemoryClient)
    client._init_from_mocks(**mock_components)
    return client


# ── Ingest flow ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_trivial_message_skips(mock_components):
    client = _make_client(mock_components)
    with patch("engram_memory.client.is_trivial", return_value=True):
        result = await client.ingest(user_id="u1", text="ok thanks")
    assert result.skipped is True
    mock_components["extractor"].extract.assert_not_called()


@pytest.mark.asyncio
async def test_ingest_factual_calls_extractor(mock_components):
    client = _make_client(mock_components)
    mock_components["embedder"].encode.return_value = [0.1, 0.2, 0.3]
    mock_components["driver"].execute.return_value = []
    mock_components["extractor"].extract.return_value = (
        [MagicMock(
            operation="create", label="Person", merge_keys={"name": "A"},
            properties={}, summary="A", cluster_hint="Work", supersedes_ref=None,
        )],
        [],
    )
    mock_components["engine"].build_upsert.return_value = ("MERGE ...", {"p": 1})

    with patch("engram_memory.client.is_trivial", return_value=False):
        result = await client.ingest(user_id="u1", text="I work at Google")
    assert result.skipped is False
    mock_components["extractor"].extract.assert_awaited_once()


@pytest.mark.asyncio
async def test_ingest_with_reference_id(mock_components):
    client = _make_client(mock_components)
    mock_components["embedder"].encode.return_value = [0.1]
    mock_components["driver"].execute.return_value = []
    mock_components["extractor"].extract.return_value = (
        [MagicMock(
            operation="create", label="P", merge_keys={"n": "A"},
            properties={}, summary="A", cluster_hint=None, supersedes_ref=None,
        )],
        [],
    )
    mock_components["engine"].build_upsert.return_value = ("Q", {})

    with patch("engram_memory.client.is_trivial", return_value=False):
        await client.ingest(user_id="u1", text="test", reference_id="ref-99")
    call_kwargs = mock_components["engine"].build_upsert.call_args
    assert "ref-99" in str(call_kwargs)


@pytest.mark.asyncio
async def test_ingest_invalidates_cache(mock_components):
    client = _make_client(mock_components)
    mock_components["embedder"].encode.return_value = [0.1]
    mock_components["driver"].execute.return_value = []
    mock_components["extractor"].extract.return_value = ([], [])

    with patch("engram_memory.client.is_trivial", return_value=False):
        await client.ingest(user_id="u1", text="test")
    mock_components["cache"].invalidate_user.assert_awaited_with("u1")


@pytest.mark.asyncio
async def test_ingest_creates_relationships(mock_components):
    client = _make_client(mock_components)
    mock_components["embedder"].encode.return_value = [0.1]
    mock_components["driver"].execute.return_value = [{"elementId": "eid1"}]
    mock_components["extractor"].extract.return_value = (
        [MagicMock(
            operation="create", label="Person", merge_keys={"name": "A"},
            properties={}, summary="A", cluster_hint=None, supersedes_ref=None,
        )],
        [MagicMock(from_ref="temp_0", to_ref="eid2", type="KNOWS", properties={})],
    )
    mock_components["engine"].build_upsert.return_value = ("MERGE ...", {})
    mock_components["engine"].build_relationship.return_value = ("MATCH ...", {})

    with patch("engram_memory.client.is_trivial", return_value=False):
        result = await client.ingest(user_id="u1", text="test")
    assert result.relationships_created == 1
    mock_components["engine"].build_relationship.assert_called_once()


# ── Batch ingest ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_batch_processes_non_trivial_only(mock_components):
    client = _make_client(mock_components)
    mock_components["embedder"].encode.return_value = [0.1]
    mock_components["driver"].execute.return_value = []
    mock_components["extractor"].extract.return_value = ([], [])

    with patch("engram_memory.client.is_trivial", side_effect=[True, False, True]):
        results = await client.ingest_batch(
            user_id="u1",
            items=[
                {"text": "ok"},
                {"text": "I work at X"},
                {"text": "thanks"},
            ],
        )
    assert mock_components["extractor"].extract.await_count == 1
    assert len(results) == 3
    assert results[0].skipped is True
    assert results[2].skipped is True


# ── Recall flow ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recall_returns_from_cache(mock_components):
    client = _make_client(mock_components)
    cached = RecallResult(nodes=[], total_candidates=5, from_cache=True)
    mock_components["cache"].get.return_value = cached
    result = await client.recall(user_id="u1", query="test")
    assert result.from_cache is True
    mock_components["driver"].execute.assert_not_called()


@pytest.mark.asyncio
async def test_recall_cache_miss_queries_graph(mock_components):
    client = _make_client(mock_components)
    mock_components["cache"].get.return_value = None
    mock_components["embedder"].encode.return_value = [0.1, 0.2]
    mock_components["driver"].execute.return_value = [
        {"elementId": "n1", "similarity": 0.9, "summary": "test", "label": "Person",
         "properties": {}, "referenceId": None, "strength": 0.8}
    ]
    mock_components["traversal"].traverse.return_value = []
    mock_components["scorer"].rank.return_value = [
        {"elementId": "n1", "final_score": 0.85, "hops": 0, "summary": "test",
         "label": "Person", "properties": {}, "referenceId": None, "is_current": True}
    ]

    result = await client.recall(user_id="u1", query="test", top_k=5)
    assert result.from_cache is False
    mock_components["cache"].set.assert_awaited()


# ── Delete ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_cascade(mock_components):
    client = _make_client(mock_components)
    mock_components["driver"].execute.return_value = [{"deleted": 1}]
    await client.delete_memory(user_id="u1", node_id="eid1", cascade=True)
    cypher = mock_components["driver"].execute.call_args.args[0]
    assert "DETACH DELETE" in cypher


@pytest.mark.asyncio
async def test_delete_no_cascade_with_rels_raises(mock_components):
    client = _make_client(mock_components)
    mock_components["driver"].execute.return_value = [{"rel_count": 3}]
    with pytest.raises(HasRelationshipsError):
        await client.delete_memory(user_id="u1", node_id="eid1", cascade=False)


@pytest.mark.asyncio
async def test_delete_no_cascade_no_rels_succeeds(mock_components):
    client = _make_client(mock_components)
    mock_components["driver"].execute.side_effect = [
        [{"rel_count": 0}],
        [{"deleted": 1}],
    ]
    await client.delete_memory(user_id="u1", node_id="eid1", cascade=False)


# ── get_graph ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_graph_returns_snapshot(mock_components):
    client = _make_client(mock_components)
    mock_components["driver"].execute.side_effect = [
        [{"elementId": "n1", "labels": ["Person"], "props": {}}],
        [{"total": 1}],
        [{"from_id": "n1", "type": "KNOWS", "to_id": "n2", "props": {}}],
        [{"total": 1}],
    ]
    snap = await client.get_graph(user_id="u1", page=1, page_size=10)
    assert snap.total_nodes == 1
    assert snap.total_relationships == 1
    assert snap.page == 1


# ── userId validation ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_rejects_invalid_user_id(mock_components):
    client = _make_client(mock_components)
    with pytest.raises(InvalidUserIdError):
        await client.ingest(user_id="bad user!!", text="test")


@pytest.mark.asyncio
async def test_recall_rejects_invalid_user_id(mock_components):
    client = _make_client(mock_components)
    with pytest.raises(InvalidUserIdError):
        await client.recall(user_id="", query="test")


@pytest.mark.asyncio
async def test_delete_rejects_invalid_user_id(mock_components):
    client = _make_client(mock_components)
    with pytest.raises(InvalidUserIdError):
        await client.delete_memory(user_id="bad user!!", node_id="x")


# ── Health check ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_check_delegates(mock_components):
    from engram_memory.models import HealthStatus

    client = _make_client(mock_components)
    mock_components["health_checker"].check.return_value = HealthStatus(
        neo4j_connected=True, llm_reachable=True,
        embedding_model_loaded=True, vector_index_exists=True,
        schema_version_current=True,
    )
    status = await client.health_check()
    assert status.is_healthy() is True


# ── Search (hierarchical) ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_uses_hierarchy(mock_components):
    client = _make_client(mock_components)
    mock_components["embedder"].encode.return_value = [0.1, 0.2]
    mock_components["hierarchy"].query_hierarchy.return_value = [
        {"elementId": "c1", "summary": "Work stuff", "score": 0.9}
    ]
    result = await client.search(user_id="u1", query="work", top_k=5)
    assert len(result.nodes) == 1
    mock_components["hierarchy"].query_hierarchy.assert_awaited_once()


# ── Sync wrapper ─────────────────────────────────────────────────────


def test_sync_client_class_exists():
    from engram_memory.client import MemoryClient

    assert hasattr(MemoryClient, "ingest")
    assert hasattr(MemoryClient, "recall")
    assert hasattr(MemoryClient, "delete_memory")
    assert hasattr(MemoryClient, "health_check")
    assert hasattr(MemoryClient, "close")
