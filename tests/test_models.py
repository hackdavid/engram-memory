"""Phase 1: Verify all Pydantic data contracts."""

import pytest

from engram_memory.models import (
    GraphSnapshot,
    HealthStatus,
    IngestResult,
    NodeInstruction,
    NodeResult,
    RecallResult,
    RelInstruction,
    ScoredNode,
)


# ── NodeInstruction ─────────────────────────────────────────────────


def test_node_instruction_create():
    ni = NodeInstruction(
        operation="create",
        label="Person",
        merge_keys={"name": "Alice"},
        properties={"age": 30},
        summary="Alice is 30 years old",
    )
    assert ni.operation == "create"
    assert ni.cluster_hint is None
    assert ni.supersedes_ref is None


def test_node_instruction_update_with_ref():
    ni = NodeInstruction(
        operation="update",
        ref="eid-123",
        label="Person",
        merge_keys={"name": "Alice"},
        properties={"age": 31},
        summary="Alice is now 31",
        supersedes_ref="eid-old",
    )
    assert ni.operation == "update"
    assert ni.ref == "eid-123"
    assert ni.supersedes_ref == "eid-old"


def test_node_instruction_invalid_operation():
    with pytest.raises(Exception):
        NodeInstruction(
            operation="delete",
            label="Person",
            merge_keys={"name": "Alice"},
            properties={},
            summary="test",
        )


# ── RelInstruction ──────────────────────────────────────────────────


def test_rel_instruction_with_properties():
    ri = RelInstruction(
        from_ref="ref1",
        to_ref="ref2",
        type="WORKS_AT",
        properties={"confidence": 0.95, "source": "direct_mention"},
    )
    assert ri.properties["confidence"] == 0.95


def test_rel_instruction_default_empty_properties():
    ri = RelInstruction(from_ref="a", to_ref="b", type="KNOWS")
    assert ri.properties == {}


# ── IngestResult ────────────────────────────────────────────────────


def test_ingest_result_skipped():
    result = IngestResult(skipped=True)
    assert result.nodes_created == []
    assert result.nodes_updated == []
    assert result.relationships_created == 0


def test_ingest_result_with_data():
    nr = NodeResult(element_id="eid1", label="Person", operation="created")
    result = IngestResult(nodes_created=[nr], relationships_created=2)
    assert len(result.nodes_created) == 1
    assert result.relationships_created == 2


# ── RecallResult ────────────────────────────────────────────────────


def test_recall_result_pagination():
    result = RecallResult(
        nodes=[], total_candidates=100, has_more=True, cursor="abc123"
    )
    assert result.has_more is True
    assert result.from_cache is False
    assert result.cursor == "abc123"


# ── ScoredNode ──────────────────────────────────────────────────────


def test_scored_node_defaults():
    sn = ScoredNode(
        element_id="eid1",
        label="Company",
        properties={"name": "Google"},
        summary="Google is a tech company",
        score=0.85,
        hops_from_seed=1,
    )
    assert sn.is_current is True
    assert sn.schema_version >= 1
    assert sn.reference_id is None


# ── GraphSnapshot ───────────────────────────────────────────────────


def test_graph_snapshot_pagination():
    gs = GraphSnapshot(
        nodes=[],
        relationships=[],
        total_nodes=500,
        total_relationships=800,
        page=2,
        page_size=100,
        has_more=True,
    )
    assert gs.page == 2
    assert gs.has_more is True


def test_graph_snapshot_defaults():
    gs = GraphSnapshot()
    assert gs.nodes == []
    assert gs.page == 1
    assert gs.page_size == 100
    assert gs.has_more is False


# ── HealthStatus ────────────────────────────────────────────────────


def test_health_status_all_healthy():
    hs = HealthStatus(
        neo4j_connected=True,
        vector_index_exists=True,
        llm_reachable=True,
        embedding_model_loaded=True,
        schema_version_current=True,
    )
    assert hs.is_healthy() is True


def test_health_status_partial_failure():
    hs = HealthStatus(
        neo4j_connected=True,
        vector_index_exists=True,
        llm_reachable=False,
        embedding_model_loaded=True,
        schema_version_current=True,
    )
    assert hs.is_healthy() is False


def test_health_status_defaults_unhealthy():
    hs = HealthStatus()
    assert hs.is_healthy() is False
