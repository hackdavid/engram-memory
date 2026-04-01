"""Phase 2: Dynamic Cypher engine tests."""

import pytest

from engram_memory.graph.engine import CypherEngine


@pytest.fixture
def engine():
    return CypherEngine()


def test_build_upsert_create(engine):
    cypher, params = engine.build_upsert(
        label="Person",
        merge_keys={"name": "Alice"},
        properties={"age": 30, "city": "London"},
        embedding=[0.1, 0.2, 0.3],
        user_id="u1",
        reference_id="ref-1",
        schema_version=1,
    )
    assert "MERGE" in cypher
    assert "Person" in cypher
    assert "_EngramNode" in cypher
    assert "name:$name" in cypher.replace(" ", "").replace("\n", "")
    assert params["userId"] == "u1"
    assert params["referenceId"] == "ref-1"
    assert "Alice" not in cypher


def test_build_upsert_parameterised_no_injection(engine):
    """Semicolons and spaces are stripped; values stay in params, not cypher."""
    cypher, params = engine.build_upsert(
        label="Person; DROP DATABASE",
        merge_keys={"name": "Alice"},
        properties={},
        embedding=[0.1],
        user_id="u1",
    )
    assert ";" not in cypher
    assert " DROP " not in cypher
    assert "Alice" not in cypher
    assert "Alice" in params.values()


def test_build_upsert_with_version_lock(engine):
    cypher, params = engine.build_upsert(
        label="Person",
        merge_keys={"name": "Alice"},
        properties={"age": 31},
        embedding=[0.1],
        user_id="u1",
        expected_version=2,
    )
    assert "_version" in cypher
    assert params.get("expected_version") == 2


def test_build_upsert_no_reference_id(engine):
    cypher, params = engine.build_upsert(
        label="Skill",
        merge_keys={"name": "Python"},
        properties={"level": "expert"},
        embedding=[0.5],
        user_id="u1",
    )
    assert "referenceId" not in params
    assert "MERGE" in cypher


def test_build_relationship(engine):
    cypher, params = engine.build_relationship(
        from_ref="eid1",
        to_ref="eid2",
        rel_type="WORKS_AT",
        properties={"confidence": 0.9},
        user_id="u1",
    )
    assert "WORKS_AT" in cypher
    assert params.get("confidence") == 0.9
    assert "MERGE" in cypher


def test_build_relationship_sanitises_type(engine):
    cypher, _ = engine.build_relationship(
        from_ref="eid1",
        to_ref="eid2",
        rel_type="WORKS AT!!",
        user_id="u1",
    )
    assert "WORKS_AT" in cypher
    assert "!!" not in cypher


def test_build_relationship_no_properties(engine):
    cypher, params = engine.build_relationship(
        from_ref="eid1",
        to_ref="eid2",
        rel_type="KNOWS",
        user_id="u1",
    )
    assert "KNOWS" in cypher
    assert params["from_ref"] == "eid1"


def test_build_batch_upsert(engine):
    items = [
        {"label": "Person", "merge_keys": {"name": "A"}, "properties": {}, "embedding": [0.1]},
        {"label": "Person", "merge_keys": {"name": "B"}, "properties": {}, "embedding": [0.2]},
    ]
    cypher, params = engine.build_batch_upsert(items, user_id="u1")
    assert "UNWIND" in cypher
    assert len(params["batch"]) == 2


def test_build_batch_upsert_empty(engine):
    cypher, params = engine.build_batch_upsert([], user_id="u1")
    assert params["batch"] == []
