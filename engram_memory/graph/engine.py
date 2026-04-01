"""Dynamic Cypher generator -- parameterised queries from arbitrary labels and properties."""

from __future__ import annotations

import json
from typing import Any

from engram_memory.constants import SDK_SCHEMA_VERSION
from engram_memory.graph.indexes import VECTOR_INDEX_NODE_LABEL
from engram_memory.graph.sanitise import sanitise_label, sanitise_rel_type


def _sanitise_property_value(value: Any) -> Any:
    """Coerce a value into a Neo4j-compatible primitive.

    Neo4j properties only accept primitives (str, int, float, bool)
    and homogeneous arrays thereof.  Nested dicts/lists-of-dicts are
    JSON-serialised to strings so the LLM's output never causes a
    CypherTypeError at write time.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return json.dumps(value, default=str)
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, (bool, int, float, str)) for v in value):
            return list(value)
        return json.dumps(value, default=str)
    return str(value)


def _sanitise_props(props: dict[str, Any]) -> dict[str, Any]:
    return {k: _sanitise_property_value(v) for k, v in props.items()}


class CypherEngine:
    """Generates parameterised Cypher for dynamic graph operations."""

    def build_upsert(
        self,
        label: str,
        merge_keys: dict[str, Any],
        properties: dict[str, Any],
        embedding: list[float],
        user_id: str,
        reference_id: str | None = None,
        schema_version: int = SDK_SCHEMA_VERSION,
        expected_version: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Build a MERGE + SET Cypher for upserting a node.

        All values are parameterised. The label is sanitised via regex.
        If expected_version is given, a WHERE clause enforces optimistic locking.
        """
        safe_label = sanitise_label(label)
        # Vector index is FOR (n:_EngramNode); merged nodes must include that label.
        typed = f"{safe_label}:{VECTOR_INDEX_NODE_LABEL}"

        # MERGE map syntax requires `propKey: $param`, not `n.propKey = $param`.
        merge_clause_parts = [f"{k}: ${k}" for k in merge_keys]
        merge_where = ", ".join(merge_clause_parts)

        params: dict[str, Any] = {
            **merge_keys,
            "userId": user_id,
            "_embedding": embedding,
            "_schemaVersion": schema_version,
        }
        if reference_id is not None:
            params["referenceId"] = reference_id

        safe_props = _sanitise_props(properties)
        set_parts = []
        for k, v in safe_props.items():
            param_key = f"prop_{k}"
            params[param_key] = v
            set_parts.append(f"n.{k} = ${param_key}")

        set_parts.extend([
            "n.userId = $userId",
            "n._embedding = $_embedding",
            "n._schemaVersion = $_schemaVersion",
            "n.strength = COALESCE(n.strength, 1.0)",
            "n.isCurrent = COALESCE(n.isCurrent, true)",
            "n.lastAccessed = datetime()",
        ])

        if reference_id is not None:
            set_parts.append("n.referenceId = $referenceId")

        if expected_version is not None:
            params["expected_version"] = expected_version
            version_clause = (
                f"MERGE (n:{typed} {{{merge_where}}})\n"
                f"  ON MATCH SET n._version = CASE WHEN n._version = $expected_version "
                f"THEN n._version + 1 ELSE n._version END\n"
                f"  ON CREATE SET n._version = 1\n"
                f"SET {', '.join(set_parts)}\n"
                f"RETURN elementId(n) AS elementId"
            )
            return version_clause, params

        cypher = (
            f"MERGE (n:{typed} {{{merge_where}}})\n"
            f"  ON CREATE SET n._version = 1\n"
            f"  ON MATCH SET n._version = n._version + 1\n"
            f"SET {', '.join(set_parts)}\n"
            f"RETURN elementId(n) AS elementId"
        )
        return cypher, params

    def build_relationship(
        self,
        from_ref: str,
        to_ref: str,
        rel_type: str,
        user_id: str,
        properties: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Build a MERGE for a relationship between two nodes by elementId."""
        safe_type = sanitise_rel_type(rel_type)
        params: dict[str, Any] = {
            "from_ref": from_ref,
            "to_ref": to_ref,
            "userId": user_id,
        }

        rel_set_parts = [
            "r._traversalCount = COALESCE(r._traversalCount, 0)",
            "r._lastTraversed = COALESCE(r._lastTraversed, datetime())",
        ]

        if properties:
            safe_rel_props = _sanitise_props(properties)
            for k, v in safe_rel_props.items():
                param_key = f"rel_{k}"
                params[param_key] = v
                rel_set_parts.append(f"r.{k} = ${param_key}")
                params[k] = v

        set_clause = ", ".join(rel_set_parts)

        cypher = (
            f"MATCH (a) WHERE elementId(a) = $from_ref AND a.userId = $userId\n"
            f"MATCH (b) WHERE elementId(b) = $to_ref AND b.userId = $userId\n"
            f"MERGE (a)-[r:{safe_type}]->(b)\n"
            f"SET {set_clause}\n"
            f"RETURN type(r) AS type"
        )
        return cypher, params

    def build_batch_upsert(
        self,
        items: list[dict[str, Any]],
        user_id: str,
        schema_version: int = SDK_SCHEMA_VERSION,
    ) -> tuple[str, dict[str, Any]]:
        """Build a batched UNWIND upsert for multiple nodes sharing the same label."""
        if not items:
            return "RETURN 0", {"batch": []}

        safe_label = sanitise_label(items[0]["label"])
        typed = f"{safe_label}:{VECTOR_INDEX_NODE_LABEL}"

        batch = []
        for item in items:
            entry = {
                "merge_keys": item["merge_keys"],
                "properties": item.get("properties", {}),
                "embedding": item["embedding"],
            }
            batch.append(entry)

        cypher = (
            f"UNWIND $batch AS item\n"
            f"MERGE (n:{typed} {{name: item.merge_keys.name}})\n"
            f"  ON CREATE SET n._version = 1, n.userId = $userId, "
            f"n._schemaVersion = $schemaVersion\n"
            f"  ON MATCH SET n._version = n._version + 1\n"
            f"SET n += item.properties, n._embedding = item.embedding, "
            f"n.strength = COALESCE(n.strength, 1.0), n.lastAccessed = datetime()\n"
            f"RETURN elementId(n) AS elementId"
        )
        return cypher, {"batch": batch, "userId": user_id, "schemaVersion": schema_version}

    def build_grouped_batch_upsert(
        self,
        items: list[dict[str, Any]],
        user_id: str,
        schema_version: int = SDK_SCHEMA_VERSION,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Build one UNWIND upsert per distinct label.

        Each item must have: label, merge_keys (dict), properties (dict),
        embedding (list[float]).  Optional: reference_id (str|None).

        Returns a list of (cypher, params) pairs — one per label group.
        """
        if not items:
            return []

        from collections import defaultdict

        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            groups[sanitise_label(item["label"])].append(item)

        queries: list[tuple[str, dict[str, Any]]] = []
        for label, group in groups.items():
            typed = f"{label}:{VECTOR_INDEX_NODE_LABEL}"

            merge_key_names = sorted(group[0]["merge_keys"].keys())
            merge_map = ", ".join(f"{k}: item.merge_keys.{k}" for k in merge_key_names)

            batch = []
            for item in group:
                safe_props = _sanitise_props(item.get("properties", {}))
                entry: dict[str, Any] = {
                    "merge_keys": item["merge_keys"],
                    "properties": safe_props,
                    "embedding": item["embedding"],
                }
                if item.get("reference_id") is not None:
                    entry["referenceId"] = item["reference_id"]
                batch.append(entry)

            cypher = (
                f"UNWIND $batch AS item\n"
                f"MERGE (n:{typed} {{{merge_map}}})\n"
                f"  ON CREATE SET n._version = 1\n"
                f"  ON MATCH SET n._version = n._version + 1\n"
                f"SET n += item.properties,\n"
                f"    n.userId = $userId,\n"
                f"    n._embedding = item.embedding,\n"
                f"    n._schemaVersion = $schemaVersion,\n"
                f"    n.strength = COALESCE(n.strength, 1.0),\n"
                f"    n.isCurrent = COALESCE(n.isCurrent, true),\n"
                f"    n.lastAccessed = datetime(),\n"
                f"    n.referenceId = COALESCE(item.referenceId, n.referenceId)\n"
                f"RETURN elementId(n) AS elementId"
            )
            queries.append((cypher, {
                "batch": batch,
                "userId": user_id,
                "schemaVersion": schema_version,
            }))

        return queries

    def build_batch_relationships(
        self,
        rels: list[dict[str, Any]],
        user_id: str,
    ) -> tuple[str, dict[str, Any]] | None:
        """Build a single UNWIND MERGE for a batch of relationships.

        Each rel dict must have: from_ref, to_ref, rel_type.
        Returns None when the list is empty.

        Because UNWIND cannot parameterise the relationship type, we group
        by rel_type and emit one UNWIND per type.
        """
        if not rels:
            return None

        from collections import defaultdict

        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for rel in rels:
            safe_type = sanitise_rel_type(rel["rel_type"])
            groups[safe_type].append({
                "from_ref": rel["from_ref"],
                "to_ref": rel["to_ref"],
            })

        queries: list[tuple[str, dict[str, Any]]] = []
        for rel_type, batch in groups.items():
            cypher = (
                f"UNWIND $batch AS rel\n"
                f"MATCH (a) WHERE elementId(a) = rel.from_ref AND a.userId = $userId\n"
                f"MATCH (b) WHERE elementId(b) = rel.to_ref AND b.userId = $userId\n"
                f"MERGE (a)-[r:{rel_type}]->(b)\n"
                f"SET r._traversalCount = COALESCE(r._traversalCount, 0),\n"
                f"    r._lastTraversed = COALESCE(r._lastTraversed, datetime())\n"
                f"RETURN type(r) AS type"
            )
            queries.append((cypher, {"batch": batch, "userId": user_id}))

        return queries
