"""Engram SDK Pydantic data contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from engram.constants import SDK_SCHEMA_VERSION


# ── LLM instruction models (ingestion input) ────────────────────────


class NodeInstruction(BaseModel):
    """Instruction from the LLM to create or update a graph node."""

    operation: Literal["create", "update"]
    ref: str | None = None
    label: str
    merge_keys: dict[str, Any]
    properties: dict[str, Any]
    summary: str
    cluster_hint: str | None = None
    supersedes_ref: str | None = None


class RelInstruction(BaseModel):
    """Instruction from the LLM to create a relationship."""

    from_ref: str
    to_ref: str
    type: str
    properties: dict[str, Any] = Field(default_factory=dict)


# ── Result models (ingestion output) ────────────────────────────────


class NodeResult(BaseModel):
    """Result of a single node create/update operation."""

    element_id: str
    label: str
    reference_id: str | None = None
    operation: Literal["created", "updated"]
    schema_version: int = SDK_SCHEMA_VERSION


class RelResult(BaseModel):
    """Result of a single relationship creation."""

    from_id: str
    to_id: str
    type: str


class IngestResult(BaseModel):
    """Aggregate result of an ingest operation."""

    skipped: bool = False
    nodes_created: list[NodeResult] = Field(default_factory=list)
    nodes_updated: list[NodeResult] = Field(default_factory=list)
    relationships_created: int = 0


# ── Retrieval models ────────────────────────────────────────────────


class ScoredNode(BaseModel):
    """A graph node with a composite relevance score."""

    element_id: str
    label: str
    properties: dict[str, Any]
    summary: str
    reference_id: str | None = None
    score: float
    hops_from_seed: int
    schema_version: int = SDK_SCHEMA_VERSION
    is_current: bool = True


class RecallResult(BaseModel):
    """Result of a recall (retrieval) operation."""

    nodes: list[ScoredNode] = Field(default_factory=list)
    total_candidates: int = 0
    from_cache: bool = False
    has_more: bool = False
    cursor: str | None = None


class GraphSnapshot(BaseModel):
    """Paginated snapshot of a user's graph."""

    nodes: list[dict[str, Any]] = Field(default_factory=list)
    relationships: list[dict[str, Any]] = Field(default_factory=list)
    total_nodes: int = 0
    total_relationships: int = 0
    page: int = 1
    page_size: int = 100
    has_more: bool = False


# ── Health ──────────────────────────────────────────────────────────


class HealthStatus(BaseModel):
    """Aggregated health-check result."""

    neo4j_connected: bool = False
    vector_index_exists: bool = False
    llm_reachable: bool = False
    embedding_model_loaded: bool = False
    schema_version_current: bool = False

    def is_healthy(self) -> bool:
        return all(
            [
                self.neo4j_connected,
                self.vector_index_exists,
                self.llm_reachable,
                self.embedding_model_loaded,
                self.schema_version_current,
            ]
        )
