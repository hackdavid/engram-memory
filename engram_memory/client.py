"""Engram SDK client -- wires all components into a single entry point."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

from engram_memory.cache.lru_cache import MemoryCache
from engram_memory.config import Config
from engram_memory.constants import SDK_SCHEMA_VERSION
from engram_memory.exceptions import HasRelationshipsError
from engram_memory.extractors.llm_extractor import LLMExtractor
from engram_memory.extractors.trivial_filter import is_trivial
from engram_memory.graph.driver import GraphDriver
from engram_memory.graph.engine import CypherEngine
from engram_memory.graph.hierarchy import HierarchyManager
from engram_memory.graph.indexes import IndexManager
from engram_memory.graph.migrations import MigrationRunner
from engram_memory.graph.sanitise import validate_user_id
from engram_memory.graph.scorer import CompositeScorer
from engram_memory.graph.traversal import TraversalEngine
from engram_memory.health.checks import HealthChecker
from engram_memory.models import (
    GraphSnapshot,
    HealthStatus,
    IngestResult,
    NodeResult,
    RecallResult,
    ScoredNode,
)
from engram_memory.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


async def _encode(embedder, text: str) -> list[float]:
    """Call the embedder, using async path if available (e.g. OpenAIEmbedding)."""
    if hasattr(embedder, "encode_async"):
        return await embedder.encode_async(text)
    return embedder.encode(text)

_INTERNAL_PROPERTY_PREFIXES = ("_embedding", "_schemaVersion", "_version")
_INTERNAL_PROPERTY_EXACT = frozenset({
    "isCurrent", "lastAccessed", "userId", "strength", "referenceId",
})

_VECTOR_SEARCH_QUERY = (
    "CALL db.index.vector.queryNodes($indexName, $topK, $queryVector) "
    "YIELD node, score AS similarity "
    "WHERE node.userId = $userId AND node.isCurrent = true "
    "RETURN elementId(node) AS elementId, similarity, "
    "       labels(node)[0] AS label, node.summary AS summary, "
    "       node.strength AS strength, node.referenceId AS referenceId, "
    "       properties(node) AS properties"
)


def _clean_properties(props: dict[str, Any] | None) -> dict[str, Any]:
    """Strip internal/bulky keys from a properties dict before surfacing to users."""
    if not props:
        return {}
    return {
        k: v
        for k, v in props.items()
        if not k.startswith("_") and k not in _INTERNAL_PROPERTY_EXACT
    }

_INGEST_CONTEXT_QUERY = (
    "CALL db.index.vector.queryNodes($indexName, $topK, $queryVector) "
    "YIELD node, score "
    "WHERE node.userId = $userId AND node.isCurrent = true "
    "WITH node, score "
    "OPTIONAL MATCH (node)-[r]-(neighbor) "
    "WHERE neighbor.userId = $userId AND neighbor.isCurrent = true "
    "RETURN elementId(node) AS elementId, "
    "       labels(node)[0] AS label, "
    "       node.summary AS summary, "
    "       collect(DISTINCT type(r)) AS rel_types, "
    "       score AS similarity "
    "ORDER BY similarity DESC"
)

_GRAPH_QUERY = (
    "MATCH (n) WHERE n.userId = $userId "
    "RETURN elementId(n) AS elementId, labels(n) AS labels, "
    "       properties(n) AS props "
    "ORDER BY n.lastAccessed DESC "
    "SKIP $skip LIMIT $limit"
)

_GRAPH_COUNT_QUERY = (
    "MATCH (n) WHERE n.userId = $userId "
    "RETURN count(n) AS total"
)

_REL_QUERY = (
    "MATCH (n)-[r]->(m) WHERE n.userId = $userId "
    "RETURN elementId(n) AS from_id, type(r) AS type, "
    "       elementId(m) AS to_id, properties(r) AS props "
    "SKIP $skip LIMIT $limit"
)

_REL_COUNT_QUERY = (
    "MATCH (n)-[r]->(m) WHERE n.userId = $userId "
    "RETURN count(r) AS total"
)

_DELETE_CASCADE = (
    "MATCH (n) WHERE elementId(n) = $nodeId AND n.userId = $userId "
    "DETACH DELETE n"
)

_DELETE_CHECK_RELS = (
    "MATCH (n) WHERE elementId(n) = $nodeId AND n.userId = $userId "
    "OPTIONAL MATCH (n)-[r]-() "
    "RETURN count(r) AS rel_count"
)

_DELETE_SAFE = (
    "MATCH (n) WHERE elementId(n) = $nodeId AND n.userId = $userId "
    "DELETE n"
)

_NODE_HISTORY = (
    "MATCH (n) WHERE elementId(n) = $nodeId AND n.userId = $userId "
    "OPTIONAL MATCH (n)-[:SUPERSEDES*]->(old) "
    "RETURN elementId(old) AS elementId, old.summary AS summary, "
    "       old._version AS version "
    "ORDER BY old._version DESC"
)


class AsyncMemoryClient:
    """Async entry point for the Engram SDK."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._driver = GraphDriver(
            uri=config.neo4j_uri,
            user=config.neo4j_user,
            password=config.neo4j_password,
            database=config.neo4j_database,
            max_pool_size=config.neo4j_max_pool_size,
        )
        self._engine = CypherEngine()

        from engram_memory.llm.litellm_adapter import LiteLLMAdapter
        llm_extras: dict[str, Any] = {}
        if config.llm_request_timeout is not None:
            llm_extras["timeout"] = float(config.llm_request_timeout)
        self._llm = LiteLLMAdapter(
            model=config.llm_model,
            api_key=config.llm_api_key,
            api_base=config.llm_api_base,
            api_version=config.llm_api_version,
            max_retries=config.llm_max_retries,
            max_tokens=config.llm_max_tokens,
            extra_params=llm_extras or None,
        )

        if config.embedding_provider == "local":
            from engram_memory.embeddings.sentence_transformer import LocalEmbedding
            self._embedder = LocalEmbedding(model_name=config.embedding_model)
        else:
            from engram_memory.embeddings.openai_embedding import OpenAIEmbedding
            self._embedder = OpenAIEmbedding(
                api_key=config.embedding_api_key,
                model=config.embedding_model,
                dimensions=config.embedding_dimensions,
                base_url=config.embedding_api_base,
            )

        if config.two_tier_embedding:
            from engram_memory.embeddings.two_tier import TwoTierEmbedder
            self._embedder = TwoTierEmbedder(base=self._embedder)

        self._extractor = LLMExtractor(llm=self._llm)
        self._traversal = TraversalEngine(
            driver=self._driver,
            decay=config.traversal_decay,
            max_depth=config.traversal_max_depth,
            min_score=config.traversal_min_score,
        )
        self._scorer = CompositeScorer(
            alpha=config.score_alpha,
            beta=config.score_beta,
            gamma=config.score_gamma,
        )
        self._hierarchy = HierarchyManager(
            driver=self._driver, embedder=self._embedder,
        )
        self._cache = MemoryCache(
            max_size=config.cache_max_size,
            ttl_seconds=config.cache_ttl_seconds,
        ) if config.cache_enabled else None
        self._rate_limiter = RateLimiter(
            rpm=config.llm_rate_limit_rpm,
            burst=config.llm_rate_limit_burst,
        )
        self._index_manager = IndexManager(
            driver=self._driver,
            embedding_dimensions=config.embedding_dimensions,
        )
        self._migration_runner = MigrationRunner(
            driver=self._driver, target_version=SDK_SCHEMA_VERSION,
        )
        self._health_checker = HealthChecker(
            driver=self._driver,
            llm=self._llm,
            embedder=self._embedder,
            schema_version=SDK_SCHEMA_VERSION,
        )
        self._user_id_pattern = config.user_id_pattern

    def _init_from_mocks(self, **components: Any) -> None:
        """Inject mocked components for testing."""
        self._driver = components.get("driver")
        self._llm = components.get("llm")
        self._embedder = components.get("embedder")
        self._extractor = components.get("extractor")
        self._engine = components.get("engine")
        self._traversal = components.get("traversal")
        self._scorer = components.get("scorer")
        self._hierarchy = components.get("hierarchy")
        self._cache = components.get("cache")
        self._health_checker = components.get("health_checker")
        self._rate_limiter = None
        self._user_id_pattern = r"^[a-zA-Z0-9_\-\.]{1,128}$"
        self._config = None

    async def __aenter__(self) -> AsyncMemoryClient:
        await self._index_manager.ensure_vector_index()
        if hasattr(self, "_migration_runner"):
            await self._migration_runner.check_and_migrate()
        return self

    async def __aexit__(self, *exc) -> None:
        await self._driver.close()

    # ── Public API ───────────────────────────────────────────────────

    async def ingest(
        self,
        user_id: str,
        text: str,
        reference_id: str | None = None,
    ) -> IngestResult:
        """Ingest a single message into the memory graph.

        Flow:
          1. embed(text) -> query_vector
          2. vector_search(query_vector, top_k=5) -> top-5 similar existing nodes
             (returns only elementId, label, summary, rel_types — no raw properties)
          3. build_user_prompt(text, context_nodes) -> slim prompt
          4. LLM extracts NodeInstructions + RelInstructions
          5. Write nodes (reuse text embedding for nodes whose summary == text;
             compute fresh embedding for node summaries that differ)
          6. Write relationships
          7. Return IngestResult with token counts from the LLM call
        """
        validate_user_id(user_id, self._user_id_pattern)

        if is_trivial(text):
            return IngestResult(skipped=True)

        if self._rate_limiter:
            await self._rate_limiter.acquire()

        # Step 1: embed the input text once — reused for context search AND node storage.
        text_embedding = await _encode(self._embedder, text)

        # Step 2: vector-search for the top-5 most semantically similar existing nodes.
        # Returns slim context: elementId, label, summary, rel_types.
        # Gracefully returns [] when the graph is empty (no index entries yet).
        context_nodes = await self._driver.execute(
            _INGEST_CONTEXT_QUERY,
            indexName="engram_embedding_index",
            topK=5,
            queryVector=text_embedding,
            userId=user_id,
        )

        # Step 3 + 4: build prompt from slim context and call the LLM.
        nodes, rels = await self._extractor.extract(
            user_id=user_id, text=text, neighbourhood=context_nodes,
        )

        # Capture token usage from the LLM call just completed.
        usage = self._llm.last_usage

        # ── Step 5: Prepare node batch ──
        # Collect embeddings + metadata; temp_id -> index mapping for rel resolution.
        batch_items: list[dict[str, Any]] = []
        node_meta: list[dict[str, Any]] = []  # parallel list: operation, label, cluster_hint
        for node in nodes:
            if node.summary and node.summary.strip() != text.strip():
                embedding = await _encode(self._embedder, node.summary)
            else:
                embedding = text_embedding

            node_props = {**node.properties, "summary": node.summary}
            batch_items.append({
                "label": node.label,
                "merge_keys": node.merge_keys,
                "properties": node_props,
                "embedding": embedding,
                "reference_id": reference_id,
            })
            node_meta.append({
                "operation": node.operation,
                "label": node.label,
                "cluster_hint": node.cluster_hint,
            })

        # Execute batched node upserts (one UNWIND per label group).
        queries = self._engine.build_grouped_batch_upsert(
            batch_items, user_id=user_id,
        )
        element_ids: list[str] = []
        for cypher, params in queries:
            rows = await self._driver.execute(cypher, params)
            element_ids.extend(r["elementId"] for r in rows)

        # Build temp_N -> real elementId mapping for relationship resolution.
        temp_to_eid: dict[str, str] = {}
        for idx, eid in enumerate(element_ids):
            temp_to_eid[f"temp_{idx}"] = eid

        created, updated = [], []
        for idx, eid in enumerate(element_ids):
            meta = node_meta[idx]
            nr = NodeResult(
                element_id=eid,
                label=meta["label"],
                reference_id=reference_id,
                operation="created" if meta["operation"] == "create" else "updated",
            )
            if meta["operation"] == "create":
                created.append(nr)
            else:
                updated.append(nr)

            if meta["cluster_hint"]:
                await self._hierarchy.assign_to_cluster(
                    user_id=user_id, node_id=eid, cluster_hint=meta["cluster_hint"],
                )

        # ── Step 6: Batch relationship writes ──
        # Resolve temp_N references to real elementIds.
        rel_batch: list[dict[str, str]] = []
        for rel in rels:
            from_ref = temp_to_eid.get(rel.from_ref, rel.from_ref)
            to_ref = temp_to_eid.get(rel.to_ref, rel.to_ref)
            rel_batch.append({
                "from_ref": from_ref,
                "to_ref": to_ref,
                "rel_type": rel.type,
            })
        rel_queries = self._engine.build_batch_relationships(rel_batch, user_id=user_id)
        rel_count = 0
        if rel_queries:
            for cypher, params in rel_queries:
                rows = await self._driver.execute(cypher, params)
                rel_count += len(rows)

        if self._cache:
            await self._cache.invalidate_user(user_id)

        return IngestResult(
            skipped=False,
            nodes_created=created,
            nodes_updated=updated,
            relationships_created=rel_count,
            tokens_prompt=usage.get("prompt_tokens", 0),
            tokens_completion=usage.get("completion_tokens", 0),
            tokens_total=usage.get("total_tokens", 0),
        )

    async def ingest_batch(
        self,
        user_id: str,
        items: list[dict[str, Any]],
    ) -> list[IngestResult]:
        """Ingest multiple messages, skipping trivial ones."""
        validate_user_id(user_id, self._user_id_pattern)
        results = []
        for item in items:
            result = await self.ingest(
                user_id=user_id,
                text=item["text"],
                reference_id=item.get("reference_id"),
            )
            results.append(result)
        return results

    async def recall(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
    ) -> RecallResult:
        """Retrieve relevant memories for a query."""
        validate_user_id(user_id, self._user_id_pattern)

        cache_key = hashlib.sha256(f"{query}:{top_k}".encode()).hexdigest()[:16]
        if self._cache:
            cached = await self._cache.get(user_id, cache_key)
            if cached is not None:
                return cached

        query_vector = await _encode(self._embedder, query)

        seeds = await self._driver.execute(
            _VECTOR_SEARCH_QUERY,
            indexName="engram_embedding_index",
            topK=top_k,
            queryVector=query_vector,
            userId=user_id,
        )

        seed_entries = [
            {"elementId": s["elementId"], "score": s.get("similarity", 0.0)}
            for s in seeds
        ]
        traversed = await self._traversal.traverse(seeds=seed_entries, user_id=user_id)

        candidates = []
        for s in seeds:
            candidates.append({
                "elementId": s["elementId"],
                "vector_similarity": s.get("similarity", 0.0),
                "hops": 0,
                "strength": s.get("strength", 1.0),
                "summary": s.get("summary", ""),
                "label": s.get("label", ""),
                "properties": _clean_properties(s.get("properties")),
                "referenceId": s.get("referenceId"),
                "is_current": True,
            })
        seen = {s["elementId"] for s in seeds}
        for t in traversed:
            if t["elementId"] not in seen:
                candidates.append({
                    "elementId": t["elementId"],
                    "vector_similarity": 0.0,
                    "hops": t.get("hops", 1),
                    "strength": t.get("score", 0.5),
                    "summary": t.get("summary", ""),
                    "label": t.get("label", ""),
                    "properties": {},
                    "referenceId": None,
                    "is_current": True,
                })
                seen.add(t["elementId"])

        ranked = self._scorer.rank(
            candidates, decay=self._traversal._decay if hasattr(self._traversal, "_decay") else 0.5,
        )[:top_k]

        scored_nodes = [
            ScoredNode(
                element_id=n["elementId"],
                label=n.get("label") or "",
                properties=n.get("properties") or {},
                summary=n.get("summary") or "",
                reference_id=n.get("referenceId"),
                score=n["final_score"],
                hops_from_seed=n.get("hops", 0),
                is_current=n.get("is_current", True),
            )
            for n in ranked
        ]

        result = RecallResult(
            nodes=scored_nodes,
            total_candidates=len(candidates),
            from_cache=False,
        )

        if self._cache:
            await self._cache.set(user_id, cache_key, result)

        return result

    async def get_graph(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 100,
    ) -> GraphSnapshot:
        """Return a paginated snapshot of a user's memory graph."""
        validate_user_id(user_id, self._user_id_pattern)
        skip = (page - 1) * page_size

        nodes = await self._driver.execute(
            _GRAPH_QUERY, userId=user_id, skip=skip, limit=page_size,
        )
        node_count_result = await self._driver.execute(
            _GRAPH_COUNT_QUERY, userId=user_id,
        )
        total_nodes = node_count_result[0]["total"] if node_count_result else 0

        rels = await self._driver.execute(
            _REL_QUERY, userId=user_id, skip=skip, limit=page_size,
        )
        rel_count_result = await self._driver.execute(
            _REL_COUNT_QUERY, userId=user_id,
        )
        total_rels = rel_count_result[0]["total"] if rel_count_result else 0

        return GraphSnapshot(
            nodes=nodes,
            relationships=rels,
            total_nodes=total_nodes,
            total_relationships=total_rels,
            page=page,
            page_size=page_size,
            has_more=(skip + page_size) < total_nodes,
        )

    async def delete_memory(
        self,
        user_id: str,
        node_id: str,
        cascade: bool = False,
    ) -> None:
        """Delete a node. If cascade=False and node has relationships, raise."""
        validate_user_id(user_id, self._user_id_pattern)

        if cascade:
            await self._driver.execute(
                _DELETE_CASCADE, nodeId=node_id, userId=user_id,
            )
        else:
            result = await self._driver.execute(
                _DELETE_CHECK_RELS, nodeId=node_id, userId=user_id,
            )
            rel_count = result[0].get("rel_count", 0) if result else 0
            if rel_count > 0:
                raise HasRelationshipsError(
                    f"Node {node_id} has {rel_count} relationship(s). "
                    "Use cascade=True to delete."
                )
            await self._driver.execute(
                _DELETE_SAFE, nodeId=node_id, userId=user_id,
            )

        if self._cache:
            await self._cache.invalidate_user(user_id)

    async def get_node_history(
        self,
        user_id: str,
        node_id: str,
    ) -> list[dict[str, Any]]:
        """Return the supersession chain for a node."""
        validate_user_id(user_id, self._user_id_pattern)
        return await self._driver.execute(
            _NODE_HISTORY, nodeId=node_id, userId=user_id,
        )

    async def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 10,
        detail_level: str = "auto",
    ) -> RecallResult:
        """Hierarchical search using the summary tree."""
        validate_user_id(user_id, self._user_id_pattern)
        query_vector = await _encode(self._embedder, query)
        results = await self._hierarchy.query_hierarchy(
            user_id=user_id,
            query_vector=query_vector,
            detail_level=detail_level,
            k=top_k,
        )
        scored_nodes = [
            ScoredNode(
                element_id=r.get("elementId", ""),
                label="",
                properties={},
                summary=r.get("summary", ""),
                score=r.get("score", 0.0),
                hops_from_seed=0,
            )
            for r in results
        ]
        return RecallResult(nodes=scored_nodes, total_candidates=len(results))

    async def health_check(self, *, ping_llm: bool = True) -> HealthStatus:
        """Run health checks against all dependencies."""
        return await self._health_checker.check(ping_llm=ping_llm)


class MemoryClient:
    """Synchronous wrapper around AsyncMemoryClient."""

    def __init__(self, config: Config) -> None:
        self._async_client = AsyncMemoryClient(config)

    def _run(self, coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return asyncio.run(coro)

    def ingest(self, user_id: str, text: str, **kwargs) -> IngestResult:
        return self._run(self._async_client.ingest(user_id, text, **kwargs))

    def ingest_batch(self, user_id: str, items: list[dict[str, Any]]) -> list[IngestResult]:
        return self._run(self._async_client.ingest_batch(user_id, items))

    def recall(self, user_id: str, query: str, **kwargs) -> RecallResult:
        return self._run(self._async_client.recall(user_id, query, **kwargs))

    def get_graph(self, user_id: str, **kwargs) -> GraphSnapshot:
        return self._run(self._async_client.get_graph(user_id, **kwargs))

    def delete_memory(self, user_id: str, node_id: str, **kwargs) -> None:
        return self._run(self._async_client.delete_memory(user_id, node_id, **kwargs))

    def search(self, user_id: str, query: str, **kwargs) -> RecallResult:
        return self._run(self._async_client.search(user_id, query, **kwargs))

    def health_check(self, **kwargs) -> HealthStatus:
        return self._run(self._async_client.health_check(**kwargs))

    def close(self) -> None:
        self._run(self._async_client.__aexit__(None, None, None))
