# API overview

## Public imports

```python
from engram import (
    __version__,
    Config,
    AsyncMemoryClient,  # preferred for asyncio
    MemoryClient,       # thin sync wrapper
)
```

Deeper types (`IngestResult`, `RecallResult`, `ScoredNode`, …) live in `engram.models` and `engram.exceptions`.

## Client lifecycle

### Async (recommended)

```python
async with AsyncMemoryClient(config) as client:
    ...
# driver closed automatically
```

### Sync

```python
with MemoryClient(config) as client:
    ...
# or: client = MemoryClient(config); ...; client.close()
```

## Core methods (`AsyncMemoryClient`)

| Method | Role |
|--------|------|
| `ingest(user_id, text, reference_id=None)` | LLM extraction + graph upsert + embed |
| `ingest_batch(user_id, items)` | Multiple dicts with `text` and optional `reference_id` |
| `recall(user_id, query, top_k=10)` | Vector seeds + BFS + composite score |
| `search(user_id, query, top_k=10, detail_level="auto")` | Hierarchical / cluster-aware search |
| `get_graph(user_id, page=1, page_size=100)` | Paginated nodes and relationships |
| `delete_memory(user_id, node_id, cascade=False)` | Delete one node |
| `get_node_history(user_id, node_id)` | Supersession chain |
| `health_check(ping_llm=True)` | Neo4j, embedder, index, schema; optional LLM ping |

Sync `MemoryClient` exposes the same operations as blocking wrappers.

## Result types (high level)

| Type | Fields you typically use |
|------|---------------------------|
| `IngestResult` | `skipped`, `nodes_created`, `nodes_updated`, `relationships_created` |
| `RecallResult` | `nodes` (`ScoredNode`), `total_candidates`, `from_cache` |
| `ScoredNode` | `element_id`, `label`, `summary`, `score`, `hops_from_seed`, `properties` |
| `GraphSnapshot` | `nodes`, `relationships`, pagination metadata |
| `HealthStatus` | `neo4j_connected`, `llm_reachable`, embedding and index flags |

## Exceptions

All inherit from `EngramError`. Common cases:

| Exception | When |
|-----------|------|
| `ConfigurationError` | Invalid or incomplete config |
| `ExtractionError` | Structured extraction failed after retries |
| `CircuitOpenError` | LLM circuit breaker open |
| `InvalidUserIdError` | `user_id` does not match `user_id_pattern` |
| `RateLimitExceededError` | LLM rate limit bucket empty |
| `EmbeddingDimensionMismatchError` | Index vs config dimension mismatch |
| `HasRelationshipsError` | Delete without `cascade` but edges exist |
| `ConcurrentModificationError` | Optimistic locking conflict |
| `MigrationError` | Schema migration failure |

Import from `engram.exceptions` when you need to catch specific failures.

## Hooks

The `Hook` protocol and `LoggerHook` live under `engram.hooks`. Use them to wrap or instrument `ingest` / `recall` in your application layer (audit, redaction, metrics). See **Plugin Hooks** in the [main README](https://github.com/hackdavid/Engram/blob/main/README.md#plugin-hooks).

## Pipelines (conceptual)

Detailed ASCII diagrams live in the [main README](https://github.com/hackdavid/Engram/blob/main/README.md#how-it-works) under **How It Works**.
