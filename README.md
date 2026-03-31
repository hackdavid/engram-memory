<div align="center">

<img src="images/logo.png" alt="Engram logo" width="280"/>

# Engram

### *Next-generation graph memory for agentic workflows*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![LiteLLM](https://img.shields.io/badge/LLM-LiteLLM-111111?style=flat)](https://github.com/BerriAI/litellm)
[![Neo4j](https://img.shields.io/badge/graph-Neo4j-008CC1?style=flat&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Docs](https://img.shields.io/badge/docs-GitHub-24292f?style=flat&logo=github)](https://github.com/hackdavid/Engram/tree/main/docs)

**Ingest once with the LLM → recall with graph + vectors + scoring — no LLM on the read path.**

🧠 **Structured** · ⚡ **Async-first** · 🔌 **100+ models via [LiteLLM](https://docs.litellm.ai/)** · 🛡️ **Decay, cache, rate limits** · 🚀 **Production-minded**

📖 **[Developer docs](https://github.com/hackdavid/Engram/tree/main/docs)** · **[GitHub](https://github.com/hackdavid/Engram)** · **[Issues](https://github.com/hackdavid/Engram/issues)**

[Why Engram](#why-engram-next-generation-memory) · [Docs](#documentation) · [Verify LiteLLM](#verify-your-model-with-litellm-first) · [Install](#installation) · [Quick start](#quick-start) · [Contributing](#contributing) · [License](#license)

</div>

---

## 🚀 Why Engram? (Next-generation memory)

Most agent memory is a **flat pile of chunks** or a **single vector index**. That breaks down when you need **structured relationships**, **incremental truth**, and **fast, cheap retrieval** at scale.

| Approach | Pain point | **Engram** |
|----------|------------|------------|
| Chunk + vector RAG | Loses who/what/how; hard to reason about entities | **Neo4j graph** with dynamic labels & relationships from the LLM |
| One big summary | Goes stale; expensive to rewrite | **Per-message extraction** + **strength decay** so unused facts fade |
| “Ask the LLM” for every recall | Cost + latency | **`recall()` = embeddings + vector index + BFS + composite score** |
| Rigid schema | Doesn’t fit every domain | **Schema emerges at runtime** from structured JSON extraction |

Engram is built for **multi-step agents, copilots, and long-running workflows**: isolated `user_id` namespaces, hooks for audit/telemetry, health checks, and a CLI smoke test (`engram-e2e`) you can run in CI against a real database.

## Documentation

Developer-focused guides live under [`docs/`](docs/):

| Guide | Topics |
|-------|--------|
| [Documentation home](docs/README.md) | Index, links, orientation |
| [Getting started](docs/getting-started.md) | Install, LiteLLM check, first `ingest` / `recall` |
| [Configuration](docs/configuration.md) | Environment variables, `Config`, `user_id` pattern |
| [API overview](docs/api-overview.md) | Clients, models, exceptions |
| [Production & operations](docs/production.md) | Health, `engram-e2e`, logging |

On **PyPI**, the package metadata includes a **Documentation** URL that points to the same [`docs/` tree on GitHub](https://github.com/hackdavid/Engram/tree/main/docs).

---

## ✅ Verify your model with LiteLLM first

All production LLM traffic in Engram goes through **[LiteLLM](https://docs.litellm.ai/docs/)** (`litellm.acompletion`). **Before** you set `LLM_MODEL` / `LLM_API_KEY` in Engram, prove the same route works in a minimal call:

```bash
pip install litellm
# Quick check (adjust model + env for your provider):
python -c "import litellm; print(litellm.completion(model='gpt-4o-mini', messages=[{'role':'user','content':'Say OK.'}]).choices[0].message.content)"
```

Or async, matching what Engram uses internally:

```python
import asyncio
import litellm

async def main():
    r = await litellm.acompletion(
        model="gpt-4o-mini",  # e.g. anthropic/claude-..., azure/deployment-name, openai/...
        messages=[{"role": "user", "content": "Reply OK."}],
        api_key="...",       # or rely on OPENAI_API_KEY / provider-specific env
        # api_base="...",   # enterprise / Azure / custom gateway
    )
    print(r.choices[0].message.content)

asyncio.run(main())
```

Use the **exact** model id (and `api_base` / `api_version` if required) in your **`LLM_*`** environment variables — see [Quick start](#quick-start).

**Integration status:** the supported path is **`LiteLLMAdapter`** (`engram/llm/litellm_adapter.py`). Legacy adapters under `engram/llm/` may exist for reference; **new provider-specific integrations should prefer LiteLLM** or land as clean PRs — we are **open to contributions** (see below).

---

## ✨ Features

- **1 LLM call to ingest · 0 LLM calls to recall** — extract structured graph once; retrieve with vectors + traversal + scoring
- **Dynamic graph schema** — labels, properties, and relationship types from the model, not hand-maintained DDL
- **Async-first** — `AsyncMemoryClient` + sync `MemoryClient` wrapper
- **Composite ranking** — `α·vector_similarity + β·decay^hops + γ·strength`
- **Memory decay & archival** — strength fades; stale nodes drop out of active recall
- **Hierarchical cluster summaries** — broad vs detailed `search()` modes
- **Two-tier embeddings** — optional int8 + float32 path for speed/quality tradeoffs
- **Optimistic locking, rate limiting, circuit breaker** — safer under load
- **Hooks + observability** — lifecycle hooks, JSON logs, metrics, optional OpenTelemetry
- **Background tasks** — decay, hierarchy rebuild, weight-learning telemetry

---

## 📦 Installation

**PyPI release is in progress.** Until the package is published, install from this repository:

```bash
git clone https://github.com/hackdavid/Engram.git
cd Engram
pip install -e .
```

When Engram is on PyPI, a normal install will be:

```bash
pip install engram
```

(PyPI treats the name as case-insensitive, so `pip install Engram` will be equivalent.)

Either path installs the **runtime stack**: Neo4j driver, Pydantic, LiteLLM, and **local embeddings** (SentenceTransformers + PyTorch) for `EMBEDDING_PROVIDER=local` (the default). If you use **`EMBEDDING_PROVIDER=openai`**, add the OpenAI SDK: `pip install engram[openai-embed]` (after PyPI) or `pip install -e ".[openai-embed]"` from a clone.

## End-to-end validation (production smoke)

Run from a directory that has `.env` or `engram/.env` configured (see [`.env.example`](.env.example)). Install first with `pip install -e .` from a clone, or from PyPI when it is available.

### How to run

| Use case | Command |
|----------|---------|
| **Default (recommended)** | `python -m engram.cli.e2e_validate` |
| Same, from a Windows clone | `scripts\engram-e2e.cmd` (repo root) |
| Pip console script (if on `PATH`) | `engram-e2e` |
| Clone, package not installed | `python scripts/e2e_validate.py` |

On **Windows**, `engram-e2e` often fails with “not recognized” because Python’s **Scripts** folder is not on `PATH`. Prefer the **`python -m …`** row above, or add that `Scripts` directory to `PATH` (conda env, `%LocalAppData%\Programs\Python\Python3xx\Scripts`, etc.).

### What the default run does

1. Health checks  
2. **Five sequential ingests** (one LLM call each), with per-ingest timing logged  
3. Ten recall / search scenarios and a graph snapshot  

### Flags and timeouts

| Goal | How |
|------|-----|
| Retrieval only (no writes) | `python -m engram.cli.e2e_validate --skip-seed --user-id <id>` or set `E2E_USER_ID` |
| One LLM call for all seed text | `--batch-seed` |
| Tune wall-clock limits | `E2E_LLM_TIMEOUT_SEC` (default `120`), `E2E_INGEST_TIMEOUT_SEC` (default LLM timeout + 45s) |

More options: `python -m engram.cli.e2e_validate --help`.

### Neo4j-only check

Bolt connectivity without the full SDK or LLM: `python scripts/neo4j_verify_connectivity.py` (same Neo4j env vars).

## Quick Start

### 1. Set Environment Variables

```bash
# Required
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your-password"

# LLM -- uses LiteLLM model naming (supports 100+ providers)
export LLM_MODEL="gpt-4o-mini"           # OpenAI
export LLM_API_KEY="sk-..."

# Or Anthropic:
#   LLM_MODEL="anthropic/claude-sonnet-4-20250514"
#   LLM_API_KEY="sk-ant-..."

# Or Azure OpenAI:
#   LLM_MODEL="azure/my-gpt4-deployment"
#   LLM_API_KEY="azure-key"
#   LLM_API_BASE="https://myresource.openai.azure.com/"
#   LLM_API_VERSION="2024-02-01"

# Optional LLM HTTP timeout (seconds) — forwarded to LiteLLM / httpx on ingest
# export LLM_REQUEST_TIMEOUT="120"

# Optional (all have sensible defaults)
export EMBEDDING_PROVIDER="local"         # or "openai"
export CACHE_ENABLED="true"
export ENABLE_BACKGROUND_TASKS="true"
export LOG_FORMAT="json"                  # or "text"
```

### 2. Async Usage (Recommended)

```python
import asyncio
from engram import AsyncMemoryClient, Config

async def main():
    config = Config()  # reads from environment variables
    async with AsyncMemoryClient(config) as client:
        # await client.health_check(ping_llm=True)  # optional wiring check (LiteLLM ping)

        # Ingest a message
        result = await client.ingest(
            user_id="user-123",
            text="I work at Google as a senior engineer in the ML team.",
            reference_id="msg-001",  # optional: link back to source message
        )
        print(f"Created {len(result.nodes_created)} nodes, "
              f"{result.relationships_created} relationships")

        # Recall relevant context
        context = await client.recall(
            user_id="user-123",
            query="What does the user do for work?",
            top_k=5,
        )
        for node in context.nodes:
            print(f"  [{node.score:.2f}] {node.summary}")

asyncio.run(main())
```

### 3. Sync Usage

```python
from engram import MemoryClient, Config

client = MemoryClient(Config())
result = client.ingest(user_id="user-123", text="I love hiking in the mountains.")
context = client.recall(user_id="user-123", query="hobbies")
client.close()
```

## How It Works

### Ingestion Pipeline

```
User text
  │
  ▼
Trivial filter ──► skip ("hi", "ok", "thanks")
  │
  ▼
Rate limiter (token bucket)
  │
  ▼
Fetch neighbourhood (existing nodes for this user)
  │
  ▼
LLM extraction ──► NodeInstruction[] + RelInstruction[]
  │                  (1 LLM call with structured JSON output)
  ▼
For each node:
  ├─ Embed summary (SentenceTransformer or OpenAI)
  ├─ Build parameterised Cypher (MERGE + SET)
  ├─ Execute against Neo4j
  └─ Assign to cluster (HierarchyManager)
  │
  ▼
Build relationships (MERGE with traversal metadata)
  │
  ▼
Invalidate user cache
  │
  ▼
Return IngestResult
```

### Retrieval Pipeline

```
Query text
  │
  ▼
Check per-user LRU cache ──► cache hit? return immediately
  │
  ▼
Embed query
  │
  ▼
Neo4j vector search (cosine similarity, top-K seeds)
  │
  ▼
Decay-weighted BFS traversal
  │  (expand from seeds, score *= decay per hop, stop at min_score)
  ▼
Composite scoring: α·similarity + β·decay^hops + γ·strength
  │
  ▼
Rank and return top-K ScoredNode[]
  │
  ▼
Cache result, return RecallResult
```

## Core Concepts

### Strength Decay

Every node has a `strength` property (starts at 1.0). A background task periodically multiplies it by a decay factor, simulating memory fading over time.

```
strength_new = strength_old × decay_factor
```

**Default:** `decay_factor=0.95`, runs every 24 hours.

Nodes that drop below the `archive_threshold` (default: 0.01) are archived -- they get `_archived=true` and `isCurrent=false`, excluding them from recall queries.

**Why this matters:** Frequently reinforced memories (re-ingested facts) stay strong because each MERGE resets strength to 1.0. Stale facts naturally fade away.

```python
# Configure decay via environment variables or Config
config = Config(
    decay_factor=0.90,           # faster decay (memories fade quicker)
    archive_threshold=0.05,      # archive earlier
    decay_interval_hours=12,     # run decay every 12 hours
)
```

### Composite Scoring

Retrieval ranks nodes using three weighted signals:

| Signal | Weight | Description |
|--------|--------|-------------|
| `vector_similarity` | α (0.50) | Cosine similarity between query embedding and node embedding |
| `decay^hops` | β (0.35) | Proximity penalty -- nodes further from seed get lower scores |
| `strength` | γ (0.15) | Memory strength -- recently reinforced facts score higher |

```
final_score = α × vector_similarity + β × (decay ^ hops) + γ × strength
```

**Example:** A node with 0.9 similarity, 1 hop away, strength 0.8:

```
score = 0.5 × 0.9 + 0.35 × (0.5^1) + 0.15 × 0.8
      = 0.45  +  0.175  +  0.12
      = 0.745
```

```python
# Customise weights
config = Config(
    score_alpha=0.6,    # prioritise semantic similarity
    score_beta=0.25,    # less weight on graph distance
    score_gamma=0.15,   # keep strength weight
)
```

### Traversal

After vector search returns seed nodes, BFS expands outward through relationships. Each hop multiplies the score by a decay factor. Expansion stops when:

- **Max depth reached** (default: 5 hops)
- **Score drops below min_score** (default: 0.1)
- **Node already visited** (cycle prevention)

```python
config = Config(
    traversal_decay=0.5,       # each hop halves the score
    traversal_max_depth=3,     # limit to 3 hops
    traversal_min_score=0.15,  # prune weak paths earlier
)
```

### Hierarchical Summary Tree

Nodes are grouped into clusters (e.g., "Work", "Hobbies", "Education"). Each cluster has a `ClusterSummary` node at Level 1 with an aggregated embedding.

Query modes:

| Mode | Behaviour |
|------|-----------|
| `broad` | Only search Level 0-1 (cluster summaries) |
| `detailed` | Search all levels including leaf nodes |
| `auto` | Start broad; if best score < 0.5, fall through to detailed |

```python
result = await client.search(
    user_id="user-123",
    query="Tell me about their career",
    detail_level="auto",  # recommended
    top_k=10,
)
```

### Trivial Filter

Before calling the LLM, Engram checks if the text is worth extracting. Short greetings, acknowledgements, and filler messages are skipped automatically, saving LLM costs.

**Automatically skipped:** "hi", "ok thanks", "lol", "sounds good", "bye", etc.

**Not skipped:** "I work at Google as a software engineer" (contains entities + facts).

```python
# Add custom trivial patterns
from engram.extractors.trivial_filter import is_trivial
is_trivial("skip this", custom_trivial_patterns=[r"skip this"])  # True
```

### Two-Tier Embeddings

Enable coarse (int8 quantized) + fine (float32) embeddings. The coarse embedding can be used for fast approximate filtering before re-ranking with fine embeddings.

```python
config = Config(
    two_tier_embedding=True,
    embedding_provider="local",
    embedding_model="all-MiniLM-L6-v2",
)
```

### Optimistic Locking

Every node carries a `_version` counter. When updating a node, you can pass `expected_version` to ensure no concurrent write has happened since you last read it. If the version doesn't match, the update is a no-op, preventing data corruption.

### Plugin Hooks

Extend Engram's lifecycle with custom hooks:

```python
from engram.hooks.base import Hook
from engram.models import IngestResult, RecallResult

class AuditHook:
    """Log all ingest/recall events to an audit service."""

    async def pre_ingest(self, user_id: str, text: str) -> str | None:
        print(f"[AUDIT] Ingesting for {user_id}: {text[:50]}...")
        return None  # return modified text, or None to keep original

    async def post_ingest(self, user_id: str, result: IngestResult) -> None:
        print(f"[AUDIT] Ingested: {len(result.nodes_created)} nodes created")

    async def pre_recall(self, user_id: str, query: str) -> str | None:
        return None

    async def post_recall(self, user_id: str, result: RecallResult) -> None:
        print(f"[AUDIT] Recalled {len(result.nodes)} nodes (cached={result.from_cache})")
```

A built-in `LoggerHook` is included that logs all lifecycle events at INFO level.

### Rate Limiting

LLM calls are rate-limited using a token-bucket algorithm to prevent cost overruns:

```python
config = Config(
    llm_rate_limit_rpm=60,    # 60 requests per minute
    llm_rate_limit_burst=10,  # allow bursts of up to 10
)
```

When the bucket is empty, `RateLimitExceededError` is raised.

### Circuit Breaker

LLM adapters include a circuit breaker. After N consecutive failures, the breaker opens and all subsequent calls immediately raise `CircuitOpenError` instead of hitting the API. This prevents cascading failures and excessive costs.

```python
# Configured per LLM adapter
config = Config(
    llm_max_retries=3,  # retry up to 3 times with exponential backoff
)
```

### Caching

Recall results are cached per-user with an LRU eviction policy and TTL:

```python
config = Config(
    cache_enabled=True,
    cache_max_size=256,        # max entries in cache
    cache_ttl_seconds=300,     # 5-minute TTL
)
```

Cache is automatically invalidated when new data is ingested for a user.

### Background Tasks

Three periodic tasks run in the background:

| Task | Default Interval | Purpose |
|------|-----------------|---------|
| **Strength Decay** | 24 hours | Multiply all node strengths by `decay_factor`, archive weak nodes |
| **Hierarchy Rebuild** | 6 hours | Re-compute cluster summary embeddings from member nodes |
| **Weight Learning** | 12 hours | Log traversal statistics for scoring weight optimisation |

```python
config = Config(
    enable_background_tasks=True,
    decay_interval_hours=24,
    hierarchy_rebuild_interval_hours=6,
    weight_learning_interval_hours=12,
)
```

## API Reference

### `AsyncMemoryClient`

| Method | Description |
|--------|-------------|
| `ingest(user_id, text, reference_id=None)` | Extract entities from text and store in the graph |
| `ingest_batch(user_id, items)` | Ingest multiple messages (each `{"text": "...", "reference_id": "..."}`) |
| `recall(user_id, query, top_k=10)` | Retrieve relevant memories ranked by composite score |
| `search(user_id, query, top_k=10, detail_level="auto")` | Hierarchical search through summary tree |
| `get_graph(user_id, page=1, page_size=100)` | Paginated snapshot of a user's memory graph |
| `delete_memory(user_id, node_id, cascade=False)` | Delete a node (cascade removes relationships too) |
| `get_node_history(user_id, node_id)` | Get the supersession chain for a node |
| `health_check(ping_llm=True)` | Check Neo4j, embedder, vector index, schema; optionally ping the LLM via LiteLLM |

### Key Models

| Model | Purpose |
|-------|---------|
| `IngestResult` | Response from `ingest()`: `skipped`, `nodes_created`, `nodes_updated`, `relationships_created` |
| `RecallResult` | Response from `recall()`: `nodes` (list of `ScoredNode`), `total_candidates`, `from_cache` |
| `ScoredNode` | A node with `element_id`, `label`, `summary`, `score`, `hops_from_seed`, `properties` |
| `GraphSnapshot` | Paginated graph view: `nodes`, `relationships`, `total_nodes`, pagination fields |
| `HealthStatus` | Aggregated health: `neo4j_connected`, `llm_reachable`, `embedding_model_loaded`, etc. |

### Exception Hierarchy

All exceptions inherit from `EngramError`:

| Exception | When |
|-----------|------|
| `ConfigurationError` | Invalid SDK configuration |
| `ExtractionError` | LLM extraction fails after all retries |
| `CircuitOpenError` | LLM circuit breaker is open |
| `InvalidUserIdError` | user_id fails pattern validation |
| `HasRelationshipsError` | Non-cascade delete on node with relationships |
| `EmbeddingDimensionMismatchError` | Vector index dimensions differ from config |
| `RateLimitExceededError` | Token bucket exhausted |
| `MigrationError` | Schema migration failed |
| `ConcurrentModificationError` | Optimistic locking conflict |

## Configuration Reference

All fields can be set via environment variables (case-insensitive):

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NEO4J_URI` | str | *required* | Neo4j connection URI |
| `NEO4J_USER` | str | *required* | Neo4j username |
| `NEO4J_PASSWORD` | str | *required* | Neo4j password |
| `NEO4J_DATABASE` | str | `neo4j` | Neo4j database name |
| `NEO4J_MAX_POOL_SIZE` | int | `50` | Connection pool size |
| `LLM_MODEL` | str | `gpt-4o-mini` | LiteLLM model string (e.g. `gpt-4o`, `anthropic/claude-sonnet-4-20250514`, `azure/<deployment>`) |
| `LLM_API_KEY` | str | None | API key for the LLM provider |
| `LLM_API_BASE` | str | None | Base URL (required for Azure, optional otherwise) |
| `LLM_API_VERSION` | str | None | API version (required for Azure) |
| `LLM_MAX_TOKENS` | int | `4096` | Max tokens per LLM response |
| `LLM_MAX_RETRIES` | int | `3` | Max LLM retry attempts |
| `LLM_RATE_LIMIT_RPM` | int | `60` | Requests per minute limit |
| `LLM_RATE_LIMIT_BURST` | int | `10` | Burst capacity |
| `LLM_REQUEST_TIMEOUT` | float | None | Optional HTTP timeout (seconds) for LLM calls (LiteLLM / httpx) |
| `EMBEDDING_PROVIDER` | str | `local` | `local` or `openai` |
| `EMBEDDING_MODEL` | str | `all-MiniLM-L6-v2` | Embedding model name |
| `EMBEDDING_DIMENSIONS` | int | `384` | Embedding vector dimensions |
| `EMBEDDING_API_KEY` | str | None | API key for OpenAI embeddings |
| `TWO_TIER_EMBEDDING` | bool | `false` | Enable coarse+fine embeddings |
| `SCORE_ALPHA` | float | `0.50` | Vector similarity weight |
| `SCORE_BETA` | float | `0.35` | Hop decay weight |
| `SCORE_GAMMA` | float | `0.15` | Strength weight |
| `TRAVERSAL_DECAY` | float | `0.5` | Score multiplier per hop |
| `TRAVERSAL_MAX_DEPTH` | int | `5` | Max BFS hops |
| `TRAVERSAL_MIN_SCORE` | float | `0.1` | Prune below this score |
| `DECAY_FACTOR` | float | `0.95` | Strength multiplier per cycle |
| `ARCHIVE_THRESHOLD` | float | `0.01` | Archive nodes below this |
| `DECAY_INTERVAL_HOURS` | int | `24` | Hours between decay runs |
| `CACHE_ENABLED` | bool | `true` | Enable recall caching |
| `CACHE_MAX_SIZE` | int | `100` | Max cache entries |
| `CACHE_TTL_SECONDS` | int | `300` | Cache entry lifetime |
| `AUTO_MIGRATE` | bool | `true` | Run schema migrations on start |
| `LOG_LEVEL` | str | `INFO` | Logging level |
| `LOG_FORMAT` | str | `text` | `json` or `text` |
| `ENABLE_TRACING` | bool | `false` | Enable OpenTelemetry tracing |

## Architecture

```
engram/
├── __init__.py               # Public exports + lazy imports
├── _version.py               # "0.1.0"
├── client.py                 # AsyncMemoryClient + MemoryClient (sync wrapper)
├── config.py                 # Pydantic BaseSettings configuration
├── constants.py              # SDK-wide defaults
├── exceptions.py             # EngramError hierarchy
├── models.py                 # Pydantic data contracts
├── rate_limiter.py           # Token-bucket rate limiter
├── graph/
│   ├── driver.py             # Async Neo4j driver wrapper
│   ├── engine.py             # Dynamic Cypher generator
│   ├── indexes.py            # Vector index management
│   ├── migrations.py         # Schema versioning
│   ├── sanitise.py           # Label/type sanitisation
│   ├── traversal.py          # Decay-weighted BFS
│   ├── scorer.py             # Composite scoring
│   └── hierarchy.py          # Cluster summary tree
├── embeddings/
│   ├── base.py               # ABC for embedders
│   ├── sentence_transformer.py  # Local (SentenceTransformers)
│   ├── openai_embedding.py   # OpenAI API
│   └── two_tier.py           # Coarse + fine embeddings
├── llm/
│   ├── base.py               # ABC with retry + circuit breaker
│   ├── litellm_adapter.py    # Production path: LiteLLM (100+ providers)
│   ├── openai_adapter.py     # Legacy reference (not wired by default)
│   └── anthropic_adapter.py  # Legacy reference (not wired by default)
├── extractors/
│   ├── base.py               # ABC for extractors
│   ├── prompts.py            # LLM prompt templates
│   ├── llm_extractor.py      # LLM-powered extraction
│   └── trivial_filter.py     # Skip greetings/filler
├── cache/
│   └── lru_cache.py          # Per-user LRU with TTL
├── hooks/
│   ├── base.py               # Hook protocol
│   └── logger_hook.py        # Built-in logging hook
├── observability/
│   ├── logging.py            # JSON structured logging
│   ├── metrics.py            # Counters + histograms
│   └── tracing.py            # Optional OpenTelemetry
├── health/
│   └── checks.py             # Aggregated health checks
└── background/
    ├── runner.py             # Asyncio task scheduler
    ├── decay_task.py         # Strength decay + archival
    ├── hierarchy_task.py     # Cluster summary rebuild
    └── weight_learning_task.py  # Scoring weight telemetry
```

## 🤝 Contributing

Engram is **open source**. We want you to **use it in production**, **report rough edges**, and **ship improvements**.

- **Issues** — bugs, design questions, or provider-specific LiteLLM quirks (include model id, env vars you set, and redacted logs).
- **Pull requests** — keep changes focused; add or extend **tests**; clone the repo and use `pip install -e ".[dev]"` for pytest/ruff, then `pytest tests/ -v`. Match existing style and typing.
- **New LLM backends** — the supported integration is **`LiteLLMAdapter`**. If you need a path LiteLLM does not cover, open an issue first; we welcome clean adapters that follow `engram/llm/base.py` and include tests with mocks.

Thank you for helping make agent memory **structured, fast, and boringly reliable**.

## 📄 License

Engram is distributed under the **MIT License**. You may use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software for **commercial or non-commercial** purposes, provided that you include the original copyright notice and permission notice in all substantial copies.

The Software is provided **“as is”**, without warranty of any kind. See the full legal terms in [`LICENSE`](LICENSE) (copyright © 2026 Daud Dewan).
