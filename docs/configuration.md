# Configuration

Engram loads settings from the **`Config`** object (`pydantic-settings`). You can construct it explicitly or rely on **environment variables** (case-insensitive).

## Required

| Variable | Purpose |
|----------|---------|
| `NEO4J_URI` | Bolt URI, e.g. `bolt://localhost:7687` or Aura URL |
| `NEO4J_USER` | Neo4j username |
| `NEO4J_PASSWORD` | Neo4j password |

## LLM (LiteLLM)

| Variable | Default | Notes |
|----------|---------|--------|
| `LLM_MODEL` | `gpt-4o-mini` | Any LiteLLM model string, e.g. `anthropic/claude-sonnet-4-20250514`, `azure/<deployment>` |
| `LLM_API_KEY` | — | Provider API key (or use provider-specific env vars LiteLLM reads) |
| `LLM_API_BASE` | — | Required for some gateways / Azure |
| `LLM_API_VERSION` | — | Often required for Azure |
| `LLM_MAX_TOKENS` | `4096` | |
| `LLM_MAX_RETRIES` | `3` | |
| `LLM_RATE_LIMIT_RPM` | `60` | Token-bucket limiter for **ingest** LLM calls |
| `LLM_RATE_LIMIT_BURST` | `10` | |
| `LLM_REQUEST_TIMEOUT` | — | Seconds; passed through to LiteLLM/HTTP client |

The wired adapter is **`LiteLLMAdapter`** (`engram/llm/litellm_adapter.py`). Other files under `engram/llm/` are legacy references unless you contribute a new integration.

## Embeddings

| Variable | Default | Notes |
|----------|---------|--------|
| `EMBEDDING_PROVIDER` | `local` | `local` (SentenceTransformers) or `openai` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Must match dimension below for existing indexes |
| `EMBEDDING_DIMENSIONS` | `384` | Must match the vector index in Neo4j |
| `EMBEDDING_API_KEY` | — | For `openai` provider |
| `TWO_TIER_EMBEDDING` | `false` | Coarse + fine embedding path |

Changing model or dimensions on an existing database may require index migration or a fresh database — plan that before production.

## Retrieval scoring & traversal

| Variable | Default | Meaning |
|----------|---------|---------|
| `SCORE_ALPHA` | `0.50` | Weight for vector similarity |
| `SCORE_BETA` | `0.35` | Weight for hop decay term |
| `SCORE_GAMMA` | `0.15` | Weight for node strength |
| `TRAVERSAL_DECAY` | `0.5` | Per-hop score multiplier during BFS |
| `TRAVERSAL_MAX_DEPTH` | `5` | Max graph hops from seed nodes |
| `TRAVERSAL_MIN_SCORE` | `0.1` | Prune paths below this score |

## Memory decay (background)

| Variable | Default | Meaning |
|----------|---------|---------|
| `ENABLE_BACKGROUND_TASKS` | `true` | Periodic decay / hierarchy / telemetry tasks |
| `DECAY_FACTOR` | `0.95` | Per-cycle strength multiplier |
| `ARCHIVE_THRESHOLD` | `0.01` | Archive nodes weaker than this |
| `DECAY_INTERVAL_HOURS` | `24` | |
| `HIERARCHY_REBUILD_INTERVAL_HOURS` | `6` | Cluster summary refresh |
| `WEIGHT_LEARNING_INTERVAL_HOURS` | `12` | Scoring telemetry |

## Cache

| Variable | Default | Meaning |
|----------|---------|---------|
| `CACHE_ENABLED` | `true` | LRU cache for recall results |
| `CACHE_MAX_SIZE` | `100` | Entries |
| `CACHE_TTL_SECONDS` | `300` | TTL per entry |

Cache is invalidated on ingest for the affected `user_id`.

## Schema & observability

| Variable | Default | Meaning |
|----------|---------|---------|
| `NEO4J_DATABASE` | `neo4j` | Neo4j database name |
| `NEO4J_MAX_POOL_SIZE` | `50` | Driver pool |
| `AUTO_MIGRATE` | `true` | Run SDK migrations on startup |
| `LOG_LEVEL` | `INFO` | |
| `LOG_FORMAT` | `text` | `json` for structured logs |
| `ENABLE_TRACING` | `false` | OpenTelemetry (optional extra) |

## User ID validation

<a id="user-id-validation"></a>

| Variable | Default |
|----------|---------|
| `USER_ID_PATTERN` | `^[a-zA-Z0-9_-]{1,128}$` |

`user_id` must match this regex. Use stable, opaque ids from your auth system (never raw PII if you can avoid it).

## Programmatic `Config`

```python
from engram import Config

config = Config(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="secret",
    llm_model="gpt-4o-mini",
    llm_api_key="sk-...",
    log_format="json",
)
```

Environment variables still override if you use `model_config` defaults from `BaseSettings` — prefer **either** env-based **or** explicit kwargs in tests.

## Full table

For a single-page dump of every variable, see the **Configuration Reference** section in the [main README](https://github.com/hackdavid/Engram/blob/main/README.md#configuration-reference).
