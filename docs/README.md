# Engram :  Developer documentation

**Engram** is an async-first Python SDK that stores agent memory as a **Neo4j knowledge graph** with **LLM-driven extraction** (via [LiteLLM](https://docs.litellm.ai/)) and **embedding-based recall** without LLM calls on the read path.

| Resource | Link |
|----------|------|
| **Repository** | [github.com/hackdavid/Engram](https://github.com/hackdavid/Engram) |
| **Install** | Clone [hackdavid/Engram](https://github.com/hackdavid/Engram) and `pip install -e .` until PyPI release; then `pip install engram_memory` |
| **License** | [MIT](https://github.com/hackdavid/Engram/blob/main/LICENSE) |

## Documentation map

| Guide | What you will learn |
|-------|---------------------|
| [Getting started](getting-started.md) | Install, environment, verify LiteLLM, first `ingest` / `recall` |
| [Configuration](configuration.md) | `Config`, environment variables, `user_id` rules, embeddings |
| [API overview](api-overview.md) | Clients, models, exceptions, async vs sync |
| [Production & operations](production.md) | Health checks, `engram_memory-e2e`, live tests, hooks, logging |

The [root README](https://github.com/hackdavid/Engram/blob/main/README.md) remains the high-level product overview, feature list, and full configuration table.

## Quick orientation

1. **Ingest** — Embed text, vector-search for top-5 similar nodes (slim context: summaries + rel types only), one LLM call for extraction, batched `UNWIND` writes to Neo4j. Token usage tracked per call.
2. **Recall** — Embed query, vector search for seeds, single variable-length Cypher traversal (1 round-trip), composite scoring. **Zero LLM calls** on the read path.
3. **Isolation** — All graph data is scoped by `user_id` (validated against a configurable regex).

Start with [Getting started](getting-started.md), then keep [Configuration](configuration.md) and [Production & operations](production.md) nearby when you deploy.

## Additional material

- [`overview.md`](overview.md) — Early design notes and roadmap (includes non-standard YAML front matter for editor tooling; treat as supplementary, not API contract).
