# Engram :  Developer documentation

**Engram** is an async-first Python SDK that stores agent memory as a **Neo4j knowledge graph** with **LLM-driven extraction** (via [LiteLLM](https://docs.litellm.ai/)) and **embedding-based recall** without LLM calls on the read path.

| Resource | Link |
|----------|------|
| **Repository** | [github.com/hackdavid/Engram](https://github.com/hackdavid/Engram) |
| **Package (PyPI)** | `pip install engram` — project links on PyPI point back here |
| **License** | [MIT](https://github.com/hackdavid/Engram/blob/main/LICENSE) |

## Documentation map

| Guide | What you will learn |
|-------|---------------------|
| [Getting started](getting-started.md) | Install, environment, verify LiteLLM, first `ingest` / `recall` |
| [Configuration](configuration.md) | `Config`, environment variables, `user_id` rules, embeddings |
| [API overview](api-overview.md) | Clients, models, exceptions, async vs sync |
| [Production & operations](production.md) | Health checks, `engram-e2e`, live tests, hooks, logging |

The [root README](https://github.com/hackdavid/Engram/blob/main/README.md) remains the high-level product overview, feature list, and full configuration table.

## Quick orientation

1. **Ingest** — One LiteLLM call per non-trivial message extracts nodes and relationships; embeddings are written to Neo4j.
2. **Recall** — Query embedding + vector index + graph traversal + composite scoring; **no** LLM on this path.
3. **Isolation** — All graph data is scoped by `user_id` (validated against a configurable regex).

Start with [Getting started](getting-started.md), then keep [Configuration](configuration.md) and [Production & operations](production.md) nearby when you deploy.

## Additional material

- [`overview.md`](overview.md) — Early design notes and roadmap (includes non-standard YAML front matter for editor tooling; treat as supplementary, not API contract).
