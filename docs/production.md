# Production and operations

## Health checks

```python
status = await client.health_check(ping_llm=True)
```

- With **`ping_llm=True`**, Engram performs a minimal LiteLLM call — use for staging or post-deploy verification; you may use **`ping_llm=False`** in environments where outbound LLM checks are restricted, as long as you monitor ingest separately.

Interpret flags on `HealthStatus` (Neo4j connectivity, embedder loaded, vector index, schema version, LLM reachability).

## Smoke test CLI

Configure `.env` / `engram_memory/.env`, then install from the repo (`pip install -e .`) or PyPI when available.

| Use case | Command |
|----------|---------|
| Recommended | `python -m engram_memory.cli.e2e_validate` |
| Windows clone helper | `scripts\engram_memory-e2e.cmd` |
| Pip script (if on `PATH`) | `engram_memory-e2e` |
| No package install | `python scripts/e2e_validate.py` |

On Windows, prefer `python -m …` if `engram_memory-e2e` is not found (Scripts not on `PATH`).

| Flag / env | Purpose |
|------------|---------|
| `--skip-seed` + `--user-id` or `E2E_USER_ID` | Retrieval-only smoke |
| `--batch-seed` | One LLM call for bundled seed content |
| `E2E_LLM_TIMEOUT_SEC`, `E2E_INGEST_TIMEOUT_SEC` | Wall-clock guardrails |

Run in CI against a dedicated Neo4j instance. Full options: `--help`.

**Bolt only (no LLM):** `python scripts/neo4j_verify_connectivity.py` with Neo4j env vars set.

## Logging

- Set **`LOG_FORMAT=json`** for centralized log aggregation.
- Correlate logs with your `user_id` and `reference_id` in application-level fields where possible.

## Rate limits and resilience

- **LLM**: token-bucket (`LLM_RATE_LIMIT_RPM`, `LLM_RATE_LIMIT_BURST`) applies to ingestion.
- **Retries and circuit breaker**: configured via `LLM_MAX_RETRIES` and adapter behaviour — protect your budget when the provider is failing.

## Embedding load

Local SentenceTransformers models download on first use. Plan container images or cached model directories for cold-start latency in Kubernetes or serverless environments.

## Security notes

- Treat **`NEO4J_PASSWORD`** and **`LLM_API_KEY`** as secrets (secret manager, not git).
- **`user_id`** should be an application-level stable identifier; avoid embedding sensitive personal data in graph keys if your threat model requires minimization.

## Where to get help

- **Issues:** [github.com/hackdavid/Engram/issues](https://github.com/hackdavid/Engram/issues)
- **Contributing:** [README — Contributing](https://github.com/hackdavid/Engram/blob/main/README.md#-contributing)
