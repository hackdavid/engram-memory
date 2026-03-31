# Production and operations

## Health checks

```python
status = await client.health_check(ping_llm=True)
```

- With **`ping_llm=True`**, Engram performs a minimal LiteLLM call — use for staging or post-deploy verification; you may use **`ping_llm=False`** in environments where outbound LLM checks are restricted, as long as you monitor ingest separately.

Interpret flags on `HealthStatus` (Neo4j connectivity, embedder loaded, vector index, schema version, LLM reachability).

## Smoke test CLI (`engram-e2e`)

After installing extras for embeddings (`engram[local-embed]` or OpenAI embeddings configured):

```bash
engram-e2e
# equivalent:
python -m engram.cli.e2e_validate
```

From a clone without installing the package: `python scripts/e2e_validate.py`.

Useful flags (see `--help` in your installed version):

- **`--skip-seed`** — retrieval-only against an existing `user_id`
- **`--batch-seed`** — one LLM call for bundled seed content
- Environment **`E2E_LLM_TIMEOUT_SEC`**, **`E2E_INGEST_TIMEOUT_SEC`** — wall-clock guardrails

Run this in CI against a dedicated Neo4j instance to catch regressions in graph writes and recall.

## Live integration tests (optional)

These tests exercise **real Neo4j + real LLM** (cost and side effects). They are **off** unless you opt in.

1. Configure `.env` / `engram/.env`.
2. Set **`ENGRAM_LIVE_TESTS=1`**.
3. Run: `pytest tests/test_live_e2e.py -m live`

See the [README section on live tests](https://github.com/hackdavid/Engram/blob/main/README.md#live-integration-tests-optional) for shell examples and cleanup behaviour.

## Neo4j connectivity only

To verify Bolt credentials without the full SDK:

```bash
python scripts/neo4j_verify_connectivity.py
```

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
