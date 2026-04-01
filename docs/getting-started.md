# Getting started

## Prerequisites

- **Python 3.11+**
- A running **Neo4j** 5.x instance (Aura or self-hosted) with credentials you can use from your app
- An **LLM** reachable through **[LiteLLM](https://docs.litellm.ai/)** (same model id and keys you will put in Engram’s config)

## Install

**PyPI release is in progress.** For now, clone and install in editable mode:

```bash
git clone https://github.com/hackdavid/Engram.git
cd Engram
pip install -e .
```

When the package is published: `pip install engram_memory` (same as `pip install Engram` on PyPI).

The install includes **local embeddings** (SentenceTransformers + PyTorch) for the default `EMBEDDING_PROVIDER=local`. For **`EMBEDDING_PROVIDER=openai`**, also run `pip install -e ".[openai-embed]"` from the clone (or `pip install engram_memory[openai-embed]` after PyPI).

**Contributors** working from a git clone: `pip install -e ".[dev]"` (pytest, ruff, etc.).

## Verify LiteLLM before Engram

Engram’s production path uses `litellm.acompletion`. Confirm your provider works **outside** Engram first (same `model`, `api_key`, and optional `api_base` / `api_version`):

```bash
pip install litellm
python -c "import litellm; print(litellm.completion(model='gpt-4o-mini', messages=[{'role':'user','content':'Say OK.'}]).choices[0].message.content)"
```

If this fails, fix provider credentials or model id before setting `LLM_MODEL` and `LLM_API_KEY` for Engram.

## Minimal environment

Create `.env` at the repo root or under `engram_memory/.env` (see [.env.example](https://github.com/hackdavid/Engram/blob/main/.env.example) in the repository). Minimum:

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your-password"

export LLM_MODEL="gpt-4o-mini"
export LLM_API_KEY="sk-..."

export EMBEDDING_PROVIDER="local"   # or openai + EMBEDDING_API_KEY
```

Optional but recommended for predictable ingest latency:

```bash
export LLM_REQUEST_TIMEOUT="120"
```

## First async client

```python
import asyncio
from engram_memory import AsyncMemoryClient, Config

async def main():
    config = Config()
    async with AsyncMemoryClient(config) as client:
        h = await client.health_check(ping_llm=True)
        assert h.neo4j_connected

        # Ingest: 1 LLM call, batched writes, token tracking
        r = await client.ingest(
            user_id="demo-user",
            text="I work on graph memory at Acme Corp.",
            reference_id="msg-1",
        )
        print(f"{len(r.nodes_created)} nodes, {r.relationships_created} rels, "
              f"{r.tokens_total} tokens")

        # Recall: 0 LLM calls, vector search + graph traversal
        out = await client.recall(
            user_id="demo-user",
            query="Where does the user work?",
            top_k=5,
        )
        for n in out.nodes:
            print(f"  [{n.score:.2f}] {n.summary}")

asyncio.run(main())
```

Use **only** `user_id` values that match your configured pattern (default: alphanumeric, `_`, `-`, up to 128 chars). See [Configuration](configuration.md#user-id-validation).

## Sync wrapper

```python
from engram_memory import MemoryClient, Config

with MemoryClient(Config()) as client:
    client.ingest(user_id="demo-user", text="Hello from sync API.")
    print(client.recall(user_id="demo-user", query="Hello"))
```

Prefer `AsyncMemoryClient` in async servers (FastAPI, asyncio agents).

## Next steps

- [Configuration](configuration.md) — all settings and tuning
- [API overview](api-overview.md) — methods, result types, errors
- [Production & operations](production.md) — smoke tests, CI, observability
