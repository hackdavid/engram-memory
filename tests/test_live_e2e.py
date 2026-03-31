"""Live end-to-end tests against real Neo4j and LLM.

Requires ENGRAM_LIVE_TESTS=1 (in env or `.env` / `engram/.env`) and valid
NEO4J_* / LLM_* (and embedding) vars. conftest loads dotenv when this file is
run or `-m live` is used. Mutates the database; teardown deletes by userId.

Run: pytest tests/test_live_e2e.py -m live
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
import pytest_asyncio

from engram.client import AsyncMemoryClient
from engram.config import Config

LIVE_ENABLED = os.environ.get("ENGRAM_LIVE_TESTS", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not LIVE_ENABLED,
        reason="Set ENGRAM_LIVE_TESTS=1 to run live Neo4j/LLM tests",
    ),
]

_CLEANUP_CYPHER = "MATCH (n) WHERE n.userId = $userId DETACH DELETE n"


def _neo4j_env_ready() -> bool:
    return all(
        os.environ.get(k, "").strip()
        for k in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")
    )


@pytest.fixture
def live_config() -> Config:
    if not _neo4j_env_ready():
        pytest.skip(
            "NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set for live tests"
        )
    return Config()


@pytest.fixture
def live_user_id() -> str:
    return f"engram-live-{uuid.uuid4().hex[:16]}"


@pytest_asyncio.fixture
async def live_client(live_config: Config, live_user_id: str) -> AsyncMemoryClient:
    client = AsyncMemoryClient(live_config)
    await client.__aenter__()
    try:
        yield client
    finally:
        # Intentional teardown: remove all graph data for this test user.
        try:
            await client._driver.execute(
                _CLEANUP_CYPHER, {"userId": live_user_id}
            )
        except Exception:
            pass
        await client.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_live_health(live_client: AsyncMemoryClient) -> None:
    status = await live_client.health_check()
    assert status.neo4j_connected is True
    assert status.embedding_model_loaded is True
    assert status.llm_reachable is True
    assert status.vector_index_exists is True


@pytest.mark.asyncio
async def test_live_ingest_and_recall(
    live_client: AsyncMemoryClient, live_user_id: str
) -> None:
    corp = f"EngramLiveCorp{uuid.uuid4().hex[:8]}"
    text = (
        f"Jordan Vega works at {corp} as a site reliability engineer since 2023."
    )
    ingest = await live_client.ingest(
        user_id=live_user_id,
        text=text,
        reference_id="live-e2e-1",
    )
    assert ingest.skipped is False
    assert len(ingest.nodes_created) + len(ingest.nodes_updated) >= 1

    query = "Where does Jordan Vega work?"
    recall = await live_client.recall(
        user_id=live_user_id, query=query, top_k=5
    )
    if not recall.nodes:
        await asyncio.sleep(2.0)
        recall = await live_client.recall(
            user_id=live_user_id, query=query, top_k=5
        )
    assert len(recall.nodes) >= 1


@pytest.mark.asyncio
async def test_live_delete_roundtrip(
    live_client: AsyncMemoryClient, live_user_id: str
) -> None:
    text = (
        f"Live delete test {uuid.uuid4().hex[:8]}: lunar base codename is Artemis-9."
    )
    ingest = await live_client.ingest(user_id=live_user_id, text=text)
    assert ingest.skipped is False
    assert ingest.nodes_created
    node_id = ingest.nodes_created[0].element_id
    await live_client.delete_memory(
        user_id=live_user_id, node_id=node_id, cascade=True
    )


@pytest.mark.asyncio
async def test_live_recall_returns_relevant_ingested_fact(
    live_client: AsyncMemoryClient, live_user_id: str
) -> None:
    """Relevant query: recalled nodes should echo the unique fact we ingested."""
    unique = f"RelevanceToken_{uuid.uuid4().hex[:10]}"
    text = (
        f"Engineer Morgan Lee maintains the {unique} pipeline "
        f"for nightly analytics batch jobs."
    )
    ingest = await live_client.ingest(
        user_id=live_user_id, text=text, reference_id="rel-1",
    )
    assert ingest.skipped is False
    assert len(ingest.nodes_created) + len(ingest.nodes_updated) >= 1

    recall = await live_client.recall(
        user_id=live_user_id,
        query="Who maintains the analytics batch pipeline?",
        top_k=8,
    )
    if not recall.nodes:
        await asyncio.sleep(2.0)
        recall = await live_client.recall(
            user_id=live_user_id,
            query="Who maintains the analytics batch pipeline?",
            top_k=8,
        )
    assert recall.nodes, "expected at least one recalled node for a relevant query"
    blob = " ".join(
        (n.summary or "") + " " + str(n.properties) for n in recall.nodes
    ).lower()
    assert unique.lower() in blob, (
        "relevant recall should surface the ingested unique token in summaries/properties"
    )


@pytest.mark.asyncio
async def test_live_cross_user_recall_does_not_leak_other_user_memory(
    live_client: AsyncMemoryClient, live_user_id: str
) -> None:
    """Off-user query: another user_id must not see the first user's secret token."""
    token = f"ZZXISO_{uuid.uuid4().hex[:12]}"
    user_a = live_user_id
    user_b = f"engram-live-b-{uuid.uuid4().hex[:16]}"
    try:
        await live_client.ingest(
            user_id=user_a,
            text=f"Confidential initiative {token} is led by Director Chen.",
        )
        r_a = await live_client.recall(
            user_id=user_a,
            query=f"Tell me about initiative {token}",
            top_k=5,
        )
        assert r_a.nodes, "sanity: user A should recall their own ingest"

        r_b = await live_client.recall(
            user_id=user_b,
            query=f"Tell me about initiative {token}",
            top_k=5,
        )
        blob_b = " ".join(
            (n.summary or "") + " " + str(n.properties) for n in r_b.nodes
        )
        assert token not in blob_b, (
            "user B's recall must not contain user A's secret token (isolation)"
        )
    finally:
        try:
            await live_client._driver.execute(
                _CLEANUP_CYPHER, {"userId": user_b}
            )
        except Exception:
            pass


@pytest.mark.asyncio
async def test_live_discriminative_recall_parking_vs_cafeteria(
    live_client: AsyncMemoryClient, live_user_id: str
) -> None:
    """Two ingests with distinct reference_id: recall should surface the matching ingest."""
    ref_park = f"engram-live-park-{uuid.uuid4().hex[:8]}"
    ref_cafe = f"engram-live-cafe-{uuid.uuid4().hex[:8]}"
    await live_client.ingest(
        user_id=live_user_id,
        text=(
            "Visitors must use garage PARKING-GARAGE-A on level 2 for badge pickup "
            "before entering the Seattle office."
        ),
        reference_id=ref_park,
    )
    await live_client.ingest(
        user_id=live_user_id,
        text=(
            "The cafeteria offers a fresh salad bar on Thursdays "
            "on the first floor mezzanine."
        ),
        reference_id=ref_cafe,
    )

    r_park = await live_client.recall(
        user_id=live_user_id,
        query="Where do I pick up my visitor parking badge?",
        top_k=6,
    )
    if not r_park.nodes:
        await asyncio.sleep(2.0)
        r_park = await live_client.recall(
            user_id=live_user_id,
            query="Where do I pick up my visitor parking badge?",
            top_k=6,
        )

    park_refs = {n.reference_id for n in r_park.nodes if n.reference_id}
    assert ref_park in park_refs, "parking query should retrieve nodes from the parking ingest"
    top3_park = {n.reference_id for n in r_park.nodes[:3] if n.reference_id}
    assert ref_cafe not in top3_park, (
        "top parking hits should not be the cafeteria ingest (graph expansion may add it lower)"
    )

    r_cafe = await live_client.recall(
        user_id=live_user_id,
        query="What food options are available on Thursday?",
        top_k=6,
    )
    if not r_cafe.nodes:
        await asyncio.sleep(2.0)
        r_cafe = await live_client.recall(
            user_id=live_user_id,
            query="What food options are available on Thursday?",
            top_k=6,
        )

    cafe_refs = {n.reference_id for n in r_cafe.nodes if n.reference_id}
    assert ref_cafe in cafe_refs, "cafeteria query should retrieve nodes from the cafeteria ingest"
    top3_cafe = {n.reference_id for n in r_cafe.nodes[:3] if n.reference_id}
    assert ref_park not in top3_cafe, (
        "top cafeteria hits should not be the parking ingest"
    )
