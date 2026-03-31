"""Phase 3: LLM extractor tests."""

import pytest
from unittest.mock import AsyncMock

from engram.exceptions import ExtractionError
from engram.extractors.llm_extractor import LLMExtractor
from engram.models import NodeInstruction, RelInstruction


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.generate_json.return_value = {
        "nodes": [
            {
                "operation": "create",
                "label": "Person",
                "merge_keys": {"name": "Alice"},
                "properties": {"occupation": "engineer"},
                "summary": "Alice is an engineer",
                "cluster_hint": "Work",
            }
        ],
        "relationships": [
            {"from_ref": "temp_0", "to_ref": "eid_google", "type": "WORKS_AT"}
        ],
    }
    return llm


@pytest.mark.asyncio
async def test_extract_returns_valid_instructions(mock_llm):
    extractor = LLMExtractor(llm=mock_llm)
    nodes, rels = await extractor.extract(
        user_id="u1",
        text="Alice is an engineer at Google",
        neighbourhood=[],
    )
    assert len(nodes) == 1
    assert isinstance(nodes[0], NodeInstruction)
    assert nodes[0].label == "Person"
    assert len(rels) == 1
    assert isinstance(rels[0], RelInstruction)


@pytest.mark.asyncio
async def test_extract_includes_cluster_hint(mock_llm):
    extractor = LLMExtractor(llm=mock_llm)
    nodes, _ = await extractor.extract(
        user_id="u1", text="Alice works at Google", neighbourhood=[]
    )
    assert nodes[0].cluster_hint == "Work"


@pytest.mark.asyncio
async def test_extract_with_neighbourhood_context(mock_llm):
    neighbourhood = [
        {"elementId": "eid_google", "label": "Company", "name": "Google"},
    ]
    extractor = LLMExtractor(llm=mock_llm)
    nodes, rels = await extractor.extract(
        user_id="u1", text="Alice works at Google", neighbourhood=neighbourhood
    )
    mock_llm.generate_json.assert_called_once()
    call_args = mock_llm.generate_json.call_args
    assert "Google" in str(call_args)


@pytest.mark.asyncio
async def test_extract_invalid_llm_output_raises():
    bad_llm = AsyncMock()
    bad_llm.generate_json.return_value = {"garbage": True}
    extractor = LLMExtractor(llm=bad_llm)
    with pytest.raises(ExtractionError):
        await extractor.extract(user_id="u1", text="test", neighbourhood=[])


@pytest.mark.asyncio
async def test_extract_supersedes_ref():
    llm = AsyncMock()
    llm.generate_json.return_value = {
        "nodes": [{
            "operation": "create",
            "label": "Employment",
            "merge_keys": {"company": "Meta"},
            "properties": {},
            "summary": "Works at Meta now",
            "supersedes_ref": "eid_old_google_job",
        }],
        "relationships": [],
    }
    extractor = LLMExtractor(llm=llm)
    nodes, _ = await extractor.extract(user_id="u1", text="I now work at Meta", neighbourhood=[])
    assert nodes[0].supersedes_ref == "eid_old_google_job"


@pytest.mark.asyncio
async def test_extract_validation_error_wraps():
    """Malformed node data should raise ExtractionError, not raw ValidationError."""
    llm = AsyncMock()
    llm.generate_json.return_value = {
        "nodes": [{"operation": "bad_op", "label": "X"}],
        "relationships": [],
    }
    extractor = LLMExtractor(llm=llm)
    with pytest.raises(ExtractionError):
        await extractor.extract(user_id="u1", text="test", neighbourhood=[])
