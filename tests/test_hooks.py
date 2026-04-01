"""Phase 4: Hook system tests."""

import pytest
from unittest.mock import AsyncMock

from engram_memory.hooks.base import Hook
from engram_memory.hooks.logger_hook import LoggerHook
from engram_memory.models import IngestResult, RecallResult


@pytest.mark.asyncio
async def test_custom_hook_pre_ingest_called():
    hook = AsyncMock(spec=Hook)
    hook.pre_ingest.return_value = "modified text"
    result = await hook.pre_ingest(user_id="u1", text="original")
    hook.pre_ingest.assert_awaited_once()
    assert result == "modified text"


@pytest.mark.asyncio
async def test_custom_hook_post_ingest_called():
    hook = AsyncMock(spec=Hook)
    result = IngestResult(skipped=False, nodes_created=[], relationships_created=1)
    await hook.post_ingest(user_id="u1", result=result)
    hook.post_ingest.assert_awaited_once()


@pytest.mark.asyncio
async def test_custom_hook_pre_recall_called():
    hook = AsyncMock(spec=Hook)
    hook.pre_recall.return_value = "modified query"
    result = await hook.pre_recall(user_id="u1", query="original")
    assert result == "modified query"


@pytest.mark.asyncio
async def test_custom_hook_post_recall_called():
    hook = AsyncMock(spec=Hook)
    result = RecallResult(nodes=[], total_candidates=0)
    await hook.post_recall(user_id="u1", result=result)
    hook.post_recall.assert_awaited_once()


@pytest.mark.asyncio
async def test_logger_hook_pre_ingest_does_not_raise():
    lh = LoggerHook()
    await lh.pre_ingest(user_id="u1", text="test")


@pytest.mark.asyncio
async def test_logger_hook_post_ingest_does_not_raise():
    lh = LoggerHook()
    await lh.post_ingest(user_id="u1", result=IngestResult(skipped=True))


@pytest.mark.asyncio
async def test_logger_hook_pre_recall_does_not_raise():
    lh = LoggerHook()
    await lh.pre_recall(user_id="u1", query="what does Alice do?")


@pytest.mark.asyncio
async def test_logger_hook_post_recall_does_not_raise():
    lh = LoggerHook()
    await lh.post_recall(user_id="u1", result=RecallResult())
