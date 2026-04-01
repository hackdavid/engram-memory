"""Phase 4: Background task runner tests."""

import pytest
import asyncio
from unittest.mock import AsyncMock

from engram_memory.background.runner import BackgroundRunner


@pytest.mark.asyncio
async def test_runner_executes_task():
    task_fn = AsyncMock()
    runner = BackgroundRunner()
    runner.register("test_task", task_fn, interval_seconds=0.1)
    await runner.start()
    await asyncio.sleep(0.35)
    await runner.stop()
    assert task_fn.call_count >= 2


@pytest.mark.asyncio
async def test_runner_stop_is_clean():
    task_fn = AsyncMock()
    runner = BackgroundRunner()
    runner.register("test_task", task_fn, interval_seconds=60)
    await runner.start()
    await runner.stop()


@pytest.mark.asyncio
async def test_run_once_for_testing():
    task_fn = AsyncMock()
    runner = BackgroundRunner()
    runner.register("test_task", task_fn, interval_seconds=3600)
    await runner.run_once("test_task")
    task_fn.assert_awaited_once()


@pytest.mark.asyncio
async def test_task_failure_does_not_crash_runner():
    fail_task = AsyncMock(side_effect=Exception("task error"))
    ok_task = AsyncMock()
    runner = BackgroundRunner()
    runner.register("fail_task", fail_task, interval_seconds=0.1)
    runner.register("ok_task", ok_task, interval_seconds=0.1)
    await runner.start()
    await asyncio.sleep(0.35)
    await runner.stop()
    assert ok_task.call_count >= 2


@pytest.mark.asyncio
async def test_run_once_unknown_task_raises():
    runner = BackgroundRunner()
    with pytest.raises(KeyError):
        await runner.run_once("nonexistent")
