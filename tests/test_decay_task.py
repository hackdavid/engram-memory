"""Phase 4: Strength decay task tests."""

import pytest
from unittest.mock import AsyncMock

from engram_memory.background.decay_task import DecayTask


@pytest.mark.asyncio
async def test_decay_reduces_strength():
    driver = AsyncMock()
    driver.execute.return_value = [{"updated": 50}]
    task = DecayTask(driver=driver, decay_factor=0.95, archive_threshold=0.01)
    await task.run()
    call_args = driver.execute.call_args_list
    assert len(call_args) >= 1
    cypher = call_args[0].args[0]
    assert "strength" in cypher.lower()


@pytest.mark.asyncio
async def test_decay_archives_below_threshold():
    driver = AsyncMock()
    driver.execute.return_value = [{"archived": 3}]
    task = DecayTask(driver=driver, decay_factor=0.95, archive_threshold=0.01)
    await task.run()
    call_args = driver.execute.call_args_list
    all_cypher = " ".join(c.args[0] for c in call_args)
    assert "_archived" in all_cypher


@pytest.mark.asyncio
async def test_decay_uses_configured_factor():
    driver = AsyncMock()
    driver.execute.return_value = [{"updated": 10}]
    task = DecayTask(driver=driver, decay_factor=0.80, archive_threshold=0.05)
    await task.run()
    call_kwargs = driver.execute.call_args_list[0]
    assert call_kwargs.kwargs.get("decayFactor") == 0.80
