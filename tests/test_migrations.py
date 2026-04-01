"""Phase 2: Schema migration runner tests."""

import pytest
from unittest.mock import AsyncMock

from engram_memory.graph.migrations import MigrationRunner


@pytest.mark.asyncio
async def test_no_migration_needed():
    driver = AsyncMock()
    driver.execute.return_value = [{"max_version": 1}]
    runner = MigrationRunner(driver, target_version=1)
    applied = await runner.check_and_migrate()
    assert applied == 0


@pytest.mark.asyncio
async def test_migration_from_v0_to_v1():
    driver = AsyncMock()
    driver.execute.return_value = [{"max_version": None}]
    runner = MigrationRunner(driver, target_version=1)
    applied = await runner.check_and_migrate()
    assert applied == 1


@pytest.mark.asyncio
async def test_migration_runs_sequentially():
    driver = AsyncMock()
    driver.execute.return_value = [{"max_version": None}]
    runner = MigrationRunner(driver, target_version=3)
    applied = await runner.check_and_migrate()
    # v1 migration exists; v2 and v3 have no scripts yet so are skipped
    assert applied >= 1
    assert driver.execute.call_count >= 2


@pytest.mark.asyncio
async def test_already_up_to_date():
    driver = AsyncMock()
    driver.execute.return_value = [{"max_version": 5}]
    runner = MigrationRunner(driver, target_version=3)
    applied = await runner.check_and_migrate()
    assert applied == 0
