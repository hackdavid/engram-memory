"""CLI argument validation for engram.cli.e2e_validate (no live services)."""

import pytest

from engram.cli.e2e_validate import main


def test_skip_seed_requires_user_id():
    with pytest.raises(SystemExit) as exc:
        main(["--skip-seed"])
    assert exc.value.code == 2


def test_help_zero_exit():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
