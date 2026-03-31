"""Shared test fixtures for Engram SDK tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from dotenv import dotenv_values, load_dotenv
except ImportError:
    dotenv_values = None
    load_dotenv = None

_ROOT = Path(__file__).resolve().parent.parent


def _pytest_targets_live_e2e(argv: list[str]) -> bool:
    """True when this pytest invocation is meant to run live E2E tests."""
    joined = " ".join(argv)
    if "test_live_e2e" in joined:
        return True
    for i, a in enumerate(argv):
        if a == "-m" and i + 1 < len(argv):
            expr = argv[i + 1].strip()
            # Only exact marker `live` — avoid matching `not live` on full suite runs.
            if expr == "live":
                return True
    return False


def _engram_live_flag_set() -> bool:
    """True if ENGRAM_LIVE_TESTS is enabled in the process env or in dotenv files."""
    v = os.environ.get("ENGRAM_LIVE_TESTS", "").strip()
    if v.lower() in ("1", "true", "yes", "on"):
        return True
    if dotenv_values is None:
        return False
    merged: dict[str, str | None] = {}
    for path in (_ROOT / ".env", _ROOT / "engram" / ".env"):
        if not path.exists():
            continue
        for k, val in dotenv_values(path).items():
            if val is not None and k not in merged:
                merged[k] = val
    v2 = str(merged.get("ENGRAM_LIVE_TESTS", "") or "").strip()
    return v2.lower() in ("1", "true", "yes", "on")


def _load_env_files() -> None:
    """Load `.env` from repo root, then `engram/.env` for any keys not already set."""
    if load_dotenv is None:
        return
    load_dotenv(_ROOT / ".env", override=False)
    load_dotenv(_ROOT / "engram" / ".env", override=False)


# Load dotenv only when (a) live E2E is targeted, and (b) ENGRAM_LIVE_TESTS is on (env or file peek).
# This avoids polluting unit tests on `pytest tests/` while allowing `pytest tests/test_live_e2e.py`
# with ENGRAM_LIVE_TESTS only in `.env`.
if _pytest_targets_live_e2e(sys.argv) and _engram_live_flag_set():
    _load_env_files()
