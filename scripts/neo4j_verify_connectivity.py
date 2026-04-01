#!/usr/bin/env python3
"""Minimal Neo4j connectivity check (sync driver), same env vars as Engram.

Loads `.env` then `engram_memory/.env` from the repo root (non-overriding).

  NEO4J_URI      e.g. neo4j+s://xxxx.databases.neo4j.io or bolt+s://...
  NEO4J_USER
  NEO4J_PASSWORD

Run from repo root: python scripts/neo4j_verify_connectivity.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from neo4j import GraphDatabase

_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    if load_dotenv:
        load_dotenv(_ROOT / ".env", override=False)
        load_dotenv(_ROOT / "engram_memory" / ".env", override=False)

    uri = os.environ.get("NEO4J_URI", "").strip()
    user = os.environ.get("NEO4J_USER", "").strip()
    password = os.environ.get("NEO4J_PASSWORD", "").strip()

    missing = [k for k, v in (("NEO4J_URI", uri), ("NEO4J_USER", user), ("NEO4J_PASSWORD", password)) if not v]
    if missing:
        print("Missing environment variables:", ", ".join(missing), file=sys.stderr)
        print("Set them in .env / engram_memory/.env or export before running.", file=sys.stderr)
        return 1

    # Log host only (no credentials)
    safe = uri.split("@")[-1] if "@" in uri else uri
    print(f"Connecting to {safe!r} as user {user!r} ...")

    try:
        with GraphDatabase.driver(uri, auth=(user, password)) as driver:
            driver.verify_connectivity()
    except Exception as exc:
        print("FAILED:", type(exc).__name__, exc, file=sys.stderr)
        return 1

    print("OK: verify_connectivity() succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
