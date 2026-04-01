#!/usr/bin/env python3
"""Launch engram_memory E2E validation from a source checkout (repo root on sys.path).

Installed package::

  engram_memory-e2e
  python -m engram_memory.cli.e2e_validate

From clone without editable install::

  python scripts/e2e_validate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from engram_memory.cli.e2e_validate import main

if __name__ == "__main__":
    raise SystemExit(main())
