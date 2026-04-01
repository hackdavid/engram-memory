"""Structured JSON logging setup for Engram."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
) -> None:
    """Configure the root Engram logger.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        json_format: If True, emit JSON lines; otherwise plain text.
    """
    root = logging.getLogger("engram_memory")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stderr)
    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s  %(message)s")
        )
    root.addHandler(handler)
