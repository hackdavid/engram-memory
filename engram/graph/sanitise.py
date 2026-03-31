"""Label, relationship-type, and user-id sanitisation utilities."""

from __future__ import annotations

import re

from engram.constants import DEFAULT_USER_ID_PATTERN
from engram.exceptions import InvalidUserIdError

_LABEL_RE = re.compile(r"[^A-Za-z0-9_]")
_SPACES_RE = re.compile(r"\s+")


def sanitise_label(raw: str) -> str:
    """Remove non-alphanumeric/underscore characters from a Neo4j label.

    Raises ValueError if the result is empty.
    """
    cleaned = _LABEL_RE.sub("", raw)
    if not cleaned:
        raise ValueError(f"Label is empty after sanitisation: {raw!r}")
    return cleaned


def sanitise_rel_type(raw: str) -> str:
    """Normalise a relationship type to UPPER_SNAKE_CASE, stripping invalid chars.

    Spaces become underscores before stripping; the result is uppercased.
    Raises ValueError if the result is empty.
    """
    with_underscores = _SPACES_RE.sub("_", raw.strip())
    cleaned = _LABEL_RE.sub("", with_underscores)
    if not cleaned:
        raise ValueError(f"Relationship type is empty after sanitisation: {raw!r}")
    return cleaned.upper()


def validate_user_id(
    user_id: str,
    pattern: str = DEFAULT_USER_ID_PATTERN,
) -> None:
    """Validate a user_id against the configured regex pattern.

    Raises InvalidUserIdError on mismatch.
    """
    if not re.match(pattern, user_id):
        raise InvalidUserIdError(
            f"user_id {user_id!r} does not match pattern {pattern!r}"
        )
