"""Rule-based trivial message filter -- skips non-factual messages."""

from __future__ import annotations

import re

_MAX_TRIVIAL_WORDS = 6

_TRIVIAL_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^(hi|hey|hello|yo|sup)(\s|!|,|$)",
        r"^(ok|okay|k|sure|yep|yeah|yes|no|nah|nope)(\s|!|$)",
        r"^(thanks|thank you|thx|ty)(\s|!|$)",
        r"^(lol|lmao|haha|hehe|rofl)",
        r"^(bye|goodbye|see you|later|cya)",
        r"^(good morning|good night|gm|gn)(\s|!|$)",
        r"^(sounds good|got it|makes sense|right|exactly|indeed)",
        r"^(hmm|hm+|ah|oh|wow|whoa)(\s|!|$)",
        r"^(nice|cool|great|awesome|perfect|wonderful)(\s|!|$)",
    ]
]

_ENTITY_MARKERS = re.compile(
    r"""
    \b\d{4}\b                  # years
    | \b\d+\s*(years?|months?) # durations
    | \b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+ # multi-word proper nouns
    | \b(?:at|for|in|from)\s+[A-Z]     # preposition + capitalised word
    """,
    re.VERBOSE,
)


def is_trivial(
    text: str,
    custom_trivial_patterns: list[str] | None = None,
) -> bool:
    """Return True if the text is unlikely to contain factual information.

    Checks: empty/whitespace, short + matches common non-factual patterns,
    no entity markers. Custom regex patterns can supplement the built-ins.
    """
    stripped = text.strip()
    if not stripped:
        return True

    if custom_trivial_patterns:
        for pat in custom_trivial_patterns:
            if re.search(pat, stripped, re.IGNORECASE):
                return True

    words = stripped.split()

    if len(words) <= _MAX_TRIVIAL_WORDS:
        for pattern in _TRIVIAL_PATTERNS:
            if pattern.search(stripped):
                return True

    if _ENTITY_MARKERS.search(stripped):
        return False

    if len(words) <= 3:
        return True

    return False
