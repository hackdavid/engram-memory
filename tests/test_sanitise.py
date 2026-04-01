"""Phase 1: Verify label/rel-type sanitisation and userId validation."""

import pytest

from engram_memory.exceptions import InvalidUserIdError
from engram_memory.graph.sanitise import sanitise_label, sanitise_rel_type, validate_user_id


# ── sanitise_label ──────────────────────────────────────────────────


def test_sanitise_label_removes_special_chars():
    assert sanitise_label("My Label!@#") == "MyLabel"


def test_sanitise_label_preserves_valid():
    assert sanitise_label("PersonNode") == "PersonNode"


def test_sanitise_label_underscores_allowed():
    assert sanitise_label("Work_Experience") == "Work_Experience"


def test_sanitise_label_numbers_allowed():
    assert sanitise_label("Level2Node") == "Level2Node"


def test_sanitise_label_empty_raises():
    with pytest.raises(ValueError):
        sanitise_label("!@#$%")


def test_sanitise_label_injection_attempt():
    result = sanitise_label("Person; DROP DATABASE")
    assert "DROP" not in result or result == "PersonDROPDATABASE"
    assert ";" not in result


# ── sanitise_rel_type ───────────────────────────────────────────────


def test_sanitise_rel_type_uppercase():
    assert sanitise_rel_type("works at") == "WORKS_AT"


def test_sanitise_rel_type_special_chars():
    result = sanitise_rel_type("LIVES-IN!")
    assert "!" not in result
    assert result == "LIVESIN"


def test_sanitise_rel_type_already_clean():
    assert sanitise_rel_type("WORKS_AT") == "WORKS_AT"


def test_sanitise_rel_type_empty_raises():
    with pytest.raises(ValueError):
        sanitise_rel_type("---!!!")


# ── validate_user_id ────────────────────────────────────────────────


def test_validate_user_id_valid():
    validate_user_id("user-123_abc")


def test_validate_user_id_simple():
    validate_user_id("u1")


def test_validate_user_id_invalid_spaces():
    with pytest.raises(InvalidUserIdError):
        validate_user_id("user with spaces!!")


def test_validate_user_id_empty():
    with pytest.raises(InvalidUserIdError):
        validate_user_id("")


def test_validate_user_id_too_long():
    with pytest.raises(InvalidUserIdError):
        validate_user_id("a" * 200)


def test_validate_user_id_special_chars():
    with pytest.raises(InvalidUserIdError):
        validate_user_id("user@domain.com")
