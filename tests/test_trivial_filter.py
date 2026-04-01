"""Phase 2: Trivial message filter tests."""

from engram_memory.extractors.trivial_filter import is_trivial


# ── Trivial messages ────────────────────────────────────────────────


def test_trivial_short_greeting():
    assert is_trivial("hi") is True
    assert is_trivial("hello there") is True
    assert is_trivial("thanks!") is True


def test_trivial_common_non_factual():
    assert is_trivial("ok sounds good") is True
    assert is_trivial("lol") is True


def test_trivial_short_acknowledgements():
    assert is_trivial("sure") is True
    assert is_trivial("yep") is True
    assert is_trivial("cool") is True


# ── Non-trivial messages ────────────────────────────────────────────


def test_non_trivial_factual_statement():
    assert is_trivial("I work at Google as a software engineer") is False
    assert is_trivial("My name is Alice and I live in London") is False


def test_non_trivial_with_entities():
    assert is_trivial("I graduated from MIT in 2020") is False


def test_non_trivial_long_text():
    assert is_trivial(
        "I have been working in machine learning for 5 years, "
        "specialising in NLP and transformer architectures"
    ) is False


def test_non_trivial_with_proper_nouns():
    assert is_trivial("I studied at Cambridge University") is False


# ── Edge cases ──────────────────────────────────────────────────────


def test_edge_case_empty_string():
    assert is_trivial("") is True


def test_edge_case_whitespace():
    assert is_trivial("   ") is True


def test_edge_case_single_word_unknown():
    assert is_trivial("xyz") is True


# ── Custom patterns ─────────────────────────────────────────────────


def test_custom_override_matches():
    assert is_trivial("skip this please", custom_trivial_patterns=[r"skip this"]) is True


def test_custom_override_no_match():
    assert is_trivial("I work at Google", custom_trivial_patterns=[r"skip this"]) is False
