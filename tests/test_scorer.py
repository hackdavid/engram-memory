"""Phase 3: Composite scorer tests."""

import pytest

from engram_memory.graph.scorer import CompositeScorer


@pytest.fixture
def scorer():
    return CompositeScorer(alpha=0.5, beta=0.35, gamma=0.15)


def test_score_calculation(scorer):
    score = scorer.score(vector_similarity=0.9, hops=1, strength=0.8, decay=0.5)
    expected = 0.5 * 0.9 + 0.35 * (0.5 ** 1) + 0.15 * 0.8
    assert abs(score - expected) < 1e-6


def test_score_at_zero_hops(scorer):
    score = scorer.score(vector_similarity=0.9, hops=0, strength=0.8, decay=0.5)
    expected = 0.5 * 0.9 + 0.35 * 1.0 + 0.15 * 0.8
    assert abs(score - expected) < 1e-6


def test_score_decreases_with_hops(scorer):
    s1 = scorer.score(vector_similarity=0.9, hops=0, strength=0.8, decay=0.5)
    s2 = scorer.score(vector_similarity=0.9, hops=3, strength=0.8, decay=0.5)
    assert s1 > s2


def test_rank_sorts_descending(scorer):
    nodes = [
        {"vector_similarity": 0.5, "hops": 2, "strength": 0.3},
        {"vector_similarity": 0.9, "hops": 0, "strength": 0.9},
        {"vector_similarity": 0.7, "hops": 1, "strength": 0.5},
    ]
    ranked = scorer.rank(nodes, decay=0.5)
    scores = [r["final_score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_preserves_node_data(scorer):
    nodes = [{"vector_similarity": 0.8, "hops": 0, "strength": 0.5, "label": "Person"}]
    ranked = scorer.rank(nodes, decay=0.5)
    assert ranked[0]["label"] == "Person"
    assert "final_score" in ranked[0]


def test_custom_weights():
    custom = CompositeScorer(alpha=1.0, beta=0.0, gamma=0.0)
    score = custom.score(vector_similarity=0.8, hops=5, strength=0.1, decay=0.5)
    assert abs(score - 0.8) < 1e-6


def test_only_strength_matters():
    custom = CompositeScorer(alpha=0.0, beta=0.0, gamma=1.0)
    score = custom.score(vector_similarity=0.0, hops=0, strength=0.75, decay=0.5)
    assert abs(score - 0.75) < 1e-6


def test_rank_empty_list(scorer):
    ranked = scorer.rank([], decay=0.5)
    assert ranked == []
