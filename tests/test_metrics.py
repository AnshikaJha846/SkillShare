"""Automated tests for evaluation metrics and statistical analysis."""

import pytest
from src.metrics import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    compute_ndcg_at_k,
    evaluate_query_results,
)
from src.stats import bootstrap_ci, paired_significance_test
from src.data_loader import Profile, Query
from src.retrieval import RetrievalResult


def test_precision_at_k():
    relevant = {"H001", "H002"}
    assert compute_precision_at_k(["H001", "H002", "H003"], relevant, k=3) == pytest.approx(2 / 3)
    assert compute_precision_at_k(["H001", "H002"], relevant, k=3) == pytest.approx(2 / 3)
    assert compute_precision_at_k(["H009"], relevant, k=3) == pytest.approx(0.0)


def test_recall_at_k():
    relevant = {"H001", "H002"}
    assert compute_recall_at_k(["H001", "H002", "H003"], relevant, k=3) == pytest.approx(1.0)
    assert compute_recall_at_k(["H001", "H003"], relevant, k=3) == pytest.approx(0.5)
    assert compute_recall_at_k(["H009"], relevant, k=3) == pytest.approx(0.0)


def test_mrr():
    relevant = {"H001", "H002"}
    assert compute_reciprocal_rank(["H001", "H003"], relevant) == pytest.approx(1.0)
    assert compute_reciprocal_rank(["H003", "H001"], relevant) == pytest.approx(0.5)
    assert compute_reciprocal_rank(["H003", "H004", "H002"], relevant) == pytest.approx(1 / 3)
    assert compute_reciprocal_rank(["H003", "H004"], relevant) == pytest.approx(0.0)


def test_ndcg_at_k():
    graded = {"H001": 2, "H002": 1, "H003": 0}
    # Ideal order: H001, H002, H003 -> NDCG should be 1.0
    assert compute_ndcg_at_k(["H001", "H002", "H003"], graded, k=3) == pytest.approx(1.0)

    # Sub-optimal order: H002, H001 -> NDCG < 1.0
    sub_ndcg = compute_ndcg_at_k(["H002", "H001", "H003"], graded, k=3)
    assert 0.0 < sub_ndcg < 1.0


def test_bootstrap_ci():
    scores = [0.8, 0.9, 0.85, 0.7, 0.95, 0.88, 0.92, 0.78]
    mean, lower, upper = bootstrap_ci(scores, num_samples=500, ci=0.95, random_seed=42)

    assert lower <= mean <= upper
    assert 0.7 <= mean <= 0.95


def test_paired_significance_test():
    # Model B consistently outperforms Model A
    scores_a = [0.2, 0.3, 0.1, 0.25, 0.2, 0.35, 0.15, 0.3, 0.2, 0.25]
    scores_b = [0.8, 0.9, 0.85, 0.7, 0.95, 0.88, 0.92, 0.78, 0.85, 0.9]

    result = paired_significance_test(scores_a, scores_b, num_permutations=1000)
    assert result["diff_mean"] > 0.5
    assert result["permutation_p_value"] < 0.05
    assert result["is_significant_p05"] is True

    # Identical models
    res_identical = paired_significance_test(scores_a, scores_a)
    assert res_identical["diff_mean"] == 0.0
    assert res_identical["permutation_p_value"] == 1.0
