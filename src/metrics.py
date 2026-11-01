"""Evaluation metrics calculation engine for SkillShare Match.

Computes:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@K) with graded relevance
- Constraint Satisfaction Rate
- No-Match Accuracy
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set
from src.data_loader import Profile, Query
from src.constraints import check_hard_constraints
from src.retrieval import RetrievalResult


def compute_precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int = 3,
) -> float:
    """Calculate Precision@K: Fraction of top-K results that are relevant."""
    if not relevant_ids:
        # If ground truth has no relevant items (e.g. no-match),
        # 1.0 if system returned 0 results, 0.0 if false positive recommendations returned
        return 1.0 if len(retrieved_ids) == 0 else 0.0

    top_k_ids = retrieved_ids[:k]
    if not top_k_ids:
        return 0.0

    hits = sum(1 for hid in top_k_ids if hid in relevant_ids)
    return hits / k


def compute_recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
    k: int = 3,
) -> float:
    """Calculate Recall@K: Fraction of all relevant items captured in top-K."""
    if not relevant_ids:
        return 1.0 if len(retrieved_ids) == 0 else 0.0

    top_k_ids = retrieved_ids[:k]
    hits = sum(1 for hid in top_k_ids if hid in relevant_ids)
    return hits / len(relevant_ids)


def compute_reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str],
) -> float:
    """Calculate Reciprocal Rank (RR): 1 / rank of the first relevant result."""
    if not relevant_ids:
        return 1.0 if len(retrieved_ids) == 0 else 0.0

    for rank_idx, hid in enumerate(retrieved_ids, start=1):
        if hid in relevant_ids:
            return 1.0 / rank_idx
    return 0.0


def compute_dcg(relevances: Sequence[int], k: int = 3) -> float:
    """Compute Discounted Cumulative Gain up to rank K."""
    dcg = 0.0
    for idx, rel in enumerate(relevances[:k], start=1):
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(idx + 1)
    return dcg


def compute_ndcg_at_k(
    retrieved_ids: Sequence[str],
    graded_relevance_map: Dict[str, int],
    k: int = 3,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K) with graded relevance."""
    if not graded_relevance_map:
        return 1.0 if len(retrieved_ids) == 0 else 0.0

    # Actual DCG
    actual_rels = [graded_relevance_map.get(hid, 0) for hid in retrieved_ids[:k]]
    actual_dcg = compute_dcg(actual_rels, k=k)

    # Ideal DCG
    all_possible_rels = sorted(graded_relevance_map.values(), reverse=True)
    ideal_dcg = compute_dcg(all_possible_rels, k=k)

    if ideal_dcg == 0.0:
        return 1.0 if actual_dcg == 0.0 else 0.0

    return actual_dcg / ideal_dcg


def compute_constraint_satisfaction_rate(
    results: Sequence[RetrievalResult],
    query: Query,
) -> float:
    """Measure the fraction of returned results that meet 100% of required constraints."""
    if not results:
        return 1.0

    satisfied_count = 0
    for res in results:
        passed, _ = check_hard_constraints(res.profile, query.constraints)
        if passed:
            satisfied_count += 1

    return satisfied_count / len(results)


def evaluate_query_results(
    results: Sequence[RetrievalResult],
    query: Query,
    graded_labels_for_query: Optional[Dict[str, int]] = None,
    k: int = 3,
) -> Dict[str, float]:
    """Calculate all standard evaluation metrics for a single query."""
    retrieved_ids = [r.helper_id for r in results]
    relevant_ids = set(query.relevant_helper_ids)

    # Use binary relevance if graded labels not explicitly provided
    if graded_labels_for_query is None:
        graded_labels_for_query = {hid: 2 for hid in relevant_ids}

    p_at_k = compute_precision_at_k(retrieved_ids, relevant_ids, k=k)
    r_at_k = compute_recall_at_k(retrieved_ids, relevant_ids, k=k)
    mrr = compute_reciprocal_rank(retrieved_ids, relevant_ids)
    ndcg = compute_ndcg_at_k(retrieved_ids, graded_labels_for_query, k=k)
    constraint_sat = compute_constraint_satisfaction_rate(results, query)

    # No-match accuracy: 1.0 if no-match query returned 0 results, else 0.0
    if query.category == "no_match":
        no_match_acc = 1.0 if len(retrieved_ids) == 0 else 0.0
    else:
        no_match_acc = 1.0

    return {
        f"precision@{k}": round(p_at_k, 4),
        f"recall@{k}": round(r_at_k, 4),
        "mrr": round(mrr, 4),
        f"ndcg@{k}": round(ndcg, 4),
        "constraint_satisfaction": round(constraint_sat, 4),
        "no_match_accuracy": round(no_match_acc, 4),
    }
