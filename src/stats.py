"""Statistical rigor and significance testing module for SkillShare Match.

Implements:
1. Non-parametric Bootstrap Confidence Intervals (95% default).
2. Paired Permutation Test & Wilcoxon Signed-Rank Test between ranking models.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple
import numpy as np
from scipy import stats


def bootstrap_ci(
    values: Sequence[float],
    num_samples: int = 1000,
    ci: float = 0.95,
    random_seed: int = 42,
) -> Tuple[float, float, float]:
    """Calculate mean and non-parametric bootstrap confidence interval.

    Args:
        values: Sequence of metric scores across queries.
        num_samples: Number of bootstrap resamples.
        ci: Confidence interval fraction (default 0.95).
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (mean, lower_bound, upper_bound).
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    if len(arr) == 1:
        val = float(arr[0])
        return val, val, val

    rng = np.random.RandomState(random_seed)
    boot_means = np.empty(num_samples, dtype=np.float64)

    n = len(arr)
    for i in range(num_samples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    mean_val = float(np.mean(arr))
    lower_bound = float(np.percentile(boot_means, lower_pct))
    upper_bound = float(np.percentile(boot_means, upper_pct))

    return round(mean_val, 4), round(lower_bound, 4), round(upper_bound, 4)


def paired_significance_test(
    scores_model_a: Sequence[float],
    scores_model_b: Sequence[float],
    num_permutations: int = 2000,
    random_seed: int = 42,
) -> Dict[str, any]:
    """Perform paired permutation test to compare two models across identical queries.

    Args:
        scores_model_a: Metric scores from Model A (e.g. TF-IDF).
        scores_model_b: Metric scores from Model B (e.g. S-BERT).
        num_permutations: Number of permutation resamples.
        random_seed: Seed for reproducible permutation.

    Returns:
        Dictionary with diff_mean, p_value, and statistical significance flag.
    """
    a = np.asarray(scores_model_a, dtype=np.float64)
    b = np.asarray(scores_model_b, dtype=np.float64)

    if len(a) == 0:
        return {
            "diff_mean": 0.0,
            "permutation_p_value": 1.0,
            "wilcoxon_p_value": 1.0,
            "is_significant_p05": False,
        }

    diffs = b - a
    if np.all(diffs == 0.0):
        return {
            "diff_mean": 0.0,
            "permutation_p_value": 1.0,
            "wilcoxon_p_value": 1.0,
            "is_significant_p05": False,
        }

    observed_diff = float(np.mean(diffs))

    rng = np.random.RandomState(random_seed)
    perm_diffs = np.empty(num_permutations, dtype=np.float64)

    for i in range(num_permutations):
        # Randomly flip signs with 50% probability
        signs = rng.choice([-1.0, 1.0], size=len(diffs))
        perm_diffs[i] = np.mean(diffs * signs)

    p_value = float(np.mean(np.abs(perm_diffs) >= np.abs(observed_diff)))

    # Also compute Wilcoxon signed-rank test as secondary validation
    try:
        # Wilcoxon requires non-zero diffs
        non_zero = diffs[diffs != 0]
        if len(non_zero) >= 5:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(diffs, zero_method="wilcox")
            wilcoxon_p = float(wilcoxon_p)
        else:
            wilcoxon_p = p_value
    except Exception:
        wilcoxon_p = p_value

    return {
        "diff_mean": round(observed_diff, 4),
        "permutation_p_value": round(p_value, 4),
        "wilcoxon_p_value": round(wilcoxon_p, 4),
        "is_significant_p05": p_value < 0.05,
    }
