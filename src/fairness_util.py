import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def demographic_parity_diff(
    df: pd.DataFrame,
    group_col: str = "group",
    score_col: str = "y_pred",
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Demographic Parity Difference (DPD).

    If threshold is None: compares mean scores by group.
    If threshold is given: compares positive rate (score >= threshold) by group.

    Returns:
        {
          "per_group": {group_name: value},
          "max_gap": float,        # max difference between any two groups
        }
    """
    d = df.copy()

    if threshold is None:
        # Use mean score
        agg = d.groupby(group_col)[score_col].mean()
    else:
        d["_selected"] = (d[score_col] >= threshold).astype(int)
        agg = d.groupby(group_col)["_selected"].mean()

    per_group = agg.to_dict()
    values = agg.values
    max_gap = float(np.max(values) - np.min(values))

    return {"per_group": per_group, "max_gap": max_gap}


def selection_rate_parity_topk(
    df: pd.DataFrame,
    group_col: str = "group",
    score_col: str = "y_pred",
    k: int = 50,
) -> Dict[str, float]:
    """
    Selection Rate Parity using Top-K selection.

    Steps:
    - Sort by score (descending).
    - Mark top-K as selected.
    - Compute selection rate per group.
    - Compute min_group_rate / max_group_rate (4/5 rule style).

    Returns:
        {
          "per_group": {group_name: selection_rate},
          "min_over_max": float    # closer to 1 is better
        }
    """
    d = df.copy().sort_values(score_col, ascending=False)
    d["_selected"] = 0
    d.loc[d.index[:k], "_selected"] = 1

    rates = d.groupby(group_col)["_selected"].mean()
    per_group = rates.to_dict()
    vals = rates.values

    min_over_max = float(np.min(vals) / np.max(vals)) if np.max(vals) > 0 else 0.0

    return {"per_group": per_group, "min_over_max": min_over_max}


def score_distribution_overlap(
    df: pd.DataFrame,
    group_a: str,
    group_b: str,
    group_col: str = "group",
    score_col: str = "y_pred",
    bins: int = 20,
) -> float:
    """
    Score Distribution Overlap (SDO) between two groups.

    Approximation:
    - Build histograms (density=True) for each group.
    - Overlap = sum over bins of min(density_a, density_b) * bin_width

    Returns a value in [0, 1] where 1 means perfect overlap.
    """
    d = df[[group_col, score_col]].dropna()

    scores_a = d.loc[d[group_col] == group_a, score_col].values
    scores_b = d.loc[d[group_col] == group_b, score_col].values

    if len(scores_a) == 0 or len(scores_b) == 0:
        return np.nan

    hist_range = (min(d[score_col]), max(d[score_col]))

    dens_a, bin_edges = np.histogram(scores_a, bins=bins, range=hist_range, density=True)
    dens_b, _ = np.histogram(scores_b, bins=bins, range=hist_range, density=True)

    bin_widths = np.diff(bin_edges)
    overlap = np.sum(np.minimum(dens_a, dens_b) * bin_widths)

    # Clamp to [0,1]
    overlap = float(max(0.0, min(1.0, overlap)))
    return overlap


def rank_ordering_bias(
    df: pd.DataFrame,
    group_col: str = "group",
    score_col: str = "y_pred",
) -> Dict[str, float]:
    """
    Rank Ordering Bias.

    - Compute ranking by descending score.
    - Compute average rank per group (1 = best).
    - Return per-group mean rank and max difference between groups.

    Returns:
        {
          "per_group_avg_rank": {group_name: avg_rank},
          "max_rank_gap": float
        }
    """
    d = df.copy().sort_values(score_col, ascending=False)
    d["_rank"] = np.arange(1, len(d) + 1)

    avg_rank = d.groupby(group_col)["_rank"].mean()
    per_group = avg_rank.to_dict()
    vals = avg_rank.values
    max_rank_gap = float(np.max(vals) - np.min(vals))

    return {"per_group_avg_rank": per_group, "max_rank_gap": max_rank_gap}


def equal_opportunity_diff(
    df: pd.DataFrame,
    group_col: str = "group",
    y_true_col: str = "y_true",
    score_col: str = "y_pred",
    positive_label: int = 1,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Equal Opportunity Difference (difference in TPR across groups).

    - Positive = (score >= threshold)
    - TPR_g = P(predicted positive | true positive, group=g)

    Returns:
        {
          "per_group_tpr": {group_name: tpr},
          "max_tpr_gap": float
        }
    """
    d = df.copy()
    d["_pred_pos"] = (d[score_col] >= threshold).astype(int)
    d["_true_pos"] = (d[y_true_col] == positive_label).astype(int)

    per_group_tpr = {}
    for g, sub in d.groupby(group_col):
        # Only consider true positives in this group
        sub_tp = sub[sub["_true_pos"] == 1]
        if len(sub_tp) == 0:
            per_group_tpr[g] = np.nan
        else:
            per_group_tpr[g] = float(sub_tp["_pred_pos"].mean())

    # compute max gap ignoring NaNs
    vals = np.array([v for v in per_group_tpr.values() if not np.isnan(v)])
    max_tpr_gap = float(np.max(vals) - np.min(vals)) if len(vals) > 0 else np.nan

    return {"per_group_tpr": per_group_tpr, "max_tpr_gap": max_tpr_gap}


def compute_fairness_drift(
    baseline_metrics: Dict[str, float],
    current_metrics: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute absolute drift between baseline and current fairness metrics.

    Expects flat dicts, e.g.:
        baseline_metrics = {"dp_max_gap": 0.03, "gender_min_over_max": 0.92}
        current_metrics  = {"dp_max_gap": 0.11, "gender_min_over_max": 0.75}

    Returns:
        {
          "metric_name": abs_diff,
          ...
        }
    """
    drift = {}
    for k in baseline_metrics:
        if k in current_metrics and baseline_metrics[k] is not None and current_metrics[k] is not None:
            try:
                drift[k] = float(abs(current_metrics[k] - baseline_metrics[k]))
            except Exception:
                # non-numeric, skip
                continue
    return drift
