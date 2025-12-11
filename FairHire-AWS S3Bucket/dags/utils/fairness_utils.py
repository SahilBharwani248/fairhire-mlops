"""
Fairness metrics utilities for recruitment model evaluation.
Based on fairness metrics from: https://arxiv.org/pdf/2405.19699
(Fairness in AI-Driven Recruitment: Challenges, Metrics, Methods, and Future Directions)
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def demographic_parity_diff(
    df: pd.DataFrame,
    group_col: str,
    score_col: str,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate demographic parity difference (hire rate by group).
    
    Args:
        df: DataFrame with predictions and groups
        group_col: Column name for demographic groups
        score_col: Column name for scores or predictions (binary)
        threshold: Optional threshold for binary classification
    
    Returns:
        Dictionary with per-group rates and max gap
    """
    df_clean = df.dropna(subset=[group_col, score_col])
    
    if threshold is not None:
        binary = (df_clean[score_col] >= threshold).astype(int)
    else:
        # Assume already binary (0/1)
        binary = df_clean[score_col]
    
    per_group = binary.groupby(df_clean[group_col]).mean().to_dict()
    rates = list(per_group.values())
    max_gap = max(rates) - min(rates) if rates else 0.0
    
    return {
        "per_group": per_group,
        "max_gap": max_gap,
    }


def selection_rate_parity_topk(
    df: pd.DataFrame,
    group_col: str,
    score_col: str,
    k: int,
) -> Dict[str, Any]:
    """
    Calculate top-K selection rate parity across groups.
    
    Args:
        df: DataFrame with scores and groups
        group_col: Column name for demographic groups
        score_col: Column name for scores
        k: Number of top candidates to consider
    
    Returns:
        Dictionary with per-group selection rates and min/max ratio
    """
    df_clean = df.dropna(subset=[group_col, score_col])
    df_sorted = df_clean.sort_values(score_col, ascending=False)
    top_k = df_sorted.head(k)
    
    total_by_group = df_clean[group_col].value_counts().to_dict()
    topk_by_group = top_k[group_col].value_counts().to_dict()
    
    per_group = {}
    for g in total_by_group.keys():
        total_count = total_by_group.get(g, 0)
        topk_count = topk_by_group.get(g, 0)
        # Avoid division by zero - if group has no members, set rate to 0
        if total_count > 0:
            per_group[g] = topk_count / total_count
        else:
            per_group[g] = 0.0
    
    rates = list(per_group.values())
    min_rate = min(rates) if rates else 0.0
    max_rate = max(rates) if rates else 0.0
    min_over_max = min_rate / max_rate if max_rate > 0 else 0.0
    
    return {
        "per_group": per_group,
        "min_over_max": min_over_max,
    }


def rank_ordering_bias(
    df: pd.DataFrame,
    group_col: str,
    score_col: str,
) -> Dict[str, Any]:
    """
    Calculate rank ordering bias (average rank by group).
    Lower average rank = appears earlier in shortlist.
    
    Args:
        df: DataFrame with scores and groups
        group_col: Column name for demographic groups
        score_col: Column name for scores
    
    Returns:
        Dictionary with per-group average ranks and max gap
    """
    df_clean = df.dropna(subset=[group_col, score_col])
    df_clean = df_clean.copy()
    df_clean['rank'] = df_clean[score_col].rank(ascending=False, method='min')
    
    per_group_avg_rank = df_clean.groupby(group_col)['rank'].mean().to_dict()
    ranks = list(per_group_avg_rank.values())
    max_rank_gap = max(ranks) - min(ranks) if ranks else 0.0
    
    return {
        "per_group_avg_rank": per_group_avg_rank,
        "max_rank_gap": max_rank_gap,
    }


def equal_opportunity_diff(
    df: pd.DataFrame,
    group_col: str,
    y_true_col: str,
    score_col: str,
    positive_label: int = 1,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Calculate equal opportunity difference (TPR parity across groups).
    
    Args:
        df: DataFrame with true labels, predictions, and groups
        group_col: Column name for demographic groups
        y_true_col: Column name for true labels
        score_col: Column name for predicted scores
        positive_label: Label considered positive (e.g., 1 for "Hire")
        threshold: Threshold for binary classification
    
    Returns:
        Dictionary with per-group TPR and max gap
    """
    df_clean = df.dropna(subset=[group_col, y_true_col, score_col])
    df_clean = df_clean.copy()
    
    # Binary predictions
    df_clean['y_pred'] = (df_clean[score_col] >= threshold).astype(int)
    df_clean['y_true'] = df_clean[y_true_col]
    
    # True positives and positives per group
    per_group_tpr = {}
    
    for group in df_clean[group_col].unique():
        group_df = df_clean[df_clean[group_col] == group]
        true_positives = ((group_df['y_true'] == positive_label) & 
                         (group_df['y_pred'] == positive_label)).sum()
        positives = (group_df['y_true'] == positive_label).sum()
        
        tpr = true_positives / positives if positives > 0 else np.nan
        per_group_tpr[group] = tpr
    
    tprs = [tpr for tpr in per_group_tpr.values() if not np.isnan(tpr)]
    max_tpr_gap = max(tprs) - min(tprs) if len(tprs) > 1 else 0.0
    
    return {
        "per_group_tpr": per_group_tpr,
        "max_tpr_gap": max_tpr_gap,
    }


def score_distribution_overlap(
    df: pd.DataFrame,
    group_a: str,
    group_b: str,
    group_col: str,
    score_col: str,
    bins: int = 20,
) -> float:
    """
    Calculate overlap between score distributions of two groups.
    
    Args:
        df: DataFrame with scores and groups
        group_a: First group name
        group_b: Second group name
        group_col: Column name for demographic groups
        score_col: Column name for scores
        bins: Number of bins for histogram
    
    Returns:
        Overlap coefficient (0-1, higher = more similar)
    """
    df_clean = df.dropna(subset=[group_col, score_col])
    
    scores_a = df_clean[df_clean[group_col] == group_a][score_col].values
    scores_b = df_clean[df_clean[group_col] == group_b][score_col].values
    
    if len(scores_a) == 0 or len(scores_b) == 0:
        return 0.0
    
    # Create histograms
    min_score = min(scores_a.min(), scores_b.min())
    max_score = max(scores_a.max(), scores_b.max())
    
    hist_a, _ = np.histogram(scores_a, bins=bins, range=(min_score, max_score), density=True)
    hist_b, _ = np.histogram(scores_b, bins=bins, range=(min_score, max_score), density=True)
    
    # Calculate overlap (intersection over union)
    intersection = np.minimum(hist_a, hist_b).sum()
    union = np.maximum(hist_a, hist_b).sum()
    
    overlap = intersection / union if union > 0 else 0.0
    return float(overlap)

