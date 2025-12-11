"""
Fairness analysis utilities for Model A.
Performs demographic parity, equal opportunity, and other fairness metrics.
ENHANCED VERSION: Saves per-group details for dashboard display
"""
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, Any, Optional

from pycaret.classification import predict_model, load_model
from utils.fairness_utils import (
    demographic_parity_diff,
    selection_rate_parity_topk,
    rank_ordering_bias,
    equal_opportunity_diff,
    score_distribution_overlap,
)

logger = logging.getLogger(__name__)


class FairnessAnalyzer:
    """Analyzer for fairness metrics on Model A."""
    
    def __init__(self, model, s3_handler):
        """
        Initialize fairness analyzer.
        
        Args:
            model: Trained PyCaret model
            s3_handler: S3Handler instance
        """
        self.model = model
        self.s3_handler = s3_handler
        self.fairness_summary = {}
    
    def prepare_fairness_dataframe(
        self,
        test_data: pd.DataFrame,
        demographics_test: pd.DataFrame,
        predictions: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Prepare DataFrame with predictions and demographics for fairness analysis.
        
        Args:
            test_data: Test data with features and target
            demographics_test: Demographics data for test set
            predictions: Pre-computed predictions (optional)
        
        Returns:
            DataFrame with y_true, y_pred, y_proba, and demographics
        """
        logger.info("Preparing fairness DataFrame...")
        
        # Get predictions if not provided
        if predictions is None:
            predictions = predict_model(self.model, data=test_data)
        
        # Map labels to binary
        label_map = {"Hire": 1, "Reject": 0, "No hire": 0, "No Hire": 0}
        y_true = test_data['Recruiter_Decision'].map(label_map).fillna(0).astype(int)
        y_pred_labels = predictions['prediction_label'].map(label_map).fillna(0).astype(int)
        y_pred = y_pred_labels.values
        
        # Extract probability
        if 'prediction_score_Hire' in predictions.columns:
            y_proba = predictions['prediction_score_Hire'].values
        elif 'prediction_score' in predictions.columns:
            y_proba = predictions['prediction_score'].values
        else:
            # Fallback: use raw scores
            predictions_raw = predict_model(self.model, data=test_data, raw_score=True)
            y_proba = predictions_raw['prediction_score'].values
        
        # Create fairness DataFrame
        fair_df = pd.DataFrame({
            "y_true": y_true.values,
            "y_pred": y_pred,
            "y_proba": y_proba,
        })
        
        # Clean and attach demographics
        demo = demographics_test.copy()
        
        # Age grouping
        if "Age" in demo.columns:
            demo["Age_Group"] = pd.cut(
                demo["Age"],
                bins=[17, 29, 39, 120],
                labels=["18-29", "30-39", "40+"],
                right=True,
                include_lowest=True,
            )
        
        # Race grouping (collapse rare categories)
        if "Race" in demo.columns:
            race_counts = demo["Race"].value_counts()
            rare_races = race_counts[race_counts < 10].index
            demo["Race_Grouped"] = demo["Race"].where(
                ~demo["Race"].isin(rare_races),
                "Other / Minority"
            )
        
        # Attach demographics
        if "Gender" in demo.columns:
            fair_df["Gender"] = demo["Gender"].values
        if "Race_Grouped" in demo.columns:
            fair_df["Race"] = demo["Race_Grouped"].values
        elif "Race" in demo.columns:
            fair_df["Race"] = demo["Race"].values
        if "Age_Group" in demo.columns:
            fair_df["Age_Group"] = demo["Age_Group"].values
        if "Disability_Status" in demo.columns:
            fair_df["Disability_Status"] = demo["Disability_Status"].values
        
        logger.info(f"Fairness DataFrame shape: {fair_df.shape}")
        return fair_df
    
    def filter_small_groups(
        self, 
        df: pd.DataFrame, 
        group_col: str, 
        min_size: int = 10
    ) -> tuple:
        """
        Filter out groups with fewer than min_size samples.
        
        Args:
            df: DataFrame to filter
            group_col: Column name for groups
            min_size: Minimum group size
        
        Returns:
            Tuple of (filtered_df, valid_groups)
        """
        counts = df[group_col].value_counts()
        valid = counts[counts >= min_size].index
        return df[df[group_col].isin(valid)], valid
    
    def analyze_fairness(
        self,
        fair_df: pd.DataFrame,
        min_group_size: int = 10,
        top_k: int = 50
    ) -> Dict[str, Any]:
        """
        Perform comprehensive fairness analysis across all demographic attributes.
        
        Args:
            fair_df: DataFrame with predictions and demographics
            min_group_size: Minimum group size to include in analysis
            top_k: Number of top candidates to analyze for selection rate parity
        
        Returns:
            Dictionary of fairness metrics by attribute
        """
        logger.info("Performing fairness analysis...")
        
        demographic_attrs = ["Gender", "Race", "Age_Group", "Disability_Status"]
        demographic_attrs = [a for a in demographic_attrs if a in fair_df.columns]
        
        fairness_summary = {}
        
        for attr in demographic_attrs:
            logger.info(f"Analyzing fairness by {attr}...")
            
            df_attr = fair_df.copy()
            df_attr["group"] = df_attr[attr]
            
            # Filter small groups
            df_attr, kept_groups = self.filter_small_groups(df_attr, "group", min_group_size)
            if len(kept_groups) < 2:
                logger.warning(f"Not enough data for {attr} after filtering. Skipping.")
                continue
            
            # 1. Demographic Parity
            dp = demographic_parity_diff(
                df_attr,
                group_col="group",
                score_col="y_pred",
                threshold=None,
            )
            
            # 2. Top-K Selection Rate Parity
            actual_top_k = min(top_k, len(df_attr))
            srp = selection_rate_parity_topk(
                df_attr,
                group_col="group",
                score_col="y_proba",
                k=actual_top_k,
            )
            
            # 3. Equal Opportunity
            eo = equal_opportunity_diff(
                df_attr,
                group_col="group",
                y_true_col="y_true",
                score_col="y_proba",
                positive_label=1,
                threshold=0.5,
            )
            
            # 4. Rank Ordering Bias
            rob = rank_ordering_bias(
                df_attr,
                group_col="group",
                score_col="y_proba",
            )
            
            # 5. Score Distribution Overlap (if exactly 2 groups)
            kept_groups_list = list(kept_groups)
            if len(kept_groups_list) == 2:
                sdo = score_distribution_overlap(
                    df_attr,
                    group_a=kept_groups_list[0],
                    group_b=kept_groups_list[1],
                    group_col="group",
                    score_col="y_proba",
                    bins=20,
                )
            else:
                sdo = None
            
            # ✨ NEW: Build per-group details dictionary
            per_group_details = {}
            for group_name in kept_groups:
                group_data = df_attr[df_attr["group"] == group_name]
                
                per_group_details[str(group_name)] = {
                    "count": int(len(group_data)),
                    "hire_rate": float(group_data["y_pred"].mean()),
                    "avg_hire_probability": float(group_data["y_proba"].mean()),
                    "top_k_rate": float(srp["per_group"].get(group_name, 0)),
                    "avg_rank": float(rob["per_group_avg_rank"].get(group_name, 0)),
                }
                
                # Add TPR if available
                if group_name in eo["per_group_tpr"]:
                    tpr_val = eo["per_group_tpr"][group_name]
                    per_group_details[str(group_name)]["tpr"] = float(tpr_val) if not np.isnan(tpr_val) else None
            
            fairness_summary[attr] = {
                "demographic_parity_max_gap": float(dp["max_gap"]),
                "topk_min_over_max": float(srp["min_over_max"]),
                "equal_opportunity_max_gap": float(eo["max_tpr_gap"]) if not np.isnan(eo["max_tpr_gap"]) else None,
                "rank_ordering_max_gap": float(rob["max_rank_gap"]),
                "score_distribution_overlap": float(sdo) if sdo is not None else None,
                "per_group_details": per_group_details,  # ✨ NEW
                "top_k_used": actual_top_k,  # ✨ NEW: Document what top-K was used
            }
            
            logger.info(f"✅ Completed fairness analysis for {attr}")
        
        self.fairness_summary = fairness_summary
        return fairness_summary
    
    def save_fairness_metrics(self, s3_handler, output_prefix: str = "models/modelA/"):
        """
        Save fairness metrics to S3.
        
        Args:
            s3_handler: S3Handler instance
            output_prefix: S3 prefix for output files
        """
        if not self.fairness_summary:
            logger.warning("No fairness metrics to save")
            return
        
        logger.info("Saving fairness metrics to S3...")
        
        # Save standalone fairness metrics
        s3_key = f"{output_prefix}modelA_fairness_metrics.json"
        s3_handler.write_json(self.fairness_summary, s3_key)
        logger.info(f"✅ Saved fairness metrics with per-group details: {s3_key}")
        
        # Update main metrics file if it exists
        try:
            metrics_key = f"{output_prefix}modelA_metrics.json"
            if s3_handler.file_exists(metrics_key):
                model_metrics = s3_handler.read_json(metrics_key)
                model_metrics["fairness"] = self.fairness_summary
                s3_handler.write_json(model_metrics, metrics_key)
                logger.info("✅ Updated main metrics file with fairness data")
        except Exception as e:
            logger.warning(f"Could not update main metrics file: {e}")
        
        logger.info("✅ Fairness metrics saved to S3")
