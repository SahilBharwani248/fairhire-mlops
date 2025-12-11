"""
Prediction utilities for generating predictions on all candidates.
Creates shortlist base with rankings.
"""
import pandas as pd
import logging
from typing import Optional

from pycaret.classification import predict_model, load_model

logger = logging.getLogger(__name__)


class PredictionGenerator:
    """Generator for predictions on full candidate dataset."""
    
    def __init__(self, model, s3_handler):
        """
        Initialize prediction generator.
        
        Args:
            model: Trained PyCaret model
            s3_handler: S3Handler instance
        """
        self.model = model
        self.s3_handler = s3_handler
    
    def load_full_dataset(self, input_prefix: str = "data/processed/") -> pd.DataFrame:
        """
        Load full processed dataset.
        
        Args:
            input_prefix: S3 prefix for input files
        
        Returns:
            Full processed DataFrame with ALL original columns
        """
        logger.info("Loading full processed dataset from S3...")
        
        # Try different possible filenames
        possible_names = [
            "Dataset_A_processed.csv",
            "full_dataset_processed.csv",
            "processed_full.csv"
        ]
        
        full_df = None
        for name in possible_names:
            try:
                full_df = self.s3_handler.read_csv(f"{input_prefix}{name}")
                logger.info(f"✅ Loaded {name}, shape: {full_df.shape}")
                break
            except Exception:
                continue
        
        if full_df is None:
            raise FileNotFoundError(
                f"Could not find full processed dataset. Tried: {possible_names}"
            )
        
        return full_df
    
    def generate_shortlist_base(
        self,
        full_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate shortlist base with predictions and rankings.
        
        CRITICAL: This function must pass data to predict_model() in the SAME FORMAT
        that was used during training. PyCaret's predict_model() expects the exact
        columns and dtypes from setup().
        
        Args:
            full_df: Full processed dataset with ALL columns
        
        Returns:
            DataFrame with predictions, probabilities, and rankings
        """
        logger.info("Generating shortlist base...")
        
        # CRITICAL: PyCaret's predict_model() needs the target column to exist
        # even though it won't use the values for prediction
        if 'Recruiter_Decision' not in full_df.columns:
            full_df = full_df.copy()
            full_df['Recruiter_Decision'] = 'Hire'  # Dummy value (won't affect predictions)
            logger.info("Added dummy Recruiter_Decision column for PyCaret compatibility")
        
        # Verify required columns exist
        # These are the columns PyCaret expects from training
        expected_feature_cols = [
            'Skills',  # PyCaret handles this as categorical/text
            'Experience', 
            'Education_Ordinal', 
            'Certifications_Encoded', 
            'Job_Role_Encoded', 
            'Salary_Expectation', 
            'Projects_Count'
        ]
        
        missing_cols = [c for c in expected_feature_cols if c not in full_df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in dataset: {missing_cols}\n"
                f"Available columns: {full_df.columns.tolist()}"
            )
        
        logger.info(f"Dataset has all {len(expected_feature_cols)} required feature columns")
        
        # IMPORTANT: Pass the ENTIRE dataframe to predict_model()
        # PyCaret will internally select and transform the correct columns
        # based on what was configured during setup()
        logger.info("Making predictions on all candidates...")
        try:
            predictions = predict_model(self.model, data=full_df)
        except Exception as e:
            logger.error(f"Prediction failed. This usually means column mismatch.")
            logger.error(f"Columns in full_df: {full_df.columns.tolist()}")
            logger.error(f"Dtypes:\n{full_df.dtypes}")
            raise RuntimeError(f"PyCaret predict_model() failed: {e}")
        
        # Extract predictions and probabilities
        pred_labels = predictions['prediction_label'].values
        
        # Handle different possible column names for probability scores
        if 'prediction_score' in predictions.columns:
            pred_scores = predictions['prediction_score'].values
            logger.info("Using 'prediction_score' column for probabilities")
        elif 'prediction_score_Hire' in predictions.columns:
            pred_scores = predictions['prediction_score_Hire'].values
            logger.info("Using 'prediction_score_Hire' column for probabilities")
        else:
            # Fallback: use binary prediction as score
            logger.warning("No probability score column found, using binary predictions")
            pred_scores = (pred_labels == 'Hire').astype(float)
        
        # Convert labels to binary
        label_map = {'Hire': 1, 'Reject': 0, 'No Hire': 0, 'No hire': 0}
        pred_binary = pd.Series(pred_labels).map(label_map).fillna(0).astype(int).values
        
        logger.info(f"Predictions complete:")
        logger.info(f"  - Predicted Hire: {(pred_binary == 1).sum()} ({(pred_binary == 1).mean()*100:.1f}%)")
        logger.info(f"  - Predicted Reject: {(pred_binary == 0).sum()} ({(pred_binary == 0).mean()*100:.1f}%)")
        
        # Build shortlist base - use ORIGINAL full_df to preserve all columns
        shortlist_base = full_df.copy()
        
        # Add prediction columns
        shortlist_base["ModelA_Hire_Prob"] = pred_scores
        shortlist_base["ModelA_Prediction"] = pred_binary
        shortlist_base["ModelA_Pred_Label"] = pred_labels
        
        # Add ranking (1 = highest probability)
        shortlist_base["ModelA_Rank"] = shortlist_base["ModelA_Hire_Prob"].rank(
            ascending=False,  # Higher probability = better rank
            method='min'      # Ties get same rank
        ).astype(int)
        
        # Sort by rank
        shortlist_base = shortlist_base.sort_values('ModelA_Rank').reset_index(drop=True)
        
        logger.info(f"✅ Shortlist base created:")
        logger.info(f"  - Total candidates: {len(shortlist_base)}")
        logger.info(f"  - Rank range: {shortlist_base['ModelA_Rank'].min()} to {shortlist_base['ModelA_Rank'].max()}")
        logger.info(f"  - Top candidate hire probability: {shortlist_base['ModelA_Hire_Prob'].max():.3f}")
        logger.info(f"  - Bottom candidate hire probability: {shortlist_base['ModelA_Hire_Prob'].min():.3f}")
        
        return shortlist_base
    
    def save_shortlist_base(
        self,
        shortlist_base: pd.DataFrame,
        output_prefix: str = "models/modelA/"
    ):
        """
        Save shortlist base to S3.
        
        Args:
            shortlist_base: DataFrame with predictions and rankings
            output_prefix: S3 prefix for output files
        """
        logger.info("Saving shortlist base to S3...")
        s3_key = f"{output_prefix}modelA_shortlist_base.csv"
        self.s3_handler.write_csv(shortlist_base, s3_key, index=False)
        logger.info(f"✅ Shortlist base saved to S3: {s3_key}")
        logger.info(f"   Columns saved: {shortlist_base.columns.tolist()}")
