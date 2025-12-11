"""
Model training utilities for Model A using PyCaret.
Handles model setup, training, evaluation, and saving.
"""
import pandas as pd
import numpy as np
import json
import os
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

from pycaret.classification import (
    setup, compare_models, tune_model, finalize_model,
    evaluate_model, predict_model, save_model, load_model, plot_model
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ModelATrainer:
    """Trainer for Model A: Recruiter Decision Classification."""
    
    def __init__(self, session_id: int = 42):
        """
        Initialize Model A trainer.
        
        Args:
            session_id: Random seed for reproducibility
        """
        self.session_id = session_id
        self.model = None
        self.metrics = None
    
    def load_data(
        self, 
        s3_handler,
        input_prefix: str = "data/processed/"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load training and test data from S3.
        
        Args:
            s3_handler: S3Handler instance
            input_prefix: S3 prefix for input files
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) DataFrames
        """
        logger.info("Loading data from S3...")
        
        # Load training and test features
        X_train = s3_handler.read_csv(f"{input_prefix}X_train.csv")
        X_test = s3_handler.read_csv(f"{input_prefix}X_test.csv")
        
        # Load target variables
        y_train_df = s3_handler.read_csv(f"{input_prefix}y_train.csv")
        y_test_df = s3_handler.read_csv(f"{input_prefix}y_test.csv")
        
        y_train = y_train_df['Recruiter_Decision'].squeeze()
        y_test = y_test_df['Recruiter_Decision'].squeeze()
        
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def load_demographics(
        self,
        s3_handler,
        input_prefix: str = "data/processed/"
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Load demographics data from S3 (optional).
        
        Args:
            s3_handler: S3Handler instance
            input_prefix: S3 prefix for input files
        
        Returns:
            Tuple of (demographics_train, demographics_test) DataFrames or (None, None)
        """
        try:
            demographics_train = s3_handler.read_csv(f"{input_prefix}demographics_train.csv")
            demographics_test = s3_handler.read_csv(f"{input_prefix}demographics_test.csv")
            logger.info("✓ Loaded demographics files")
            return demographics_train, demographics_test
        except Exception as e:
            logger.warning(f"Demographics files not found (optional): {e}")
            return None, None
    
    def prepare_data(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for PyCaret by combining features and targets.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
        
        Returns:
            Tuple of (train_data, test_data) DataFrames
        """
        train_data = X_train.copy()
        train_data['Recruiter_Decision'] = y_train.values
        
        test_data = X_test.copy()
        test_data['Recruiter_Decision'] = y_test.values
        
        logger.info(f"Prepared training data: {train_data.shape}")
        logger.info(f"Prepared test data: {test_data.shape}")
        
        return train_data, test_data
    
    def setup_pycaret(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame
    ):
        """
        Setup PyCaret classification environment.
        
        Args:
            train_data: Training DataFrame with target column
            test_data: Test DataFrame with target column
        """
        logger.info("Setting up PyCaret classification environment...")
        
        clf = setup(
            data=train_data,
            target='Recruiter_Decision',
            test_data=test_data,
            session_id=self.session_id,
            normalize=True,
            transformation=True,
            feature_selection=True,
            remove_multicollinearity=True,
            multicollinearity_threshold=0.95,
            remove_outliers=False,
            fix_imbalance=False,
            verbose=False,  # Reduce verbosity for Airflow logs
            index=False
        )
        
        logger.info("✓ PyCaret setup completed")
    
    def train_model(
        self,
        include_models: Optional[list] = None,
        sort_by: str = 'AUC',
        n_select: int = 5,
        tune: bool = True,
        tune_iterations: int = 50
    ):
        """
        Train and select best model.
        
        Args:
            include_models: List of model IDs to include (None = all)
            sort_by: Metric to sort by
            n_select: Number of top models to select
            tune: Whether to tune the best model
            tune_iterations: Number of tuning iterations
        """
        logger.info("Comparing models...")
        
        if include_models is None:
            include_models = ['rf', 'lightgbm', 'et', 'ada', 'dt', 'lr', 'ridge', 'nb', 'knn']
        
        best_models = compare_models(
            include=include_models,
            sort=sort_by,
            n_select=n_select,
            verbose=False
        )
        
        # Get the best model
        best_model = best_models[0] if isinstance(best_models, list) else best_models
        logger.info(f"Best model: {type(best_model).__name__}")
        
        # Tune if requested
        if tune:
            logger.info("Tuning best model...")
            tuned_model = tune_model(
                best_model,
                optimize=sort_by,
                n_iter=tune_iterations,
                verbose=False
            )
            self.model = tuned_model
            logger.info("✓ Model tuning completed")
        else:
            self.model = best_model
        
        logger.info(f"Final model: {type(self.model).__name__}")
    
    def evaluate_model(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate model on test data and calculate metrics.
        
        Args:
            test_data: Test DataFrame with target column
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model on test set...")
        
        # Make predictions
        predictions = predict_model(self.model, data=test_data)
        
        # Extract predictions and probabilities
        y_true = test_data['Recruiter_Decision'].values
        y_pred = predictions['prediction_label'].values
        
        if 'prediction_score' in predictions.columns:
            y_pred_proba = predictions['prediction_score'].values
        else:
            y_pred_proba = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label='Hire', zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label='Hire', zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label='Hire', zero_division=0)
        
        # AUC requires probability scores
        if y_pred_proba is not None:
            y_true_binary = (y_true == 'Hire').astype(int)
            try:
                auc = roc_auc_score(y_true_binary, y_pred_proba)
            except ValueError:
                auc = None
        else:
            auc = None
        
        # Store metrics
        self.metrics = {
            'model_type': type(self.model).__name__,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc) if auc is not None else None,
            'test_samples': int(len(y_true)),
            'target_variable': 'Recruiter_Decision',
            'task_type': 'classification',
            'positive_class': 'Hire',
            'class_distribution': pd.Series(y_true).value_counts().to_dict()
        }
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        if auc is not None:
            logger.info(f"AUC-ROC: {auc:.4f}")
        
        return self.metrics
    
    def generate_plots(self, output_dir: str = "/tmp/plots"):
        """
        Generate model visualization plots.
        
        Args:
            output_dir: Local directory to save plots
        """
        logger.info("Generating model plots...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        plots = [
            ('auc', 'AUC'),
            ('confusion_matrix', 'Confusion Matrix'),
            ('class_report', 'Classification Report'),
            ('pr', 'Precision-Recall Curve'),
            ('feature', 'Feature Importance')
        ]
        
        for plot_type, plot_name in plots:
            try:
                plot_model(
                    self.model,
                    plot=plot_type,
                    save=True,
                    verbose=False
                )
                logger.info(f"✓ Generated {plot_name}")
            except Exception as e:
                logger.warning(f"Could not generate {plot_name}: {e}")
    
    def save_model(self, s3_handler, output_prefix: str = "models/modelA/"):
        """
        Save model to S3.
        
        Args:
            s3_handler: S3Handler instance
            output_prefix: S3 prefix for output files
        """
        logger.info("Saving model to S3...")
        
        # Save to temporary local directory first
        local_dir = "/tmp/modelA"
        os.makedirs(local_dir, exist_ok=True)
        local_path = f"{local_dir}/modelA_final"
        
        # PyCaret save_model saves to disk
        save_model(self.model, local_path, verbose=False)
        
        # Upload model files to S3
        # PyCaret saves multiple files, so we need to upload all
        for file_path in Path(local_dir).glob("modelA_final*"):
            s3_key = f"{output_prefix}{file_path.name}"
            s3_handler.upload_file(str(file_path), s3_key)
            logger.info(f"✓ Uploaded {file_path.name} to S3")
        
        logger.info("✓ Model saved to S3")
    
    def save_metrics(self, s3_handler, output_prefix: str = "models/modelA/"):
        """
        Save metrics to S3.
        
        Args:
            s3_handler: S3Handler instance
            output_prefix: S3 prefix for output files
        """
        if self.metrics is None:
            logger.warning("No metrics to save")
            return
        
        logger.info("Saving metrics to S3...")
        s3_key = f"{output_prefix}modelA_metrics.json"
        s3_handler.write_json(self.metrics, s3_key)
        logger.info("✓ Metrics saved to S3")

