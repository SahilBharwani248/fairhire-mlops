"""
Configuration file for Model A Pipeline.
Set these as Airflow Variables or environment variables in MWAA.
"""
import os
from typing import Optional

# S3 Configuration
S3_BUCKET: str = os.environ.get('S3_BUCKET', 'airflow-bucket-amu')
S3_INPUT_PREFIX: str = os.environ.get('S3_INPUT_PREFIX', 'data/processed/')
S3_OUTPUT_PREFIX: str = os.environ.get('S3_OUTPUT_PREFIX', 'models/modelA/')

# AWS Configuration
AWS_REGION: str = os.environ.get('AWS_REGION', 'us-east-1')

# Model Configuration
MODEL_SESSION_ID: int = int(os.environ.get('MODEL_SESSION_ID', '42'))
MODEL_TUNE_ITERATIONS: int = int(os.environ.get('MODEL_TUNE_ITERATIONS', '50'))

# Fairness Analysis Configuration
MIN_GROUP_SIZE: int = int(os.environ.get('MIN_GROUP_SIZE', '10'))
TOP_K_FOR_FAIRNESS: int = int(os.environ.get('TOP_K_FOR_FAIRNESS', '50'))

# Required input files in S3 (relative to S3_INPUT_PREFIX)
REQUIRED_INPUT_FILES = [
    'X_train.csv',
    'X_test.csv',
    'y_train.csv',
    'y_test.csv',
]

# Optional input files
OPTIONAL_INPUT_FILES = [
    'demographics_train.csv',
    'demographics_test.csv',
    'Dataset_A_processed.csv',  # For shortlist generation
    'ai_score_train.csv',
    'ai_score_test.csv',
]

