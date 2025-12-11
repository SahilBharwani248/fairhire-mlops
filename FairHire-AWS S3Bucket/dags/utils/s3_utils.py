"""
S3 Utilities for reading and writing files to/from S3 buckets.
Designed for use with AWS Managed Workflows for Apache Airflow (MWAA).
"""
import os
import boto3
import pandas as pd
import json
import pickle
from io import BytesIO
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class S3Handler:
    """Handler for S3 operations."""
    
    def __init__(self, bucket_name: str, aws_access_key_id: Optional[str] = None, 
                 aws_secret_access_key: Optional[str] = None, 
                 region_name: str = 'us-east-1'):
        """
        Initialize S3 handler.
        
        Args:
            bucket_name: S3 bucket name
            aws_access_key_id: AWS access key (optional if using IAM role)
            aws_secret_access_key: AWS secret key (optional if using IAM role)
            region_name: AWS region
        """
        self.bucket_name = bucket_name
        
        # Use credentials if provided, otherwise rely on IAM role (for MWAA)
        if aws_access_key_id and aws_secret_access_key:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name
            )
        else:
            # MWAA uses IAM role, so no credentials needed
            self.s3_client = boto3.client('s3', region_name=region_name)
    
    def read_csv(self, s3_key: str) -> pd.DataFrame:
        """Read CSV file from S3."""
        try:
            logger.info(f"Reading CSV from s3://{self.bucket_name}/{s3_key}")
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            df = pd.read_csv(BytesIO(obj['Body'].read()))
            logger.info(f"Successfully read CSV: {s3_key}, shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error reading CSV from S3: {e}")
            raise
    
    def write_csv(self, df: pd.DataFrame, s3_key: str, index: bool = False):
        """Write DataFrame to S3 as CSV."""
        try:
            logger.info(f"Writing CSV to s3://{self.bucket_name}/{s3_key}")
            buffer = BytesIO()
            df.to_csv(buffer, index=index)
            buffer.seek(0)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=buffer.getvalue()
            )
            logger.info(f"Successfully wrote CSV: {s3_key}")
        except Exception as e:
            logger.error(f"Error writing CSV to S3: {e}")
            raise
    
    def read_json(self, s3_key: str) -> Dict[str, Any]:
        """Read JSON file from S3."""
        try:
            logger.info(f"Reading JSON from s3://{self.bucket_name}/{s3_key}")
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            content = obj['Body'].read().decode('utf-8')
            data = json.loads(content)
            logger.info(f"Successfully read JSON: {s3_key}")
            return data
        except Exception as e:
            logger.error(f"Error reading JSON from S3: {e}")
            raise
    
    def write_json(self, data: Dict[str, Any], s3_key: str):
        """Write dictionary to S3 as JSON."""
        try:
            logger.info(f"Writing JSON to s3://{self.bucket_name}/{s3_key}")
            json_str = json.dumps(data, indent=2)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_str.encode('utf-8'),
                ContentType='application/json'
            )
            logger.info(f"Successfully wrote JSON: {s3_key}")
        except Exception as e:
            logger.error(f"Error writing JSON to S3: {e}")
            raise
    
    def read_pickle(self, s3_key: str) -> Any:
        """Read pickle file from S3."""
        try:
            logger.info(f"Reading pickle from s3://{self.bucket_name}/{s3_key}")
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            data = pickle.loads(obj['Body'].read())
            logger.info(f"Successfully read pickle: {s3_key}")
            return data
        except Exception as e:
            logger.error(f"Error reading pickle from S3: {e}")
            raise
    
    def write_pickle(self, obj: Any, s3_key: str):
        """Write Python object to S3 as pickle."""
        try:
            logger.info(f"Writing pickle to s3://{self.bucket_name}/{s3_key}")
            buffer = BytesIO()
            pickle.dump(obj, buffer)
            buffer.seek(0)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=buffer.getvalue()
            )
            logger.info(f"Successfully wrote pickle: {s3_key}")
        except Exception as e:
            logger.error(f"Error writing pickle to S3: {e}")
            raise
    
    def upload_file(self, local_path: str, s3_key: str):
        """Upload a local file to S3."""
        try:
            logger.info(f"Uploading file from {local_path} to s3://{self.bucket_name}/{s3_key}")
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            logger.info(f"Successfully uploaded file: {s3_key}")
        except Exception as e:
            logger.error(f"Error uploading file to S3: {e}")
            raise
    
    def download_file(self, s3_key: str, local_path: str):
        """Download a file from S3 to local path."""
        try:
            logger.info(f"Downloading file from s3://{self.bucket_name}/{s3_key} to {local_path}")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"Successfully downloaded file: {local_path}")
        except Exception as e:
            logger.error(f"Error downloading file from S3: {e}")
            raise
    
    def list_files(self, prefix: str) -> list:
        """List files in S3 with given prefix."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            return []
        except Exception as e:
            logger.error(f"Error listing files from S3: {e}")
            raise
    
    def file_exists(self, s3_key: str) -> bool:
        """Check if file exists in S3."""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except self.s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            raise

