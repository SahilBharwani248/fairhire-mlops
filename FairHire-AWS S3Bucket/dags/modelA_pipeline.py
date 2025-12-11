"""
Model A Training Pipeline DAG for AWS Managed Workflows for Apache Airflow (MWAA).

This DAG trains Model A using PyCaret to predict Recruiter Decision (Hire/Reject)
based on resume features. All inputs are read from S3 and outputs are written back to S3.

Tasks:
1. Load data from S3
2. Train Model A using PyCaret
3. Evaluate model and save metrics
4. Generate visualizations
5. Perform fairness analysis
6. Generate predictions on full dataset
7. Save model and all outputs to S3
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import logging

logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'max_active_runs': 1,
}

# DAG configuration
dag = DAG(
    'modelA_training_pipeline',
    default_args=default_args,
    description='Train Model A (Recruiter Decision Classifier) using PyCaret',
    start_date=datetime.now() - timedelta(days=1),
    catchup=False,
    tags=['machine-learning', 'recruitment', 'fairness', 'pycaret'],
)

def get_config(context=None):
    """Lazy load configuration to avoid imports at DAG parse time.
    
    Args:
        context: Task context (optional) - if provided, adds run_id to output prefix
    """
    try:
        from config import (
            S3_BUCKET, S3_INPUT_PREFIX, S3_OUTPUT_PREFIX, 
            AWS_REGION, MODEL_SESSION_ID, MODEL_TUNE_ITERATIONS,
            MIN_GROUP_SIZE, TOP_K_FOR_FAIRNESS
        )
        base_output_prefix = S3_OUTPUT_PREFIX
        bucket_name = S3_BUCKET
        input_prefix = S3_INPUT_PREFIX
        region = AWS_REGION
        session_id = MODEL_SESSION_ID
        tune_iter = MODEL_TUNE_ITERATIONS
        min_group = MIN_GROUP_SIZE
        top_k = TOP_K_FOR_FAIRNESS
    except ImportError:
        # Fallback to environment variables
        bucket_name = os.environ.get('S3_BUCKET', 'your-mwaa-bucket')
        input_prefix = os.environ.get('S3_INPUT_PREFIX', 'data/processed/')
        base_output_prefix = os.environ.get('S3_OUTPUT_PREFIX', 'models/modelA/')
        region = os.environ.get('AWS_REGION', 'us-east-1')
        session_id = int(os.environ.get('MODEL_SESSION_ID', '42'))
        tune_iter = int(os.environ.get('MODEL_TUNE_ITERATIONS', '50'))
        min_group = int(os.environ.get('MIN_GROUP_SIZE', '10'))
        top_k = int(os.environ.get('TOP_K_FOR_FAIRNESS', '50'))
    
    # Append run_id to output prefix if context is provided
    if context is not None:
        # Try to get run_id from different possible locations in context
        run_id = None
        if 'dag_run' in context and hasattr(context['dag_run'], 'run_id'):
            run_id = context['dag_run'].run_id
        elif 'run_id' in context:
            run_id = context['run_id']
        elif 'ti' in context and hasattr(context['ti'], 'dag_run') and context['ti'].dag_run:
            run_id = context['ti'].dag_run.run_id
        
        if run_id:
            # Ensure output prefix ends with / and add run_id
            output_prefix = base_output_prefix.rstrip('/') + f'/{run_id}/'
            logger.info(f"Using run_id '{run_id}' in output prefix: {output_prefix}")
        else:
            output_prefix = base_output_prefix
            logger.warning("Could not extract run_id from context, using base output prefix")
    else:
        output_prefix = base_output_prefix
    
    return {
        'S3_BUCKET': bucket_name,
        'S3_INPUT_PREFIX': input_prefix,
        'S3_OUTPUT_PREFIX': output_prefix,
        'AWS_REGION': region,
        'MODEL_SESSION_ID': session_id,
        'MODEL_TUNE_ITERATIONS': tune_iter,
        'MIN_GROUP_SIZE': min_group,
        'TOP_K_FOR_FAIRNESS': top_k,
    }


def load_data_task(**context):
    """Task 1: Load data from S3."""
    # Import heavy libraries only when task runs
    import pandas as pd
    from utils.s3_utils import S3Handler
    from utils.model_training import ModelATrainer
    
    logger.info("Starting data loading task...")
    
    config = get_config(context)
    s3_handler = S3Handler(bucket_name=config['S3_BUCKET'], region_name=config['AWS_REGION'])
    trainer = ModelATrainer(session_id=config['MODEL_SESSION_ID'])
    
    # Load training and test data
    X_train, X_test, y_train, y_test = trainer.load_data(
        s3_handler=s3_handler,
        input_prefix=config['S3_INPUT_PREFIX']
    )
    
    # Load demographics (optional)
    demographics_train, demographics_test = trainer.load_demographics(
        s3_handler=s3_handler,
        input_prefix=config['S3_INPUT_PREFIX']
    )
    
    # Prepare data
    train_data, test_data = trainer.prepare_data(
        X_train, X_test, y_train, y_test
    )
    
    # Store in XCom for downstream tasks
    context['ti'].xcom_push(key='train_data_path', value='/tmp/train_data.parquet')
    context['ti'].xcom_push(key='test_data_path', value='/tmp/test_data.parquet')
    
    # Save to temporary files (since we can't pass DataFrames directly via XCom)
    train_data.to_parquet('/tmp/train_data.parquet', index=False)
    test_data.to_parquet('/tmp/test_data.parquet', index=False)
    
    if demographics_test is not None:
        demographics_test.to_parquet('/tmp/demographics_test.parquet', index=False)
        context['ti'].xcom_push(key='demographics_test_path', value='/tmp/demographics_test.parquet')
    
    logger.info("✓ Data loading completed")


def train_model_task(**context):
    """Task 2: Train Model A using PyCaret."""
    # Import heavy libraries only when task runs
    import pandas as pd
    from utils.s3_utils import S3Handler
    from utils.model_training import ModelATrainer
    from pycaret.classification import save_model
    
    logger.info("Starting model training task...")
    
    config = get_config(context)
    s3_handler = S3Handler(bucket_name=config['S3_BUCKET'], region_name=config['AWS_REGION'])
    trainer = ModelATrainer(session_id=config['MODEL_SESSION_ID'])
    
    # Load data from previous task
    train_data = pd.read_parquet('/tmp/train_data.parquet')
    test_data = pd.read_parquet('/tmp/test_data.parquet')
    
    # Setup PyCaret
    trainer.setup_pycaret(train_data, test_data)
    
    # Train model
    # FIX APPLIED: Changed tune=True to tune=False to match notebook logic
    # The notebook found that the default model performed better than the tuned one.
    trainer.train_model(
        include_models=['rf', 'lightgbm', 'et', 'ada', 'dt', 'lr', 'ridge', 'nb', 'knn'],
        sort_by='AUC',
        n_select=5,
        tune=False,  # <--- CHANGED FROM True TO False
        tune_iterations=config['MODEL_TUNE_ITERATIONS']
    )
    
    # Evaluate model
    metrics = trainer.evaluate_model(test_data)
    
    # Save model and metrics to S3
    trainer.save_model(s3_handler=s3_handler, output_prefix=config['S3_OUTPUT_PREFIX'])
    trainer.save_metrics(s3_handler=s3_handler, output_prefix=config['S3_OUTPUT_PREFIX'])
    
    # Save model locally for downstream tasks (PyCaret models need special handling)
    # Save to a directory that PyCaret can load from
    model_dir = '/tmp/modelA'
    os.makedirs(model_dir, exist_ok=True)
    save_model(trainer.model, f'{model_dir}/modelA_final', verbose=False)
    
    context['ti'].xcom_push(key='model_dir', value=model_dir)
    context['ti'].xcom_push(key='test_data_path', value='/tmp/test_data.parquet')
    
    logger.info("✓ Model training completed")


def generate_visualizations_task(**context):
    """Task 3: Generate model visualization plots."""
    # Import heavy libraries only when task runs
    import glob
    from utils.s3_utils import S3Handler
    from utils.model_training import ModelATrainer
    from pycaret.classification import load_model
    
    logger.info("Starting visualization generation task...")
    
    config = get_config(context)
    s3_handler = S3Handler(bucket_name=config['S3_BUCKET'], region_name=config['AWS_REGION'])
    
    # Load model (PyCaret models need to be loaded using load_model)
    model_dir = context['ti'].xcom_pull(key='model_dir', task_ids='train_model_a')
    if model_dir is None:
        model_dir = '/tmp/modelA'
    
    model = load_model(f'{model_dir}/modelA_final', verbose=False)
    
    # Create trainer instance to use plot functions
    trainer = ModelATrainer(session_id=config['MODEL_SESSION_ID'])
    trainer.model = model
    
    # Generate plots
    plots_dir = "/tmp/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Change to plots directory for PyCaret
    original_cwd = os.getcwd()
    os.chdir(plots_dir)
    
    try:
        trainer.generate_plots(output_dir=plots_dir)
        
        # Upload plots to S3
        for plot_file in glob.glob(f"{plots_dir}/*.png"):
            filename = os.path.basename(plot_file)
            s3_key = f"{config['S3_OUTPUT_PREFIX']}plots/{filename}"
            s3_handler.upload_file(plot_file, s3_key)
            logger.info(f"✓ Uploaded plot: {filename}")
    finally:
        os.chdir(original_cwd)
    
    logger.info("✓ Visualization generation completed")


def fairness_analysis_task(**context):
    """Task 4: Perform fairness analysis."""
    # Import heavy libraries only when task runs
    import pandas as pd
    from utils.s3_utils import S3Handler
    from utils.fairness_analysis import FairnessAnalyzer
    from pycaret.classification import load_model
    
    logger.info("Starting fairness analysis task...")
    
    config = get_config(context)
    s3_handler = S3Handler(bucket_name=config['S3_BUCKET'], region_name=config['AWS_REGION'])
    
    # Load model and test data
    model_dir = context['ti'].xcom_pull(key='model_dir', task_ids='train_model_a')
    if model_dir is None:
        model_dir = '/tmp/modelA'
    
    model = load_model(f'{model_dir}/modelA_final', verbose=False)
    
    test_data = pd.read_parquet('/tmp/test_data.parquet')
    
    # Load demographics if available
    demographics_test = None
    try:
        demographics_test = pd.read_parquet('/tmp/demographics_test.parquet')
    except FileNotFoundError:
        logger.warning("Demographics data not found, skipping fairness analysis")
        return
    
    # Perform fairness analysis
    analyzer = FairnessAnalyzer(model=model, s3_handler=s3_handler)
    
    fair_df = analyzer.prepare_fairness_dataframe(
        test_data=test_data,
        demographics_test=demographics_test
    )
    
    fairness_summary = analyzer.analyze_fairness(
        fair_df, 
        min_group_size=config['MIN_GROUP_SIZE'],
        top_k=config['TOP_K_FOR_FAIRNESS']
    )
    if not fairness_summary:
        logger.warning("No fairness metrics to save")
        return
    # Save fairness metrics
    analyzer.save_fairness_metrics(s3_handler=s3_handler, output_prefix=config['S3_OUTPUT_PREFIX'])
    
    logger.info("✓ Fairness analysis completed")


def generate_predictions_task(**context):
    """Task 5: Generate predictions on full dataset for shortlist."""
    # Import heavy libraries only when task runs
    from utils.s3_utils import S3Handler
    from utils.prediction_utils import PredictionGenerator
    from pycaret.classification import load_model
    
    logger.info("Starting prediction generation task...")
    
    config = get_config(context)
    s3_handler = S3Handler(bucket_name=config['S3_BUCKET'], region_name=config['AWS_REGION'])
    
    # Load model
    model_dir = context['ti'].xcom_pull(key='model_dir', task_ids='train_model_a')
    if model_dir is None:
        model_dir = '/tmp/modelA'
    
    model = load_model(f'{model_dir}/modelA_final', verbose=False)
    
    # Generate predictions
    generator = PredictionGenerator(model=model, s3_handler=s3_handler)
    
    # Load full dataset
    full_df = generator.load_full_dataset(input_prefix=config['S3_INPUT_PREFIX'])
    
    # Generate shortlist base
    shortlist_base = generator.generate_shortlist_base(full_df)
    
    # Save to S3
    generator.save_shortlist_base(
        shortlist_base=shortlist_base,
        output_prefix=config['S3_OUTPUT_PREFIX']
    )
    
    logger.info("✓ Prediction generation completed")


# Define tasks
load_data = PythonOperator(
    task_id='load_data_from_s3',
    python_callable=load_data_task,
    dag=dag,
)

train_model = PythonOperator(
    task_id='train_model_a',
    python_callable=train_model_task,
    dag=dag,
)

generate_visualizations = PythonOperator(
    task_id='generate_model_visualizations',
    python_callable=generate_visualizations_task,
    dag=dag,
)

fairness_analysis = PythonOperator(
    task_id='perform_fairness_analysis',
    python_callable=fairness_analysis_task,
    dag=dag,
)

generate_predictions = PythonOperator(
    task_id='generate_full_predictions',
    python_callable=generate_predictions_task,
    dag=dag,
)

# Define task dependencies
load_data >> train_model
train_model >> [generate_visualizations, fairness_analysis, generate_predictions]
