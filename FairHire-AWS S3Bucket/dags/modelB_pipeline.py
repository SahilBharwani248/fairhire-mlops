"""
Model B Training Pipeline DAG for AWS Managed Workflows for Apache Airflow (MWAA).

This DAG trains Model B using PyCaret to predict Recruiter Decision (Hire/Reject)
based on resume features. All inputs are read from S3 and outputs are written back to S3.

Tasks:
1. Load data from S3
2. Train Model B using PyCaret
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
    'modelB_training_pipeline',
    default_args=default_args,
    description='Train Model B (Recruiter Decision Classifier) using PyCaret',
    start_date=datetime.now() - timedelta(days=1),
    catchup=False,
    tags=['machine-learning', 'recruitment', 'fairness', 'pycaret', 'modelB'],
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
        # Model B specific output prefix
        base_output_prefix = S3_OUTPUT_PREFIX.replace('modelA', 'modelB') if 'modelA' in S3_OUTPUT_PREFIX else 'models/modelB/'
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
        base_output_prefix = os.environ.get('S3_OUTPUT_PREFIX_MODELB', 'models/modelB/')
        region = os.environ.get('AWS_REGION', 'us-east-1')
        session_id = int(os.environ.get('MODEL_SESSION_ID', '42'))
        tune_iter = int(os.environ.get('MODEL_TUNE_ITERATIONS', '50'))
        min_group = int(os.environ.get('MIN_GROUP_SIZE', '10'))
        top_k = int(os.environ.get('TOP_K_FOR_FAIRNESS', '50'))
    
    # Append run_id to output prefix if context is provided
    if context is not None:
        # Try to get run_id from different possible locations in context
        run_id = None
        if 'dag_run' in context:
            dag_run = context['dag_run']
            if hasattr(dag_run, 'run_id'):
                run_id = dag_run.run_id
            elif isinstance(dag_run, dict) and 'run_id' in dag_run:
                run_id = dag_run['run_id']
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
    
    logger.info("Starting data loading task for Model B...")
    
    config = get_config(context)
    s3_handler = S3Handler(bucket_name=config['S3_BUCKET'], region_name=config['AWS_REGION'])
    trainer = ModelATrainer(session_id=config['MODEL_SESSION_ID'])
    
    # Load Model B specific training and test data
    input_prefix = config['S3_INPUT_PREFIX']
    X_train = s3_handler.read_csv(f"{input_prefix}X_train_modelB.csv")
    X_test = s3_handler.read_csv(f"{input_prefix}X_test_modelB.csv")
    
    y_train_df = s3_handler.read_csv(f"{input_prefix}y_train_modelB.csv")
    y_test_df = s3_handler.read_csv(f"{input_prefix}y_test_modelB.csv")
    y_train = y_train_df['Recruiter_Decision'].squeeze()
    y_test = y_test_df['Recruiter_Decision'].squeeze()
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    
    # Load demographics (optional)
    demographics_test = None
    try:
        demographics_test = s3_handler.read_csv(f"{input_prefix}demographics_test_modelB.csv")
        logger.info("✓ Loaded demographics files")
    except Exception as e:
        logger.warning(f"Demographics files not found (optional): {e}")
    
    # Prepare data
    train_data = X_train.copy()
    train_data['Recruiter_Decision'] = y_train.values
    
    test_data = X_test.copy()
    test_data['Recruiter_Decision'] = y_test.values
    
    # Save to temporary files
    train_data.to_parquet('/tmp/train_data_modelB.parquet', index=False)
    test_data.to_parquet('/tmp/test_data_modelB.parquet', index=False)
    
    if demographics_test is not None:
        demographics_test.to_parquet('/tmp/demographics_test_modelB.parquet', index=False)
        context['ti'].xcom_push(key='demographics_test_path', value='/tmp/demographics_test_modelB.parquet')
    
    context['ti'].xcom_push(key='train_data_path', value='/tmp/train_data_modelB.parquet')
    context['ti'].xcom_push(key='test_data_path', value='/tmp/test_data_modelB.parquet')
    
    logger.info("✓ Data loading completed for Model B")


def train_model_task(**context):
    """Task 2: Train Model B using PyCaret."""
    # Import heavy libraries only when task runs
    import pandas as pd
    from utils.s3_utils import S3Handler
    from utils.model_training import ModelATrainer
    from pycaret.classification import save_model
    
    logger.info("Starting model training task for Model B...")
    
    config = get_config(context)
    s3_handler = S3Handler(bucket_name=config['S3_BUCKET'], region_name=config['AWS_REGION'])
    trainer = ModelATrainer(session_id=config['MODEL_SESSION_ID'])
    
    # Load data from previous task
    train_data = pd.read_parquet('/tmp/train_data_modelB.parquet')
    test_data = pd.read_parquet('/tmp/test_data_modelB.parquet')
    
    # Setup PyCaret
    trainer.setup_pycaret(train_data, test_data)
    
    # Train model
    # FIX APPLIED: Changed tune=True to tune=False to match notebook logic
    # In modelB_final.ipynb, the tuned model performed worse, so the original best model was kept.
    trainer.train_model(
        include_models=['rf', 'lightgbm', 'et', 'ada', 'dt', 'lr', 'ridge', 'nb', 'knn'],
        sort_by='AUC',
        n_select=5,
        tune=False,  # <--- CHANGED FROM True TO False
        tune_iterations=config['MODEL_TUNE_ITERATIONS']
    )
    
    # Evaluate model
    metrics = trainer.evaluate_model(test_data)
    
    # Save model and metrics to S3 (with modelB naming)
    logger.info("Saving Model B to S3...")
    model_dir_local = '/tmp/modelB'
    os.makedirs(model_dir_local, exist_ok=True)
    save_model(trainer.model, f'{model_dir_local}/modelB_final', verbose=False)
    
    # Upload model files to S3
    import glob
    for file_path in glob.glob(f"{model_dir_local}/modelB_final*"):
        s3_key = f"{config['S3_OUTPUT_PREFIX']}{os.path.basename(file_path)}"
        s3_handler.upload_file(str(file_path), s3_key)
        logger.info(f"✓ Uploaded {os.path.basename(file_path)} to S3")
    
    # Save metrics to S3
    if trainer.metrics:
        metrics_key = f"{config['S3_OUTPUT_PREFIX']}modelB_metrics.json"
        s3_handler.write_json(trainer.metrics, metrics_key)
        logger.info("✓ Metrics saved to S3")
    
    # Save model locally for downstream tasks (PyCaret models need special handling)
    model_dir = '/tmp/modelB'
    os.makedirs(model_dir, exist_ok=True)
    save_model(trainer.model, f'{model_dir}/modelB_final', verbose=False)
    
    context['ti'].xcom_push(key='model_dir', value=model_dir)
    context['ti'].xcom_push(key='test_data_path', value='/tmp/test_data_modelB.parquet')
    
    logger.info("✓ Model B training completed")


def generate_visualizations_task(**context):
    """Task 3: Generate model visualization plots."""
    # Import heavy libraries only when task runs
    import glob
    from utils.s3_utils import S3Handler
    from utils.model_training import ModelATrainer
    from pycaret.classification import load_model
    
    logger.info("Starting visualization generation task for Model B...")
    
    config = get_config(context)
    s3_handler = S3Handler(bucket_name=config['S3_BUCKET'], region_name=config['AWS_REGION'])
    
    # Load model
    model_dir = context['ti'].xcom_pull(key='model_dir', task_ids='train_model_b')
    if model_dir is None:
        model_dir = '/tmp/modelB'
    
    model = load_model(f'{model_dir}/modelB_final', verbose=False)
    
    # Create trainer instance to use plot functions
    trainer = ModelATrainer(session_id=config['MODEL_SESSION_ID'])
    trainer.model = model
    
    # Generate plots
    plots_dir = "/tmp/plots_modelB"
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
    
    logger.info("✓ Visualization generation completed for Model B")


def fairness_analysis_task(**context):
    """Task 4: Perform fairness analysis."""
    # Import heavy libraries only when task runs
    import pandas as pd
    from utils.s3_utils import S3Handler
    from utils.fairness_analysis import FairnessAnalyzer
    from pycaret.classification import load_model
    
    logger.info("Starting fairness analysis task for Model B...")
    
    config = get_config(context)
    s3_handler = S3Handler(bucket_name=config['S3_BUCKET'], region_name=config['AWS_REGION'])
    
    # Load model and test data
    model_dir = context['ti'].xcom_pull(key='model_dir', task_ids='train_model_b')
    if model_dir is None:
        model_dir = '/tmp/modelB'
    
    model = load_model(f'{model_dir}/modelB_final', verbose=False)
    
    test_data = pd.read_parquet('/tmp/test_data_modelB.parquet')
    
    # Load demographics if available
    demographics_test = None
    try:
        demographics_test = pd.read_parquet('/tmp/demographics_test_modelB.parquet')
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
    
    # Save fairness metrics with Model B naming
    if fairness_summary:
        logger.info("Saving fairness metrics to S3...")
        
        # Save standalone fairness metrics
        fairness_key = f"{config['S3_OUTPUT_PREFIX']}modelB_fairness_metrics.json"
        s3_handler.write_json(fairness_summary, fairness_key)
        
        # Update main metrics file if it exists
        try:
            metrics_key = f"{config['S3_OUTPUT_PREFIX']}modelB_metrics.json"
            if s3_handler.file_exists(metrics_key):
                model_metrics = s3_handler.read_json(metrics_key)
                model_metrics["fairness"] = fairness_summary
                s3_handler.write_json(model_metrics, metrics_key)
                logger.info("✓ Updated main metrics file with fairness data")
        except Exception as e:
            logger.warning(f"Could not update main metrics file: {e}")
        
        logger.info("✓ Fairness metrics saved to S3")
    
    logger.info("✓ Fairness analysis completed for Model B")


def generate_predictions_task(**context):
    """Task 5: Generate predictions on full dataset for shortlist."""
    # Import heavy libraries only when task runs
    from utils.s3_utils import S3Handler
    from utils.prediction_utils import PredictionGenerator
    from pycaret.classification import load_model
    
    logger.info("Starting prediction generation task for Model B...")
    
    config = get_config(context)
    s3_handler = S3Handler(bucket_name=config['S3_BUCKET'], region_name=config['AWS_REGION'])
    
    # Load model
    model_dir = context['ti'].xcom_pull(key='model_dir', task_ids='train_model_b')
    if model_dir is None:
        model_dir = '/tmp/modelB'
    
    model = load_model(f'{model_dir}/modelB_final', verbose=False)
    
    # Generate predictions
    generator = PredictionGenerator(model=model, s3_handler=s3_handler)
    
    # Load full dataset - Model B specific
    input_prefix = config['S3_INPUT_PREFIX']
    possible_names = [
        "Dataset_B_processed.csv",
        "full_dataset_processed_modelB.csv",
        "processed_full_modelB.csv"
    ]
    
    full_df = None
    for name in possible_names:
        try:
            full_df = s3_handler.read_csv(f"{input_prefix}{name}")
            logger.info(f"✓ Loaded {name}, shape: {full_df.shape}")
            break
        except Exception:
            continue
    
    if full_df is None:
        raise FileNotFoundError(
            f"Could not find full processed dataset. Tried: {possible_names}"
        )
    
    # Generate shortlist base
    shortlist_base = generator.generate_shortlist_base(full_df)
    
    # Rename columns to ModelB prefix
    shortlist_base = shortlist_base.rename(columns={
        'ModelA_Hire_Prob': 'ModelB_Hire_Prob',
        'ModelA_Prediction': 'ModelB_Prediction',
        'ModelA_Pred_Label': 'ModelB_Pred_Label',
        'ModelA_Rank': 'ModelB_Rank'
    })
    
    # Save to S3
    s3_key = f"{config['S3_OUTPUT_PREFIX']}modelB_shortlist_base.csv"
    s3_handler.write_csv(shortlist_base, s3_key, index=False)
    logger.info(f"✓ Saved shortlist base to S3: {s3_key}")
    
    logger.info("✓ Prediction generation completed for Model B")


# Define tasks
load_data = PythonOperator(
    task_id='load_data_from_s3',
    python_callable=load_data_task,
    dag=dag,
)

train_model = PythonOperator(
    task_id='train_model_b',
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
