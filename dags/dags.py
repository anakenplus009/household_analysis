from datetime import datetime, timedelta
from airflow import models
from airflow.providers.google.cloud.operators.cloud_run import CloudRunExecuteJobOperator


# DAGの基本設定
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# 
with models.DAG(
    'daily_power_inference_dag',
    default_args=default_args,
    schedule_interval='0 22 * * *',
    start_date=datetime(2026, 6, 18),
    catchup=False,
    tags=['power-pipeline']
) as dag:
    
    run_inference_job = CloudRunExecuteJobOperator(
        task_id='execute_cloud_run_inference',
        project_id='noted-cortex-460601-c3',
        region='asia-northeast1',
        job_name='daily-interence-job'
    )