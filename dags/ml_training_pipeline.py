"""
dags/ml_training_pipeline.py
Apache Airflow DAG для управління процесом навчання моделі (ЛР5).
Включає перевірку наявності даних, запуск DVC pipelines (prepare, train),
оцінку результатів та умовну реєстрацію у MLflow.
"""

import json
import logging
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.dummy import DummyOperator

import mlflow
from mlflow.tracking import MlflowClient

# Загальні налаштування DAG
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Шляхи
PROJECT_ROOT = "/opt/airflow"
DATA_FILE = "data/raw/train.csv"
METRICS_FILE = "artifacts/metrics.json"

def _evaluate_and_branch(**kwargs):
    """
    Зчитує metrics.json. Якщо Accuracy > 0.85, повертає id наступної задачі:
    'register_model', інакше 'stop_pipeline'.
    """
    metrics_path = os.path.join(PROJECT_ROOT, METRICS_FILE)
    if not os.path.exists(metrics_path):
        logging.warning("Metrics file not found. Stop pipeline.")
        return 'stop_pipeline'
        
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    accuracy = metrics.get('test_accuracy', 0.0)
    logging.info(f"Model accuracy: {accuracy}")
    
    if accuracy > 0.85:
        logging.info("Quality Gate PASSED. Proceeding to registration.")
        return 'register_model'
    else:
        logging.warning(f"Quality Gate FAILED (accuracy {accuracy} <= 0.85). Stopping.")
        return 'stop_pipeline'


def _register_best_model(**kwargs):
    """
    Реєструє фінальну модель (яка зберігається в Model Registry)
    до MLflow зі стадією Staging.
    """
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}")
    client = MlflowClient()
    
    # Реєструємо модель. Для спрощення створюємо/оновлюємо нову версію
    # Якщо використовуємо DVC train.py, він міг не залогувати в Registry автоматично.
    # В рамках ЛР5 використаємо пошук останнього run.
    experiment = client.get_experiment_by_name("Baseline_Training")
    if not experiment:
        logging.info("Experiment Baseline_Training not found. Cannot register.")
        return
        
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_accuracy DESC"],
        max_results=1
    )
    if not runs:
        logging.info("No runs found in Baseline_Training experiment.")
        return
        
    best_run = runs[0]
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    model_name = "TwitterHateSpeechModel"
    
    # Реєстрація моделі
    logging.info(f"Registering model from run {run_id}")
    mv = mlflow.register_model(model_uri, name=model_name)
    
    # Перехід до Staging
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Staging"
    )
    logging.info(f"Model {model_name} v{mv.version} marked as Staging.")


with DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='Continuous Training Pipeline with DVC and MLflow',
    schedule_interval=None, # Запуск вручну або через зовнішній тригер
    catchup=False,
    tags=['mlops', 'training'],
) as dag:

    # 1. Sensor / Check: перевірка доступності даних
    # FileSensor очікує, поки файл data/raw/train.csv з'явиться. 
    # (fs_conn_id за замовчуванням 'fs_default' - шлях відносно кореня сервера).
    check_data = FileSensor(
        task_id='check_data_availability',
        filepath=os.path.join(PROJECT_ROOT, DATA_FILE),
        poke_interval=30,
        timeout=600,
        mode='poke'
    )

    # 2. Data Preparation: запуск dvc repro stage: prepare
    prep_data = BashOperator(
        task_id='prepare_data',
        bash_command=f'cd {PROJECT_ROOT} && dvc config core.analytics false && dvc repro prepare',
    )

    # 3. Model Training: запуск dvc repro stage: train
    train_model = BashOperator(
        task_id='train_model',
        bash_command=f'cd {PROJECT_ROOT} && dvc config core.analytics false && dvc repro train',
    )

    # 4. Evaluation & Branching
    evaluate_branch = BranchPythonOperator(
        task_id='evaluate_and_branch',
        python_callable=_evaluate_and_branch,
    )

    # 5. Model Registration або Зупинка
    register_model = PythonOperator(
        task_id='register_model',
        python_callable=_register_best_model,
    )

    stop_pipeline = DummyOperator(
        task_id='stop_pipeline'
    )

    # Визначення залежностей (DAG edges)
    check_data >> prep_data >> train_model >> evaluate_branch
    evaluate_branch >> [register_model, stop_pipeline]
