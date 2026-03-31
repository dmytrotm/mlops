"""
tests/test_airflow_dags.py
Тест для перевірки цілісності (integrity) DAG файлів.
"""

import os
import pytest
from airflow.models import DagBag


def test_dag_import():
    """
    Перевіряє, чи DAG файл(и) завантажуються без помилок імпорту та синтаксису.
    """
    dag_folder = os.getenv("AIRFLOW_HOME", "dags/")
    if not os.path.exists(dag_folder):
        # Якщо тестуємо локально без встановленого Airflow і папка dags на рівень вище
        dag_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dags'))

    dag_bag = DagBag(dag_folder=dag_folder, include_examples=False)

    assert len(dag_bag.import_errors) == 0, f"DAG import errors: {dag_bag.import_errors}"
    assert dag_bag.size() >= 1, "No DAGs found in the folder"
