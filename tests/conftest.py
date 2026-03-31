"""
tests/conftest.py
Shared fixtures for pre-train and post-train tests.
"""
import os

import pytest


@pytest.fixture
def data_dir():
    return os.getenv("DATA_DIR", "data/prepared")


@pytest.fixture
def raw_data_dir():
    return os.getenv("RAW_DATA_DIR", "data/raw")


@pytest.fixture
def model_path():
    return os.getenv("MODEL_PATH", "models/best_model.pkl")


@pytest.fixture
def metrics_path():
    return os.getenv("METRICS_PATH", "artifacts/metrics.json")


@pytest.fixture
def confusion_matrix_path():
    return os.getenv("CM_PATH", "artifacts/confusion_matrix.png")
