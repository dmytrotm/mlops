"""
tests/test_model.py
Post-train tests: artifacts existence and Quality Gate.
Ці тести працюють ПІСЛЯ тренування моделі.
"""
import json
import os

import pytest


# ---------------------------------------------------------------------------
# Artifact Existence (Post-train)
# ---------------------------------------------------------------------------

class TestArtifacts:
    """Перевірка наявності артефактів після тренування."""

    def test_model_artifact_exists(self):
        """Артефакт моделі має існувати."""
        paths = ["models/best_model.pkl", "data/models/model.joblib"]
        found = any(os.path.exists(p) for p in paths)
        assert found, f"No model artifact found. Checked: {paths}"

    def test_metrics_json_exists(self):
        """metrics.json повинен існувати."""
        assert os.path.exists("artifacts/metrics.json"), \
            "artifacts/metrics.json not found"

    def test_confusion_matrix_exists(self):
        """Confusion matrix має існувати."""
        assert os.path.exists("artifacts/confusion_matrix.png"), \
            "artifacts/confusion_matrix.png not found"

    def test_classification_report_exists(self):
        """Classification report має існувати."""
        assert os.path.exists("artifacts/classification_report.txt"), \
            "artifacts/classification_report.txt not found"

    def test_metrics_json_valid(self):
        """metrics.json має бути валідним JSON з потрібними ключами."""
        path = "artifacts/metrics.json"
        if not os.path.exists(path):
            pytest.skip("metrics.json not found — run training first")
        with open(path, "r") as f:
            metrics = json.load(f)
        required_keys = {"test_accuracy", "test_f1", "test_precision",
                         "test_recall", "test_roc_auc"}
        missing = required_keys - set(metrics.keys())
        assert not missing, f"Missing metric keys: {missing}"


# ---------------------------------------------------------------------------
# Quality Gate (Post-train)
# ---------------------------------------------------------------------------

class TestQualityGate:
    """Quality Gate — мінімальні вимоги до якості моделі."""

    def _load_metrics(self):
        path = "artifacts/metrics.json"
        if not os.path.exists(path):
            pytest.skip("metrics.json not found — run training first")
        with open(path, "r") as f:
            return json.load(f)

    def test_quality_gate_f1(self):
        """F1 >= threshold (default 0.50 для цього датасету, зважаючи на дисбаланс)."""
        threshold = float(os.getenv("F1_THRESHOLD", "0.50"))
        metrics = self._load_metrics()
        f1 = float(metrics["test_f1"])
        assert f1 >= threshold, \
            f"Quality Gate FAILED: f1={f1:.4f} < {threshold:.2f}"

    def test_quality_gate_accuracy(self):
        """Accuracy >= 0.75 (базовий рівень для датасету)."""
        threshold = float(os.getenv("ACC_THRESHOLD", "0.75"))
        metrics = self._load_metrics()
        acc = float(metrics["test_accuracy"])
        assert acc >= threshold, \
            f"Quality Gate FAILED: accuracy={acc:.4f} < {threshold:.2f}"

    def test_quality_gate_roc_auc(self):
        """ROC-AUC >= 0.70."""
        threshold = float(os.getenv("AUC_THRESHOLD", "0.70"))
        metrics = self._load_metrics()
        auc = float(metrics["test_roc_auc"])
        assert auc >= threshold, \
            f"Quality Gate FAILED: roc_auc={auc:.4f} < {threshold:.2f}"

    def test_no_overfitting(self):
        """Train F1 не повинен бути занадто далекий від Test F1 (< 0.40 gap)."""
        metrics = self._load_metrics()
        train_f1 = float(metrics.get("train_f1", 0))
        test_f1 = float(metrics["test_f1"])
        if train_f1 > 0:
            gap = train_f1 - test_f1
            assert gap < 0.40, \
                f"Overfitting detected: train_f1={train_f1:.4f}, " \
                f"test_f1={test_f1:.4f}, gap={gap:.4f}"
