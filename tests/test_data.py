"""
tests/test_data.py
Pre-train tests: data validation (schema, quality, ranges).
Ці тести НЕ потребують тренування моделі.
"""
import os

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Data Schema & Quality (Pre-train)
# ---------------------------------------------------------------------------

class TestDataSchema:
    """Перевірка структури та якості підготовлених даних."""

    def test_train_file_exists(self, data_dir):
        train_path = os.path.join(data_dir, "train.csv")
        assert os.path.exists(train_path), f"Train data not found: {train_path}"

    def test_required_columns_present(self, data_dir):
        """Перевіряє наявність обов'язкових колонок."""
        df = pd.read_csv(os.path.join(data_dir, "train.csv"), nrows=10)
        required = {"label", "tweet_clean", "num_hashtags", "num_mentions",
                     "tweet_len", "sentiment_score"}
        missing = required - set(df.columns)
        assert not missing, f"Missing columns: {sorted(missing)}"

    def test_label_values_binary(self, data_dir):
        """Label повинен бути 0 або 1."""
        df = pd.read_csv(os.path.join(data_dir, "train.csv"), usecols=["label"])
        unique_labels = set(df["label"].unique())
        assert unique_labels <= {0, 1}, f"Unexpected label values: {unique_labels}"

    def test_no_null_labels(self, data_dir):
        """Немає пропусків у label."""
        df = pd.read_csv(os.path.join(data_dir, "train.csv"), usecols=["label"])
        assert df["label"].notna().all(), "label contains NaN values"

    def test_sufficient_rows(self, data_dir):
        """Мінімум 100 рядків для навчання."""
        df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        assert len(df) >= 100, f"Too few rows: {len(df)}"

    def test_numeric_features_no_nulls(self, data_dir):
        """Числові ознаки не мають NaN."""
        numeric_cols = ["num_hashtags", "num_mentions", "tweet_len", "sentiment_score"]
        df = pd.read_csv(os.path.join(data_dir, "train.csv"), usecols=numeric_cols)
        for col in numeric_cols:
            assert df[col].notna().all(), f"Column '{col}' has NaN values"

    def test_tweet_len_positive(self, data_dir):
        """tweet_len повинен бути > 0."""
        df = pd.read_csv(os.path.join(data_dir, "train.csv"), usecols=["tweet_len"])
        assert (df["tweet_len"] > 0).all(), "tweet_len should be > 0"

    def test_sentiment_score_range(self, data_dir):
        """VADER sentiment score в діапазоні [-1, 1]."""
        df = pd.read_csv(os.path.join(data_dir, "train.csv"), usecols=["sentiment_score"])
        assert df["sentiment_score"].between(-1, 1).all(), \
            "sentiment_score out of range [-1, 1]"

    def test_class_balance_not_extreme(self, data_dir):
        """Мінорний клас >= 1% (не повністю відсутній)."""
        df = pd.read_csv(os.path.join(data_dir, "train.csv"), usecols=["label"])
        class_ratio = df["label"].value_counts(normalize=True).min()
        assert class_ratio >= 0.01, \
            f"Minority class ratio too low: {class_ratio:.4f}"


class TestRawData:
    """Перевірка наявності сирих даних."""

    def test_raw_train_exists(self, raw_data_dir):
        path = os.path.join(raw_data_dir, "train.csv")
        assert os.path.exists(path), f"Raw data not found: {path}"

    def test_raw_data_has_tweet_column(self, raw_data_dir):
        path = os.path.join(raw_data_dir, "train.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, nrows=5)
            assert "tweet" in df.columns, "Raw data missing 'tweet' column"
