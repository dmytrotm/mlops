"""
Stage 2 — Train a Cyberbullying Detection model on prepared data.

Reads prepared train/test CSVs (output of prepare.py),
builds TF-IDF + numeric features, trains a RandomForest,
evaluates, and logs everything to MLflow.
"""

import argparse
import os
import re
import json

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)
from scipy.sparse import hstack, csr_matrix

import mlflow
import mlflow.sklearn
import joblib


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = "artifacts"
NUMERIC_FEATURES = ["num_hashtags", "num_mentions", "tweet_len", "sentiment_score"]


def build_features(df: pd.DataFrame, tfidf: TfidfVectorizer = None,
                   tfidf_max_features: int = 5000, fit: bool = False):
    """Build sparse feature matrix from a prepared DataFrame."""
    X_numeric = csr_matrix(df[NUMERIC_FEATURES].values)

    # Ensure no NaN in text column
    tweets = df["tweet_clean"].fillna("")

    if fit:
        tfidf = TfidfVectorizer(max_features=tfidf_max_features, stop_words="english")
        X_tfidf = tfidf.fit_transform(tweets)
    else:
        X_tfidf = tfidf.transform(tweets)

    X = hstack([X_numeric, X_tfidf])
    feature_names = NUMERIC_FEATURES + [f"tfidf_{w}" for w in tfidf.get_feature_names_out()]
    return X, feature_names, tfidf


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    # 1. Load prepared data
    train_path = os.path.join(args.data_dir, "train.csv")
    test_path = os.path.join(args.data_dir, "test.csv")
    for p in [train_path, test_path]:
        if not os.path.exists(p):
            print(f"Error: {p} not found. Run the prepare stage first.")
            return

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    print(f"Train: {len(df_train)}  |  Test: {len(df_test)}")

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # 2. Build features (fit TF-IDF on train, transform test)
    X_train, feature_names, tfidf = build_features(
        df_train, tfidf_max_features=args.tfidf_features, fit=True
    )
    X_test, _, _ = build_features(df_test, tfidf=tfidf)

    y_train = df_train["label"]
    y_test = df_test["label"]

    # 3. Handle imbalance with SMOTE (optional)
    if args.smote:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=args.random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {pd.Series(y_train).value_counts().to_dict()}")

    # 4. MLflow
    mlflow.set_experiment("Cyberbullying_Detection")

    mode_label = "smote" if args.smote else "balanced"
    run_name = f"RF_depth{args.max_depth}_est{args.n_estimators}_{mode_label}"

    with mlflow.start_run(run_name=run_name):
        # Tags
        mlflow.set_tag("author", "Dmytro")
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("data_version", "1.0")
        mlflow.set_tag("imbalance_strategy", "smote" if args.smote else "class_weight_balanced")

        # 5. Train
        max_depth = None if args.max_depth == 0 else args.max_depth
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=args.random_state,
            n_jobs=-1,
        )

        if args.cv > 0:
            print(f"Running {args.cv}-fold Cross Validation...")
            cv_scores = cross_val_score(model, X_train, y_train, cv=args.cv, scoring="f1")
            mlflow.log_metric("cv_f1_mean", cv_scores.mean())
            mlflow.log_metric("cv_f1_std", cv_scores.std())
            print(f"CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        model.fit(X_train, y_train)

        # 6. Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_train_pred = model.predict(X_train)

        metrics = {
            "test_accuracy":  accuracy_score(y_test, y_pred),
            "test_f1":        f1_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall":    recall_score(y_test, y_pred),
            "test_roc_auc":   roc_auc_score(y_test, y_pred_proba),
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "train_f1":       f1_score(y_train, y_train_pred),
        }

        # 7. Log params & metrics
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth if args.max_depth != 0 else "None")
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("smote", args.smote)
        mlflow.log_param("tfidf_max_features", args.tfidf_features)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_metrics(metrics)

        # 8. Log model
        mlflow.sklearn.log_model(model, "model")

        # 9. Save model locally (DVC-tracked output)
        model_path = os.path.join(args.model_dir, "model.joblib")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

        # 10. Save metrics JSON (DVC metrics file)
        metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # 11. Artifacts --------------------------------------------------------

        # Classification report (text)
        report = classification_report(y_test, y_pred, target_names=["Not Hate", "Hate"])
        report_path = os.path.join(ARTIFACTS_DIR, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)
        print(f"\n{report}")

        # Feature importance (top 20)
        importance = model.feature_importances_
        top_n = 20
        top_idx = np.argsort(importance)[-top_n:]
        top_features = [feature_names[i] for i in top_idx]
        top_importance = importance[top_idx]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_importance, y=top_features, orient="h")
        plt.title("Top-20 Feature Importance")
        plt.tight_layout()
        fi_path = os.path.join(ARTIFACTS_DIR, "feature_importance.png")
        plt.savefig(fi_path, dpi=100)
        mlflow.log_artifact(fi_path)
        plt.close()

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Hate", "Hate"])
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=100)
        mlflow.log_artifact(cm_path)
        plt.close()

        print(f"\nRun completed:")
        print(f"  Test F1:        {metrics['test_f1']:.4f}")
        print(f"  Test Precision: {metrics['test_precision']:.4f}")
        print(f"  Test Recall:    {metrics['test_recall']:.4f}")
        print(f"  Test ROC-AUC:   {metrics['test_roc_auc']:.4f}")
        print(f"  Test Accuracy:  {metrics['test_accuracy']:.4f}")
        print(f"  Train F1:       {metrics['train_f1']:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cyberbullying Detection Model")
    parser.add_argument("--data_dir", type=str, default="data/prepared",
                        help="Directory with prepared train.csv / test.csv")
    parser.add_argument("--model_dir", type=str, default="data/models",
                        help="Directory to save the trained model")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=10, help="Max depth (0 = None)")
    parser.add_argument("--random_state", type=int, default=42, help="Random state")
    parser.add_argument("--cv", type=int, default=0, help="Cross-validation folds (0 = skip)")
    parser.add_argument("--smote", action="store_true", help="Use SMOTE oversampling")
    parser.add_argument("--tfidf_features", type=int, default=5000, help="Max TF-IDF features")

    args = parser.parse_args()
    train(args)
