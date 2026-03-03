import argparse
import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report,
)
from scipy.sparse import hstack, csr_matrix

import mlflow
import mlflow.sklearn

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_nltk():
    """Download NLTK data with SSL workaround."""
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download("vader_lexicon", quiet=True)


def clean_tweet(text: str) -> str:
    """Basic tweet cleaning for TF-IDF."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)           # URLs
    text = re.sub(r"@\w+", "", text)                      # mentions
    text = re.sub(r"#(\w+)", r"\1", text)                 # keep hashtag text
    text = re.sub(r"[^a-zA-Z\s]", "", text)               # non-alpha
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_symbols(text: str):
    hashtag_count = len(re.findall(r"#", str(text)))
    mention_count = len(re.findall(r"@", str(text)))
    return hashtag_count, mention_count


ARTIFACTS_DIR = "artifacts"


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def load_and_preprocess(data_path: str, tfidf_max_features: int = 5000):
    """Load CSV, engineer features, return X (sparse), y, feature_names."""
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["tweet"])

    setup_nltk()

    # --- numeric features ---
    df[["num_hashtags", "num_mentions"]] = df["tweet"].apply(
        lambda x: pd.Series(count_symbols(x))
    )
    df["tweet_len"] = df["tweet"].apply(lambda x: len(str(x)))

    sia = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df["tweet"].apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )

    numeric_features = ["num_hashtags", "num_mentions", "tweet_len", "sentiment_score"]
    X_numeric = csr_matrix(df[numeric_features].values)

    # --- TF-IDF features ---
    df["tweet_clean"] = df["tweet"].apply(clean_tweet)
    tfidf = TfidfVectorizer(max_features=tfidf_max_features, stop_words="english")
    X_tfidf = tfidf.fit_transform(df["tweet_clean"])

    # --- combine ---
    X = hstack([X_numeric, X_tfidf])
    y = df["label"]

    feature_names = numeric_features + [f"tfidf_{w}" for w in tfidf.get_feature_names_out()]

    return X, y, feature_names, tfidf


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    # 1. Load data
    data_path = "data/raw/train.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    X, y, feature_names, tfidf = load_and_preprocess(data_path, args.tfidf_features)

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state, stratify=y
    )

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

        # 9. Artifacts ---------------------------------------------------------

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
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=10, help="Max depth (0 = None)")
    parser.add_argument("--random_state", type=int, default=42, help="Random state")
    parser.add_argument("--cv", type=int, default=0, help="Cross-validation folds (0 = skip)")
    parser.add_argument("--smote", action="store_true", help="Use SMOTE oversampling")
    parser.add_argument("--tfidf_features", type=int, default=5000, help="Max TF-IDF features")

    args = parser.parse_args()
    train(args)
