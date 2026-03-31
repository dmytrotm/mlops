import argparse
import os
import re

import pandas as pd
import numpy as np

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


# ---------------------------------------------------------------------------
# Main preparation logic
# ---------------------------------------------------------------------------

def prepare(input_file: str, output_dir: str, test_size: float, random_state: int):
    """Load raw data, engineer features, split, and save."""
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")

    # Drop rows with missing tweets
    df = df.dropna(subset=["tweet"])
    print(f"After dropna: {len(df)} rows")

    # --- Feature engineering ---
    setup_nltk()

    df[["num_hashtags", "num_mentions"]] = df["tweet"].apply(
        lambda x: pd.Series(count_symbols(x))
    )
    df["tweet_len"] = df["tweet"].apply(lambda x: len(str(x)))

    sia = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df["tweet"].apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )

    # Clean tweet text (stored for TF-IDF in the train stage)
    df["tweet_clean"] = df["tweet"].apply(clean_tweet).fillna("")

    # --- Train / test split ---
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )
    print(f"Train: {len(df_train)}  |  Test: {len(df_test)}")

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    print(f"Saved prepared data to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument("--input", type=str, default="data/raw/train.csv",
                        help="Path to raw CSV")
    parser.add_argument("--output_dir", type=str, default="data/prepared",
                        help="Directory for prepared CSVs")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction for test split")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random state for reproducibility")

    args = parser.parse_args()
    prepare(args.input, args.output_dir, args.test_size, args.random_state)
