"""Step 0.5 (extra): train on STAR RATINGS of scraped reviews, test on your 200 labels.

Run: python -m scripts.p0_weak_baseline

Star ratings are free, noisy labels (1-2 negative, 3 neutral, 4-5 positive).
This checks whether noisy in-domain labels beat clean out-of-domain tweets.
"""
import glob
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from training.baselines import macro_f1, tfidf_lr

ROOT = Path(__file__).resolve().parent.parent
RATING_TO_SENTIMENT = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}


def main():
    hand = pd.read_csv(ROOT / "data" / "feasibility" / "hand_200.csv").dropna(subset=["sentiment"])

    files = sorted(glob.glob(str(ROOT / "data" / "raw" / "sample_*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["content"] = df["content"].fillna("").astype(str)
    df = df[df["content"].str.split().str.len() >= 3]
    df = df[~df["review_id"].isin(hand["review_id"])]          # never train on the test reviews
    df = df[~df["content"].isin(hand["content"])]              # ...or on exact copies of them
    df = df.drop_duplicates("content")
    df["weak"] = df["rating"].map(RATING_TO_SENTIMENT)

    print(f"Training on {len(df)} reviews, star-rating labels: {df['weak'].value_counts().to_dict()}")
    model = tfidf_lr().fit(df["content"].tolist(), df["weak"].tolist())

    y = hand["sentiment"].tolist()
    pred = list(model.predict(hand["content"].astype(str).tolist()))
    print(f"\nMacro-F1 on your 200 app reviews: {macro_f1(y, pred):.3f}")
    print(classification_report(y, pred, zero_division=0))


if __name__ == "__main__":
    main()