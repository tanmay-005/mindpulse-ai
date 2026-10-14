"""Step 0.5: first baseline. Train on SentiMix tweets, test on your 200 app reviews.

Run: python -m scripts.p0_baseline
(The first run downloads SentiMix from Hugging Face, about 5 MB.)
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from training import sentimix
from training.baselines import macro_f1, majority_class, tfidf_lr

ROOT = Path(__file__).resolve().parent.parent
RATING_TO_SENTIMENT = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}


def main():
    # 1. Load SentiMix and sanity-check it
    splits = {s: sentimix.load_hf(s) for s in ("train", "validation", "test")}
    for name, tweets in splits.items():
        print(f"SentiMix {name}: {len(tweets)} tweets, labels {pd.Series([t.label for t in tweets]).value_counts().to_dict()}")
    for w in sentimix.check_counts(splits):
        print("WARNING:", w)
    print("Example:", splits["train"][0])

    # 2. Train on SentiMix train
    train = [t for t in splits["train"] if t.label]
    model = tfidf_lr().fit([t.text for t in train], [t.label for t in train])
    majority = majority_class([t.label for t in train])
    results = {"train_tweets": len(train), "majority_class": majority}

    # 3. Score on SentiMix's own test split (or validation if test has no labels)
    name = "test" if any(t.label for t in splits["test"]) else "validation"
    tw = [t for t in splits[name] if t.label]
    y, pred = [t.label for t in tw], list(model.predict([t.text for t in tw]))
    results[f"sentimix_{name}"] = {
        "n": len(y),
        "majority_macro_f1": round(macro_f1(y, [majority] * len(y)), 3),
        "tfidf_lr_macro_f1": round(macro_f1(y, pred), 3),
    }

    # 4. Score on your 200 hand-labelled app reviews
    hand = pd.read_csv(ROOT / "data" / "feasibility" / "hand_200.csv").dropna(subset=["sentiment"])
    y = hand["sentiment"].tolist()
    pred = list(model.predict(hand["content"].astype(str).tolist()))
    stars = hand["rating"].map(RATING_TO_SENTIMENT).tolist()
    results["app_reviews"] = {
        "n": len(y),
        "majority_macro_f1": round(macro_f1(y, [majority] * len(y)), 3),
        "tfidf_lr_macro_f1": round(macro_f1(y, pred), 3),
        "star_rating_as_label_macro_f1": round(macro_f1(y, stars), 3),
        "star_rating_agrees_with_text_%": round(100 * sum(a == b for a, b in zip(y, stars)) / len(y), 1),
    }

    print("\nPer-class results on your 200 app reviews (TF-IDF + LR):")
    print(classification_report(y, pred, zero_division=0))

    out = ROOT / "data" / "feasibility" / "baseline.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()