"""Step 0.4: hand-label the sentiment of 200 random app reviews.

Run: python -m scripts.p0_hand_label

The first run picks 200 reviews (3+ words, equal share per app, fixed seed)
and saves them to data/feasibility/hand_200.csv. Then it shows one at a time:

    p = positive   u = neutral   n = negative   b = back   q = quit

Progress is saved after every answer, so you can stop and resume any time.
Star ratings are hidden on purpose: label what the TEXT says.
"""
import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "data" / "feasibility" / "hand_200.csv"
N_TOTAL = 200
SEED = 42
KEYS = {"p": "positive", "u": "neutral", "n": "negative"}


def make_sample():
    files = sorted(glob.glob(str(ROOT / "data" / "raw" / "sample_*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["content"] = df["content"].fillna("").astype(str)
    df = df[df["content"].str.split().str.len() >= 3]          # real reviews only
    df = df.drop_duplicates("content")

    per_app = N_TOTAL // df["app_name"].nunique()
    parts = [g.sample(min(per_app, len(g)), random_state=SEED) for _, g in df.groupby("app_name")]
    sample = pd.concat(parts)
    if len(sample) < N_TOTAL:                                  # top up if 200 doesn't divide evenly
        rest = df.drop(sample.index)
        sample = pd.concat([sample, rest.sample(N_TOTAL - len(sample), random_state=SEED)])

    sample = sample.sample(frac=1, random_state=SEED)          # mix the apps together
    out = sample[["review_id", "app_name", "content", "rating"]].copy()
    out["sentiment"] = ""
    return out.reset_index(drop=True)


def main():
    PATH.parent.mkdir(parents=True, exist_ok=True)
    if PATH.exists():
        sheet = pd.read_csv(PATH, dtype={"sentiment": str}).fillna({"sentiment": ""})
    else:
        sheet = make_sample()
        sheet.to_csv(PATH, index=False, encoding="utf-8")
        print(f"Picked {len(sheet)} reviews -> {PATH}")

    history = []
    while True:
        todo = sheet.index[sheet["sentiment"] == ""].tolist()
        if not todo:
            print(f"\nAll {len(sheet)} reviews labelled.")
            break
        i = todo[0]
        done = len(sheet) - len(todo)
        print(f"\n[{done + 1}/{len(sheet)}] {sheet.at[i, 'app_name']}")
        print(f"  {sheet.at[i, 'content']}")
        key = input("  p=positive  u=neutral  n=negative  b=back  q=quit > ").strip().lower()

        if key == "q":
            break
        if key == "b":
            if history:
                sheet.at[history.pop(), "sentiment"] = ""
            else:
                print("  Nothing to undo.")
        elif key in KEYS:
            sheet.at[i, "sentiment"] = KEYS[key]
            history.append(i)
        else:
            print("  Please type p, u, n, b or q.")
            continue
        sheet.to_csv(PATH, index=False, encoding="utf-8")       # save after every answer

    counts = sheet["sentiment"].replace("", "not labelled yet").value_counts()
    print("\nCounts so far:\n" + counts.to_string())


if __name__ == "__main__":
    main()