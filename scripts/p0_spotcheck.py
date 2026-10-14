"""Step 0.3: spot-check the language tagger by hand.

Run: python -m scripts.p0_spotcheck
Answer y or n for each review. The script counts for you.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def ask(question):
    while True:
        a = input(question).strip().lower()
        if a in ("y", "n"):
            return a == "y"
        print("  Please type y or n.")


def main():
    df = pd.read_csv(ROOT / "data" / "feasibility" / "reviews_tagged.csv")

    # Check 1: Google said Hindi, our tagger said English. Who is right?
    doubt = df[(df["lang_query"] == "hi") & (df["lang_mix"] == "english") & (df["n_words"] >= 3)]
    doubt = doubt.sample(min(30, len(doubt)), random_state=1)
    print("\nCHECK 1: the tagger called these ENGLISH. Is each one actually Hinglish?\n")
    really_hinglish = 0
    for i, text in enumerate(doubt["content"], 1):
        print(f"[{i}/{len(doubt)}] {text}")
        really_hinglish += ask("  Actually Hinglish? (y/n) ")

    # Check 2: random reviews with 3+ words. Is the tagger's label right?
    rand = df[df["n_words"] >= 3].sample(20, random_state=2)
    print("\nCHECK 2: is the tagger's label correct?\n")
    wrong = 0
    for i, (text, label) in enumerate(zip(rand["content"], rand["lang_mix"]), 1):
        print(f"[{i}/20] {text}\n  tagger says: {label}")
        wrong += not ask("  Correct? (y/n) ")

    missed = input("\nHinglish words the tagger missed (separate with spaces, or press Enter): ")

    print("\n=== Copy everything below and send it ===")
    print(f"Check 1: {really_hinglish} of {len(doubt)} 'english' reviews were actually Hinglish")
    print(f"Check 2: {wrong} of 20 labels were wrong")
    print(f"Missed words: {missed.strip() or '(none)'}")


if __name__ == "__main__":
    main()