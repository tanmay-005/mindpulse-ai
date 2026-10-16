"""Step 1.4: find Roman-script Hindi words the tagger is missing.

Run: python -m scripts.mine_hindi_words

Google sorts reviews into an 'en' and a 'hi' stream. A word that is much more
common in the 'hi' stream than in the 'en' stream is very likely Hindi.
We score each word with a smoothed log-odds ratio and print the top candidates
that are not in hindi_words.txt yet. YOU decide which ones are real Hindi.
"""
import math
import re
from collections import Counter

from ingestion.store import connect
from processing.langmix import roman_hindi

WORD_RE = re.compile(r"[a-z]{2,}")
MIN_COUNT = 5        # ignore rare words: too noisy to judge
TOP_N = 150


def main():
    with connect() as conn:
        rows = conn.execute(
            "SELECT lang_query, lower(coalesce(content_norm, content_raw)) "
            "FROM reviews WHERE n_words >= 3"
        ).fetchall()

    counts = {"en": Counter(), "hi": Counter()}
    for lang, text in rows:
        if lang in counts:
            counts[lang].update(set(WORD_RE.findall(text)))    # count each word once per review

    n_hi, n_en = sum(counts["hi"].values()), sum(counts["en"].values())
    vocab = set(counts["hi"]) | set(counts["en"])
    known = roman_hindi()

    scored = []
    for w in vocab:
        c_hi, c_en = counts["hi"][w], counts["en"][w]
        if c_hi < MIN_COUNT or w in known:
            continue
        # log-odds with add-one smoothing: how much more likely in 'hi' than in 'en'
        score = math.log((c_hi + 1) / (n_hi + len(vocab))) - math.log((c_en + 1) / (n_en + len(vocab)))
        scored.append((score, w, c_hi, c_en))

    scored.sort(reverse=True)
    print(f"{len(rows)} reviews scanned. Top {TOP_N} candidate Hindi words (not yet in the list):\n")
    print(f"{'word':<14}{'in hi':>7}{'in en':>7}{'score':>8}")
    for score, w, c_hi, c_en in scored[:TOP_N]:
        print(f"{w:<14}{c_hi:>7}{c_en:>7}{score:>8.2f}")
    print("\nAdd the real Hindi words to processing/hindi_words.txt (one per line).")
    print("Skip English words, brand names and words that are both (e.g. 'me', 'to').")


if __name__ == "__main__":
    main()