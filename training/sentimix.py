"""Load the SemEval-2020 Task 9 (SentiMix) Hinglish sentiment dataset.

The data is one token per line. Each tweet starts with a 'meta' line:

    meta    <tweet_id>    <sentiment>
    <token> <language tag>
    <token> <language tag>
    ...

We download it from the Hugging Face mirror 'RTT1/SentiMix'. Its dataset card is
empty, so check_counts() makes sure it looks like the official release:
about 14k train, 3k validation and 3k test tweets, labelled positive/neutral/negative.
"""
from dataclasses import dataclass

LABELS = ("negative", "neutral", "positive")
HF_ID = "RTT1/SentiMix"


@dataclass
class Tweet:
    tweet_id: str
    text: str
    label: str | None


def parse_conll(lines):
    """Turn token-per-line text into a list of Tweets."""
    tweets = []
    cur_id, cur_label, tokens = None, None, []

    def flush():
        if cur_id is not None and tokens:
            tweets.append(Tweet(cur_id, " ".join(tokens), cur_label))

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "meta":                 # start of a new tweet
            flush()
            cur_id = parts[1] if len(parts) > 1 else str(len(tweets))
            last = parts[-1].lower() if len(parts) > 2 else None
            cur_label = last if last in LABELS else None
            tokens = []
        elif cur_id is not None:               # a token line: keep the word, drop the tag
            tokens.append(parts[0])
    flush()
    return tweets


def load_hf(split):
    """Download one split ('train', 'validation' or 'test') and parse it."""
    from datasets import load_dataset
    ds = load_dataset(HF_ID, split=split)
    return parse_conll(ds["text"])


def load_file(path):
    with open(path, encoding="utf-8") as f:
        return parse_conll(f)


def check_counts(splits):
    """Return warnings if the data doesn't look like the official SentiMix."""
    expected = {"train": (12000, 16000), "validation": (2500, 3500), "test": (2500, 3500)}
    warnings = []
    for name, tweets in splits.items():
        lo, hi = expected.get(name, (0, 10**9))
        if not lo <= len(tweets) <= hi:
            warnings.append(f"{name}: {len(tweets)} tweets, expected {lo}-{hi}")
        if name != "test" and not any(t.label for t in tweets):
            warnings.append(f"{name}: no sentiment labels found")
    return warnings