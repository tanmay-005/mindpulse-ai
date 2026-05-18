"""
src/vocab_updater.py
Vocabulary learning — tracks unknown tokens and expands
the tokenizer vocabulary when words appear frequently enough.
"""

import os
import json
import shutil
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "mindpulse model")
)
UNKNOWN_WORDS_PATH = os.path.join(MODEL_PATH, "unknown_words.json")


def _load_unknown_words() -> dict:
    if os.path.exists(UNKNOWN_WORDS_PATH):
        try:
            with open(UNKNOWN_WORDS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_unknown_words(data: dict):
    with open(UNKNOWN_WORDS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def track_unknown_words(text: str, tokenizer):
    """Tracks words that get tokenized as [UNK]. Called from predict.py."""
    data = _load_unknown_words()
    changed = False
    for word in text.lower().split():
        tokens = tokenizer.tokenize(word)
        if not tokens or "[UNK]" in tokens:
            data[word] = data.get(word, 0) + 1
            changed = True
    if changed:
        _save_unknown_words(data)


def get_unknown_words(min_frequency: int = 3) -> dict:
    """Returns words seen at least min_frequency times."""
    data = _load_unknown_words()
    return {
        word: count
        for word, count in data.items()
        if count >= min_frequency
    }


def update_vocabulary(min_frequency: int = 3) -> bool:
    """
    Adds frequent unknown words to tokenizer vocabulary
    and resizes model embeddings. No retraining needed.
    Returns True if vocabulary was updated, False otherwise.
    """
    new_words = list(get_unknown_words(min_frequency).keys())

    if not new_words:
        return False

    print(f"Adding {len(new_words)} new words: {new_words}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    num_added = tokenizer.add_tokens(new_words)
    print(f"Added {num_added} new tokens")

    model.resize_token_embeddings(len(tokenizer))
    print(f"Embeddings resized to {len(tokenizer)}")

    # Backup before overwriting
    backup = MODEL_PATH + "_vocab_backup"
    if os.path.exists(backup):
        shutil.rmtree(backup)
    shutil.copytree(MODEL_PATH, backup)

    tokenizer.save_pretrained(MODEL_PATH)
    model.save_pretrained(MODEL_PATH)

    # Clear processed words
    _save_unknown_words({})
    print("Vocabulary updated ✅")
    return True


def show_unknown_words():
    """Prints tracked unknown words sorted by frequency."""
    data = _load_unknown_words()
    if not data:
        print("No unknown words tracked yet.")
        return
    sorted_words = sorted(data.items(), key=lambda x: x[1], reverse=True)
    print(f"\n{'Word':<20} {'Seen':<10}")
    print("-" * 30)
    for word, count in sorted_words:
        print(f"{word:<20} {count:<10}")