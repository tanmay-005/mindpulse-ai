"""Text normalisation for app reviews (Phase 1, step 1.3).

normalize(text) runs these steps in order:
  1. strip personal info  (URLs, e-mails, UPI IDs, phone and long account numbers)
  2. tag emojis           ("😡😡😡" -> "😡😡 [angry]")
  3. shorten elongations  ("bakwaaaas" -> "bakwaas", "!!!!" -> "!!")
  4. unify spellings      ("nhi" -> "nahi", "plz" -> "please")
  5. tidy whitespace

The raw text is never changed in the database: the result goes to content_norm.
"""
import re
from functools import lru_cache
from pathlib import Path

import yaml

SLANG_PATH = Path(__file__).resolve().parent / "slang.yaml"

# ── 1. personal info ──────────────────────────────────────────
URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
UPI_RE = re.compile(r"\b[\w.-]{2,}@[a-z]{2,}\b", re.IGNORECASE)          # e.g. name@okaxis
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")  # Indian mobiles
LONG_NUM_RE = re.compile(r"(?<!\d)\d{9,}(?!\d)")                         # account / txn numbers


def strip_personal_info(text):
    text = URL_RE.sub(" <url> ", text)
    text = EMAIL_RE.sub(" <email> ", text)
    text = UPI_RE.sub(" <upi> ", text)
    text = PHONE_RE.sub(" <phone> ", text)
    text = LONG_NUM_RE.sub(" <num> ", text)
    return text


# ── 2. emojis ─────────────────────────────────────────────────
EMOJI_TAGS = {
    "😡": "angry", "🤬": "angry", "😠": "angry", "👿": "angry",
    "😤": "frustrated", "😒": "annoyed", "🙄": "annoyed", "😑": "annoyed",
    "😢": "sad", "😭": "crying", "😞": "disappointed", "😔": "disappointed", "☹": "sad", "🙁": "sad",
    "😟": "worried", "😰": "anxious", "😨": "scared", "😱": "shocked",
    "😂": "laughing", "🤣": "laughing", "😆": "laughing", "😹": "laughing",
    "😊": "happy", "😀": "happy", "😃": "happy", "😄": "happy", "😁": "happy", "🙂": "happy", "☺": "happy",
    "😍": "love", "🥰": "love", "😘": "love", "❤": "love", "💕": "love", "💖": "love", "♥": "love",
    "👍": "thumbs_up", "👎": "thumbs_down", "👌": "ok", "🙏": "please",
    "🔥": "fire", "💯": "perfect", "⭐": "star", "🌟": "star",
    "🤮": "disgusted", "🤢": "disgusted", "💩": "poop",
    "🤔": "confused", "😕": "confused", "🥺": "pleading", "😐": "neutral",
    "🤡": "clown", "😴": "bored", "🤦": "facepalm",
}
_VS = "\ufe0f"   # variation selector that often follows ❤ ☺ ☹ ♥
_SKIN = "[\U0001F3FB-\U0001F3FF]"
EMOJI_RUN_RE = re.compile(
    "((?:" + "|".join(re.escape(e) for e in sorted(EMOJI_TAGS, key=len, reverse=True))
    + ")(?:" + _VS + "|" + _SKIN + ")?)(?:\\1|(?:" + _VS + "))*"
)


def tag_emojis(text):
    """Keep at most two of a repeated emoji and add a text tag after it."""
    def repl(m):
        base = m.group(1).replace(_VS, "")
        base = re.sub(_SKIN, "", base)
        count = m.group(0).count(base)
        return f" {base * min(count, 2)} [{EMOJI_TAGS[base]}] "
    return EMOJI_RUN_RE.sub(repl, text)


# ── 3. elongations ────────────────────────────────────────────
ELONG_RE = re.compile(r"([^\d\s])\1{2,}")      # 3+ repeats of any non-digit, non-space char


def shorten_elongations(text):
    return ELONG_RE.sub(r"\1\1", text)


# ── 4. spellings ──────────────────────────────────────────────
WORD_RE = re.compile(r"[A-Za-z]+")
DOUBLE_RE = re.compile(r"([a-z])\1")


@lru_cache(maxsize=1)
def load_slang():
    data = yaml.safe_load(open(SLANG_PATH, encoding="utf-8"))
    table = {}
    for section in data.values():
        table.update({str(k).lower(): str(v) for k, v in section.items()})
    return table


def unify_spellings(text):
    table = load_slang()
    canonical = set(table.values())

    def repl(m):
        word = m.group(0)
        low = word.lower()
        if low in table:
            return table[low]
        single = DOUBLE_RE.sub(r"\1", low)        # "bhaii" -> "bhai" after elongation
        if single != low and (single in table or single in canonical):
            return table.get(single, single)
        return word
    return WORD_RE.sub(repl, text)


# ── pipeline ──────────────────────────────────────────────────
def normalize(text):
    if not text:
        return ""
    text = strip_personal_info(text)
    text = tag_emojis(text)
    text = shorten_elongations(text)
    text = unify_spellings(text)
    return " ".join(text.split())