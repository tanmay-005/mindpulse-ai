"""
src/emoji_utils.py
Emoji processing — extraction, text conversion, sentiment scoring,
and emotion classification using the trained DistilBERT model.

Key fixes:
- Hyphen replaced in descriptions so 'heart-eyes' -> 'heart eyes' -> love detected
- _get_emoji_description and _classify_emoji_description use same cleaned desc
- Love keywords checked on fully cleaned description (no colors, no hyphens)
- Love priority logic simplified and reliable
"""

import emoji
import os
import torch
from functools import lru_cache

# ── Model path ───────────────────────────────────────────────
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "mindpulse model")
)

# ── Lazy-loaded classifier ───────────────────────────────────
_classifier = None

def _get_classifier():
    global _classifier
    if _classifier is None:
        from transformers import pipeline
        _classifier = pipeline(
            "text-classification",
            model=MODEL_PATH,
            device=0 if torch.cuda.is_available() else -1,
            top_k=1,
        )
    return _classifier


# ── Label normalization ──────────────────────────────────────
LABEL_INDEX_MAP = {
    "LABEL_0": "sadness",
    "LABEL_1": "joy",
    "LABEL_2": "love",
    "LABEL_3": "anger",
    "LABEL_4": "fear",
    "LABEL_5": "surprise",
    "LABEL_6": "neutral",
}

def _normalize_label(label: str) -> str:
    label = label.lower().strip()
    return LABEL_INDEX_MAP.get(label.upper(), label)


# ── Color words that confuse the model ──────────────────────
_COLOR_WORDS = {
    "red", "blue", "black", "white", "green", "yellow",
    "orange", "purple", "pink", "brown", "grey", "gray"
}

# ── Love-related keywords ────────────────────────────────────
_LOVE_KEYWORDS = {
    "heart", "hearts", "love", "kiss", "kissing",
    "hug", "hugging", "blush", "blushing",
    "adore", "affection", "romantic", "couple"
}

# ── Joy-only keywords — never reclassify these as love ───────
_JOY_ONLY = {
    "tears", "laughing", "grinning", "party",
    "celebration", "tada", "popper"
}


def _clean_desc(raw: str) -> str:
    """
    Full cleaning pipeline for emoji descriptions:
    1. Remove colons
    2. Replace underscores with spaces
    3. Replace hyphens with spaces  <- KEY FIX: 'heart-eyes' -> 'heart eyes'
    4. Lowercase and strip
    5. Remove color words
    """
    desc = (raw
            .replace(":", "")
            .replace("_", " ")
            .replace("-", " ")
            .lower()
            .strip())
    words = [w for w in desc.split() if w not in _COLOR_WORDS]
    return " ".join(words) if words else desc


@lru_cache(maxsize=512)
def _classify_emoji_description(desc: str) -> tuple:
    """
    Classifies a cleaned emoji description using the trained model.
    Cached — each unique description classified only once per session.
    Returns (label, confidence).
    """
    if not desc:
        return ("neutral", 0.5)
    try:
        clf = _get_classifier()
        result = clf(desc)[0]
        if isinstance(result, list):
            result = result[0]
        label = _normalize_label(result["label"])
        conf  = float(result["score"])
        return (label, conf)
    except Exception:
        return ("neutral", 0.5)


def _get_emoji_description(e: str) -> str:
    data = emoji.EMOJI_DATA.get(e, {})
    raw  = data.get("en", "")
    return _clean_desc(raw)


def _is_love_emoji(desc: str) -> bool:

    words = set(desc.split())
    has_love     = bool(words & _LOVE_KEYWORDS)
    has_joy_only = bool(words & _JOY_ONLY)
    return has_love and not has_joy_only


def extract_emojis(text: str) -> list:
    return [ch for ch in text if ch in emoji.EMOJI_DATA]


def convert_emojis_to_text(text: str) -> str:
    return emoji.demojize(text, delimiters=(" ", " ")).strip()


POSITIVE_EMOTIONS = {"joy", "love", "surprise"}
NEGATIVE_EMOTIONS = {"sadness", "anger", "fear"}


def get_emoji_sentiment_score(text: str):
    emojis = extract_emojis(text)
    if not emojis:
        return None

    scores = []
    for e in emojis:
        desc = _get_emoji_description(e)
        label, conf = _classify_emoji_description(desc)
        if label in POSITIVE_EMOTIONS:
            scores.append(conf)
        elif label in NEGATIVE_EMOTIONS:
            scores.append(-conf)
        else:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else None


def get_dominant_emoji_emotion(text: str):

    emojis = extract_emojis(text)
    if not emojis:
        return None

    emotions       = []
    has_love_emoji = False

    for e in emojis:
        desc  = _get_emoji_description(e)
        label, _ = _classify_emoji_description(desc)

        if _is_love_emoji(desc):
            has_love_emoji = True
            emotions.append("love")
        else:
            emotions.append(label)

    if not emotions:
        return None

    if has_love_emoji:
        positive_count = sum(1 for e in emotions if e in POSITIVE_EMOTIONS)
        if positive_count >= len(emotions) / 2:
            return "love"

    return max(set(emotions), key=emotions.count)


def get_emoji_details(text: str) -> list:
    emojis = extract_emojis(text)
    if not emojis:
        return []

    details = []
    for e in emojis:
        desc = _get_emoji_description(e)
        if _is_love_emoji(desc):
            label, conf = "love", 0.85
        else:
            label, conf = _classify_emoji_description(desc)
        details.append({
            "emoji":       e,
            "description": desc,
            "emotion":     label,
            "confidence":  conf,
        })
    return details