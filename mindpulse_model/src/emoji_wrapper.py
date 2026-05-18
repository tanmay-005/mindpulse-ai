import re
from src.emoji_utils import (
    convert_emojis_to_text,
    get_emoji_sentiment_score,
    get_dominant_emoji_emotion,
    get_emoji_details,
    POSITIVE_EMOTIONS,
    NEGATIVE_EMOTIONS,
)
from src.slang import normalize_slang
from src.predict import predict_emotion as _predict_emotion
from src.sentiment import predict_sentiment as _predict_sentiment

# ── Emotion polarity maps ────────────────────────────────────
POSITIVE_LABEL = {"joy", "love", "surprise"}
NEGATIVE_LABEL = {"sadness", "anger", "fear"}

# ── Love guard — text must contain these for love to be valid ─
# Without romantic context in text, love prediction is overridden to joy
_LOVE_TEXT_KEYWORDS = {
    "love", "adore", "cherish", "darling", "sweetheart",
    "romantic", "affection", "devoted", "miss you",
    "care for", "fond", "crush", "beloved", "dear",
    "boyfriend", "girlfriend", "partner", "husband", "wife",
    "soulmate", "babe", "baby", "honey", "cuddle",
    "infatuated", "passionate", "tenderly", "heartfelt"
}

# ── Neutral phrases — whole word matching ────────────────────
NEUTRAL_PHRASES = [
    r"\baverage\b", r"\bokay\b", r"\bok\b", r"\bfine\b",
    r"\balright\b", r"\bso.so\b", r"\bnot bad\b", r"\bmediocre\b",
    r"\bmoderate\b", r"\bdecent\b", r"\btolerable\b", r"\bmeh\b",
    r"\bindifferent\b", r"\bneutral\b",
]

# ── Surprise phrases ─────────────────────────────────────────
SURPRISE_PHRASES = [
    r"\bhad no idea\b", r"\bdidn.t know\b", r"\bcan.t believe\b",
    r"\bno way\b", r"\bout of nowhere\b", r"\bnever expected\b",
    r"\bdidn.t expect\b", r"\bso unexpected\b", r"\bwhat a surprise\b",
    r"\bplot twist\b", r"\bshocked\b", r"\bblown away\b",
    r"\bflabbergasted\b", r"\bflabbergasting\b",
]

# ── Text-only sarcasm phrases (no emoji needed) ──────────────
SARCASM_PHRASES = [
    r"\bjust perfect\b", r"\bjust great\b", r"\bwow thanks\b",
    r"\boh great\b", r"\boh wonderful\b", r"\bsure sure\b",
    r"\byeah right\b", r"\bwow really\b", r"\boh how lovely\b",
    r"\btotally fine\b", r"\babsolutely perfect\b",
    r"\bjust what i needed\b", r"\boh fantastic\b",
    r"\bwow so helpful\b", r"\bgreat another\b",
    r"\banother delay\b", r"\banother problem\b",
    r"\banother issue\b", r"\banother monday\b",
]

# Compile all patterns once
_NEUTRAL_RE  = re.compile("|".join(NEUTRAL_PHRASES),  re.IGNORECASE)
_SURPRISE_RE = re.compile("|".join(SURPRISE_PHRASES), re.IGNORECASE)
_SARCASM_RE  = re.compile("|".join(SARCASM_PHRASES),  re.IGNORECASE)


def _text_is_neutral(text: str)  -> bool: return bool(_NEUTRAL_RE.search(text))
def _text_is_surprise(text: str) -> bool: return bool(_SURPRISE_RE.search(text))
def _text_is_sarcasm(text: str)  -> bool: return bool(_SARCASM_RE.search(text))


def _love_is_valid(text: str, emoji_emotion: str) -> bool:
    """
    Returns True only if love prediction is justified.
    Love requires EITHER romantic keywords in text OR love emojis present.
    Prevents strong positive sentences being misclassified as love.
    """
    # Love emoji present — valid
    if emoji_emotion == "love":
        return True
    # Romantic keywords in text — valid
    text_words = set(text.lower().split())
    if text_words & _LOVE_TEXT_KEYWORDS:
        return True
    return False


def _is_emoji_sarcasm(
    text_emotion: str, emoji_emotion: str, emoji_score: float
) -> bool:
    """
    Detects sarcasm via emoji contradiction:
    text positive + emoji strongly negative = sarcasm
    text negative + emoji strongly positive = sarcasm
    """
    if emoji_score is None:
        return False
    text_pos   = text_emotion in POSITIVE_LABEL
    emoji_neg  = emoji_score < -0.45
    text_neg   = text_emotion in NEGATIVE_LABEL
    emoji_pos  = emoji_score > 0.45
    return (text_pos and emoji_neg) or (text_neg and emoji_pos)


def predict_emotion(text: str):
    """
    Full emotion prediction pipeline:
    1.  Slang normalization
    2.  Extract emoji signals from ORIGINAL text
    3.  Convert emojis to text
    4.  DistilBERT prediction
    4b. Love guard — override love if no romantic context
    5.  Phrase overrides (neutral / surprise / sarcasm)
    6.  Emoji blending (sarcasm-aware)
    """
    # ── Step 1: Slang normalization ──────────────────────────
    text_normalized = normalize_slang(text)

    # ── Step 2: Extract emoji signals from ORIGINAL text ────
    emoji_score   = get_emoji_sentiment_score(text)
    emoji_emotion = get_dominant_emoji_emotion(text)

    # ── Step 3: Convert emojis to text ──────────────────────
    clean_text = convert_emojis_to_text(text_normalized)

    # ── Step 4: DistilBERT prediction ────────────────────────
    label, conf = _predict_emotion(clean_text)

    # ── Step 4b: Love guard ──────────────────────────────────
    # If model predicts love but there is no romantic context
    # in text AND no love emoji → override to joy
    if label == "love" and not _love_is_valid(clean_text, emoji_emotion):
        label = "joy"
        conf  = conf * 0.85

    text_lower = text.lower()

    # ── Step 5a: Text-only sarcasm override ──────────────────
    # Must check BEFORE neutral/surprise — sarcasm takes priority
    if _text_is_sarcasm(text_lower) and emoji_emotion is None:
        label = "anger"
        conf  = max(conf, 0.65)

    # ── Step 5b: Neutral override ────────────────────────────
    elif _text_is_neutral(text_lower) and emoji_emotion is None:
        label = "neutral"
        conf  = max(conf, 0.70)

    # ── Step 5c: Surprise override ───────────────────────────
    elif _text_is_surprise(text_lower) and emoji_emotion is None:
        label = "surprise"
        conf  = max(conf, 0.75)

    # ── Step 6: Emoji blending (sarcasm-aware) ───────────────
    if emoji_emotion is not None:

        # Emoji sarcasm — emoji wins
        if emoji_score is not None and _is_emoji_sarcasm(
            label, emoji_emotion, emoji_score
        ):
            label = emoji_emotion
            conf  = conf * 0.5

        # Emoji disagrees — override at low threshold
        elif emoji_emotion != label:
            if emoji_score is not None and abs(emoji_score) > 0.3:
                label = emoji_emotion
                conf  = conf * 0.65
            else:
                label = emoji_emotion
                conf  = conf * 0.7

        # Emoji agrees — boost confidence
        elif emoji_emotion == label:
            conf = min(conf + (1 - conf) * 0.35, 0.99)

    return label, conf


def predict_sentiment(text: str):
    """
    Full sentiment prediction pipeline with VADER + DistilBERT blending.
    """
    # ── Step 1: Slang normalization ──────────────────────────
    text_normalized = normalize_slang(text)

    # ── Step 2: Extract emoji signals ───────────────────────
    emoji_score = get_emoji_sentiment_score(text)

    # ── Step 3: Convert emojis to text ──────────────────────
    clean_text = convert_emojis_to_text(text_normalized)

    # ── Step 4: Sentiment prediction ────────────────────────
    label, score = _predict_sentiment(clean_text)

    text_lower = text.lower()

    # ── Step 5: Dampen neutral phrases ──────────────────────
    if _text_is_neutral(text_lower):
        score = score * 0.45

    # ── Step 5b: Sarcasm → force NEGATIVE ───────────────────
    if _text_is_sarcasm(text_lower) and emoji_score is None:
        label = "NEGATIVE"
        score = max(score, 0.70)

    # ── Step 6: Emoji blending ───────────────────────────────
    if emoji_score is not None:
        if emoji_score > 0.45 and label == "NEGATIVE":
            label = "POSITIVE"
            score = score * 0.5
        elif emoji_score < -0.45 and label == "POSITIVE":
            label = "NEGATIVE"
            score = score * 0.5
        elif (emoji_score > 0 and label == "POSITIVE") or \
             (emoji_score < 0 and label == "NEGATIVE"):
            score = min(score + (1 - score) * 0.25, 0.99)

    return label, score