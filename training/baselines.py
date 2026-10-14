"""Classical baselines every later model must beat (docs/ML_DESIGN.md section 2)."""
import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
REPEAT_RE = re.compile(r"(.)\1{2,}")


def light_clean(text):
    """Minimal cleanup for tweets and reviews (full normalisation comes in Phase 1)."""
    text = URL_RE.sub(" ", text or "")
    text = MENTION_RE.sub(" ", text)
    text = REPEAT_RE.sub(r"\1\1", text.lower())     # 'sooooo' -> 'soo'
    return " ".join(text.split())


def tfidf_lr(seed=42):
    """TF-IDF on words (1-2 grams) + characters (2-5 grams), then logistic regression.

    Character n-grams handle spelling variants like nahi / nhi / nai.
    """
    features = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=light_clean, ngram_range=(1, 2),
                                 min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=light_clean, analyzer="char_wb",
                                 ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ])
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=2.0, random_state=seed)
    return Pipeline([("features", features), ("clf", clf)])


def majority_class(labels):
    return Counter(labels).most_common(1)[0][0]


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))