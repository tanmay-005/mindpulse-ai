"""
src/sentiment.py
Dual sentiment: VADER + DistilBERT blended for best accuracy.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

_vader = SentimentIntensityAnalyzer()

_distilbert_sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=512,
)


def predict_sentiment(text: str):
    """
    Blends VADER and DistilBERT:
    - Both agree       → boosted confidence
    - VADER very strong (>0.7) → trust VADER
    - Disagree         → trust DistilBERT
    - VADER neutral    → trust DistilBERT alone
    """
    vader_scores   = _vader.polarity_scores(text)
    vader_compound = vader_scores["compound"]

    if vader_compound >= 0.05:
        vader_label = "POSITIVE"
        vader_conf  = vader_compound
    elif vader_compound <= -0.05:
        vader_label = "NEGATIVE"
        vader_conf  = abs(vader_compound)
    else:
        vader_label = None
        vader_conf  = 0.0

    result     = _distilbert_sentiment(text)[0]
    bert_label = result["label"]
    bert_conf  = result["score"]

    if vader_label is None:
        return bert_label, bert_conf

    if vader_label == bert_label:
        blended = min((vader_conf + bert_conf) / 2 + 0.1, 0.99)
        return bert_label, blended

    if vader_conf > 0.7:
        return vader_label, vader_conf

    return bert_label, bert_conf