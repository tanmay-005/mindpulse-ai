import pickle
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "mindpulse model")
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

with open(os.path.join(MODEL_PATH, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)


def predict_emotion(text: str):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)
    label = label_encoder.inverse_transform([predicted_class.item()])[0]

    return label, float(confidence)