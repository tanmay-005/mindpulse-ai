"""Step 0.6: pick the base encoder. Run on Google Colab with a GPU.

Fine-tunes each candidate model on star-rating-labelled app reviews (the
labels that worked best in 0.5) and scores it on your 200 hand-labelled reviews.

Colab:  !python -m scripts.p0_encoders
Quick test on any machine:  python -m scripts.p0_encoders --models distilbert-base-uncased --max-train 200 --epochs 1
"""
import argparse
import gc
import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["negative", "neutral", "positive"]
RATING_TO_LABEL = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}

CANDIDATES = [
    "l3cube-pune/hing-bert",            # BERT trained on Roman-script Hinglish
    "l3cube-pune/hing-roberta",         # RoBERTa trained on Roman-script Hinglish
    "l3cube-pune/hing-roberta-mixed",   # trained on Roman + Devanagari Hinglish
    "bert-base-multilingual-cased",     # multilingual, no Hinglish-specific training
    "distilbert-base-uncased",          # English only: shows what Hinglish pretraining adds
]


def load_data(max_train=0, seed=42):
    """Train = star-labelled scraped reviews. Test = your 200 hand labels. No overlap."""
    hand = pd.read_csv(ROOT / "data" / "feasibility" / "hand_200.csv").dropna(subset=["sentiment"])
    files = sorted(glob.glob(str(ROOT / "data" / "raw" / "sample_*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["content"] = df["content"].fillna("").astype(str)
    df = df[df["content"].str.split().str.len() >= 3]
    df = df[~df["review_id"].isin(hand["review_id"]) & ~df["content"].isin(hand["content"])]
    df = df.drop_duplicates("content")
    if max_train:
        df = df.sample(min(max_train, len(df)), random_state=seed)
    train_texts, train_y = df["content"].tolist(), df["rating"].map(RATING_TO_LABEL).tolist()
    test_texts = hand["content"].astype(str).tolist()
    test_y = [LABELS.index(s) for s in hand["sentiment"]]
    return train_texts, train_y, test_texts, test_y


def make_loader(tok, texts, labels, batch_size, shuffle):
    enc = tok(texts, truncation=True, max_length=128)
    items = [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
              "labels": labels[i]} for i in range(len(texts))]
    return DataLoader(items, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=DataCollatorWithPadding(tok))


def run_one(model_id, data, args, device):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_texts, train_y, test_texts, test_y = data

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=3).to(device)

    train_dl = make_loader(tok, train_texts, train_y, args.batch_size, shuffle=True)
    test_dl = make_loader(tok, test_texts, test_y, 64, shuffle=False)

    # class weights: rare classes (neutral) count more, like class_weight="balanced" in 0.5
    counts = np.bincount(train_y, minlength=3)
    weights = torch.tensor(len(train_y) / (3 * np.maximum(counts, 1)), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start = time.time()
    model.train()
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(**batch).logits
                loss = loss_fn(logits.float(), labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if step % 50 == 0:
                print(f"  epoch {epoch + 1} step {step}/{len(train_dl)} loss {loss.item():.3f}")

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop("labels")
            preds.extend(model(**batch).logits.argmax(-1).tolist())

    f1 = f1_score(test_y, preds, average="macro", zero_division=0)
    per_class = f1_score(test_y, preds, average=None, labels=[0, 1, 2], zero_division=0)
    print(classification_report(test_y, preds, labels=[0, 1, 2], target_names=LABELS, zero_division=0))

    del model, optimizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"macro_f1": round(float(f1), 3),
            **{f"f1_{name}": round(float(v), 3) for name, v in zip(LABELS, per_class)},
            "train_minutes": round((time.time() - start) / 60, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=CANDIDATES)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-train", type=int, default=0, help="use only N training reviews (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else
                                 "  <- no GPU: in Colab use Runtime > Change runtime type > T4 GPU"))
    data = load_data(args.max_train, args.seed)
    print(f"Train: {len(data[0])} star-labelled reviews | Test: {len(data[2])} hand-labelled reviews")

    out = ROOT / "data" / "feasibility" / "encoders.json"
    results = json.loads(out.read_text()) if out.exists() else {}
    for model_id in args.models:
        print(f"\n=== {model_id} ===")
        results[model_id] = run_one(model_id, data, args, device)
        print(results[model_id])
        out.write_text(json.dumps(results, indent=2))       # save after each model

    print("\nSUMMARY (macro-F1 on your 200 reviews; TF-IDF + LR on star labels was 0.677)")
    print(pd.DataFrame(results).T.sort_values("macro_f1", ascending=False).to_string())


if __name__ == "__main__":
    main()