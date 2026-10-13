# MindPulse AI — Hinglish App-Review Intelligence

> A continually learning NLP system that reads live Google Play reviews of Indian apps — most of them Hinglish, slang-heavy and emoji-filled — and tells product, support and release teams what users are angry about, how urgent it is, and what changed since the last app update.

**Status:** 🚧 In development (v2 rebuild). v1 — a general-purpose emotion classifier — is preserved under the `v1-emotion-classifier` tag.

---

## Why this exists

Indian consumer apps receive thousands of Play Store reviews every day. Most are written in **Hinglish** (Hindi–English code-mixed, usually in Roman script), full of slang and emojis:

> *"bhai refund 5 din se nahi aaya, paisa kat gaya order cancel 😡 customer care bakwas"*

Off-the-shelf sentiment tools are trained on clean English and misread this text. Teams that need to react — to a broken release, a payment outage, a refund backlog — find out late, from social media.

MindPulse AI turns that review stream into decisions:

| Who | Question it answers |
|---|---|
| Release / QA engineers | Did the latest version break something? |
| Support leads | Which reviews need a reply first? |
| Product managers | What should we fix next, and is it getting better? |
| Competitive analysts | How do we compare with a rival in the same category? |

## What it does

- **Understands Hinglish** — slang, emoji and script normalisation, then a transformer fine-tuned on code-mixed text.
- **Tags every review** with four predictions from one multi-task model:
  - **Sentiment** — positive / neutral / negative
  - **Emotion** — joy, anger, disappointment, anxiety, surprise, neutral
  - **Aspect** (multi-label) — payment, refund, delivery, order accuracy, app performance, UI/UX, customer support, pricing & offers, account & login
  - **Urgency** — low / medium / high / critical
- **Detects spikes** — e.g. payment complaints jumping 3× after a new release.
- **Discovers new issues** — clusters complaints that don't fit any known aspect yet.
- **Keeps learning** — collects new labels every day, detects drift, retrains, and promotes a new model **only if it beats the current one** on a frozen, human-labelled test set.
- **Shows it all live** — a public dashboard tracking real apps, including a model-health page with every model version's metrics.

## Tracked apps (v2)

Six apps, in same-category pairs so comparisons are meaningful:

| Category | Apps |
|---|---|
| Food delivery | Zomato, Swiggy |
| Payments | PhonePe, Paytm |
| Quick commerce | Blinkit, Zepto |

Adding an app is a one-line config change. All figures shown for these apps are **model estimates from public reviews**, not official company data.

## Architecture at a glance

```
Play Store ──► Ingestion (daily, GitHub Actions) ──► Cleaning & normalisation ──► PostgreSQL
                                                                              │
              ┌─────────────── Inference (ONNX student model) ◄──────────────┤
              ▼                                                               │
       Predictions ──► Spike detection · Issue clustering · Urgency ranking ──┤
              │                                                               │
              ▼                                                               ▼
   Active-learning queue ──► Labels ──► Retraining (Colab/Kaggle GPU) ──► Promotion gate ──► Model registry
                                                                                               │
                                     FastAPI service  ◄────────────────────────────────────────┘
                                     Streamlit dashboard
```

Full details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Tech stack

| Layer | Tools |
|---|---|
| Data collection | `google-play-scraper`, GitHub Actions (scheduled) |
| Storage | PostgreSQL (hosted free tier), Parquet snapshots |
| NLP / DL | PyTorch, Hugging Face Transformers, HingBERT, sentence-transformers |
| Classical ML | scikit-learn, LightGBM |
| Unsupervised | UMAP, HDBSCAN, c-TF-IDF |
| Experiment tracking | MLflow |
| Serving | FastAPI, ONNX Runtime, Docker |
| Dashboard | Streamlit, Plotly |
| Training compute | Google Colab / Kaggle GPUs |

## Documentation

| Document | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Components, data flow, deployment, tech choices |
| [`docs/DATA_PLAN.md`](docs/DATA_PLAN.md) | Data sources, schema, label taxonomy, labeling strategy, gold test set |
| [`docs/ML_DESIGN.md`](docs/ML_DESIGN.md) | Models, training, evaluation, ablations |
| [`docs/RETRAINING.md`](docs/RETRAINING.md) | Continual-learning loop, drift detection, promotion gate |
| [`docs/DASHBOARD_SPEC.md`](docs/DASHBOARD_SPEC.md) | Every dashboard page and what it shows |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Phased plan with tasks and exit criteria |

## Results

To be filled in as phases complete. Every number here will come from the frozen gold test set (time-held-out, human-labelled) and will be compared against classical baselines.

| Task | Baseline (TF-IDF + LR) | MindPulse | Metric |
|---|---|---|---|
| Sentiment | — | — | Macro-F1 |
| Emotion | — | — | Macro-F1 |
| Aspect | — | — | Micro / Macro-F1 |
| Urgency | — | — | Macro-F1 |

## Planned repository structure

```
mindpulse-ai/
├── README.md
├── docs/                    # design documents
├── configs/                 # tracked apps, label taxonomy, thresholds
├── ingestion/               # scraper, dedup, scheduling
├── processing/              # normalisation (slang, emoji, script), language ID
├── labeling/                # weak labels, LLM-assisted labeling, gold set tools
├── training/                # baselines, DAPT, multi-task model, distillation
├── evaluation/              # gold-set eval, calibration, ablations
├── retraining/              # drift detection, AL queue, promotion gate
├── analytics/               # spike detection, issue clustering
├── serving/                 # FastAPI app, ONNX export
├── dashboard/               # Streamlit app
├── db/                      # schema and migrations
├── notebooks/               # exploration and training notebooks
├── tests/
├── .github/workflows/       # scheduled ingestion and inference
├── requirements.txt
└── Dockerfile
```

## Responsible use

- Only **public** reviews are collected, at low volume, for non-commercial portfolio and research purposes.
- Reviewer names are **not stored**; only review text, rating, date and app version.
- Dashboard figures are labelled as model estimates and shown with sample sizes and confidence.
- LLMs may assist with labeling; every LLM label is checked against human labels, and LLMs never make the production predictions.

## Author

**Tanmay Rana** — [GitHub](https://github.com/tanmay-005) · [LinkedIn](https://linkedin.com/in/tanmay-rana-13a7b4249)
