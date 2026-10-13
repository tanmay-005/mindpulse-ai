# Architecture

MindPulse AI is a batch-plus-online system: reviews are collected and scored on a schedule, while a FastAPI service and a Streamlit dashboard serve results and live predictions. Training runs separately on free GPU notebooks, and new models reach production only through a promotion gate.

---

## 1. Design principles

1. **Free to run.** Every component fits a free tier or runs on a laptop / free GPU notebook.
2. **Training and serving are separate.** GPUs are needed only for training; inference runs on CPU via a distilled, quantised ONNX model.
3. **Nothing ships without evidence.** A model is promoted only after beating the current one on a frozen gold test set.
4. **Every prediction is traceable.** Each stored prediction records the model version that produced it.
5. **Config over code.** Tracked apps, label taxonomy and thresholds live in `configs/`, not in code.

---

## 2. Component overview

```
                              ┌──────────────────────────────────────────────┐
                              │              SCHEDULED (GitHub Actions)        │
 Google Play ──► [1] Ingestion ──► [2] Processing ──► [3] Batch inference ──► [4] Analytics jobs
   (public)        scraper,          dedup, slang/      ONNX student          spikes, clusters,
                   cursor per app    emoji/script       model                 daily aggregates
                              └───────────┬──────────────────┬────────────────────┬────────────┘
                                          ▼                  ▼                    ▼
                                  ┌──────────────────── [5] PostgreSQL ────────────────────┐
                                  │ apps · reviews · predictions · labels · label_queue ·   │
                                  │ model_versions · alerts · clusters · daily_aggregates · │
                                  │ drift_reports                                            │
                                  └───────┬──────────────────────┬──────────────────┬──────┘
                                          │                      │                  │
                         [6] Active-learning queue      [9] FastAPI service   [10] Streamlit dashboard
                                          │                      ▲                  │
                                          ▼                      │        label corrections
                         [7] Training pipeline (Colab/Kaggle GPU)│                  │
                          DAPT → multi-task → distil → ONNX      │                  ▼
                                          │                      │           back to [5] labels
                                          ▼                      │
                         [8] Promotion gate + Model registry ────┘
                          (gold-set eval, MLflow, HF Hub)
```

| # | Component | Responsibility | Runs on |
|---|---|---|---|
| 1 | Ingestion | Pull newest reviews per tracked app; resume from last seen review | GitHub Actions (daily) |
| 2 | Processing | Deduplicate, normalise slang/emoji/script, language-ID, filter spam | GitHub Actions |
| 3 | Batch inference | Score new reviews with the current champion model | GitHub Actions (CPU) |
| 4 | Analytics jobs | Daily aggregates, spike detection, weekly issue clustering, urgency ranking | GitHub Actions |
| 5 | PostgreSQL | System of record | Hosted free tier |
| 6 | Active-learning queue | Select reviews whose labels would help the model most | Part of jobs 3–4 |
| 7 | Training pipeline | Baselines, domain-adaptive pretraining, multi-task fine-tuning, distillation, ONNX export | Colab / Kaggle GPU |
| 8 | Promotion gate + registry | Compare challenger vs champion; register versions and metrics | Training notebook + MLflow |
| 9 | FastAPI service | `/predict`, `/health`, `/model` endpoints | Docker container |
| 10 | Dashboard | Eight pages; label-correction UI | Streamlit host |

---

## 3. Data flow, step by step

### 3.1 Daily ingestion and scoring

1. **Fetch.** For each app in `configs/apps.yaml`, call `google_play_scraper.reviews(app_id, lang, country="in", sort=Sort.NEWEST, count=N)` for both `lang="en"` and `lang="hi"`, paging with the continuation token until a review older than the stored cursor is reached.
2. **Store raw.** Insert into `reviews` with `review_id` as the unique key (duplicates ignored). Reviewer names are discarded at this step.
3. **Process.** Normalise text (see `DATA_PLAN.md` §4), detect language mix, flag spam / empty / emoji-only reviews.
4. **Score.** Run the champion ONNX model on new reviews; write to `predictions` with the `model_version`.
5. **Queue.** Low-confidence and high-disagreement reviews go to `label_queue` (see `RETRAINING.md` §3).
6. **Aggregate.** Update `daily_aggregates` per app × aspect × sentiment × version.
7. **Detect spikes.** Compare today's aspect rates with the rolling baseline; write `alerts`.
8. **Weekly:** cluster unexplained negative reviews into candidate new issues (`clusters`).

### 3.2 Retraining (triggered, not continuous)

1. A trigger fires (schedule + enough new labels, drift, or gold-set decline — see `RETRAINING.md` §2).
2. Training notebook pulls labelled data + replay buffer from PostgreSQL / Parquet snapshots.
3. Trains the challenger (teacher → student → ONNX).
4. Evaluates challenger and champion on the frozen gold set.
5. If the gate passes: register the version in MLflow, upload the ONNX file to a Hugging Face Hub model repo with a version tag, and run it in shadow mode for a week alongside the champion.
6. After the shadow week, update `model_versions.is_champion`; the next scheduled scoring run and the API pick up the new champion automatically. Failed or rejected challengers are still registered, with the reason.

### 3.3 Online requests

- **API:** `POST /predict` → normalise → ONNX inference → returns the four predictions, confidences and top contributing tokens.
- **Dashboard:** reads aggregates and alerts from PostgreSQL; the *Try it live* page calls the API; admin label corrections are written to `labels` with `source = "correction"`, while public visitors' corrections are stored as `source = "suggestion"` and count only after admin approval.

---

## 4. Deployment

| Piece | Where | Notes |
|---|---|---|
| Scheduled jobs | GitHub Actions cron | Scraping, scoring, aggregates, spike detection; well within free minutes at this volume |
| Database | Hosted PostgreSQL free tier (e.g. Supabase or Neon) | Free hosts often wipe local files, so SQLite is not used in production |
| Model files | Hugging Face Hub model repo | Versioned by tag; downloaded at job / container start |
| Experiment tracking | MLflow (local or a free hosted MLflow server) | Training runs, metrics, artifacts |
| API | Docker container on a free host (e.g. Hugging Face Spaces, Render) | CPU only |
| Dashboard | Streamlit Community Cloud or Hugging Face Spaces | Public link for recruiters |
| Training | Google Colab / Kaggle notebooks | GPU only when retraining |

Free-tier limits change; confirm current limits before deploying (Roadmap Phase 7).

---

## 5. Key technical decisions

| Decision | Choice | Reason |
|---|---|---|
| Base encoder | HingBERT (Hinglish-pretrained BERT) | Pretrained on real Roman-script code-mixed text; plain DistilBERT is English-only |
| One model or four | One shared encoder, four task heads | Shared representation, one forward pass, cheaper serving |
| Serving model | Distilled student, ONNX, int8 quantised | Runs on free CPU hosts with acceptable latency |
| Database | PostgreSQL | Durable on free hosts; real SQL for analytics |
| Retraining cadence | Trigger-based, at most weekly | Free GPU time is limited; avoids churning models on noise |
| LLM role | Labeling assistant only | Keeps predictions reproducible, cheap and fully owned |
| Spike detection | Statistical baseline per app × aspect | Explainable, works with small daily counts |

---

## 6. Non-functional targets

| Target | Value |
|---|---|
| Ingestion freshness | Reviews on the dashboard within 24 hours |
| API latency (CPU, single review) | p95 under 300 ms |
| Daily volume handled | Up to ~10,000 new reviews across all apps |
| Reproducibility | Every model version reproducible from its MLflow run + data snapshot ID |
| Cost | ₹0 per month on free tiers |

These targets are goals to verify during development, not measured results.

---

## 7. Repository layout

See the README for the full tree. Each top-level folder maps to one component above: `ingestion/` (1), `processing/` (2), `serving/` and `training/` (3, 7, 9), `analytics/` (4), `db/` (5), `retraining/` (6, 8), `dashboard/` (10).
