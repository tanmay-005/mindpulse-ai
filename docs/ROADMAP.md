# Roadmap

Nine phases, about 15–17 weeks of part-time work. Each phase ends with a usable, demo-able result and clear exit criteria, so the project is presentable even if later phases slip. Durations are estimates.

| Phase | Name | Est. duration | Demo-able outcome |
|---|---|---|---|
| 0 | Feasibility check | 1 week | Real reviews scraped; baseline runs; go / no-go decision |
| 1 | Data pipeline | 1.5 weeks | Daily automated ingestion into PostgreSQL |
| 2 | Labels and gold set | 2 weeks | Frozen gold set; first labelled training data |
| 3 | Baselines and DL model | 3 weeks | Multi-task teacher beating baselines on gold |
| 4 | Student model and API | 1.5 weeks | Fast ONNX model behind FastAPI |
| 5 | Analytics | 1.5 weeks | Spike alerts and weekly issue clusters |
| 6 | Retraining loop | 2 weeks | First model promoted (or rejected) by the gate |
| 7 | Dashboard and deployment | 2 weeks | Public live dashboard |
| 8 | Polish and write-up | 1 week | Final README, results, resume bullets |

---

## Before Phase 0 — repository housekeeping

- [x] Tag the v1 code (it lives in `mindpulse_model/` on `main`) as `v1-emotion-classifier`.
- [x] Create a `v2` branch; add the new README and `docs/`. v1 files are removed from this branch and stay available under the tag.

---

## Phase 0 — Feasibility check (1 week)

**Goal:** prove the two riskiest assumptions before investing weeks.

- [ ] Verify Play Store package IDs for all six apps.
- [ ] Scrape ~2,000 recent reviews per app (`lang="en"` and `lang="hi"`); save to CSV.
- [ ] Measure: share of Hinglish vs English vs Devanagari; share with `reviewCreatedVersion`; duplicates; spam.
- [ ] Download SentiMix; train TF-IDF + LR sentiment baseline; test on 200 hand-labelled app reviews.
- [ ] Load 2–3 HingBERT checkpoints; quick zero-shot / few-shot sanity check; pick the base model.
- [ ] Write a one-page feasibility note in `docs/feasibility.md`.

**Exit criteria (go):** scraper reliably returns new reviews daily; at least ~30% of reviews are Hinglish or Hindi; baseline sentiment macro-F1 on app reviews is clearly above the majority class.

**No-go fallback:** if scraping proves unreliable, switch to the GitHub Issue Triage direction (public GH Archive data). Phases 3–8 carry over with a different data source and heads.

---

## Phase 1 — Data pipeline (1.5 weeks)

- [ ] Set up hosted PostgreSQL; apply the schema from `DATA_PLAN.md` §3.
- [ ] `ingestion/`: per-app incremental scraper with cursor, paging, back-off, page cap.
- [ ] `processing/`: cleaning, emoji tags, elongation, slang normalisation (written fresh; v1 used only as a reference for word lists), Devanagari transliteration, language-ID, spam filter, MinHash near-dup detection — each with unit tests.
- [ ] GitHub Actions workflow: daily ingest + process.
- [ ] Backfill: collect as much recent history as is politely possible for DAPT and trends.
- [ ] Parquet snapshot export script.

**Exit criteria:** workflow runs unattended for 5 consecutive days; no duplicate reviews; processing tests pass.

---

## Phase 2 — Labels and gold set (2 weeks)

- [ ] Write the labelling guide (`docs/labelling_guide.md`) with definitions and 5+ examples per label.
- [ ] Build a simple labelling tool (Streamlit page or Label Studio).
- [ ] Label the **gold set**: ~1,500 reviews from a held-out time window, stratified by app and rating.
- [ ] Re-label 200 after a week; report intra-annotator κ; fix the guide where it was ambiguous.
- [ ] **Freeze** `gold_v1`.
- [ ] Label 300 non-gold reviews by hand for LLM validation.
- [ ] Run LLM labelling on ~5,000 reviews (two passes); compute κ / F1 vs human labels per task; accept tasks with κ ≥ 0.6.
- [ ] Map GoEmotions to the 6 emotions; build the Hinglish-24k sentiment merge.

**Exit criteria:** gold set frozen; LLM label quality measured and documented; training data for all four tasks exists.

---

## Phase 3 — Baselines and DL model (3 weeks)

- [ ] Baselines: majority, VADER, TF-IDF + LR, TF-IDF + SVM, and English DistilBERT retrained on the v2 data (all four tasks) — all on gold, logged in MLflow.
- [ ] DAPT: continue MLM on unlabelled reviews (pre-gold window only).
- [ ] Multi-task teacher with the four heads, masked losses, trust weights, uncertainty task weighting.
- [ ] Noisy-label handling: label smoothing, confident learning on star-rating labels, small-loss filtering.
- [ ] Temperature scaling; ECE and reliability diagrams.
- [ ] Ablations A1–A8 (3 seeds each where compute allows; prioritise A1, A3, A4, A6).
- [ ] Error analysis on 100+ gold errors.
- [ ] Bootstrap confidence intervals for all headline numbers.

**Exit criteria:** teacher beats TF-IDF + LR on every task's macro-F1 on gold (or the shortfall is explained); ablation table and error analysis written up.

---

## Phase 4 — Student model and API (1.5 weeks)

- [ ] Distil the teacher into a smaller student (soft + hard targets, pseudo-labelled unlabelled pool).
- [ ] ONNX export + int8 quantisation; ablation A9 (teacher vs student).
- [ ] Occlusion-based word importance.
- [ ] Escalation scorer (LightGBM) with SHAP; precision@20/@50 on gold.
- [ ] FastAPI: `POST /predict`, `GET /health`, `GET /model`; Dockerfile; tests.
- [ ] Daily batch scoring job writes `predictions` with `model_version`.
- [ ] Register `v1` in MLflow and `model_versions`; upload to Hugging Face Hub.

**Exit criteria:** student within ~2 macro-F1 points of teacher; p95 latency ≤ 300 ms on CPU; daily scoring runs unattended.

---

## Phase 5 — Analytics (1.5 weeks)

- [ ] Daily aggregates job.
- [ ] Spike detection (rolling median/MAD, ratio + z-score + minimum count); version linking; alert open/close logic.
- [ ] Tune thresholds against spikes found in backfilled history; document examples.
- [ ] Frozen embedding model; weekly UMAP + HDBSCAN clustering of unexplained complaints; c-TF-IDF keywords; stable cluster IDs.
- [ ] Manual coherence check of clusters for 3 weeks.

**Exit criteria:** at least one real historical spike detected and correctly linked to a version (if one exists in the data); clusters judged coherent in most weeks.

---

## Phase 6 — Retraining loop (2 weeks)

- [ ] Drift monitoring job: embedding MMD, PSI, OOV rate → `drift_reports`.
- [ ] Active-learning selector (uncertainty, disagreement, diversity, new clusters, new versions) → `label_queue`.
- [ ] Admin labelling page; start the ~50/day labelling habit.
- [ ] Snapshot builder with replay buffer and contamination unit test.
- [ ] Promotion gate script with all checks from `RETRAINING.md` §5.
- [ ] Shadow-mode scoring; promote and rollback scripts.
- [ ] First full cycle: collect labels → retrain → gate → shadow → promote or reject.
- [ ] First `recent_YYYY_MM` evaluation set.

**Exit criteria:** one complete, logged retraining cycle with a documented gate decision.

---

## Phase 7 — Dashboard and deployment (2 weeks)

- [ ] Confirm current free-tier limits for the database, API and dashboard hosts.
- [ ] Deploy the API container.
- [ ] Build pages in the order from `DASHBOARD_SPEC.md` §5.
- [ ] Admin login; public suggestions vs trusted corrections.
- [ ] Footer disclaimers, sample sizes, consistent colours, mobile check.
- [ ] Deploy the dashboard; public URL in the README.

**Exit criteria:** public URL live; all eight pages load in under ~3 seconds; data refreshes daily without manual work.

---

## Phase 8 — Polish and write-up (1 week)

- [ ] Fill the README results table from MLflow (gold-set numbers with confidence intervals).
- [ ] Add screenshots / a short GIF of the dashboard.
- [ ] Architecture diagram image.
- [ ] Short technical blog post or `docs/REPORT.md`: problem, data, model, retraining, results, limitations.
- [ ] Resume bullets — written only from measured numbers.
- [ ] Clean-up: tests passing, linting, `requirements.txt` pinned.

**Exit criteria:** a stranger can understand the project from the README in 2 minutes and reproduce a model from the docs.

---

## Ongoing after launch

- ~15 minutes a day of labelling from the active-learning queue.
- Weekly: check alerts, clusters, drift report.
- Monthly: new `recent` evaluation set; LLM-label audit.
- Retrain when triggered; keep the Model Health page growing — a longer version history is the strongest proof the system works.

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Play Store markup change breaks the scraper | Medium | Pin the library version; alert on zero-review runs; keep backfilled data so the project still works offline |
| Too few Hinglish reviews in some apps | Medium | Measured in Phase 0; adjust tracked apps |
| Labelling takes longer than planned | High | Gold set first; LLM labels validated early; active learning to spend human time well |
| Free GPU limits slow training | Medium | Short max length (128), mixed precision, fewer seeds for minor ablations |
| Free hosting limits | Medium | Pre-computed aggregates; small ONNX student; confirm limits in Phase 7 |
| Model doesn't beat TF-IDF on some task | Low–medium | Report honestly; error analysis; it is still a valid finding |
