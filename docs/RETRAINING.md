# Retraining and Continual Learning

MindPulse keeps learning from new reviews, but a new model replaces the current one only when it is measurably better on data it has never seen. This document defines the loop, what triggers it, how new labels are chosen, and the safeguards that stop the model from learning its own mistakes.

---

## 1. The loop

```
 ┌────────────────────────────────────────────────────────────────────────────┐
 │                                                                            │
 ▼                                                                            │
New reviews ──► Champion predicts ──► Active-learning selection ──► Labels    │
 (daily)          + confidences        (uncertain, diverse, new)    (human,   │
                                                                     correction,
                                                                     LLM, rating)
                                                                        │     │
          Drift monitor ──► Trigger? ──yes──► Train challenger ◄────────┘     │
          (weekly)                             (new labels + replay buffer)   │
                                                        │                     │
                                                        ▼                     │
                                               Promotion gate (gold set)      │
                                                │                 │           │
                                             fails             passes         │
                                                │                 ▼           │
                                     log + keep champion    Shadow week ──► Promote ─┘
                                                                     (rollback ready)
```

1. **Predict.** The champion scores every new review and stores its confidence.
2. **Select.** Active learning picks the reviews whose labels will teach the model most (§3).
3. **Label.** Labels arrive from the human queue, dashboard corrections, validated LLM labelling, and star ratings, each with a trust weight.
4. **Monitor.** Drift and performance are checked weekly (§2).
5. **Train.** When a trigger fires, a challenger is trained on new labels plus a replay buffer of older data (§4).
6. **Gate.** The challenger must beat the champion on the frozen gold set without regressing anywhere important (§5).
7. **Shadow, then promote.** The challenger runs silently alongside the champion for a week before switching (§6).
8. **Roll back** instantly if production signals go wrong (§7).

---

## 2. Retraining triggers

Checked once a week. At most one retrain per week (free GPU time is limited, and frequent swaps would make trends hard to read).

| Trigger | Condition (starting threshold) | Signal source |
|---|---|---|
| Enough new labels | ≥ 500 new trusted labels (`human`, `correction`, accepted `llm`) since the last training run | `labels` table |
| Embedding drift | This week's embedding drift score above the 95th percentile of historical week-to-week values | `drift_reports.embedding_drift` |
| Prediction drift | PSI > 0.2 on the predicted sentiment or aspect distribution vs the training period | `drift_reports.psi_by_feature` |
| Vocabulary drift | Out-of-vocabulary rate after normalisation up by more than 50% relative to the training period | `drift_reports.oov_rate` |
| Performance decay | Macro-F1 on the latest `recent_YYYY_MM` set drops > 3 points below its value at promotion time | Monthly recent set |
| Taxonomy change | A new aspect promoted from issue clustering | `clusters.promoted_to` |

**Embedding drift** is computed with the frozen embedding model (`ML_DESIGN.md` §7.1): the maximum mean discrepancy (MMD) between a sample of this week's review embeddings and a sample from the champion's training data. The threshold comes from the project's own history, not a fixed number.

Drift alone does not mean the model is worse; it means the data moved. Retraining on drift still has to pass the gate.

---

## 3. Active learning

Human labelling time is the scarcest resource. Target budget: about **50 reviews per day** of human labelling (adjustable).

### 3.1 Selection strategies

| Strategy | How it picks | Share of daily budget |
|---|---|---|
| Uncertainty | Smallest margin between the top two classes (sentiment, emotion, urgency) or aspect probabilities closest to 0.5 | 40% |
| Disagreement | Teacher and student disagree, or the LLM label disagrees with the model | 20% |
| Diversity | k-center sampling over embeddings among uncertain reviews, so the batch isn't 50 near-identical complaints | 20% |
| New clusters | Reviews from this week's emerging-issue clusters | 10% |
| New app versions | Reviews from versions released in the last 7 days | 10% |

Selected reviews enter `label_queue` with a reason and priority.

### 3.2 Labelling interface

- **Admin labelling page** (dashboard, login-protected): shows the review, the model's predictions, and one-click buttons for each task.
- Keyboard shortcuts, target pace about 15 seconds per review (≈ 12–15 minutes per day).
- Queued items that receive no label within 14 days expire, so the queue reflects current data.

---

## 4. Training data assembly

Each retrain builds a dataset snapshot (`snapshot_id`), saved as Parquet, containing:

| Part | Content | Why |
|---|---|---|
| New labels | All trusted labels since the last run, with trust weights | Learn new patterns |
| Replay buffer | A stratified sample of older labels (by app, class and month), about 40% of the training set | Prevent catastrophic forgetting of older patterns |
| Bootstrap data | SentiMix / Hinglish-24k, mapped GoEmotions, at reduced weight | Stable foundation |
| Weak labels | Star-rating sentiment for recent reviews, weight 0.3 | Volume |

Excluded always: the gold set, the current `recent_YYYY_MM` evaluation set, spam, and near-duplicates of excluded items.

**Forgetting check:** the challenger is also evaluated on older label slices (by month). A drop of more than 3 macro-F1 points on any older month is flagged at the gate.

---

## 5. Promotion gate

All conditions must hold. Evaluated on the frozen gold set unless stated.

| # | Check | Pass condition |
|---|---|---|
| 1 | Overall improvement | Mean macro-F1 across the four tasks ≥ champion + 0.5 points, and the paired bootstrap shows the challenger ahead in ≥ 90% of resamples |
| 2 | No task regression | No single task drops by more than 1.0 macro-F1 point |
| 3 | No slice regression | No app and no language-mix slice drops by more than 3 points |
| 4 | Recent data | Not worse than the champion on the current `recent_YYYY_MM` set |
| 5 | Calibration | ECE per head not worse by more than 0.02 |
| 6 | Escalation | Inbox precision@20 not worse |
| 7 | Forgetting | No older monthly slice drops by more than 3 points (§4) |
| 8 | Serving | Student p95 latency ≤ 300 ms on CPU; model size within the hosting budget |

Every gate decision — pass or fail — is written to `model_versions.promotion_note` and shown on the dashboard's Model Health page. Rejected challengers are kept in the registry, which shows the gate actually rejects models.

Thresholds are starting values; any change to them is recorded in this file with a date and reason.

---

## 6. Shadow week and promotion

1. The challenger scores all new reviews for 7 days alongside the champion; its predictions are stored but not shown.
2. Compared: prediction distributions, alerts each model would have raised, and the reviews where they disagree.
3. Tanmay reviews 30 random disagreements by hand. If the challenger is right in most of them, promote.
4. Promotion = set `is_champion`, tag the Hugging Face Hub revision, and point the scoring job and API at it. Historical predictions keep their original `model_version`.

---

## 7. Rollback

- The previous champion stays available; rollback is a single config change (`configs/serving.yaml: champion_version`).
- **Roll back if**, within two weeks of promotion: the dashboard correction rate doubles, alerts fire on more than twice the usual number of app × aspect pairs without a real event, or `recent` performance drops > 3 points.
- Every rollback is logged with its reason.

---

## 8. Safeguards against feedback loops

| Risk | Safeguard |
|---|---|
| Model learns its own mistakes | Model predictions are never used as training labels for the next model. Teacher pseudo-labels are used only to distil the student within one version. |
| Public users poisoning labels | The public dashboard's "correct this label" button stores **suggestions** with `source = "suggestion"`; they become trusted only after admin approval. Only the logged-in admin's corrections count directly. |
| LLM labels drifting in quality | Every month, 100 fresh LLM labels are compared with human labels; if κ falls below 0.6 for a task, that task's LLM labels are paused. |
| Test-set contamination | Gold and recent sets are excluded by `review_id` and near-duplicate check in every snapshot build; a unit test enforces it. |
| Selection bias from active learning | Uncertainty-selected labels skew toward hard cases, so the replay buffer and a small random-sample share keep the training distribution realistic. |
| Silent degradation | Monthly `recent` set evaluation, even when no retrain is triggered. |

---

## 9. Retraining runbook

1. Weekly trigger job writes `drift_reports` and, if triggered, opens a GitHub issue "Retrain requested: <reasons>".
2. Run `notebooks/retrain.ipynb` on Colab/Kaggle: builds the snapshot, trains teacher → student → ONNX, evaluates, logs to MLflow.
3. Notebook runs the gate (`retraining/gate.py`) and prints the decision table.
4. If passed: upload to Hub as a candidate, set `shadow_version` in config.
5. After the shadow week: run `retraining/promote.py`, which updates the database and config.
6. Update the README results table if metrics changed.
