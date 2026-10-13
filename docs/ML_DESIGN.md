# ML Design

MindPulse uses deep learning where it is needed (understanding noisy Hinglish text), classical ML where it fits naturally (escalation scoring on structured features), and unsupervised learning where no labels exist yet (new-issue discovery). Every component is measured against a simpler baseline on the frozen gold test set.

---

## 1. Model stack at a glance

| # | Component | Type | Input | Output |
|---|---|---|---|---|
| 1 | Baselines | Classical ML | Normalised text | Per-task predictions to beat |
| 2 | Domain-adapted encoder | DL (self-supervised) | Unlabelled reviews | HingBERT adapted to app-review Hinglish |
| 3 | Multi-task teacher | DL (supervised) | Text | Sentiment, emotion, aspects, urgency + probabilities |
| 4 | Student model | DL (distillation) | Text | Same outputs, small and fast; ONNX int8 |
| 5 | Escalation scorer | Classical ML (LightGBM) | Model outputs + structured features | `urgency_score` for ranking the inbox |
| 6 | Embedding model (frozen) | DL (pretrained) | Text | Sentence vectors for clustering and drift |
| 7 | Issue discovery | Unsupervised | Embeddings of unexplained complaints | Weekly clusters + keywords |
| 8 | Spike detection | Statistical | Daily aggregates | Alerts |

---

## 2. Baselines (Phase 3, first)

Built before any transformer, so every later number has something to beat.

| Baseline | Tasks | Notes |
|---|---|---|
| Majority class | All | Floor; shows class imbalance |
| VADER (from v1) | Sentiment | English lexicon; expected to struggle on Hinglish — that is the point |
| TF-IDF (word 1–2 grams + char 2–5 grams) + Logistic Regression | All (one-vs-rest for aspects) | Strong, cheap; char n-grams handle spelling variants |
| TF-IDF + Linear SVM | All | Second classical reference |
| DistilBERT (English), retrained on the v2 training data | All | Transformer baseline; shows what a Hinglish-pretrained encoder adds |

If the transformer does not clearly beat TF-IDF + LR on a task, that is reported honestly and investigated.

---

## 3. Domain-adaptive pretraining (DAPT)

**Why:** HingBERT learned Hinglish from general social media. App reviews have their own vocabulary ("refund nahi aaya", "OTP", "COD", "rider", "UPI lite"). Continuing masked-language-model training on in-domain text usually improves downstream accuracy, and this project measures by how much.

| Setting | Starting value |
|---|---|
| Base model | HingBERT family from L3Cube (e.g. `l3cube-pune/hing-bert` or `hing-roberta`; exact checkpoint chosen in Phase 0 by quick comparison) |
| Data | All collected, non-spam reviews **before** the gold-set window (no test leakage) |
| Objective | Masked LM, 15% masking (whole-word masking if supported) |
| Length | Max 128 tokens (most reviews are short) |
| Epochs | 1–3, stop when validation perplexity plateaus |
| Compute | Colab/Kaggle GPU, mixed precision (fp16) |

Output: `mindpulse-encoder-dapt-vN`. The ablation table reports results with and without DAPT.

---

## 4. Multi-task teacher model

### 4.1 Architecture

```
            normalised review text
                     │
           Tokenizer (HingBERT)
                     │
     Encoder (DAPT HingBERT, 12 layers)
                     │
         mean-pooled hidden states
          ┌────────┬───────┼────────┬──────────┐
          ▼        ▼       ▼        ▼
     Sentiment  Emotion  Aspect   Urgency
     softmax(3) softmax(6) sigmoid(9) ordinal(4)
```

Each head is a small MLP (hidden 256, dropout 0.1). Urgency uses an **ordinal** head (cumulative logits, CORAL-style) because critical > high > medium > low is an ordering, not four unrelated classes.

### 4.2 Losses

| Head | Loss | Why |
|---|---|---|
| Sentiment | Cross-entropy with label smoothing (ε ≈ 0.1) | Much sentiment data is weak star-rating labels |
| Emotion | Focal loss (γ ≈ 2) with class weights | Strong imbalance (neutral/anger dominate) |
| Aspect | Binary cross-entropy with `pos_weight` per aspect | Multi-label, rare aspects |
| Urgency | Ordinal (CORAL) loss | Respects the ordering; penalises "critical→low" more than "critical→high" |

**Missing labels are masked.** Not every review has labels for every task (a SentiMix tweet has only sentiment). A task's loss is computed only on samples that have that task's label.

**Sample weights** = trust weight of the label source (`DATA_PLAN.md` §6.1).

**Task weighting:** start with learned uncertainty weighting (one learnable log-variance per task); compare against fixed equal weights in the ablations.

### 4.3 Training recipe (starting points)

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW, weight decay 0.01 |
| Learning rate | 2e-5 encoder, 1e-3 heads |
| Schedule | Linear warm-up 6%, linear decay |
| Batch size | 32 (gradient accumulation if memory-limited) |
| Epochs | 3–5, early stopping on validation mean macro-F1 |
| Max length | 128 tokens |
| Seeds | 3 seeds for final reported runs; mean ± std reported |

### 4.4 Learning from noisy labels

Star ratings disagree with review text fairly often (e.g. 5 stars with a complaint). Three techniques, each tested as an ablation:

1. **Label smoothing** on the weak-label task (above).
2. **Confident learning** — use out-of-fold predicted probabilities (e.g. with `cleanlab`) to find likely mislabelled weak examples; down-weight or drop them; report how many were found and spot-check 100 by hand.
3. **Small-loss filtering** — in later epochs, ignore the highest-loss fraction of weak-label samples, which are disproportionately label errors.

The rating-vs-text disagreement rate is itself shown on the dashboard Overview.

### 4.5 Calibration

Confidences drive the active-learning queue and dashboard confidence bars, so they must mean something. Temperature scaling per head is fitted on the validation set. Expected Calibration Error (ECE) and reliability diagrams are reported before and after.

---

## 5. Student model and serving

| Step | Detail |
|---|---|
| Student architecture | A smaller encoder (e.g. a 6-layer model initialised from the teacher, or a small multilingual MiniLM) with the same four heads |
| Distillation data | All labelled data + a large pool of unlabelled reviews pseudo-labelled by the teacher |
| Distillation loss | KL divergence to teacher soft targets (temperature T ≈ 2) + hard-label loss where labels exist |
| Export | ONNX, dynamic int8 quantisation |
| Acceptance | Student within ~2 macro-F1 points of the teacher on gold per task, and p95 latency under 300 ms on CPU |

The teacher is used for training and analysis; the student is what runs in production.

**Explanations in serving:** the student runs in ONNX (no gradients), so token importance is computed by **occlusion** — re-score the review with each word masked and measure the drop in the predicted class probability. Reviews are short, so this is cheap. Integrated Gradients on the PyTorch teacher is used in offline error analysis.

---

## 6. Escalation scorer (LightGBM)

Ranks the support inbox. The transformer's urgency head reads only the text; real triage also depends on context.

| Feature group | Examples |
|---|---|
| Text-model outputs | Urgency probabilities, P(anxiety), P(anger), P(payment), P(refund), P(account_login) |
| Money signals | ₹ amounts mentioned, money phrases ("paisa kat gaya", "debit ho gaya", "refund nahi") |
| Review metadata | Star rating, length, thumbs-up count, has the company replied, hours since posted |
| Version context | Days since this app version first appeared, current alert active for this aspect |

- **Target:** high/critical urgency (binary) from human, correction and gold-quality labels only — never trained on its own previous predictions.
- **Evaluation:** precision@20 and precision@50 on the gold set (how many of the top-ranked reviews are truly urgent), plus ROC-AUC.
- **Explanation:** SHAP values per review, shown as "why it's urgent" in the inbox.

---

## 7. Embeddings, issue discovery and drift

### 7.1 A frozen embedding model

Clustering and drift use a **fixed** pretrained multilingual sentence-embedding model (e.g. a multilingual MiniLM from sentence-transformers), not the retrained encoder. If the embedding model changed with every retrain, drift scores would mix "the data changed" with "the model changed".

### 7.2 Weekly issue discovery

1. Select last week's negative reviews where no aspect probability exceeds 0.5 ("unexplained complaints").
2. Embed → UMAP to ~5 dimensions → HDBSCAN (min cluster size ~15).
3. Keywords per cluster via c-TF-IDF; three representative reviews nearest the centroid.
4. Match to last week's clusters by centroid cosine similarity (> ~0.8) to keep stable IDs; compute growth.
5. A cluster can be **promoted** to a new aspect from the dashboard, which adds it to the taxonomy and queues its reviews for labelling.

Quality is checked by hand each week in early phases (are clusters coherent?), and reported with a silhouette score as a rough guide.

### 7.3 Spike detection

For each app × aspect, the daily rate is negative aspect mentions divided by total reviews that day.

- **Baseline:** rolling 28-day median and median absolute deviation (MAD).
- **Alert when all hold:** observed/expected ratio ≥ 2, robust z-score ≥ 3, and at least ~15 mentions (avoids alerts on tiny counts).
- **Version linking:** if most spike reviews come from a version released in the last few days, the alert names that version.
- **Close** after 3 consecutive normal days.

Thresholds are starting values, tuned against obvious real spikes found in historical data during Phase 5.

---

## 8. Evaluation protocol

| What | How |
|---|---|
| Primary metric | Macro-F1 per task on the frozen gold set |
| Aspects | Micro-F1 and macro-F1, plus per-aspect F1 |
| Urgency | Macro-F1 + mean absolute error on the ordinal scale |
| Uncertainty | 95% bootstrap confidence intervals (1,000 resamples of the gold set) |
| Slices | Per app, per `lang_mix` (Hinglish / English / transliterated Hindi), per rating, short vs long reviews |
| Calibration | ECE per head, reliability diagrams |
| Serving | p50/p95 latency on CPU, model size |
| Generalisation | Leave-one-app-out: train without one app, test on it |

Accuracy is reported only as a secondary number; with this much imbalance it hides failures on rare classes.

---

## 9. Ablation plan

Each row removes or swaps one thing, everything else held fixed, 3 seeds each.

| # | Variant | Question it answers |
|---|---|---|
| A0 | Full teacher | Reference |
| A1 | No DAPT (plain HingBERT) | How much does domain-adaptive pretraining help? |
| A2 | Multilingual BERT instead of HingBERT | Does Hinglish pretraining matter? |
| A3 | Four separate single-task models | Does multi-task learning help or hurt each task? |
| A4 | No noisy-label handling | Do smoothing / confident learning / filtering help? |
| A5 | No weak star-rating labels | Do noisy-but-plentiful labels help at all? |
| A6 | No LLM labels (human only) | What do LLM-assisted labels add? |
| A7 | No text normalisation | Does the slang/emoji/script layer still matter with a transformer? |
| A8 | Equal task weights | Does uncertainty weighting help? |
| A9 | Student vs teacher | What does distillation + quantisation cost in accuracy and gain in speed? |

---

## 10. Error analysis

After each major model version, review at least 100 gold-set errors and tag each with a cause:

- sarcasm ("wah kya service hai, 2 ghante late 👏")
- mixed sentiment in one review
- Devanagari / transliteration problems
- rare aspect confusion (refund vs payment)
- label ambiguity (the guide itself needs clarifying)
- very short reviews

Counts per cause go into the README's results section. The guide and taxonomy are updated when ambiguity is the cause.

---

## 11. Experiment tracking conventions

- Every run logs to MLflow: git commit, data `snapshot_id`, config file, seeds, all metrics, confusion matrices, calibration plots.
- Run names: `<component>-<variant>-<date>` (e.g. `teacher-A1-noDAPT-2026-11-20`).
- Configs in `configs/train/*.yaml`; no hyperparameters hard-coded in notebooks.
- A model version is reproducible from its run ID alone.
