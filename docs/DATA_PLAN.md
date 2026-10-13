# Data Plan

This document defines where MindPulse's data comes from, how it is stored and cleaned, what the labels mean, and how labels are produced and trusted. The single most important artefact here is the **gold test set** (§7): every model decision in the project is made against it.

---

## 1. Data sources

### 1.1 Live stream — Google Play reviews

| Item | Detail |
|---|---|
| Library | [`google-play-scraper`](https://pypi.org/project/google-play-scraper/) (`pip install google-play-scraper`, no API key) |
| Call | `reviews(app_id, lang=..., country="in", sort=Sort.NEWEST, count=200, continuation_token=...)` |
| Languages | Fetched with `lang="en"` and `lang="hi"`; Hinglish appears under both |
| Frequency | Once a day per app |
| Volume control | Stop paging once reviews are older than the stored cursor; cap pages per run |

**Tracked apps** (`configs/apps.yaml`):

| Category | App | Play Store package ID |
|---|---|---|
| Food delivery | Zomato | `com.application.zomato` |
| Food delivery | Swiggy | `in.swiggy.android` |
| Payments | PhonePe | `com.phonepe.app` |
| Payments | Paytm | `net.one97.paytm` |
| Quick commerce | Blinkit | `com.grofers.customerapp` |
| Quick commerce | Zepto | `com.zeptoconsumerapp` |

Zepto's ID is confirmed from its Play Store listing. The other five IDs must be verified against their Play Store URLs in Phase 0 before the first scrape.

**Fields kept:** `reviewId`, `content`, `score`, `at`, `reviewCreatedVersion`, `thumbsUpCount`, `replyContent`, `repliedAt`.
**Fields dropped at ingestion:** `userName`, `userImage`. No personal identifiers are stored.

### 1.2 Bootstrap datasets (public, labelled)

| Dataset | What it gives | Used for |
|---|---|---|
| [SemEval-2020 Task 9 (SentiMix), Hinglish](https://arxiv.org/abs/2008.04277) | Hinglish tweets, sentiment labels (pos/neg/neutral), word-level language tags | Sentiment pre-training; language-ID features |
| [Hinglish-24k merge](https://arxiv.org/abs/2601.05091) recipe | SentiMix train (~14k) + an 11k public Hinglish tweet set = 24,111 tweets | A larger sentiment starting set (rebuild the merge ourselves) |
| [COMI-LINGUA](https://arxiv.org/pdf/2503.21670) | 376K manually annotated Hinglish instances across several NLP tasks, CC-BY-4.0 | Language-ID and normalisation support; check which task annotations fit before use |
| [L3Cube-HingCorpus + HingBERT](https://arxiv.org/abs/2204.08398) | Large unlabelled Roman-script Hinglish corpus and pretrained models | Base encoder; optional extra DAPT text |
| GoEmotions (English) | 58K Reddit comments, 27 emotions | Auxiliary emotion signal, mapped to our 6 emotions (§5.2) |

Bootstrap data is tweets and Reddit comments, not app reviews, so it only **warms up** the model. App-review labels (§6) carry the real weight.

---

## 2. Storage

- **PostgreSQL** is the system of record (schema in §3).
- **Parquet snapshots** are written for every training run (`snapshot_id` = date + hash) so any model can be retrained on exactly the same data.
- **Raw text is never overwritten.** Normalised text is stored alongside it.

---

## 3. Database schema

```sql
CREATE TABLE apps (
    app_id          TEXT PRIMARY KEY,          -- Play Store package ID
    display_name    TEXT NOT NULL,             -- e.g. 'Zomato'
    category        TEXT NOT NULL,             -- 'food_delivery' | 'payments' | 'quick_commerce'
    active          BOOLEAN DEFAULT TRUE,
    last_cursor_at  TIMESTAMPTZ                -- newest review timestamp already ingested
);

CREATE TABLE reviews (
    review_id         TEXT PRIMARY KEY,        -- Play Store reviewId
    app_id            TEXT REFERENCES apps(app_id),
    content_raw       TEXT NOT NULL,
    content_norm      TEXT,                    -- after processing (§4)
    rating            SMALLINT CHECK (rating BETWEEN 1 AND 5),
    app_version       TEXT,                    -- reviewCreatedVersion, may be NULL
    thumbs_up         INTEGER DEFAULT 0,
    has_reply         BOOLEAN DEFAULT FALSE,
    replied_at        TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL,    -- review date
    ingested_at       TIMESTAMPTZ DEFAULT now(),
    lang_mix          TEXT,                    -- 'hinglish' | 'english' | 'hindi_dev' | 'other'
    code_mix_index    REAL,                    -- share of Hindi tokens, 0–1
    is_spam           BOOLEAN DEFAULT FALSE,
    near_dup_of       TEXT                     -- review_id of an earlier near-duplicate
);

CREATE TABLE model_versions (
    version          TEXT PRIMARY KEY,         -- e.g. 'v3'
    created_at       TIMESTAMPTZ DEFAULT now(),
    mlflow_run_id    TEXT,
    snapshot_id      TEXT,
    hub_revision     TEXT,                     -- Hugging Face Hub tag
    gold_metrics     JSONB,                    -- per-task metrics on the gold set
    is_champion      BOOLEAN DEFAULT FALSE,
    promotion_note   TEXT                      -- why promoted / rejected
);

CREATE TABLE predictions (
    review_id        TEXT REFERENCES reviews(review_id),
    model_version    TEXT REFERENCES model_versions(version),
    sentiment        TEXT,  sentiment_conf REAL,
    emotion          TEXT,  emotion_conf   REAL,
    aspects          TEXT[],                   -- multi-label
    aspect_conf      JSONB,                    -- {aspect: probability}
    urgency          TEXT,  urgency_conf   REAL,
    urgency_score    REAL,                     -- escalation model score, 0–1
    embedding_ref    TEXT,                     -- pointer to stored embedding (Parquet)
    predicted_at     TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (review_id, model_version)
);

CREATE TABLE labels (
    label_id         BIGSERIAL PRIMARY KEY,
    review_id        TEXT REFERENCES reviews(review_id),
    task             TEXT NOT NULL,            -- 'sentiment' | 'emotion' | 'aspect' | 'urgency'
    value            TEXT NOT NULL,            -- aspect labels stored one row per aspect
    source           TEXT NOT NULL,            -- 'gold' | 'human' | 'correction' | 'suggestion' | 'llm' | 'weak_rating'
    trust            REAL NOT NULL,            -- sample weight used in training (§6.1)
    labeller         TEXT,                     -- 'tanmay' | LLM name + prompt version | 'rule'
    created_at       TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE label_queue (
    review_id        TEXT REFERENCES reviews(review_id),
    reason           TEXT,                     -- 'low_confidence' | 'disagreement' | 'new_cluster' | 'diversity'
    priority         REAL,
    status           TEXT DEFAULT 'pending',   -- 'pending' | 'labelled' | 'skipped'
    queued_at        TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (review_id, reason)
);

CREATE TABLE daily_aggregates (
    day              DATE,
    app_id           TEXT REFERENCES apps(app_id),
    app_version      TEXT,
    aspect           TEXT,                     -- 'ALL' for app totals
    n_reviews        INTEGER,
    n_negative       INTEGER,
    avg_rating       REAL,
    emotion_counts   JSONB,
    PRIMARY KEY (day, app_id, app_version, aspect)
);

CREATE TABLE alerts (
    alert_id         BIGSERIAL PRIMARY KEY,
    app_id           TEXT REFERENCES apps(app_id),
    aspect           TEXT,
    started_on       DATE,
    ended_on         DATE,
    peak_ratio       REAL,                     -- observed / expected
    linked_version   TEXT,
    status           TEXT DEFAULT 'active',    -- 'active' | 'resolved'
    evidence_ids     TEXT[]                    -- sample review_ids
);

CREATE TABLE clusters (
    cluster_id       TEXT PRIMARY KEY,         -- stable ID across weeks
    week             DATE,
    app_ids          TEXT[],
    keywords         TEXT[],
    size             INTEGER,
    growth           REAL,                     -- vs previous week
    promoted_to      TEXT                      -- aspect name if promoted, else NULL
);

CREATE TABLE drift_reports (
    report_date      DATE PRIMARY KEY,
    model_version    TEXT REFERENCES model_versions(version),
    embedding_drift  REAL,
    psi_by_feature   JSONB,
    oov_rate         REAL,                     -- share of unseen normalised tokens
    triggered        BOOLEAN
);
```

Indexes: `reviews(app_id, created_at)`, `predictions(model_version)`, `labels(review_id, task)`, `label_queue(status, priority)`.

---

## 4. Processing pipeline

Applied in order; each step is a small, unit-tested function in `processing/`.

1. **Cleaning** — strip URLs, phone numbers and e-mail addresses (privacy), collapse whitespace.
2. **Emoji handling** — keep the emoji and add a text tag from an emoji lexicon (`😡` → `😡 [angry]`). Written fresh for v2; v1's emoji layer is only a reference.
3. **Elongation** — cap repeated characters at two (`bakwaaaas` → `bakwaas`, `!!!!!` → `!!`).
4. **Slang and spelling normalisation** — a new dictionary built for v2 (v1's slang map used as a starting reference) plus entries mined from reviews (`plz/pls` → `please`, `nhi/nai/nahin` → `nahi`). Hinglish words are normalised to one common spelling, not translated.
5. **Script handling** — detect Devanagari. Reviews mostly in Devanagari are transliterated to Roman script (e.g. with an Indic transliteration library) so the Roman-script encoder can read them; the original stays in `content_raw`.
6. **Language ID** — token-level Hindi/English tagging (a HingBERT-LID-style model, or a lexicon baseline first) to compute `lang_mix` and `code_mix_index`.
7. **Spam and junk filter** — flag empty, emoji-only, very short ("good"), repeated-character or promotional reviews. Flagged reviews are stored but excluded from training and alerts.
8. **Deduplication** — exact duplicates by `review_id`; near-duplicates (copy-pasted reviews) by MinHash on character shingles, recorded in `near_dup_of`.

The unknown-word tracker from v1 becomes the `oov_rate` drift signal (§3, `drift_reports`).

---

## 5. Label taxonomy

All definitions live in `configs/taxonomy.yaml` and in the labelling guide used for the gold set. Examples below are illustrative.

### 5.1 Sentiment (single label)

| Label | Definition | Example |
|---|---|---|
| positive | Overall satisfied | "delivery super fast tha, maza aa gaya 👍" |
| neutral | Factual, mixed with no lean, or a question | "new update me dark mode kaha hai?" |
| negative | Overall dissatisfied | "har baar order late, bakwas service" |

### 5.2 Emotion (single label, dominant emotion)

| Label | Definition | Example | GoEmotions mapping |
|---|---|---|---|
| joy | Happy, grateful, delighted | "thank you zomato, mast experience" | joy, gratitude, admiration, amusement, love, excitement, approval |
| anger | Angry, outraged, insulting | "chor company hai, paisa wapas karo 😡" | anger, annoyance, disgust |
| disappointment | Let down, sad, resigned | "pehle accha tha, ab bekar ho gaya" | disappointment, sadness, remorse, grief |
| anxiety | Worried, especially about money or safety | "₹2000 kat gaye, account me nahi aaye, dar lag raha hai" | fear, nervousness |
| surprise | Unexpected (good or bad) | "itna discount? sach me?" | surprise, realization, confusion |
| neutral | No clear emotion | "please add UPI lite option" | neutral |

*Anxiety* is kept separate from *anger* deliberately: money-lost reviews written in fear are often the most urgent.

### 5.3 Aspect (multi-label, zero or more)

| Aspect | Covers |
|---|---|
| payment | Payment failure, money deducted, UPI/card/wallet issues, double charge |
| refund | Refund delayed, refused or partial |
| delivery | Late, not delivered, delivery partner behaviour |
| order_accuracy | Wrong, missing, damaged or expired items |
| app_performance | Crash, hang, slow, bug, update broke something |
| ui_ux | Confusing layout, missing feature, design change |
| customer_support | Unreachable, unhelpful, bot loops, no reply |
| pricing_offers | Prices, fees, surge, coupons, membership |
| account_login | OTP not received, login, KYC, blocked account |

A review with none of these (e.g. pure praise) has an empty aspect list. Unexplained negative reviews are what issue clustering looks at.

### 5.4 Urgency (single label)

| Level | Rule of thumb | Example |
|---|---|---|
| critical | Money lost or account blocked **and** unresolved, or safety issue | "₹1500 debit ho gaye, order nahi hua, 3 din se support reply nahi" |
| high | Core flow broken for the user (can't order, can't pay, can't log in) | "OTP aa hi nahi raha, login nahi ho raha" |
| medium | Real problem with a workaround or a one-off bad experience | "order 40 min late aaya" |
| low | Suggestions, mild complaints, praise | "dark mode add karo" |

---

## 6. Labelling strategy

### 6.1 Label sources and trust weights

| Source | Produced by | Trust weight (training) | Used where |
|---|---|---|---|
| `gold` | Tanmay, using the labelling guide | — (test only, never trained on) | Promotion gate, all reported metrics |
| `human` | Tanmay, from the active-learning queue | 1.0 | Training |
| `correction` | Admin's "correct this label" button on the dashboard | 1.0 | Training |
| `suggestion` | Public visitors' "correct this label" button | 0 until approved; then becomes `correction` | Training only after admin approval |
| `llm` | LLM with the taxonomy prompt, validated (§6.3) | 0.6–0.8 per task, set by measured agreement | Training |
| `weak_rating` | Star rating → sentiment (1–2 negative, 3 neutral, 4–5 positive) | 0.3 | Sentiment training only |

Trust weights are sample weights in the loss. They are starting values, tuned once in Phase 3.

### 6.2 How each task gets labels

| Task | Bootstrap | Ongoing |
|---|---|---|
| Sentiment | SentiMix / Hinglish-24k + star ratings | Star ratings on every new review + active learning |
| Emotion | GoEmotions (mapped) + LLM labels on reviews | Active learning + corrections |
| Aspect | LLM labels on ~5,000 reviews | Active learning, corrections, promoted clusters |
| Urgency | LLM labels guided by the §5.4 rules | Active learning + corrections |

### 6.3 LLM-assisted labelling protocol

1. Fixed prompt containing the taxonomy, definitions and ~15 worked Hinglish examples; prompt versions are stored (`labeller = "<llm>@<prompt-version>"`).
2. Each review is labelled twice with different example orders; disagreements go to the human queue instead of training.
3. Before any LLM labels are used for a task, they are compared with 300 human labels for that task. Agreement is reported as Cohen's κ (single-label tasks) or per-aspect F1 (aspects).
4. A task's LLM labels are accepted only if κ ≥ 0.6; otherwise the prompt is revised or that task falls back to human labels.
5. LLM labels never touch the gold set.

---

## 7. Gold test set

The gold set is what makes every reported number trustworthy.

| Property | Value |
|---|---|
| Size | ~1,500 reviews (250 per app) |
| Time window | Reviews from a fixed window **after** all training data used in v1 of the new model |
| Sampling | Stratified by app and star rating; extra weight on negative reviews so rare aspects and urgency levels appear |
| Labeller | Tanmay, all four tasks, using the written guide |
| Consistency check | 200 reviews re-labelled after at least a week; intra-annotator agreement (κ) reported |
| Status | **Frozen.** Versioned (`gold_v1`), never trained on, never edited after freezing except for documented error fixes |

**Rolling "recent" set:** each month, ~150 newly labelled recent reviews form a separate `recent_YYYY_MM` set. It is reported alongside gold to show how the model handles newer data, but the promotion gate uses gold (see `RETRAINING.md` §5).

---

## 8. Splits and leakage rules

- **Time-based splits only:** train on older reviews, validate on a later window, test on gold (later still). Random splits would leak future slang and issues into training.
- **App leakage check:** one ablation trains without one app and tests on it, to show generalisation to an unseen app.
- **Near-duplicates** stay on one side of any split.
- **Bootstrap tweets** appear only in training, never in validation or test.

---

## 9. Data governance

- Public reviews only; no login-gated data.
- No reviewer names, images or other identifiers stored.
- Polite collection: one run per app per day, capped pages, back-off on errors.
- Non-commercial portfolio and research use.
- Raw review text is not published in bulk; the dashboard shows small samples as evidence for aggregates.
