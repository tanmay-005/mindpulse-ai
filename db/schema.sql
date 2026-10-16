-- MindPulse AI database schema (docs/DATA_PLAN.md section 3).
-- Safe to run more than once: every statement uses IF NOT EXISTS.

CREATE TABLE IF NOT EXISTS apps (
    app_id          TEXT PRIMARY KEY,                  -- Play Store package ID
    display_name    TEXT NOT NULL,                     -- e.g. 'Zomato'
    category        TEXT NOT NULL,                     -- food_delivery | payments | quick_commerce
    active          BOOLEAN NOT NULL DEFAULT TRUE,
    last_cursor_at  TIMESTAMPTZ                        -- newest review already ingested
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id       TEXT PRIMARY KEY,                  -- Play Store reviewId
    app_id          TEXT NOT NULL REFERENCES apps(app_id),
    content_raw     TEXT NOT NULL,
    content_norm    TEXT,                              -- after processing
    rating          SMALLINT CHECK (rating BETWEEN 1 AND 5),
    app_version     TEXT,                              -- may be NULL
    thumbs_up       INTEGER NOT NULL DEFAULT 0,
    has_reply       BOOLEAN NOT NULL DEFAULT FALSE,
    replied_at      TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL,              -- when the review was posted
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    lang_query      TEXT,                              -- 'en' or 'hi': which Google stream it came from
    lang_mix        TEXT,                              -- english | hinglish | hindi_dev | other
    code_mix_index  REAL,                              -- share of Hindi words, 0-1
    n_words         INTEGER,
    is_spam         BOOLEAN NOT NULL DEFAULT FALSE,
    near_dup_of     TEXT                               -- review_id of an earlier near-duplicate
);

CREATE TABLE IF NOT EXISTS model_versions (
    version         TEXT PRIMARY KEY,                  -- e.g. 'v1'
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    mlflow_run_id   TEXT,
    snapshot_id     TEXT,
    hub_revision    TEXT,                              -- Hugging Face Hub tag
    gold_metrics    JSONB,
    is_champion     BOOLEAN NOT NULL DEFAULT FALSE,
    promotion_note  TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    review_id       TEXT NOT NULL REFERENCES reviews(review_id),
    model_version   TEXT NOT NULL REFERENCES model_versions(version),
    sentiment       TEXT,  sentiment_conf REAL,
    emotion         TEXT,  emotion_conf   REAL,
    aspects         TEXT[],
    aspect_conf     JSONB,
    urgency         TEXT,  urgency_conf   REAL,
    urgency_score   REAL,
    predicted_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (review_id, model_version)
);

CREATE TABLE IF NOT EXISTS labels (
    label_id        BIGSERIAL PRIMARY KEY,
    review_id       TEXT NOT NULL REFERENCES reviews(review_id),
    task            TEXT NOT NULL CHECK (task IN ('sentiment', 'emotion', 'aspect', 'urgency')),
    value           TEXT NOT NULL,                     -- one row per aspect for multi-label
    source          TEXT NOT NULL CHECK (source IN ('gold', 'human', 'correction', 'suggestion', 'llm', 'weak_rating')),
    trust           REAL NOT NULL,                     -- sample weight in training
    labeller        TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS label_queue (
    review_id       TEXT NOT NULL REFERENCES reviews(review_id),
    reason          TEXT NOT NULL,                     -- low_confidence | disagreement | diversity | new_cluster | new_version
    priority        REAL,
    status          TEXT NOT NULL DEFAULT 'pending',   -- pending | labelled | skipped | expired
    queued_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (review_id, reason)
);

CREATE TABLE IF NOT EXISTS daily_aggregates (
    day             DATE NOT NULL,
    app_id          TEXT NOT NULL REFERENCES apps(app_id),
    app_version     TEXT NOT NULL DEFAULT 'unknown',   -- primary keys can't be NULL
    aspect          TEXT NOT NULL,                     -- 'ALL' for app totals
    n_reviews       INTEGER NOT NULL,
    n_negative      INTEGER NOT NULL,
    avg_rating      REAL,
    emotion_counts  JSONB,
    PRIMARY KEY (day, app_id, app_version, aspect)
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id        BIGSERIAL PRIMARY KEY,
    app_id          TEXT NOT NULL REFERENCES apps(app_id),
    aspect          TEXT NOT NULL,
    started_on      DATE NOT NULL,
    ended_on        DATE,
    peak_ratio      REAL,                              -- observed / expected
    linked_version  TEXT,
    status          TEXT NOT NULL DEFAULT 'active',    -- active | resolved
    evidence_ids    TEXT[]
);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id      TEXT PRIMARY KEY,
    week            DATE NOT NULL,
    app_ids         TEXT[],
    keywords        TEXT[],
    size            INTEGER,
    growth          REAL,
    promoted_to     TEXT
);

CREATE TABLE IF NOT EXISTS drift_reports (
    report_date     DATE PRIMARY KEY,
    model_version   TEXT REFERENCES model_versions(version),
    embedding_drift REAL,
    psi_by_feature  JSONB,
    oov_rate        REAL,
    triggered       BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_reviews_app_created ON reviews (app_id, created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_version  ON predictions (model_version);
CREATE INDEX IF NOT EXISTS idx_labels_review_task   ON labels (review_id, task);
CREATE INDEX IF NOT EXISTS idx_queue_status         ON label_queue (status, priority);