# Dashboard Specification

The dashboard is the public face of MindPulse: a live Streamlit app tracking six real Indian apps. Pages 1–6 are what an app company would use day to day; pages 7–8 show reviewers and interviewers that the ML is real and keeps improving.

---

## 1. Global elements

**Filter bar (all pages except 7 and 8):**

| Filter | Options | Default |
|---|---|---|
| App | Zomato, Swiggy, PhonePe, Paytm, Blinkit, Zepto | Zomato |
| Date range | Last 7 / 30 / 90 days, custom | Last 30 days |
| App version | All, or specific versions (latest first) | All |
| Language mix | All, Hinglish, English, Hindi (transliterated) | All |

**On every page:**
- Footer: "Figures are model estimates from public Google Play reviews, not official company data. Model `vN`, data updated `<timestamp>`."
- Every number shows its sample size (`n = 1,240`); percentages based on fewer than 30 reviews are greyed out.
- No rankings or labels such as "worst app".

**Access levels:**

| Role | Can do |
|---|---|
| Public visitor | View all pages; use *Try it live*; submit label **suggestions** (stored as unverified) |
| Admin (Tanmay, login) | Everything above + trusted label corrections, labelling queue, promote clusters, approve suggestions |

---

## 2. Pages

### Page 1 — Overview

*Question answered: how is this app doing right now?*

| Element | Type | Data |
|---|---|---|
| KPI tiles | 5 tiles | Reviews today, reviews this week, % negative (with change vs previous period), average rating, active alerts |
| Sentiment trend | Stacked area or line chart, daily | Share of positive / neutral / negative |
| Rating vs text disagreement | Single stat + small trend line | % of reviews where star rating and predicted sentiment disagree (e.g. 5★ but angry text) |
| Top complaints this week | Ranked list (top 3) | Aspect, count, % change vs last week, one sample review |
| Latest alert | Banner (if any) | Links to Page 3 |

### Page 2 — Aspect Analysis

*Question answered: what exactly are people complaining about?*

| Element | Type | Data |
|---|---|---|
| Aspect breakdown | Horizontal bar chart | Mentions per aspect, split by sentiment |
| Aspect × version heatmap | Heatmap (rows = aspects, columns = recent versions) | Negative-mention rate per version; highlights versions that made something worse |
| Emotion mix per aspect | 100% stacked bars | Emotion distribution within each aspect |
| Aspect trend | Line chart (select aspects) | Daily negative-mention rate over the date range |
| Sample reviews | Table, click an aspect to filter | Review text, rating, date, version, predictions with confidence |

### Page 3 — Alerts and Spikes

*Question answered: did something just break?*

| Element | Type | Data |
|---|---|---|
| Active alerts | Cards | App, aspect, "3.2× normal since <date>", linked version, duration |
| Alert detail | Line chart with expected band | Daily rate vs rolling baseline, spike shaded |
| Evidence | Review list + top phrases | Representative reviews and most frequent phrases during the spike |
| Alert history | Table, newest first | Start, end, peak ratio, linked version, status |

### Page 4 — Emerging Issues

*Question answered: is there a new problem we don't have a name for yet?*

| Element | Type | Data |
|---|---|---|
| New clusters | Cards sorted by growth | Auto keywords (e.g. "OTP · not coming · login"), size, growth vs last week, apps affected |
| Cluster detail | Review list + weekly size chart | Three representative reviews, full list, size over weeks |
| Promote to aspect | Button (admin only) | Adds a new aspect to the taxonomy and queues reviews for labelling |
| Cluster map | 2-D scatter (UMAP) | Optional view of how clusters relate |

### Page 5 — Urgent Review Inbox

*Question answered: which reviews should support answer first?*

| Element | Type | Data |
|---|---|---|
| Ranked queue | Table sorted by `urgency_score` | Review, rating, age, urgency level, emotion, aspects, company replied? |
| Why it's urgent | Highlighted text + top factors | Occlusion-based word highlights from the text model; top SHAP factors from the escalation model |
| Correct this label | Inline buttons per task | Admin: trusted label. Public: suggestion |
| Queue stats | Small tiles | Labels collected this week, pending suggestions, labelling queue size |

### Page 6 — Competitor Comparison

*Question answered: how does this app compare with its rival?*

Only apps in the same category are compared (Zomato vs Swiggy, PhonePe vs Paytm, Blinkit vs Zepto). A category selector replaces the app filter on this page.

| Element | Type | Data |
|---|---|---|
| Sentiment trend | Two lines on one chart | % negative over time per app |
| Aspect comparison | Grouped bars | Negative-mention rate per aspect, side by side |
| Who's improving | Small multiples per aspect | Change in negative rate over the last 4 weeks per app |
| Alert counts | Table | Alerts per app in the period |

All values are rates (per 100 reviews), not raw counts, because apps receive different review volumes.

### Page 7 — Model Health

*Question answered: is the model good, and is it getting better?*

| Element | Type | Data |
|---|---|---|
| Current champion | Info card | Version, promotion date, training snapshot size, gold-set macro-F1 per task |
| Version history | Line chart + table | Gold macro-F1 per task for every version, including rejected challengers (marked) |
| Last gate decision | Checklist table | Each gate check from `RETRAINING.md` §5 with pass/fail and values |
| Drift monitor | Line chart with threshold line | Weekly embedding drift, PSI, OOV rate |
| Label sources | Stacked bars per week | Labels from human, correction, LLM, star rating |
| Confusion matrices | Heatmaps, task selector | Current champion on the gold set |
| Calibration | Reliability diagram | Per head, with ECE |
| Ablation results | Table | From `ML_DESIGN.md` §9, mean ± std over seeds |
| Baseline comparison | Bar chart | TF-IDF + LR vs teacher vs student per task |

### Page 8 — Try It Live

*Question answered: does it really understand Hinglish?*

| Element | Type | Data |
|---|---|---|
| Input box | Text area | Any review in Hinglish, English or Hindi; example chips to click |
| Normalised text | Side-by-side text | Original vs normalised (slang, emoji tags, transliteration) |
| Predictions | Four result cards | Sentiment, emotion, aspects, urgency, each with a confidence bar |
| Word importance | Highlighted text | Occlusion-based highlights per prediction |
| Feedback | "Was this right?" | Stored as a public suggestion |

Calls `POST /predict` on the FastAPI service. Inputs are not stored unless the visitor submits feedback.

---

## 3. Data sources per page

| Page | Main tables |
|---|---|
| 1 Overview | `daily_aggregates`, `alerts`, `predictions` |
| 2 Aspects | `daily_aggregates`, `predictions`, `reviews` |
| 3 Alerts | `alerts`, `daily_aggregates`, `reviews` |
| 4 Emerging | `clusters`, `reviews`, `predictions` |
| 5 Inbox | `predictions`, `reviews`, `labels`, `label_queue` |
| 6 Comparison | `daily_aggregates`, `alerts` |
| 7 Model health | `model_versions`, `drift_reports`, `labels` |
| 8 Try it live | FastAPI `/predict` |

---

## 4. Performance and implementation notes

- Pages read **pre-computed aggregates**; no page runs the model over thousands of reviews on load.
- Query results are cached for 10 minutes (`st.cache_data`).
- Charts in Plotly with one consistent colour scheme: sentiment colours fixed across all pages (negative always the same colour), one colour per app on the comparison page.
- Works on a phone screen: KPI tiles wrap, tables scroll horizontally.
- Target: every page loads in under 3 seconds on the free host.

---

## 5. Build order

1. Page 8 (needs only the API) — first demo-able piece.
2. Pages 1 and 2 (need aggregates).
3. Page 7 (needs model versions — core recruiter value).
4. Pages 3 and 5.
5. Pages 4 and 6.
6. Admin login and labelling tools.
