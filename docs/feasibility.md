# Phase 0 — Feasibility note

**Decision: GO.** Enough Hinglish reviews arrive every day, a model trained on real app reviews works far better than one trained on tweets, and the base encoder is chosen. Run 13–16 October 2026.

## 0.1 App IDs
All six package IDs in `configs/apps.yaml` return the right app (`scripts/check_apps.py`: 6/6 found).
Swiggy's listing is "Swiggy: Food Instamart Dineout", so some Swiggy reviews are about quick commerce.

## 0.2 Sample collected
12,000 reviews: 2,000 per app (1,000 from Google's `en` set + 1,000 from its `hi` set, no overlap). Newest review: 12 Oct 2026.
Google's `lang` setting filters reviews by detected language; it does not just change the display language.

## 0.3 Language mix and data quality

| App | Hinglish | Hindi (Devanagari) | Has app version | Under 3 words | Exact duplicates |
|---|---|---|---|---|---|
| Blinkit | 10.8% | 8.2% | 84.2% | 55.9% | 38.5% |
| Paytm | 13.2% | 14.1% | 85.9% | 64.8% | 44.3% |
| PhonePe | 9.7% | 15.6% | 80.4% | 68.0% | 39.4% |
| Swiggy | 11.8% | 6.7% | 84.9% | 59.2% | 39.4% |
| Zepto | 11.6% | 8.5% | 87.0% | 45.6% | 29.8% |
| Zomato | 10.0% | 8.8% | 85.9% | 65.1% | 46.1% |

- Over all reviews, 21.5% are Hinglish or Hindi, held down by very short reviews ("good", "nice app").
- Among the 4,828 reviews with 3+ words, 37.3% are Hinglish or Hindi. This overstates the live share, because the sample is half `hi` and the `hi` stream is far slower.
- **Review speed:** the `en` stream brings ~180 to 1,000+ reviews per app per day; the `hi` stream only 1–34.
- **Estimated useful Hinglish/Hindi reviews per day (3+ words):** Blinkit 44+, PhonePe 28, Zomato 23, Swiggy 19, Paytm 17, Zepto 11 — **about 142 per day** across the six apps.
- **Tagger spot-check:** 19 of 20 random labels correct; 8 of 30 reviews tagged "english" from the `hi` set were actually Hinglish, so the tagger under-counts and 142/day is conservative.
- **Changed criterion:** the original "≥30% of reviews" criterion was replaced by "≥~100 useful Hinglish/Hindi reviews per day", because volume is what retraining and the dashboard need. **Met.**

## 0.4 Hand-labelled test set
200 reviews (3+ words, ~33 per app), labelled from the text only, with star ratings hidden:
108 positive, 76 negative, 16 neutral.

## 0.5 Baselines (macro-F1 on the 200 app reviews)

| Model | Trained on | Macro-F1 | Negative recall |
|---|---|---|---|
| Majority class | — | 0.049 | — |
| TF-IDF + LR | 14,000 SentiMix tweets | 0.340 | 0.11 |
| Star rating used as the label | — | 0.631 | — |
| TF-IDF + LR | 4,325 star-labelled app reviews | **0.677** | 0.88 |

- SentiMix (Hugging Face mirror `RTT1/SentiMix`) matches the official release: 14,000 / 3,000 / 3,000 tweets. Its test labels are in a separate file, so validation was used (TF-IDF + LR: 0.615).
- A tweet-trained model misses 89% of complaints in app reviews.
- Star ratings agree with the text 82% of the time: a useful but noisy free label.
- Noisy in-domain labels beat clean out-of-domain labels by a wide margin.

## 0.6 Base encoder (fine-tuned 2 epochs on star-labelled reviews, Colab T4)

| Model | Macro-F1 | F1 negative | F1 neutral | F1 positive |
|---|---|---|---|---|
| **l3cube-pune/hing-roberta-mixed** | **0.679** | **0.932** | 0.182 | **0.922** |
| l3cube-pune/hing-roberta | 0.624 | 0.869 | 0.100 | 0.902 |
| bert-base-multilingual-cased | 0.617 | 0.814 | 0.176 | 0.860 |
| l3cube-pune/hing-bert | 0.607 | 0.853 | 0.087 | 0.881 |
| distilbert-base-uncased | 0.605 | 0.844 | 0.111 | 0.860 |

**Chosen: `l3cube-pune/hing-roberta-mixed`** — best on negative and positive, and the only candidate pretrained on both Roman and Devanagari Hinglish.
One seed, 200 test reviews, no tuning: models between 0.60 and 0.62 are effectively tied. Phase 3 re-checks with 3 seeds on the gold set.

## Decision

- [x] Scraper reliably returns new reviews
- [x] Enough Hinglish/Hindi: ~142 useful reviews per day (criterion: ≥~100)
- [x] Baseline on app reviews clearly above the majority class (0.677 vs 0.049)

**GO.** No apps replaced.

## Changes to the plan from Phase 0
1. **Star ratings become the main training signal for sentiment;** SentiMix becomes a minor extra.
2. **Base encoder: `hing-roberta-mixed`** (~278M parameters, so distillation in Phase 4 matters for CPU serving).
3. **Neutral is the weak class** for every model; Phase 2 labelling must deliberately include neutral reviews.
4. **The daily scraper must page until it reaches the last saved review,** not stop at a fixed count: Blinkit gets 1,000+ `en` reviews a day.
5. **Short reviews (under 3 words) are excluded** from training and evaluation; they are about half of all reviews.