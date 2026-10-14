"""Step 0.3: measure the language mix and data quality of the scraped sample.

Run: python -m scripts.p0_profile
"""
import glob
from pathlib import Path

import pandas as pd

from processing.langmix import lang_mix

ROOT = Path(__file__).resolve().parent.parent


def main():
    files = sorted(glob.glob(str(ROOT / "data" / "raw" / "sample_*.csv")))
    if not files:
        raise SystemExit("No sample files. Run: python -m scripts.p0_scrape")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # tag every review
    df["content"] = df["content"].fillna("").astype(str)
    mix = df["content"].map(lang_mix)
    df["lang_mix"] = mix.map(lambda m: m.label)
    df["code_mix_index"] = mix.map(lambda m: m.code_mix_index)
    df["n_words"] = df["content"].str.split().str.len()
    df["_text"] = df["content"].str.lower().str.strip()
    # same text posted again within the same app (copy-pasted reviews)
    df["exact_dup"] = df.duplicated(["app_name", "_text"]) & (df["_text"] != "")

    def pct(mask):
        return round(100 * mask.mean(), 1)

    rows = []
    for app, g in df.groupby("app_name"):
        rows.append({
            "app": app,
            "reviews": len(g),
            "hinglish_%": pct(g["lang_mix"] == "hinglish"),
            "hindi_dev_%": pct(g["lang_mix"] == "hindi_dev"),
            "english_%": pct(g["lang_mix"] == "english"),
            "other_%": pct(g["lang_mix"] == "other"),
            "has_version_%": pct(g["app_version"].notna()),
            "exact_dup_%": pct(g["exact_dup"]),
            "under_3_words_%": pct(g["n_words"] < 3),
            "oldest": str(g["created_at"].min())[:10],
            "newest": str(g["created_at"].max())[:10],
        })
    table = pd.DataFrame(rows)

    out = ROOT / "data" / "feasibility"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "profile.csv", index=False)
    df.drop(columns="_text").to_csv(out / "reviews_tagged.csv", index=False, encoding="utf-8")

    pd.set_option("display.width", 200)
    print(table.to_string(index=False))

    hindi_share = 100 * df["lang_mix"].isin(["hinglish", "hindi_dev"]).mean()
    print(f"\nOverall Hinglish or Hindi: {hindi_share:.1f}%   (go criterion: at least ~30%)")

    print("\nBy query language (how Google sorted them):")
    print(pd.crosstab(df["lang_query"], df["lang_mix"], normalize="index").round(2).to_string())

    # Same question, only for reviews with 3+ words (the ones a model can learn from)
    long = df[df["n_words"] >= 3]
    share_long = 100 * long["lang_mix"].isin(["hinglish", "hindi_dev"]).mean()
    print(f"\nReviews with 3+ words: {len(long)} of {len(df)}")
    print(f"Hinglish or Hindi among them: {share_long:.1f}%")
    print(pd.crosstab(long["app_name"], long["lang_mix"], normalize="index").round(2).to_string())

    # How fast do reviews arrive? Newest-first sample, so time span = time to collect them
    df["created_at"] = pd.to_datetime(df["created_at"])
    grp = df.groupby(["app_name", "lang_query"])["created_at"]
    days = (grp.max() - grp.min()).dt.total_seconds() / 86400
    per_day = (grp.size() / days.clip(lower=1)).round(0)
    print("\nNew reviews per day (estimated):")
    print(per_day.unstack().to_string())

    # The live stream is mostly 'en', so estimate useful Hinglish/Hindi per day directly
    df["useful_hi"] = (df["n_words"] >= 3) & df["lang_mix"].isin(["hinglish", "hindi_dev"])
    frac = df.groupby(["app_name", "lang_query"])["useful_hi"].mean()
    est = (per_day * frac).unstack().sum(axis=1).round(0)
    print("\nEstimated useful Hinglish/Hindi reviews per day (3+ words):")
    print(est.to_string())
    print(f"Total across apps: {est.sum():.0f} per day")

if __name__ == "__main__":
    main()