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


if __name__ == "__main__":
    main()