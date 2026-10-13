"""Step 0.2: scrape a sample of recent reviews for each tracked app.

Run from the project root:
    python -m scripts.p0_scrape --per-lang 200    # quick test
    python -m scripts.p0_scrape                   # full sample (~1,000 per language per app)
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from ingestion.playstore import fetch_reviews

ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("p0_scrape")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-lang", type=int, default=1000, help="reviews per language per app")
    parser.add_argument("--apps", nargs="*", help="only these app names, e.g. --apps Zepto Blinkit")
    args = parser.parse_args()

    config = yaml.safe_load(open(ROOT / "configs" / "apps.yaml", encoding="utf-8"))
    settings = config["scrape"]
    out_dir = ROOT / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    for app in config["apps"]:
        if args.apps and app["name"] not in args.apps:
            continue

        rows = []
        for lang in settings["langs"]:
            got = fetch_reviews(
                app["id"], lang=lang, country=settings["country"],
                target=args.per_lang, batch_size=settings["batch_size"],
                sleep_seconds=settings["sleep_seconds"], max_retries=settings["max_retries"],
            )
            log.info("%s [%s]: %d reviews", app["name"], lang, len(got))
            rows.extend(got)

        if not rows:
            log.error("%s: no reviews returned", app["name"])
            continue

        # the same review can come back under both 'en' and 'hi' — keep one copy
        df = pd.DataFrame(rows).drop_duplicates("review_id")
        df.insert(1, "app_name", app["name"])
        df.insert(2, "category", app["category"])

        path = out_dir / f"sample_{app['name'].lower()}.csv"
        df.to_csv(path, index=False, encoding="utf-8")
        log.info("%s: %d unique reviews, newest %s -> %s",
                 app["name"], len(df), df["created_at"].max()[:10], path.name)


if __name__ == "__main__":
    main()