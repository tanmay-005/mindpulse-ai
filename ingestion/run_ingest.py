"""Step 1.2: fetch new reviews for every tracked app and save them to the database.

Run: python -m ingestion.run_ingest
     python -m ingestion.run_ingest --apps Zepto      (just one app)

First run for an app: takes the newest `first_run_per_lang` reviews per language.
Later runs: pages back until it reaches reviews it already has, so no fixed
limit (Blinkit gets 1,000+ a day). Safe to run any number of times.
"""
import argparse
import logging
from datetime import timedelta
from pathlib import Path

import yaml

from ingestion.playstore import fetch_reviews
from ingestion.store import connect, get_cursor, insert_reviews, update_cursor

ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("ingest")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apps", nargs="*", help="only these app names")
    args = parser.parse_args()

    config = yaml.safe_load(open(ROOT / "configs" / "apps.yaml", encoding="utf-8"))
    s = config["scrape"]
    total_new = 0

    with connect() as conn:
        for app in config["apps"]:
            if args.apps and app["name"] not in args.apps:
                continue

            cursor = get_cursor(conn, app["id"])
            if cursor is None:
                since, target = None, s["first_run_per_lang"]
                log.info("%s: first run, taking newest %d per language", app["name"], target)
            else:
                # step back a few hours: reviews sometimes appear late; duplicates are skipped
                since, target = cursor - timedelta(hours=s["overlap_hours"]), s["max_per_run_per_lang"]
                log.info("%s: fetching reviews newer than %s", app["name"], since.isoformat(timespec="minutes"))

            for lang in s["langs"]:
                got = fetch_reviews(
                    app["id"], lang=lang, country=s["country"], target=target,
                    batch_size=s["batch_size"], sleep_seconds=s["sleep_seconds"],
                    max_retries=s["max_retries"], since=since,
                )
                new = insert_reviews(conn, got) if got else 0
                total_new += new
                log.info("  %s [%s]: fetched %d, new %d", app["name"], lang, len(got), new)
                if since is not None and len(got) == target:
                    log.warning("  %s [%s]: hit the %d cap before reaching old reviews; "
                                "some reviews may be missing", app["name"], lang, target)

            update_cursor(conn, app["id"])
            conn.commit()          # save each app's work before moving to the next
            log.info("  %s: cursor now %s", app["name"], get_cursor(conn, app["id"]))

    log.info("Done. %d new reviews saved.", total_new)


if __name__ == "__main__":
    main()