"""Fill in the processed columns for reviews that haven't been processed yet.

Run: python -m processing.run_processing
     python -m processing.run_processing --all     (re-process every review, e.g. after editing slang.yaml)

Step 1.3 fills content_norm. Later steps add language mix, spam and duplicates here.
"""
import argparse
import logging

from ingestion.store import connect
from processing.normalize import normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("processing")
BATCH = 2000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="re-process every review")
    args = parser.parse_args()

    where = "" if args.all else "WHERE content_norm IS NULL"
    with connect() as conn:
        rows = conn.execute(f"SELECT review_id, content_raw FROM reviews {where}").fetchall()
        log.info("%d reviews to process", len(rows))
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            updates = [(normalize(text), rid) for rid, text in batch]
            with conn.cursor() as cur:
                cur.executemany("UPDATE reviews SET content_norm = %s WHERE review_id = %s", updates)
            conn.commit()
            log.info("  processed %d / %d", min(start + BATCH, len(rows)), len(rows))
    log.info("Done.")


if __name__ == "__main__":
    main()