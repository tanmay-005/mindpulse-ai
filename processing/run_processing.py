"""Fill in the processed columns for reviews.

Run: python -m processing.run_processing
     python -m processing.run_processing --all     (redo every review, e.g. after editing a word list)

For each new review (1.3-1.5):  content_norm, lang_mix, code_mix_index, is_spam
Then for the last DUP_WINDOW_DAYS of reviews (1.5):  near_dup_of
"""
import argparse
import logging

from ingestion.store import connect
from processing.langmix import lang_mix
from processing.normalize import normalize
from processing.quality import find_near_duplicates, is_spam

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("processing")
BATCH = 2000
DUP_WINDOW_DAYS = 90    # copy-paste campaigns are bursty; comparing against 3 months is plenty
DUP_MIN_WORDS = 8       # short reviews match by chance ("this is very good app"); copies are long


def process_text(conn, redo_all):
    where = "" if redo_all else "WHERE content_norm IS NULL"
    rows = conn.execute(f"SELECT review_id, content_raw FROM reviews {where}").fetchall()
    log.info("%d reviews to process", len(rows))
    for start in range(0, len(rows), BATCH):
        updates = []
        for rid, raw in rows[start:start + BATCH]:
            norm = normalize(raw)
            mix = lang_mix(norm)
            updates.append((norm, mix.label, mix.code_mix_index, is_spam(raw), rid))
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE reviews SET content_norm = %s, lang_mix = %s, code_mix_index = %s, "
                "is_spam = %s WHERE review_id = %s", updates)
        conn.commit()
        log.info("  processed %d / %d", min(start + BATCH, len(rows)), len(rows))


def mark_near_duplicates(conn):
    recent = f"created_at >= (SELECT max(created_at) FROM reviews) - interval '{DUP_WINDOW_DAYS} days'"
    window = f"{recent} AND n_words >= {DUP_MIN_WORDS} AND NOT is_spam"
    rows = conn.execute(
        f"SELECT review_id, content_norm FROM reviews WHERE {window} ORDER BY created_at, review_id"
    ).fetchall()
    dup_of = find_near_duplicates(rows)
    conn.execute(f"UPDATE reviews SET near_dup_of = NULL WHERE {recent}")   # clear old marks, short reviews too
    with conn.cursor() as cur:
        cur.executemany("UPDATE reviews SET near_dup_of = %s WHERE review_id = %s",
                        [(orig, rid) for rid, orig in dup_of.items()])
    conn.commit()
    log.info("near-duplicates: %d of %d checked reviews are copies", len(dup_of), len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="redo every review")
    args = parser.parse_args()
    with connect() as conn:
        process_text(conn, args.all)
        mark_near_duplicates(conn)
    log.info("Done.")


if __name__ == "__main__":
    main()