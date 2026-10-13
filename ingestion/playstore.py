"""Fetch Google Play reviews politely, newest first.

Reviewer names and profile pictures are dropped here and never stored.
"""
import logging
import time
from datetime import datetime

from google_play_scraper import Sort, reviews

log = logging.getLogger(__name__)

# Fields we keep from each review → our column names
FIELD_MAP = {
    "reviewId": "review_id",
    "content": "content",
    "score": "rating",
    "at": "created_at",
    "reviewCreatedVersion": "app_version",
    "thumbsUpCount": "thumbs_up",
    "replyContent": "reply_content",
    "repliedAt": "replied_at",
}


def to_record(raw, app_id, lang):
    """Turn one raw review into our format, dropping personal fields."""
    rec = {ours: raw.get(theirs) for theirs, ours in FIELD_MAP.items()}
    rec["app_id"] = app_id
    rec["lang_query"] = lang
    for key in ("created_at", "replied_at"):
        if isinstance(rec[key], datetime):
            rec[key] = rec[key].isoformat()
    return rec


def fetch_reviews(app_id, lang="en", country="in", target=1000,
                  batch_size=200, sleep_seconds=1.5, max_retries=4, since=None):
    """Return up to `target` newest reviews for one app and one language.

    Stops early if the store runs out, or when it reaches a review
    older than `since` (used later by the daily scraper).
    """
    out = []
    token = None
    while len(out) < target:
        count = min(batch_size, target - len(out))
        batch, token = _fetch_with_retries(app_id, lang, country, count, token, max_retries)
        if not batch:
            break

        reached_old = False
        for raw in batch:
            if since is not None and raw.get("at") is not None and raw["at"] <= since:
                reached_old = True
                break
            out.append(to_record(raw, app_id, lang))

        if reached_old or token is None:
            break
        time.sleep(sleep_seconds)          # be polite between pages
    return out[:target]


def _fetch_with_retries(app_id, lang, country, count, token, max_retries):
    delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            if token is None:   # first page
                return reviews(app_id, lang=lang, country=country,
                               sort=Sort.NEWEST, count=count)
            return reviews(app_id, continuation_token=token)  # next pages
        except Exception as e:
            if attempt == max_retries:
                raise
            log.warning("fetch %s/%s failed (%s); retrying in %.0fs", app_id, lang, e, delay)
            time.sleep(delay)
            delay *= 2     # wait twice as long each retry: 2s, 4s, 8s, 16s