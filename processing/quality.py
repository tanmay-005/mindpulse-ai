"""Spam flags and near-duplicate detection (step 1.5).

is_spam(text)        -> True for junk: no real content, character spam, referral/promo spam.
find_near_duplicates -> for copy-pasted reviews, the review_id of the earlier copy.

Flagged reviews stay in the database (they still count on the dashboard as
"reviews received") but are left out of training data and spike alerts.
Short reviews like "good" are NOT spam: they are real opinions, just too short
to learn from, and training already skips reviews under 3 words.
"""
import re

from datasketch import MinHash, MinHashLSH

# ── spam rules ────────────────────────────────────────────────
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B50\u2B55]")
# Only clear promotion phrasing. A complaint that merely MENTIONS a referral or
# promo code ("referral link not showing") is a real review, not spam.
PROMO_RE = re.compile(
    r"(\buse\s+my\s+(referral\s+|invite\s+|promo\s+)?(code|link)\b"
    r"|\bmy\s+(referral|invite)\s+code\s+is\b"
    r"|\brefer\s*(and|&|n)\s*earn\b"
    r"|\b(referral|invite|promo)\s+code\s*[:\-]?\s*[A-Z0-9]{5,}\b"
    r"|\bjoin\s+(my\s+|our\s+)?(telegram|whatsapp)\b"
    r"|\btelegram\s+(channel|group)\b"
    r"|\bearn\s+(money\s+)?(daily|online)\b|\bwork\s+from\s+home\b)",
    re.IGNORECASE,
)


def is_spam(text):
    text = text or ""
    letters = [c for c in text.lower() if c.isalpha()]       # letters in ANY script
    if not text.strip():
        return True                                   # empty
    if not letters and not EMOJI_RE.search(text):
        return True                                   # only digits / punctuation: "....", "1234"
    if len(letters) >= 6 and len(set(letters)) <= 2:
        return True                                   # keyboard spam: "hhhhhhhh", "aaaaab"
    if PROMO_RE.search(text):
        return True                                   # referral / promotion spam
    return False


# ── near-duplicates ───────────────────────────────────────────
NUM_PERM = 128          # hash functions per signature: more = more accurate, slower
THRESHOLD = 0.8         # Jaccard similarity above which two reviews count as copies
SHINGLE = 5             # character 5-grams


def minhash(text):
    t = " ".join(text.lower().split())
    m = MinHash(num_perm=NUM_PERM)
    shingles = {t[i:i + SHINGLE] for i in range(max(1, len(t) - SHINGLE + 1))}
    m.update_batch([s.encode("utf-8") for s in shingles])
    return m


def find_near_duplicates(rows, threshold=THRESHOLD):
    """rows: (review_id, text) in time order, oldest first.
    Returns {review_id: id_of_earliest_earlier_copy} for every later copy."""
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    first_seen = {}                 # review_id -> its position, so we can pick the earliest match
    dup_of = {}
    for pos, (rid, text) in enumerate(rows):
        m = minhash(text)
        matches = lsh.query(m)
        if matches:
            dup_of[rid] = min(matches, key=first_seen.get)
        else:
            lsh.insert(rid, m)      # only originals go into the index
            first_seen[rid] = pos
    return dup_of