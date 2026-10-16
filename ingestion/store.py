"""Read and write reviews in PostgreSQL."""
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

INSERT_SQL = """
INSERT INTO reviews (review_id, app_id, content_raw, rating, app_version, thumbs_up,
                     has_reply, replied_at, created_at, lang_query, n_words)
VALUES (%(review_id)s, %(app_id)s, %(content)s, %(rating)s, %(app_version)s, %(thumbs_up)s,
        %(has_reply)s, %(replied_at)s, %(created_at)s, %(lang_query)s, %(n_words)s)
ON CONFLICT (review_id) DO NOTHING
"""


def connect():
    """Open a connection using DATABASE_URL from .env (or the environment)."""
    load_dotenv(ROOT / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set. Put it in the .env file in the project root.")
    return psycopg.connect(url)


def get_cursor(conn, app_id):
    """Newest review timestamp already stored for this app (None on the first run)."""
    row = conn.execute("SELECT last_cursor_at FROM apps WHERE app_id = %s", (app_id,)).fetchone()
    return row[0] if row else None


def insert_reviews(conn, records):
    """Insert reviews, skipping ones already stored. Returns how many were new."""
    rows = []
    for r in records:
        content = r.get("content") or ""
        rows.append({
            **r,
            "content": content,
            "thumbs_up": r.get("thumbs_up") or 0,
            "has_reply": bool(r.get("reply_content")),
            "n_words": len(content.split()),
        })
    app_ids = {r["app_id"] for r in rows}
    count_sql = "SELECT count(*) FROM reviews WHERE app_id = ANY(%s)"
    before = conn.execute(count_sql, (list(app_ids),)).fetchone()[0]
    with conn.cursor() as cur:
        cur.executemany(INSERT_SQL, rows)
    after = conn.execute(count_sql, (list(app_ids),)).fetchone()[0]
    return after - before


def update_cursor(conn, app_id):
    """Move the app's cursor to its newest stored review."""
    conn.execute(
        """UPDATE apps SET last_cursor_at =
               (SELECT max(created_at) FROM reviews WHERE app_id = %s)
           WHERE app_id = %s""",
        (app_id, app_id),
    )