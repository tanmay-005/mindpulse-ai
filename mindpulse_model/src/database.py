import sqlite3
import os

DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data.db")
)


def _get_conn():
    """Returns a new thread-safe connection."""
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def _init_db():
    """Create table if not exists."""
    conn = _get_conn()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            text       TEXT,
            emotion    TEXT,
            sentiment  TEXT,
            confidence REAL
        )
    """)

    conn.commit()
    conn.close()


# Initialize on import
_init_db()


def save_result(text: str, emotion: str, sentiment: str, confidence: float):
    """Save a prediction result to the database."""
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO history (text, emotion, sentiment, confidence) VALUES (?, ?, ?, ?)",
            (text, emotion, sentiment, confidence)
        )
        conn.commit()
    finally:
        conn.close()


def get_history():
    """Returns all history rows ordered by most recent first."""
    conn = _get_conn()
    try:
        cursor = conn.execute(
            "SELECT * FROM history ORDER BY id DESC"
        )
        return cursor.fetchall()
    finally:
        conn.close()


def get_recent(n: int = 50):
    """Returns the N most recent predictions."""
    conn = _get_conn()
    try:
        cursor = conn.execute(
            "SELECT * FROM history ORDER BY id DESC LIMIT ?",
            (n,)
        )
        return cursor.fetchall()
    finally:
        conn.close()