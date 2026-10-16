"""Step 1.1: create the tables and register the tracked apps.

Run: python -m db.setup_db
Reads DATABASE_URL from the .env file. Safe to run again at any time.
"""
import os
from pathlib import Path

import psycopg
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def main():
    load_dotenv(ROOT / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set. Put it in the .env file in the project root.")

    schema = (ROOT / "db" / "schema.sql").read_text(encoding="utf-8")
    apps = yaml.safe_load(open(ROOT / "configs" / "apps.yaml", encoding="utf-8"))["apps"]

    with psycopg.connect(url) as conn:
        conn.execute(schema)
        for a in apps:
            # insert the app, or update its name/category if it's already there
            conn.execute(
                """INSERT INTO apps (app_id, display_name, category)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (app_id) DO UPDATE
                   SET display_name = EXCLUDED.display_name, category = EXCLUDED.category""",
                (a["id"], a["name"], a["category"]),
            )
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        ).fetchall()
        n_apps = conn.execute("SELECT count(*) FROM apps").fetchone()[0]

    print(f"{len(tables)} tables: {', '.join(t[0] for t in tables)}")
    print(f"{n_apps} apps registered")


if __name__ == "__main__":
    main()