# make_tokens.py
import glob
import os
import secrets
import sqlite3
from datetime import datetime

PAPERS_GLOB = "data/papers/*.json"
DB_PATH = "data/survey.db"


def init_db(conn: sqlite3.Connection):
    conn.execute("""
    CREATE TABLE IF NOT EXISTS surveys (
        token TEXT PRIMARY KEY,
        json_path TEXT NOT NULL,
        created_at TEXT NOT NULL,
        submitted_at TEXT
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS responses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        token TEXT NOT NULL,
        payload TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(token) REFERENCES surveys(token)
    )
    """)
    conn.commit()


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/papers", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    json_files = sorted(glob.glob(PAPERS_GLOB))
    if not json_files:
        raise SystemExit(f"No JSON files found under {PAPERS_GLOB}")

    created = 0
    for jp in json_files:
        token = secrets.token_urlsafe(16)  # short but unguessable enough for prototype
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        conn.execute(
            "INSERT INTO surveys(token, json_path, created_at) VALUES (?, ?, ?)",
            (token, jp, now),
        )
        created += 1

    conn.commit()
    conn.close()

    print(f"Created {created} survey tokens in {DB_PATH}")
    print("Example links (local):")
    print("  http://127.0.0.1:8000/<token>")
    print("To list tokens, run: sqlite3 data/survey.db 'select token, json_path from surveys;'")


if __name__ == "__main__":
    main()
