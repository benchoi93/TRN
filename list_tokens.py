import sqlite3

DB_PATH = "data/survey.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.execute("SELECT token, json_path, created_at, submitted_at FROM surveys ORDER BY created_at DESC;")
rows = cur.fetchall()
conn.close()

print(f"Found {len(rows)} tokens\n")
for token, path, created_at, submitted_at in rows:
    print(f"{token}\t{path}\tcreated={created_at}\tsubmitted={submitted_at}")
