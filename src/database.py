import os
import sqlite3
DB_PATH = "alerts/alerts.db"


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at      TEXT NOT NULL,
                student_id      INTEGER NOT NULL,
                behavior        TEXT NOT NULL,
                confidence      REAL,
                screenshot_path TEXT,
                video_name      TEXT,
                video_time_sec  REAL
            )
        """)


def insert_alert(created_at, student_id, behavior, confidence,
                 screenshot_path, video_name, video_time_sec):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT INTO alerts (created_at, student_id, behavior, confidence,
                                   screenshot_path, video_name, video_time_sec)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (created_at, student_id, behavior, confidence,
             screenshot_path, video_name, video_time_sec),
        )


def get_alerts():
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute(
            "SELECT created_at, behavior, confidence, screenshot_path "
            "FROM alerts ORDER BY id DESC"
        ).fetchall()


if __name__ == "__main__":
    init_db()
    for row in get_alerts():
        print(row)