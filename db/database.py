"""
db/database.py

SQLite database setup and connection management.
The database file is created at eval_platform.db in the project root.
All tables are created on first run if they don't exist.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "eval_platform.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns that were introduced after initial schema creation."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(test_cases)")}
    if "source_type" not in existing:
        conn.execute(
            "ALTER TABLE test_cases ADD COLUMN source_type TEXT NOT NULL DEFAULT 'internal'"
        )
        conn.commit()


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS benchmarks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL UNIQUE,
                description TEXT    NOT NULL DEFAULT '',
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS test_cases (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                benchmark_id   INTEGER NOT NULL REFERENCES benchmarks(id) ON DELETE CASCADE,
                question       TEXT    NOT NULL,
                reference_text TEXT    NOT NULL,
                domain         TEXT    NOT NULL DEFAULT 'general',
                source_type    TEXT    NOT NULL DEFAULT 'internal',
                created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS eval_runs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                benchmark_id INTEGER NOT NULL REFERENCES benchmarks(id) ON DELETE CASCADE,
                provider     TEXT    NOT NULL,
                model        TEXT    NOT NULL,
                status       TEXT    NOT NULL DEFAULT 'pending',
                avg_score    REAL,
                grounded_pct REAL,
                run_at       TEXT    NOT NULL DEFAULT (datetime('now')),
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS run_results (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id              INTEGER NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
                test_case_id        INTEGER NOT NULL REFERENCES test_cases(id),
                response            TEXT    NOT NULL,
                overall_label       TEXT    NOT NULL,
                hallucination_score REAL    NOT NULL,
                grounded_count      INTEGER NOT NULL DEFAULT 0,
                ungrounded_count    INTEGER NOT NULL DEFAULT 0,
                contradicted_count  INTEGER NOT NULL DEFAULT 0,
                total_sentences     INTEGER NOT NULL DEFAULT 0,
                sentence_results    TEXT    NOT NULL DEFAULT '[]'
            );
        """)
        _migrate(conn)
