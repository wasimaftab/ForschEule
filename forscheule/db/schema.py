"""SQLite schema initialisation."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT    NOT NULL CHECK(source IN ('pubmed','arxiv')),
    source_id   TEXT    NOT NULL,
    title       TEXT    NOT NULL,
    abstract    TEXT    NOT NULL DEFAULT '',
    authors     TEXT    NOT NULL DEFAULT '',
    published_at TEXT,
    url         TEXT    NOT NULL DEFAULT '',
    doi         TEXT,
    embedding   BLOB,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source, source_id)
);

CREATE TABLE IF NOT EXISTS recommendations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    date         TEXT    NOT NULL,
    paper_id     INTEGER NOT NULL REFERENCES papers(id),
    score        REAL    NOT NULL,
    rank         INTEGER NOT NULL,
    matched_terms TEXT,
    UNIQUE(date, rank)
);

CREATE TABLE IF NOT EXISTS profiles (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL UNIQUE,
    description   TEXT    NOT NULL DEFAULT '',
    boosted_terms TEXT    NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    date        TEXT PRIMARY KEY,
    signature   TEXT NOT NULL,
    window_days INTEGER NOT NULL,
    top_k       INTEGER NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS paper_summaries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id        INTEGER NOT NULL REFERENCES papers(id),
    model           TEXT NOT NULL,
    schema_version  TEXT NOT NULL,
    input_hash      TEXT NOT NULL,
    summary_json    TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(paper_id, model, schema_version, input_hash)
);

CREATE TABLE IF NOT EXISTS weekly_digests (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    date                TEXT NOT NULL,
    window_days         INTEGER NOT NULL,
    top_n               INTEGER NOT NULL,
    paper_set_signature TEXT NOT NULL,
    paper_model         TEXT NOT NULL,
    synthesis_model     TEXT NOT NULL,
    digest_json         TEXT NOT NULL,
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(date, window_days, top_n, paper_set_signature, paper_model, synthesis_model)
);

CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source, source_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_date ON recommendations(date);
CREATE INDEX IF NOT EXISTS idx_paper_summaries_paper ON paper_summaries(paper_id);
CREATE INDEX IF NOT EXISTS idx_weekly_digests_date ON weekly_digests(date);
"""


def init_db(db_path: Path) -> sqlite3.Connection:
    """Create / open the database and ensure the schema exists."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn
