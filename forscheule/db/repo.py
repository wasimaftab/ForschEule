"""Database repository – CRUD helpers for papers and recommendations."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field


@dataclass
class Paper:
    source: str
    source_id: str
    title: str
    abstract: str = ""
    authors: str = ""
    published_at: str | None = None
    url: str = ""
    doi: str | None = None
    embedding: bytes | None = None
    id: int | None = None


@dataclass
class Recommendation:
    date: str
    paper_id: int
    score: float
    rank: int
    matched_terms: list[str] = field(default_factory=list)
    id: int | None = None


def upsert_paper(conn: sqlite3.Connection, p: Paper) -> int:
    """Insert or ignore a paper; return its row id."""
    conn.execute(
        """INSERT INTO papers (source, source_id, title, abstract, authors,
                               published_at, url, doi, embedding)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(source, source_id) DO UPDATE SET
               title=excluded.title,
               abstract=excluded.abstract,
               authors=excluded.authors,
               published_at=excluded.published_at,
               url=excluded.url,
               doi=excluded.doi,
               embedding=COALESCE(excluded.embedding, papers.embedding)
        """,
        (
            p.source,
            p.source_id,
            p.title,
            p.abstract,
            p.authors,
            p.published_at,
            p.url,
            p.doi,
            p.embedding,
        ),
    )
    conn.commit()
    # fetch the id
    row = conn.execute(
        "SELECT id FROM papers WHERE source=? AND source_id=?",
        (p.source, p.source_id),
    ).fetchone()
    return row["id"]


def save_recommendations(
    conn: sqlite3.Connection, date: str, recs: list[Recommendation]
) -> None:
    """Replace recommendations for a given date (idempotent)."""
    conn.execute("DELETE FROM recommendations WHERE date=?", (date,))
    for r in recs:
        conn.execute(
            """INSERT INTO recommendations (date, paper_id, score, rank, matched_terms)
               VALUES (?, ?, ?, ?, ?)""",
            (r.date, r.paper_id, r.score, r.rank, json.dumps(r.matched_terms)),
        )
    conn.commit()


def get_recommendations_for_date(
    conn: sqlite3.Connection, date: str
) -> list[dict]:
    """Return recommendations joined with paper data for a given date."""
    rows = conn.execute(
        """SELECT r.rank, r.score, r.matched_terms,
                  p.source, p.source_id, p.title, p.abstract, p.authors,
                  p.published_at, p.url, p.doi
           FROM recommendations r
           JOIN papers p ON r.paper_id = p.id
           WHERE r.date = ?
           ORDER BY r.rank""",
        (date,),
    ).fetchall()
    results = []
    for row in rows:
        d = dict(row)
        d["matched_terms"] = json.loads(d["matched_terms"]) if d["matched_terms"] else []
        results.append(d)
    return results


def get_recommendations_for_date_range(
    conn: sqlite3.Connection, end_date: str, window_days: int
) -> list[dict]:
    """Return deduplicated recommendations across *window_days* days ending on *end_date*.

    The range is inclusive: ``window_days=7`` means 7 days total (end_date
    minus 6 days through end_date).  Papers are deduplicated by paper_id,
    keeping the highest-scoring occurrence.  Results are ordered by score
    descending.
    """
    rows = conn.execute(
        """SELECT r.paper_id, r.rank, r.score, r.matched_terms,
                  p.source, p.source_id, p.title, p.abstract, p.authors,
                  p.published_at, p.url, p.doi
           FROM recommendations r
           JOIN papers p ON r.paper_id = p.id
           WHERE r.date BETWEEN date(?, '-' || (? - 1) || ' days') AND ?
           ORDER BY r.score DESC""",
        (end_date, window_days, end_date),
    ).fetchall()
    seen: set[int] = set()
    results = []
    for row in rows:
        d = dict(row)
        pid = d["paper_id"]
        if pid in seen:
            continue
        seen.add(pid)
        d["matched_terms"] = json.loads(d["matched_terms"]) if d["matched_terms"] else []
        results.append(d)
    return results


def get_all_papers(conn: sqlite3.Connection) -> list[Paper]:
    """Return all papers."""
    rows = conn.execute("SELECT * FROM papers").fetchall()
    return [Paper(**{k: row[k] for k in row.keys() if k != "created_at"}) for row in rows]


def has_recommendations_for_date(conn: sqlite3.Connection, date: str) -> bool:
    """Check if recommendations already exist for a date."""
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM recommendations WHERE date=?", (date,)
    ).fetchone()
    return row["cnt"] > 0


# ---------------------------------------------------------------------------
# Settings CRUD
# ---------------------------------------------------------------------------

def get_setting(conn: sqlite3.Connection, key: str) -> str | None:
    """Get a single setting value by key."""
    row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def get_all_settings(conn: sqlite3.Connection) -> dict[str, str]:
    """Return all settings as a dict."""
    rows = conn.execute("SELECT key, value FROM settings").fetchall()
    return {row["key"]: row["value"] for row in rows}


def put_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Insert or update a single setting."""
    conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()


def put_settings_bulk(conn: sqlite3.Connection, settings: dict[str, str]) -> None:
    """Insert or update multiple settings."""
    for k, v in settings.items():
        conn.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (k, v),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Pipeline run tracking (signature-based idempotency)
# ---------------------------------------------------------------------------

def get_pipeline_run_signature(conn: sqlite3.Connection, date: str) -> str | None:
    """Return the signature for a previous pipeline run on this date, or None."""
    row = conn.execute(
        "SELECT signature FROM pipeline_runs WHERE date=?", (date,)
    ).fetchone()
    return row["signature"] if row else None


def upsert_pipeline_run(
    conn: sqlite3.Connection,
    date: str,
    signature: str,
    window_days: int,
    top_k: int,
) -> None:
    """Record or update the pipeline run metadata for a date."""
    conn.execute(
        """INSERT INTO pipeline_runs (date, signature, window_days, top_k, updated_at)
           VALUES (?, ?, ?, ?, datetime('now'))
           ON CONFLICT(date) DO UPDATE SET
               signature=excluded.signature,
               window_days=excluded.window_days,
               top_k=excluded.top_k,
               updated_at=datetime('now')
        """,
        (date, signature, window_days, top_k),
    )
    conn.commit()


def get_distinct_signatures_in_range(
    conn: sqlite3.Connection, end_date: str, window_days: int
) -> list[dict]:
    """Return distinct pipeline configurations within a date window.

    Groups by (signature, window_days, top_k) so the same configuration
    used across multiple dates counts as one entry.  Each dict has keys:
    ``signature``, ``window_days``, ``top_k``, ``first_date``, ``last_date``,
    ``run_count``.
    """
    rows = conn.execute(
        """SELECT signature, window_days, top_k,
                  MIN(date) AS first_date,
                  MAX(date) AS last_date,
                  COUNT(*)  AS run_count
           FROM pipeline_runs
           WHERE date BETWEEN date(?, '-' || (? - 1) || ' days') AND ?
           GROUP BY signature, window_days, top_k
           ORDER BY first_date""",
        (end_date, window_days, end_date),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Paper summary caching (OpenAI)
# ---------------------------------------------------------------------------

def get_paper_summary(
    conn: sqlite3.Connection,
    paper_id: int,
    model: str,
    schema_version: str,
    input_hash: str,
) -> dict | None:
    """Return cached per-paper summary or None."""
    row = conn.execute(
        """SELECT summary_json FROM paper_summaries
           WHERE paper_id=? AND model=? AND schema_version=? AND input_hash=?""",
        (paper_id, model, schema_version, input_hash),
    ).fetchone()
    return json.loads(row["summary_json"]) if row else None


def save_paper_summary(
    conn: sqlite3.Connection,
    paper_id: int,
    model: str,
    schema_version: str,
    input_hash: str,
    summary: dict,
) -> None:
    """Cache a per-paper summary (upsert)."""
    conn.execute(
        """INSERT INTO paper_summaries
           (paper_id, model, schema_version, input_hash, summary_json)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(paper_id, model, schema_version, input_hash) DO UPDATE SET
               summary_json=excluded.summary_json,
               created_at=datetime('now')
        """,
        (paper_id, model, schema_version, input_hash, json.dumps(summary)),
    )
    conn.commit()


def get_weekly_digest(
    conn: sqlite3.Connection,
    date: str,
    window_days: int,
    top_n: int,
    paper_set_signature: str,
    paper_model: str,
    synthesis_model: str,
) -> dict | None:
    """Return cached weekly digest or None."""
    row = conn.execute(
        """SELECT digest_json FROM weekly_digests
           WHERE date=? AND window_days=? AND top_n=? AND paper_set_signature=?
                 AND paper_model=? AND synthesis_model=?""",
        (date, window_days, top_n, paper_set_signature, paper_model, synthesis_model),
    ).fetchone()
    return json.loads(row["digest_json"]) if row else None


def save_weekly_digest(
    conn: sqlite3.Connection,
    date: str,
    window_days: int,
    top_n: int,
    paper_set_signature: str,
    paper_model: str,
    synthesis_model: str,
    digest: dict,
) -> None:
    """Cache a weekly digest (upsert)."""
    conn.execute(
        """INSERT INTO weekly_digests
           (date, window_days, top_n, paper_set_signature,
            paper_model, synthesis_model, digest_json)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(date, window_days, top_n, paper_set_signature,
                       paper_model, synthesis_model)
           DO UPDATE SET digest_json=excluded.digest_json, created_at=datetime('now')
        """,
        (date, window_days, top_n, paper_set_signature,
         paper_model, synthesis_model, json.dumps(digest)),
    )
    conn.commit()
