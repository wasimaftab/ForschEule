"""Tests for database schema and repository."""

from __future__ import annotations

import tempfile
from pathlib import Path

from forscheule.db.repo import (
    Paper,
    Recommendation,
    get_all_settings,
    get_distinct_signatures_in_range,
    get_recommendations_for_date,
    get_recommendations_for_date_range,
    get_setting,
    has_recommendations_for_date,
    put_setting,
    put_settings_bulk,
    save_recommendations,
    upsert_paper,
    upsert_pipeline_run,
)
from forscheule.db.schema import init_db


def _tmp_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
    tmp.close()
    return init_db(Path(tmp.name))


def test_init_db_creates_tables():
    conn = _tmp_db()
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    names = {row["name"] for row in tables}
    assert "papers" in names
    assert "recommendations" in names
    assert "profiles" in names
    assert "settings" in names
    assert "pipeline_runs" in names
    assert "paper_summaries" in names
    assert "weekly_digests" in names
    conn.close()


def test_upsert_paper_insert_and_update():
    conn = _tmp_db()
    p = Paper(source="pubmed", source_id="12345", title="Test Paper", abstract="An abstract")
    pid1 = upsert_paper(conn, p)
    assert pid1 > 0

    # Upsert same paper with updated title
    p.title = "Updated Title"
    pid2 = upsert_paper(conn, p)
    assert pid1 == pid2

    row = conn.execute("SELECT title FROM papers WHERE id=?", (pid1,)).fetchone()
    assert row["title"] == "Updated Title"
    conn.close()


def test_save_and_get_recommendations():
    conn = _tmp_db()
    pid = upsert_paper(
        conn,
        Paper(source="arxiv", source_id="2401.00001", title="Paper A", abstract="Abstract A"),
    )
    recs = [
        Recommendation(
            date="2024-01-15", paper_id=pid, score=0.95, rank=1, matched_terms=["spatial"]
        )
    ]
    save_recommendations(conn, "2024-01-15", recs)

    assert has_recommendations_for_date(conn, "2024-01-15")
    assert not has_recommendations_for_date(conn, "2024-01-16")

    result = get_recommendations_for_date(conn, "2024-01-15")
    assert len(result) == 1
    assert result[0]["title"] == "Paper A"
    assert result[0]["matched_terms"] == ["spatial"]
    conn.close()


def test_idempotent_recommendations():
    """Saving recommendations for same date replaces them (idempotent)."""
    conn = _tmp_db()
    pid = upsert_paper(
        conn,
        Paper(source="pubmed", source_id="99999", title="Paper B", abstract="Abstract B"),
    )
    recs = [Recommendation(date="2024-01-15", paper_id=pid, score=0.8, rank=1)]
    save_recommendations(conn, "2024-01-15", recs)
    save_recommendations(conn, "2024-01-15", recs)  # re-save

    result = get_recommendations_for_date(conn, "2024-01-15")
    assert len(result) == 1  # no duplicates
    conn.close()


def test_settings_crud():
    conn = _tmp_db()

    # Initially empty
    assert get_setting(conn, "lab_profile") is None
    assert get_all_settings(conn) == {}

    # Put single
    put_setting(conn, "lab_profile", "My lab")
    assert get_setting(conn, "lab_profile") == "My lab"

    # Update single
    put_setting(conn, "lab_profile", "Updated lab")
    assert get_setting(conn, "lab_profile") == "Updated lab"

    # Bulk put
    put_settings_bulk(conn, {"top_k": "10", "lab_profile": "Bulk lab"})
    s = get_all_settings(conn)
    assert s["top_k"] == "10"
    assert s["lab_profile"] == "Bulk lab"

    conn.close()


def test_get_recommendations_for_date_range():
    """Date range query aggregates across multiple dates, deduplicates by paper_id."""
    conn = _tmp_db()
    p1 = upsert_paper(conn, Paper(source="pubmed", source_id="A1", title="P1", abstract="A1"))
    p2 = upsert_paper(conn, Paper(source="pubmed", source_id="A2", title="P2", abstract="A2"))
    p3 = upsert_paper(conn, Paper(source="pubmed", source_id="A3", title="P3", abstract="A3"))

    # p1 on day 1 with score 0.9, p2 on day 1 with 0.8
    save_recommendations(conn, "2026-02-08", [
        Recommendation(date="2026-02-08", paper_id=p1, score=0.9, rank=1),
        Recommendation(date="2026-02-08", paper_id=p2, score=0.8, rank=2),
    ])
    # p1 again on day 2 with lower score 0.7, p3 new with 0.85
    save_recommendations(conn, "2026-02-09", [
        Recommendation(date="2026-02-09", paper_id=p1, score=0.7, rank=2),
        Recommendation(date="2026-02-09", paper_id=p3, score=0.85, rank=1),
    ])

    results = get_recommendations_for_date_range(conn, "2026-02-09", 7)
    paper_ids = [r["paper_id"] for r in results]

    # All 3 papers present, no duplicates
    assert len(results) == 3
    assert len(set(paper_ids)) == 3

    # Ordered by score desc: p1(0.9), p3(0.85), p2(0.8)
    assert results[0]["paper_id"] == p1
    assert results[0]["score"] == 0.9  # kept highest score
    assert results[1]["paper_id"] == p3
    assert results[2]["paper_id"] == p2

    conn.close()


def test_date_range_outside_window_excluded():
    """Papers outside the date window should not appear."""
    conn = _tmp_db()
    p1 = upsert_paper(conn, Paper(source="pubmed", source_id="X1", title="Old", abstract="Old"))
    p2 = upsert_paper(conn, Paper(source="pubmed", source_id="X2", title="New", abstract="New"))

    save_recommendations(conn, "2026-01-01", [
        Recommendation(date="2026-01-01", paper_id=p1, score=0.9, rank=1),
    ])
    save_recommendations(conn, "2026-02-09", [
        Recommendation(date="2026-02-09", paper_id=p2, score=0.8, rank=1),
    ])

    results = get_recommendations_for_date_range(conn, "2026-02-09", 7)
    assert len(results) == 1
    assert results[0]["paper_id"] == p2

    conn.close()


def test_date_range_window_is_exact_n_days():
    """window_days=7 should span exactly 7 days (end_date - 6 through end_date)."""
    conn = _tmp_db()
    # end_date = 2026-02-09, window=7 → range is 2026-02-03 through 2026-02-09
    p_inside = upsert_paper(
        conn, Paper(source="pubmed", source_id="W1", title="Inside", abstract="I")
    )
    p_boundary = upsert_paper(
        conn, Paper(source="pubmed", source_id="W2", title="Boundary", abstract="B")
    )
    p_outside = upsert_paper(
        conn, Paper(source="pubmed", source_id="W3", title="Outside", abstract="O")
    )

    # 2026-02-03 = exactly 6 days before 2026-02-09 (first day in 7-day window)
    save_recommendations(conn, "2026-02-03", [
        Recommendation(date="2026-02-03", paper_id=p_boundary, score=0.7, rank=1),
    ])
    # 2026-02-02 = 7 days before 2026-02-09 (outside the 7-day window)
    save_recommendations(conn, "2026-02-02", [
        Recommendation(date="2026-02-02", paper_id=p_outside, score=0.9, rank=1),
    ])
    save_recommendations(conn, "2026-02-09", [
        Recommendation(date="2026-02-09", paper_id=p_inside, score=0.8, rank=1),
    ])

    results = get_recommendations_for_date_range(conn, "2026-02-09", 7)
    result_ids = {r["paper_id"] for r in results}

    assert p_inside in result_ids, "end_date itself should be included"
    assert p_boundary in result_ids, "first day of window (end - 6) should be included"
    assert p_outside not in result_ids, "day before window (end - 7) should be excluded"
    assert len(results) == 2

    conn.close()


def test_get_distinct_signatures_in_range():
    """Detect multiple pipeline configurations within a date window."""
    conn = _tmp_db()
    upsert_pipeline_run(conn, "2026-02-07", "sig_aaa111", window_days=7, top_k=10)
    upsert_pipeline_run(conn, "2026-02-08", "sig_aaa111", window_days=7, top_k=10)
    upsert_pipeline_run(conn, "2026-02-09", "sig_bbb222", window_days=7, top_k=5)

    sigs = get_distinct_signatures_in_range(conn, "2026-02-09", 7)
    signatures = [s["signature"] for s in sigs]

    # GROUP BY collapses the two sig_aaa111 rows into one group
    assert len(sigs) == 2
    assert "sig_aaa111" in signatures
    assert "sig_bbb222" in signatures

    # Verify aggregated metadata for the grouped entry
    aaa = next(s for s in sigs if s["signature"] == "sig_aaa111")
    assert aaa["first_date"] == "2026-02-07"
    assert aaa["last_date"] == "2026-02-08"
    assert aaa["run_count"] == 2

    bbb = next(s for s in sigs if s["signature"] == "sig_bbb222")
    assert bbb["first_date"] == "2026-02-09"
    assert bbb["last_date"] == "2026-02-09"
    assert bbb["run_count"] == 1

    # Narrow window to just 1 day should only see the last signature
    sigs_one = get_distinct_signatures_in_range(conn, "2026-02-09", 1)
    assert len(sigs_one) == 1
    assert sigs_one[0]["signature"] == "sig_bbb222"

    conn.close()


def test_get_distinct_signatures_empty_range():
    """No pipeline runs in range should return empty list."""
    conn = _tmp_db()
    sigs = get_distinct_signatures_in_range(conn, "2026-02-09", 7)
    assert sigs == []
    conn.close()


def test_identical_signature_across_dates_yields_one_group():
    """Same config on every day in the window should yield exactly one group."""
    conn = _tmp_db()
    for day in ("2026-02-03", "2026-02-04", "2026-02-05", "2026-02-06",
                "2026-02-07", "2026-02-08", "2026-02-09"):
        upsert_pipeline_run(conn, day, "same_sig", window_days=7, top_k=10)

    sigs = get_distinct_signatures_in_range(conn, "2026-02-09", 7)
    assert len(sigs) == 1
    assert sigs[0]["signature"] == "same_sig"
    assert sigs[0]["run_count"] == 7
    assert sigs[0]["first_date"] == "2026-02-03"
    assert sigs[0]["last_date"] == "2026-02-09"

    conn.close()
