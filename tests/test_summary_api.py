"""Tests for summary-related API endpoints."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from forscheule.db.repo import (
    Paper,
    Recommendation,
    save_recommendations,
    save_weekly_digest,
    upsert_paper,
)
from forscheule.db.schema import init_db


def _make_test_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
    tmp.close()
    db_path = Path(tmp.name)
    conn = init_db(db_path)

    pids = []
    for i in range(5):
        pid = upsert_paper(
            conn,
            Paper(
                source="pubmed",
                source_id=str(2000 + i),
                title=f"Summary Paper {i}",
                abstract=f"Abstract for summary paper {i}",
                published_at="2026-02-09",
                url=f"https://example.com/s{i}",
            ),
        )
        pids.append(pid)

    save_recommendations(
        conn,
        "2026-02-09",
        [
            Recommendation(
                date="2026-02-09",
                paper_id=pids[j],
                score=round(0.95 - j * 0.05, 2),
                rank=j + 1,
            )
            for j in range(5)
        ],
    )

    conn.close()
    return db_path


@pytest.fixture()
def client_no_key():
    """Client with OpenAI unavailable."""
    db_path = _make_test_db()
    patcher_db = patch("forscheule.api.app.DB_PATH", db_path)
    patcher_status = patch(
        "forscheule.api.app.openai_status",
        return_value=(False, "OpenAI API key is not configured."),
    )
    patcher_db.start()
    patcher_status.start()
    from forscheule.api.app import app

    yield TestClient(app)
    patcher_status.stop()
    patcher_db.stop()


@pytest.fixture()
def client_with_key():
    """Client with OpenAI available (thread mocked)."""
    db_path = _make_test_db()
    patcher_db = patch("forscheule.api.app.DB_PATH", db_path)
    patcher_status = patch(
        "forscheule.api.app.openai_status",
        return_value=(True, ""),
    )
    patcher_thread = patch("forscheule.api.app._run_job_in_thread")
    patcher_db.start()
    patcher_status.start()
    patcher_thread.start()

    from forscheule.api.app import _job_lock, _jobs, app

    yield TestClient(app), _jobs, _job_lock
    patcher_thread.stop()
    patcher_status.stop()
    patcher_db.stop()


# ---------------------------------------------------------------------------
# POST /jobs/summary-weekly
# ---------------------------------------------------------------------------

def test_summary_weekly_no_api_key(client_no_key):
    resp = client_no_key.post(
        "/jobs/summary-weekly",
        json={"date": "2026-02-09"},
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 503
    assert "API key" in resp.json()["error"]


def test_summary_weekly_returns_job(client_with_key):
    client, _jobs, _job_lock = client_with_key
    resp = client.post(
        "/jobs/summary-weekly",
        json={"date": "2026-02-09", "window": 7, "top_n": 5},
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "queued"
    # Clean up
    with _job_lock:
        _jobs.clear()


def test_summary_weekly_blocked_by_running_job(client_with_key):
    client, _jobs, _job_lock = client_with_key
    with _job_lock:
        _jobs["fake"] = {"type": "run-daily", "status": "running", "error": None}
    resp = client.post(
        "/jobs/summary-weekly",
        json={"date": "2026-02-09"},
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 409
    with _job_lock:
        _jobs.clear()


# ---------------------------------------------------------------------------
# GET /weekly-summary
# ---------------------------------------------------------------------------

def test_weekly_summary_not_found(client_no_key):
    resp = client_no_key.get(
        "/weekly-summary",
        params={"date": "2026-02-09"},
    )
    assert resp.status_code == 404


def test_weekly_summary_bad_date(client_no_key):
    resp = client_no_key.get(
        "/weekly-summary",
        params={"date": "bad-date"},
    )
    assert resp.status_code == 400


def test_summary_weekly_bad_window(client_with_key):
    """Non-integer window should return 400, not 500."""
    client, _jobs, _job_lock = client_with_key
    resp = client.post(
        "/jobs/summary-weekly",
        json={"date": "2026-02-09", "window": "abc"},
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert "window" in resp.json()["error"]
    with _job_lock:
        _jobs.clear()


def test_summary_weekly_bad_top_n(client_with_key):
    """Non-integer top_n should return 400, not 500."""
    client, _jobs, _job_lock = client_with_key
    resp = client.post(
        "/jobs/summary-weekly",
        json={"date": "2026-02-09", "top_n": "xyz"},
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert "top_n" in resp.json()["error"]
    with _job_lock:
        _jobs.clear()


def test_run_daily_bad_window():
    """Non-integer window on run-daily should return 400."""
    db_path = _make_test_db()
    patcher_db = patch("forscheule.api.app.DB_PATH", db_path)
    patcher_thread = patch("forscheule.api.app._run_job_in_thread")
    patcher_db.start()
    patcher_thread.start()
    from forscheule.api.app import _job_lock, _jobs, app

    c = TestClient(app)
    resp = c.post(
        "/jobs/run-daily",
        json={"window": "not_a_number"},
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert "window" in resp.json()["error"]
    with _job_lock:
        _jobs.clear()
    patcher_thread.stop()
    patcher_db.stop()


def test_backfill_bad_days():
    """Non-integer days on backfill should return 400."""
    db_path = _make_test_db()
    patcher_db = patch("forscheule.api.app.DB_PATH", db_path)
    patcher_thread = patch("forscheule.api.app._run_job_in_thread")
    patcher_db.start()
    patcher_thread.start()
    from forscheule.api.app import _job_lock, _jobs, app

    c = TestClient(app)
    resp = c.post(
        "/jobs/backfill",
        json={"days": "bad"},
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert "days" in resp.json()["error"]
    with _job_lock:
        _jobs.clear()
    patcher_thread.stop()
    patcher_db.stop()


def test_summary_weekly_bad_date(client_with_key):
    """Invalid date should return 400 before queuing."""
    client, _jobs, _job_lock = client_with_key
    resp = client.post(
        "/jobs/summary-weekly",
        json={"date": "not-a-date"},
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert "date" in resp.json()["error"].lower()
    # No job should have been queued
    with _job_lock:
        assert not any(j["type"] == "summary-weekly" for j in _jobs.values())
        _jobs.clear()


def test_summary_weekly_force_string_false(client_with_key):
    """String 'false' should NOT be treated as force=True."""
    client, _jobs, _job_lock = client_with_key
    resp = client.post(
        "/jobs/summary-weekly",
        json={"date": "2026-02-09", "force": "false"},
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 200
    data = resp.json()
    # Verify the job was queued (we can't inspect kwargs directly, but the
    # job should be created without error)
    assert data["status"] == "queued"
    with _job_lock:
        _jobs.clear()


def test_summary_weekly_blocked_by_queued_job(client_with_key):
    """A queued (not yet running) job should also block new submissions."""
    client, _jobs, _job_lock = client_with_key
    with _job_lock:
        _jobs["fake-q"] = {"type": "run-daily", "status": "queued", "error": None}
    resp = client.post(
        "/jobs/summary-weekly",
        json={"date": "2026-02-09"},
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 409
    with _job_lock:
        _jobs.clear()


def test_run_daily_blocked_by_queued_job():
    """A queued job should prevent run-daily from creating a second job."""
    db_path = _make_test_db()
    patcher_db = patch("forscheule.api.app.DB_PATH", db_path)
    patcher_thread = patch("forscheule.api.app._run_job_in_thread")
    patcher_db.start()
    patcher_thread.start()
    from forscheule.api.app import _job_lock, _jobs, app

    c = TestClient(app)
    with _job_lock:
        _jobs["queued-fake"] = {"type": "backfill", "status": "queued", "error": None}
    resp = c.post(
        "/jobs/run-daily",
        json={},
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 409
    with _job_lock:
        _jobs.clear()
    patcher_thread.stop()
    patcher_db.stop()


def test_parse_int_rejects_boolean_window(client_with_key):
    """Boolean values in integer fields should return 400, not silently cast."""
    client, _jobs, _job_lock = client_with_key
    resp = client.post(
        "/jobs/summary-weekly",
        json={"date": "2026-02-09", "window": True},
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert "boolean" in resp.json()["error"].lower()
    with _job_lock:
        _jobs.clear()


def test_parse_int_rejects_boolean_top_n(client_with_key):
    """Boolean top_n should return 400."""
    client, _jobs, _job_lock = client_with_key
    resp = client.post(
        "/jobs/summary-weekly",
        json={"date": "2026-02-09", "top_n": False},
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert "boolean" in resp.json()["error"].lower()
    with _job_lock:
        _jobs.clear()


def test_weekly_summary_found():
    """Insert a digest directly, then fetch via API."""
    db_path = _make_test_db()
    conn = init_db(db_path)

    digest = {
        "themes": [{"title": "Theme1", "summary": "S1", "paper_ids": []}],
        "contradictions_or_tensions": [],
        "what_to_read_first": [],
        "methods_trends": [],
        "recommended_next_queries": [],
    }
    save_weekly_digest(
        conn, "2026-02-09", 7, 10, "sig123", "gpt-5-mini", "gpt-5.2", digest
    )
    conn.close()

    patcher_db = patch("forscheule.api.app.DB_PATH", db_path)
    patcher_db.start()
    from forscheule.api.app import app

    client = TestClient(app)
    resp = client.get(
        "/weekly-summary",
        params={
            "date": "2026-02-09",
            "window": 7,
            "top_n": 10,
            "paper_model": "gpt-5-mini",
            "synthesis_model": "gpt-5.2",
        },
    )
    patcher_db.stop()

    assert resp.status_code == 200
    data = resp.json()
    assert data["date"] == "2026-02-09"
    assert data["digest"]["themes"][0]["title"] == "Theme1"
    assert "created_at" in data


def test_summary_weekly_malformed_json(client_with_key):
    """Malformed JSON body on summary-weekly should return 400."""
    client, _jobs, _job_lock = client_with_key
    resp = client.post(
        "/jobs/summary-weekly",
        content=b"{bad json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert "Invalid JSON" in resp.json()["error"]
    with _job_lock:
        _jobs.clear()


def test_summary_weekly_rejects_number_body(client_with_key):
    """JSON number body on summary-weekly should return 400."""
    client, _jobs, _job_lock = client_with_key
    resp = client.post(
        "/jobs/summary-weekly",
        content=b"123",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert "object" in resp.json()["error"].lower()
    with _job_lock:
        _jobs.clear()
