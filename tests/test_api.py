"""Tests for the FastAPI endpoints."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from forscheule.db.repo import (
    Paper,
    Recommendation,
    save_recommendations,
    upsert_paper,
)
from forscheule.db.schema import init_db


def _make_test_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
    tmp.close()
    db_path = Path(tmp.name)
    conn = init_db(db_path)

    pids = []
    for i in range(10):
        pid = upsert_paper(
            conn,
            Paper(
                source="pubmed",
                source_id=str(1000 + i),
                title=f"Test Paper {i}",
                abstract=f"Abstract for test paper {i}",
                published_at="2024-01-15",
                url=f"https://example.com/{i}",
                doi=f"10.1234/test{i}",
            ),
        )
        pids.append(pid)

    save_recommendations(
        conn,
        "2024-01-15",
        [
            Recommendation(
                date="2024-01-15",
                paper_id=pids[j],
                score=round(0.95 - j * 0.05, 2),
                rank=j + 1,
            )
            for j in range(10)
        ],
    )

    conn.close()
    return db_path


@pytest.fixture()
def client():
    db_path = _make_test_db()
    patcher = patch("forscheule.api.app.DB_PATH", db_path)
    patcher.start()
    from forscheule.api.app import app

    yield TestClient(app)
    patcher.stop()


# ---------------------------------------------------------------------------
# Backward-compatible /daily tests
# ---------------------------------------------------------------------------

def test_daily_returns_results(client):
    resp = client.get("/daily", params={"date": "2024-01-15"})
    assert resp.status_code == 200
    assert len(resp.json()) == 10


def test_daily_bad_date(client):
    resp = client.get("/daily", params={"date": "not-a-date"})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /daily with top_k
# ---------------------------------------------------------------------------

def test_daily_top_k_override(client):
    resp = client.get("/daily", params={"date": "2024-01-15", "top_k": "3"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 3
    assert data[0]["rank"] == 1


def test_daily_default_returns_all_stored(client):
    """Without top_k param, returns all stored recommendations."""
    resp = client.get("/daily", params={"date": "2024-01-15"})
    assert resp.status_code == 200
    assert len(resp.json()) == 10


# ---------------------------------------------------------------------------
# UI route
# ---------------------------------------------------------------------------

def test_ui_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "ForschEule" in resp.text
    assert "Results" in resp.text
    assert "Settings" in resp.text


# ---------------------------------------------------------------------------
# Settings endpoints
# ---------------------------------------------------------------------------

def test_get_settings_defaults(client):
    resp = client.get("/settings")
    assert resp.status_code == 200
    data = resp.json()
    assert "lab_profile" in data
    assert "boosted_phrases" in data
    assert "top_k" in data
    assert data["top_k"] == 5


def test_put_and_get_settings(client):
    resp = client.put("/settings", json={
        "lab_profile": "Test profile",
        "boosted_phrases": ["term1", "term2"],
        "top_k": 8,
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    data = client.get("/settings").json()
    assert data["lab_profile"] == "Test profile"
    assert data["boosted_phrases"] == ["term1", "term2"]
    assert data["top_k"] == 8


def test_put_settings_partial(client):
    resp = client.put("/settings", json={"top_k": 3})
    assert resp.status_code == 200
    assert client.get("/settings").json()["top_k"] == 3


def test_put_settings_validation(client):
    assert client.put("/settings", json={"top_k": -1}).status_code == 400
    assert client.put("/settings", json={"lab_profile": ""}).status_code == 400
    assert client.put(
        "/settings", json={"boosted_phrases": "not a list"}
    ).status_code == 400


# ---------------------------------------------------------------------------
# Job endpoints
# ---------------------------------------------------------------------------

def test_job_not_found(client):
    assert client.get("/jobs/nonexistent").status_code == 404


def test_backfill_requires_days(client):
    resp = client.post(
        "/jobs/backfill",
        json={},
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert "days" in resp.json()["error"]


def test_run_daily_returns_job_id():
    db_path = _make_test_db()
    patcher_db = patch("forscheule.api.app.DB_PATH", db_path)
    patcher_thread = patch("forscheule.api.app._run_job_in_thread")
    patcher_db.start()
    patcher_thread.start()
    try:
        from forscheule.api.app import _job_lock, _jobs, app

        c = TestClient(app)
        resp = c.post(
            "/jobs/run-daily",
            json={"window": 7},
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "queued"

        with _job_lock:
            _jobs[data["job_id"]]["status"] = "success"

        assert c.get(f"/jobs/{data['job_id']}").json()["status"] == "success"

        with _job_lock:
            _jobs.clear()
    finally:
        patcher_thread.stop()
        patcher_db.stop()


def test_concurrent_job_blocked():
    db_path = _make_test_db()
    patcher_db = patch("forscheule.api.app.DB_PATH", db_path)
    patcher_thread = patch("forscheule.api.app._run_job_in_thread")
    patcher_db.start()
    patcher_thread.start()
    try:
        from forscheule.api.app import _job_lock, _jobs, app

        c = TestClient(app)
        with _job_lock:
            _jobs["fake"] = {"type": "run-daily", "status": "running", "error": None}

        resp = c.post(
            "/jobs/run-daily",
            json={},
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 409

        with _job_lock:
            _jobs.clear()
    finally:
        patcher_thread.stop()
        patcher_db.stop()


# ---------------------------------------------------------------------------
# Atomic enqueue (TOCTOU fix) and job pruning tests
# ---------------------------------------------------------------------------

def test_try_enqueue_is_atomic():
    """_try_enqueue_job check-and-insert is atomic – no TOCTOU race."""
    from forscheule.api.app import _job_lock, _jobs, _try_enqueue_job

    with _job_lock:
        _jobs.clear()

    # First enqueue should succeed
    jid1 = _try_enqueue_job("run-daily")
    assert jid1 is not None

    # Second enqueue while first is still queued should fail
    jid2 = _try_enqueue_job("backfill")
    assert jid2 is None

    with _job_lock:
        _jobs.clear()


def test_try_enqueue_succeeds_after_job_completes():
    """After a job finishes, a new one can be enqueued."""
    from forscheule.api.app import _job_lock, _jobs, _try_enqueue_job

    with _job_lock:
        _jobs.clear()

    jid1 = _try_enqueue_job("run-daily")
    assert jid1 is not None

    # Simulate job completing
    with _job_lock:
        _jobs[jid1]["status"] = "success"
        _jobs[jid1]["_finished_at"] = time.monotonic()

    jid2 = _try_enqueue_job("backfill")
    assert jid2 is not None

    with _job_lock:
        _jobs.clear()


def test_stale_jobs_are_pruned():
    """Terminal jobs older than TTL are removed on next enqueue."""
    from forscheule.api import app as app_module
    from forscheule.api.app import _job_lock, _jobs, _try_enqueue_job

    with _job_lock:
        _jobs.clear()

    # Inject a stale completed job with _finished_at far in the past
    old_time = time.monotonic() - app_module._STALE_JOB_TTL - 1
    with _job_lock:
        _jobs["stale-1"] = {
            "type": "run-daily",
            "status": "success",
            "error": None,
            "_finished_at": old_time,
        }
        _jobs["stale-2"] = {
            "type": "backfill",
            "status": "failed",
            "error": "some error",
            "_finished_at": old_time,
        }

    assert len(_jobs) == 2

    # Enqueue a new job — stale entries should be pruned
    jid = _try_enqueue_job("run-daily")
    assert jid is not None

    with _job_lock:
        # Stale jobs should be gone, only the new job remains
        assert "stale-1" not in _jobs
        assert "stale-2" not in _jobs
        assert jid in _jobs

    with _job_lock:
        _jobs.clear()


def test_recent_completed_jobs_not_pruned():
    """Recently completed jobs should NOT be pruned (still queryable)."""
    from forscheule.api.app import _job_lock, _jobs, _try_enqueue_job

    with _job_lock:
        _jobs.clear()

    # Insert a recently finished job
    with _job_lock:
        _jobs["recent-ok"] = {
            "type": "run-daily",
            "status": "success",
            "error": None,
            "_finished_at": time.monotonic(),  # just now
        }

    # Enqueue new job — recent finished job should remain
    jid = _try_enqueue_job("backfill")
    assert jid is not None

    with _job_lock:
        assert "recent-ok" in _jobs  # not pruned
        assert jid in _jobs

    with _job_lock:
        _jobs.clear()


# ---------------------------------------------------------------------------
# Malformed JSON body tests
# ---------------------------------------------------------------------------

def test_run_daily_malformed_json(client):
    """Malformed JSON body should return 400, not 500."""
    resp = client.post(
        "/jobs/run-daily",
        content=b"{not valid json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert "Invalid JSON" in resp.json()["error"]


def test_backfill_malformed_json(client):
    """Malformed JSON body should return 400, not 500."""
    resp = client.post(
        "/jobs/backfill",
        content=b"{{bad",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert "Invalid JSON" in resp.json()["error"]


def test_run_daily_no_content_type(client):
    """No content-type header should be treated as empty body (no error)."""
    patcher_thread = patch("forscheule.api.app._run_job_in_thread")
    patcher_thread.start()
    from forscheule.api.app import _job_lock, _jobs

    resp = client.post("/jobs/run-daily")
    # Should succeed with defaults (empty kwargs)
    assert resp.status_code == 200
    with _job_lock:
        _jobs.clear()
    patcher_thread.stop()


def test_put_settings_malformed_json(client):
    """Malformed JSON on PUT /settings should return 400."""
    resp = client.put(
        "/settings",
        content=b"not json at all",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert "Invalid JSON" in resp.json()["error"]


def test_put_settings_no_content_type(client):
    """No content-type → empty body → 'No valid settings provided.'"""
    resp = client.put("/settings")
    assert resp.status_code == 400
    assert "No valid settings" in resp.json()["error"]


# ---------------------------------------------------------------------------
# PUT /settings boolean top_k guard
# ---------------------------------------------------------------------------

def test_put_settings_rejects_boolean_top_k(client):
    """top_k: true should return 400, not silently cast to 1."""
    resp = client.put("/settings", json={"top_k": True})
    assert resp.status_code == 400
    assert "boolean" in resp.json()["error"].lower()


def test_put_settings_rejects_boolean_false_top_k(client):
    """top_k: false should return 400, not silently cast to 0."""
    resp = client.put("/settings", json={"top_k": False})
    assert resp.status_code == 400
    assert "boolean" in resp.json()["error"].lower()


# ---------------------------------------------------------------------------
# Non-object JSON body tests
# ---------------------------------------------------------------------------

def test_run_daily_rejects_array_body(client):
    """JSON array body should return 400."""
    resp = client.post(
        "/jobs/run-daily",
        content=b"[]",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert "object" in resp.json()["error"].lower()


def test_backfill_rejects_string_body(client):
    """JSON string body should return 400."""
    resp = client.post(
        "/jobs/backfill",
        content=b'"abc"',
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert "object" in resp.json()["error"].lower()


def test_put_settings_rejects_array_body(client):
    """JSON array body on PUT /settings should return 400."""
    resp = client.put(
        "/settings",
        content=b"[]",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert "object" in resp.json()["error"].lower()
