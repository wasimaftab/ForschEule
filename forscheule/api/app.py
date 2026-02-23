"""ForschEule API – read endpoint, web console, settings, and job triggers."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import date as date_cls
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from forscheule.config import DB_PATH, get_runtime_settings, openai_status
from forscheule.db.repo import (
    get_distinct_signatures_in_range,
    get_recommendations_for_date,
    get_recommendations_for_date_range,
    put_settings_bulk,
)
from forscheule.db.schema import init_db

logger = logging.getLogger(__name__)

app = FastAPI(title="ForschEule", version="0.2.0")

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

# ---------------------------------------------------------------------------
# Job state (in-memory, single-process)
# ---------------------------------------------------------------------------
_jobs: dict[str, dict] = {}
_job_lock = threading.Lock()


async def _safe_json_body(request: Request) -> dict | JSONResponse:
    """Parse JSON body safely, returning 400 on decode errors.

    Returns ``{}`` when there is no body or the content-type is not JSON.
    Rejects non-object JSON values (arrays, strings, numbers, etc.).
    """
    content_type = request.headers.get("content-type", "")
    if "application/json" not in content_type:
        # No JSON content-type → treat as empty body (optional params use defaults)
        return {}
    try:
        parsed = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON body."},
        )
    if not isinstance(parsed, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "JSON body must be an object."},
        )
    return parsed


def _parse_int(value, name: str, min_val: int = 1, max_val: int = 1000) -> int | JSONResponse:
    """Parse an int from a request body value, returning a 400 JSONResponse on failure."""
    if isinstance(value, bool):
        return JSONResponse(
            status_code=400,
            content={"error": f"'{name}' must be an integer, not a boolean."},
        )
    try:
        result = int(value)
    except (ValueError, TypeError):
        return JSONResponse(
            status_code=400,
            content={"error": f"'{name}' must be an integer."},
        )
    if result < min_val or result > max_val:
        return JSONResponse(
            status_code=400,
            content={"error": f"'{name}' must be between {min_val} and {max_val}."},
        )
    return result


_STALE_JOB_TTL = 3600  # prune completed/failed jobs older than 1 hour


def _try_enqueue_job(job_type: str) -> str | None:
    """Atomically check no active job exists and enqueue a new one.

    Returns the new ``job_id`` on success, or ``None`` if a job is already
    active (queued or running).  Also prunes terminal jobs older than
    ``_STALE_JOB_TTL`` seconds.
    """
    now = time.monotonic()
    job_id = str(uuid.uuid4())[:8]
    with _job_lock:
        # Prune stale terminal jobs
        stale = [
            jid for jid, j in _jobs.items()
            if j["status"] in ("success", "failed")
            and now - j.get("_finished_at", now) > _STALE_JOB_TTL
        ]
        for jid in stale:
            del _jobs[jid]

        # Reject if any job is still active
        if any(j["status"] in ("running", "queued") for j in _jobs.values()):
            return None

        _jobs[job_id] = {
            "type": job_type,
            "status": "queued",
            "error": None,
            "_enqueued_at": now,
        }
    return job_id


def _run_job_in_thread(job_id: str, job_type: str, kwargs: dict) -> None:
    """Execute a pipeline job in a background thread."""
    with _job_lock:
        _jobs[job_id]["status"] = "running"

    try:
        from forscheule.config import setup_logging
        from forscheule.pipeline import backfill, run_daily

        setup_logging()

        if job_type == "run-daily":
            run_daily(**kwargs)
        elif job_type == "backfill":
            backfill(**kwargs)
        elif job_type == "summary-weekly":
            _run_summary_job(**kwargs)
        with _job_lock:
            _jobs[job_id]["status"] = "success"
            _jobs[job_id]["_finished_at"] = time.monotonic()
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        with _job_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(exc)
            _jobs[job_id]["_finished_at"] = time.monotonic()


def _run_summary_job(
    date: str,
    window: int = 7,
    top_n: int = 10,
    paper_model: str = "gpt-5-mini",
    synthesis_model: str = "gpt-5.2",
    force: bool = False,
) -> None:
    """Execute the two-stage summary pipeline.

    Aggregates the top *top_n* recommended papers across all dates in the
    window ending on *date* (deduplicated by paper_id, keeping highest score).
    """
    from forscheule.config import get_openai_client
    from forscheule.summary.digest import generate_digest
    from forscheule.summary.per_paper import summarize_papers

    client = get_openai_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    conn = init_db(DB_PATH)

    # Warn if the window spans pipeline runs with different settings
    sigs = get_distinct_signatures_in_range(conn, date, window)
    if len(sigs) > 1:
        sig_parts = ", ".join(
            f"{s['signature'][:8]}(dates {s['first_date']}..{s['last_date']}, "
            f"top_k={s['top_k']})"
            for s in sigs
        )
        logger.warning(
            "Weekly summary window contains %d distinct pipeline configurations "
            "(different settings were used on different days): %s. "
            "The digest will mix scoring regimes.",
            len(sigs), sig_parts,
        )

    recs = get_recommendations_for_date_range(conn, date, window)
    if not recs:
        conn.close()
        raise RuntimeError(
            f"No recommendations found in the {window}-day window ending {date}. "
            "Run the pipeline first."
        )

    papers_for_summary = recs[:top_n]

    paper_dicts = [
        {
            "paper_id": rec["paper_id"],
            "title": rec["title"],
            "abstract": rec["abstract"],
            "source": rec["source"],
            "source_id": rec["source_id"],
            "published_at": rec.get("published_at", ""),
            "url": rec.get("url", ""),
        }
        for rec in papers_for_summary
    ]

    paired = summarize_papers(
        client, paper_dicts, conn, model=paper_model, force=force,
    )
    if not paired:
        conn.close()
        raise RuntimeError("All per-paper summarizations failed.")

    summarized_papers = [p for p, _s in paired]
    summaries = [s for _p, s in paired]

    generate_digest(
        client, summarized_papers, summaries, conn,
        date=date, window_days=window, top_n=top_n,
        paper_model=paper_model, synthesis_model=synthesis_model,
        force=force,
    )
    conn.close()
    logger.info("Summary job completed for date=%s: %d summaries", date, len(summaries))


# ---------------------------------------------------------------------------
# UI route
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def ui_index(request: Request):
    """Render the web console."""
    conn = init_db(DB_PATH)
    settings = get_runtime_settings(conn)
    logger.debug(
        "Loaded settings for UI: top_k=%d, profile_len=%d, phrases=%d",
        settings["top_k"], len(settings["lab_profile"]), len(settings["boosted_phrases"]),
    )
    conn.close()

    oai_ok, oai_reason = openai_status()

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "today": date_cls.today().isoformat(),
            "top_k": settings["top_k"],
            "lab_profile": settings["lab_profile"],
            "boosted_phrases_text": "\n".join(settings["boosted_phrases"]),
            "openai_configured": oai_ok,
            "openai_reason": oai_reason,
        },
    )


# ---------------------------------------------------------------------------
# Results endpoint (backward-compatible, extended with top_k)
# ---------------------------------------------------------------------------
@app.get("/daily")
def daily(
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
    top_k: int | None = Query(None, description="Number of results (default: from settings)"),
):
    """Return paper recommendations for a given date."""
    try:
        date_cls.fromisoformat(date)
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid date format. Use YYYY-MM-DD."},
        )

    conn = init_db(DB_PATH)
    recs = get_recommendations_for_date(conn, date)
    conn.close()

    if top_k is not None and top_k > 0:
        recs = recs[:top_k]

    return recs


# ---------------------------------------------------------------------------
# Settings endpoints
# ---------------------------------------------------------------------------
@app.get("/settings")
def get_settings():
    """Return current runtime settings."""
    conn = init_db(DB_PATH)
    settings = get_runtime_settings(conn)
    conn.close()
    return settings


@app.put("/settings")
async def put_settings(request: Request):
    """Update runtime settings."""
    body = await _safe_json_body(request)
    if isinstance(body, JSONResponse):
        return body

    lab_profile = body.get("lab_profile")
    boosted_phrases = body.get("boosted_phrases")
    top_k = body.get("top_k")

    to_save: dict[str, str] = {}
    if lab_profile is not None:
        if not isinstance(lab_profile, str) or not lab_profile.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "lab_profile must be a non-empty string."},
            )
        to_save["lab_profile"] = lab_profile.strip()

    if boosted_phrases is not None:
        if not isinstance(boosted_phrases, list):
            return JSONResponse(
                status_code=400,
                content={"error": "boosted_phrases must be a list of strings."},
            )
        to_save["boosted_phrases"] = json.dumps(boosted_phrases)

    if top_k is not None:
        if isinstance(top_k, bool):
            return JSONResponse(
                status_code=400,
                content={"error": "top_k must be an integer, not a boolean."},
            )
        try:
            k = int(top_k)
            if k < 1 or k > 50:
                raise ValueError
        except (ValueError, TypeError):
            return JSONResponse(
                status_code=400,
                content={"error": "top_k must be an integer between 1 and 50."},
            )
        to_save["top_k"] = str(k)

    if not to_save:
        return JSONResponse(
            status_code=400,
            content={"error": "No valid settings provided."},
        )

    conn = init_db(DB_PATH)
    logger.info("Updating settings: keys=%s", list(to_save.keys()))
    put_settings_bulk(conn, to_save)
    conn.close()

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Job endpoints
# ---------------------------------------------------------------------------
@app.post("/jobs/run-daily")
async def trigger_run_daily(request: Request):
    """Trigger a run-daily pipeline job in the background."""
    body = await _safe_json_body(request)
    if isinstance(body, JSONResponse):
        return body

    kwargs: dict = {}
    window = body.get("window")
    if window is not None:
        parsed = _parse_int(window, "window", 1, 90)
        if isinstance(parsed, JSONResponse):
            return parsed
        kwargs["window_days"] = parsed
    top_k = body.get("top_k")
    if top_k is not None:
        parsed = _parse_int(top_k, "top_k", 1, 50)
        if isinstance(parsed, JSONResponse):
            return parsed
        kwargs["top_k"] = parsed

    job_id = _try_enqueue_job("run-daily")
    if job_id is None:
        return JSONResponse(
            status_code=409,
            content={"error": "A job is already running. Wait for it to finish."},
        )

    t = threading.Thread(
        target=_run_job_in_thread, args=(job_id, "run-daily", kwargs), daemon=True
    )
    t.start()

    return {"job_id": job_id, "status": "queued"}


@app.post("/jobs/backfill")
async def trigger_backfill(request: Request):
    """Trigger a backfill pipeline job in the background."""
    body = await _safe_json_body(request)
    if isinstance(body, JSONResponse):
        return body

    days_raw = body.get("days")
    if days_raw is None:
        return JSONResponse(
            status_code=400,
            content={"error": "'days' is required and must be >= 1."},
        )
    days = _parse_int(days_raw, "days", 1, 365)
    if isinstance(days, JSONResponse):
        return days

    kwargs: dict = {"days": days}
    window = body.get("window")
    if window is not None:
        parsed = _parse_int(window, "window", 1, 90)
        if isinstance(parsed, JSONResponse):
            return parsed
        kwargs["window_days"] = parsed
    top_k = body.get("top_k")
    if top_k is not None:
        parsed = _parse_int(top_k, "top_k", 1, 50)
        if isinstance(parsed, JSONResponse):
            return parsed
        kwargs["top_k"] = parsed

    job_id = _try_enqueue_job("backfill")
    if job_id is None:
        return JSONResponse(
            status_code=409,
            content={"error": "A job is already running. Wait for it to finish."},
        )

    t = threading.Thread(
        target=_run_job_in_thread, args=(job_id, "backfill", kwargs), daemon=True
    )
    t.start()

    return {"job_id": job_id, "status": "queued"}


@app.post("/jobs/summary-weekly")
async def trigger_summary_weekly(request: Request):
    """Trigger a weekly summary generation job."""
    oai_ok, oai_reason = openai_status()
    if not oai_ok:
        return JSONResponse(status_code=503, content={"error": oai_reason})

    body = await _safe_json_body(request)
    if isinstance(body, JSONResponse):
        return body

    date_str = body.get("date", date_cls.today().isoformat())
    try:
        date_cls.fromisoformat(date_str)
    except (ValueError, TypeError):
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid date format. Use YYYY-MM-DD."},
        )

    window = _parse_int(body.get("window", 7), "window", 1, 90)
    if isinstance(window, JSONResponse):
        return window
    top_n = _parse_int(body.get("top_n", 10), "top_n", 1, 50)
    if isinstance(top_n, JSONResponse):
        return top_n

    kwargs = {
        "date": date_str,
        "window": window,
        "top_n": top_n,
        "paper_model": body.get("paper_model", "gpt-5-mini"),
        "synthesis_model": body.get("synthesis_model", "gpt-5.2"),
        "force": body.get("force") is True,
    }

    job_id = _try_enqueue_job("summary-weekly")
    if job_id is None:
        return JSONResponse(
            status_code=409,
            content={"error": "A job is already running. Wait for it to finish."},
        )

    t = threading.Thread(
        target=_run_job_in_thread, args=(job_id, "summary-weekly", kwargs), daemon=True
    )
    t.start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/weekly-summary")
def get_weekly_summary_endpoint(
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
    window: int = Query(7, description="Window in days"),
    top_n: int = Query(10, description="Number of top papers"),
    paper_model: str = Query("gpt-5-mini"),
    synthesis_model: str = Query("gpt-5.2"),
):
    """Return cached weekly summary if available."""
    try:
        date_cls.fromisoformat(date)
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "Invalid date format."})

    conn = init_db(DB_PATH)
    row = conn.execute(
        """SELECT digest_json, paper_set_signature, created_at FROM weekly_digests
           WHERE date=? AND window_days=? AND top_n=? AND paper_model=? AND synthesis_model=?
           ORDER BY created_at DESC LIMIT 1""",
        (date, window, top_n, paper_model, synthesis_model),
    ).fetchone()
    conn.close()

    if not row:
        return JSONResponse(
            status_code=404,
            content={"error": "No digest found for these parameters. Generate one first."},
        )

    return {
        "date": date,
        "window_days": window,
        "top_n": top_n,
        "paper_model": paper_model,
        "synthesis_model": synthesis_model,
        "digest": json.loads(row["digest_json"]),
        "created_at": row["created_at"],
    }


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    """Check the status of a pipeline job."""
    with _job_lock:
        job = _jobs.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found."})
    return {
        "job_id": job_id,
        "type": job["type"],
        "status": job["status"],
        "error": job.get("error"),
    }
