"""Stage A: Per-paper structured summaries using OpenAI."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from typing import TYPE_CHECKING

from forscheule.db.repo import get_paper_summary, save_paper_summary

if TYPE_CHECKING:
    from openai import OpenAI
from forscheule.summary.schemas import PER_PAPER_SCHEMA, SCHEMA_VERSION

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-5-mini"
_TIMEOUT = 60


def _compute_input_hash(title: str, abstract: str, model: str, schema_version: str) -> str:
    """Deterministic hash of per-paper summary inputs."""
    raw = json.dumps(
        {"title": title, "abstract": abstract, "model": model, "schema_version": schema_version},
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def summarize_paper(
    client: OpenAI,
    paper: dict,
    paper_id: int,
    conn: sqlite3.Connection,
    model: str = _DEFAULT_MODEL,
    force: bool = False,
) -> dict:
    """Summarize a single paper. Uses cache if available.

    *paper*: dict with keys title, abstract, source, published_at, url, source_id.
    *paper_id*: database ID for cache keying.
    *force*: skip cache lookup and regenerate.
    Returns structured summary dict.
    """
    input_hash = _compute_input_hash(paper["title"], paper["abstract"], model, SCHEMA_VERSION)

    if not force:
        cached = get_paper_summary(conn, paper_id, model, SCHEMA_VERSION, input_hash)
        if cached is not None:
            logger.debug("Cache hit for paper_id=%d", paper_id)
            return cached

    system_msg = (
        "You are an expert scientific reviewer for spatial transcriptomics research. "
        "Analyze the following paper and produce a structured summary."
    )
    user_msg = (
        f"Title: {paper['title']}\n"
        f"Abstract: {paper['abstract']}\n"
        f"Source: {paper['source']}\n"
        f"Published: {paper.get('published_at', 'unknown')}\n"
        f"URL: {paper.get('url', '')}\n\n"
        f"Paper ID for reference: {paper['source']}:{paper.get('source_id', str(paper_id))}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        response_format=PER_PAPER_SCHEMA,
        timeout=_TIMEOUT,
    )

    summary = json.loads(response.choices[0].message.content)
    save_paper_summary(conn, paper_id, model, SCHEMA_VERSION, input_hash, summary)
    logger.info("Summarized paper_id=%d (model=%s)", paper_id, model)
    return summary


def summarize_papers(
    client: OpenAI,
    papers: list[dict],
    conn: sqlite3.Connection,
    model: str = _DEFAULT_MODEL,
    force: bool = False,
) -> list[tuple[dict, dict]]:
    """Summarize a batch of papers sequentially.

    *papers*: list of dicts with keys paper_id, title, abstract, source, etc.
    *force*: bypass cache and regenerate all summaries.
    Returns list of ``(paper, summary)`` tuples so that papers and summaries
    stay paired even when individual summarizations fail.
    """
    results: list[tuple[dict, dict]] = []
    for paper in papers:
        try:
            summary = summarize_paper(
                client, paper, paper["paper_id"], conn, model=model, force=force,
            )
            results.append((paper, summary))
        except Exception:
            logger.exception("Failed to summarize paper_id=%d", paper["paper_id"])
    return results
