"""Stage B: Weekly synthesis digest using OpenAI."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from typing import TYPE_CHECKING

from forscheule.db.repo import get_weekly_digest, save_weekly_digest

if TYPE_CHECKING:
    from openai import OpenAI
from forscheule.summary.schemas import WEEKLY_DIGEST_SCHEMA

logger = logging.getLogger(__name__)

_DEFAULT_SYNTHESIS_MODEL = "gpt-5.2"
_TIMEOUT = 120


def _compute_paper_set_signature(summaries: list[dict]) -> str:
    """Hash the set of per-paper summaries to detect changes."""
    raw = json.dumps(summaries, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def generate_digest(
    client: OpenAI,
    papers: list[dict],
    summaries: list[dict],
    conn: sqlite3.Connection,
    date: str,
    window_days: int = 7,
    top_n: int = 10,
    paper_model: str = "gpt-5-mini",
    synthesis_model: str = _DEFAULT_SYNTHESIS_MODEL,
    force: bool = False,
) -> dict:
    """Generate weekly digest from per-paper summaries and original data.

    *papers*: list of dicts with title, abstract, source, url, etc.
    *summaries*: list of per-paper structured summaries (from Stage A).
    *force*: bypass cache and regenerate.
    Returns digest dict with themes, contradictions, etc.
    """
    paper_set_sig = _compute_paper_set_signature(summaries)

    if not force:
        cached = get_weekly_digest(
            conn, date, window_days, top_n, paper_set_sig, paper_model, synthesis_model
        )
        if cached is not None:
            logger.info("Cache hit for weekly digest date=%s", date)
            return cached

    system_msg = (
        "You are an expert research synthesizer for a spatial transcriptomics lab. "
        "Given structured summaries and original abstracts of the top papers from the "
        "last week, produce a synthesis identifying themes, tensions, reading "
        "priorities, and methodology trends."
    )

    paper_blocks = []
    for paper, summary in zip(papers, summaries):
        block = (
            f"--- Paper: {summary.get('paper_id', 'unknown')} ---\n"
            f"Title: {paper['title']}\n"
            f"Abstract: {paper['abstract']}\n"
            f"One-line takeaway: {summary['one_line_takeaway']}\n"
            f"Methods: {summary['methods']}\n"
            f"Main findings: {summary['main_findings']}\n"
            f"Limitations: {summary['limitations']}\n"
            f"Relevance to lab: {summary['relevance_to_lab']}\n"
            f"Novelty: {summary['novelty_level']}\n"
            f"Read priority: {summary['read_priority']}\n"
        )
        paper_blocks.append(block)

    user_msg = (
        f"Date range: {date} (window: {window_days} days)\n"
        f"Number of papers: {len(papers)}\n\n"
        + "\n".join(paper_blocks)
    )

    response = client.chat.completions.create(
        model=synthesis_model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        response_format=WEEKLY_DIGEST_SCHEMA,
        timeout=_TIMEOUT,
    )

    digest = json.loads(response.choices[0].message.content)
    save_weekly_digest(
        conn, date, window_days, top_n, paper_set_sig, paper_model, synthesis_model, digest
    )
    logger.info("Generated weekly digest for date=%s", date)
    return digest
