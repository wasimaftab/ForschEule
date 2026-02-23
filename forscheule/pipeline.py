"""Core pipeline: fetch -> normalise -> embed -> rank -> store."""

from __future__ import annotations

import logging
from datetime import date, timedelta

from forscheule.config import (
    ARTICLE_MAX_LENGTH,
    ARTICLE_MODEL_NAME,
    DB_PATH,
    FETCH_WINDOW_DAYS,
    QUERY_MAX_LENGTH,
    QUERY_MODEL_NAME,
    compute_pipeline_signature,
    get_runtime_settings,
)
from forscheule.db.repo import (
    Recommendation,
    get_pipeline_run_signature,
    save_recommendations,
    upsert_paper,
    upsert_pipeline_run,
)
from forscheule.db.schema import init_db
from forscheule.rank.dedup import deduplicate
from forscheule.rank.score import rank_papers
from forscheule.sources.arxiv import fetch_arxiv
from forscheule.sources.pubmed import fetch_pubmed

logger = logging.getLogger(__name__)


def run_pipeline(
    target_date: date,
    window_days: int | None = None,
    force: bool = False,
    top_k: int | None = None,
) -> None:
    """Run the full pipeline for a single target date."""
    window = window_days or FETCH_WINDOW_DAYS
    date_str = target_date.isoformat()

    conn = init_db(DB_PATH)

    # Load runtime settings FIRST (needed for signature computation)
    settings = get_runtime_settings(conn)
    effective_top_k = top_k if top_k is not None else settings["top_k"]

    # Compute signature for this configuration (includes embedding model info)
    current_sig = compute_pipeline_signature(
        target_date=date_str,
        window_days=window,
        top_k=effective_top_k,
        lab_profile=settings["lab_profile"],
        boosted_phrases=settings["boosted_phrases"],
        query_model=QUERY_MODEL_NAME,
        article_model=ARTICLE_MODEL_NAME,
        query_max_length=QUERY_MAX_LENGTH,
        article_max_length=ARTICLE_MAX_LENGTH,
    )

    # Idempotency: skip only if date exists AND signature matches
    if not force:
        prev_sig = get_pipeline_run_signature(conn, date_str)
        if prev_sig == current_sig:
            logger.info(
                "Recommendations for %s already computed with same settings – skipping.",
                date_str,
            )
            conn.close()
            return
        if prev_sig is not None:
            logger.info(
                "Settings changed for %s (sig %s→%s) – recomputing.",
                date_str,
                prev_sig[:8],
                current_sig[:8],
            )

    logger.info("Running pipeline for %s (window=%d days)", date_str, window)

    # 1. Fetch (anchored to target_date for correct backfill)
    pubmed_papers = fetch_pubmed(window_days=window, reference_date=target_date)
    arxiv_papers = fetch_arxiv(window_days=window, reference_date=target_date)
    all_papers = pubmed_papers + arxiv_papers
    logger.info(
        "Total fetched: %d (PubMed=%d, arXiv=%d)",
        len(all_papers),
        len(pubmed_papers),
        len(arxiv_papers),
    )

    if not all_papers:
        logger.warning("No papers fetched – nothing to rank.")
        conn.close()
        return

    # 2. Deduplicate
    unique_papers = deduplicate(all_papers)

    # 3. Store papers
    paper_id_map: dict[str, int] = {}
    for p in unique_papers:
        pid = upsert_paper(conn, p)
        paper_id_map[f"{p.source}:{p.source_id}"] = pid

    # 4. Rank (using runtime settings, anchored to target_date)
    top = rank_papers(
        unique_papers,
        lab_profile=settings["lab_profile"],
        boosted_phrases=settings["boosted_phrases"],
        top_k=effective_top_k,
        reference_date=target_date,
    )

    # 5. Store recommendations
    recs = []
    for rank_pos, (paper, score, matched) in enumerate(top, 1):
        key = f"{paper.source}:{paper.source_id}"
        recs.append(
            Recommendation(
                date=date_str,
                paper_id=paper_id_map[key],
                score=score,
                rank=rank_pos,
                matched_terms=matched,
            )
        )
    save_recommendations(conn, date_str, recs)
    upsert_pipeline_run(conn, date_str, current_sig, window, effective_top_k)
    logger.info("Saved %d recommendations for %s", len(recs), date_str)

    conn.close()


def run_daily(window_days: int | None = None, top_k: int | None = None) -> None:
    """Run pipeline for today."""
    run_pipeline(date.today(), window_days=window_days, top_k=top_k)


def backfill(
    days: int, window_days: int | None = None, top_k: int | None = None
) -> None:
    """Run pipeline for each of the last N days (inclusive of today)."""
    today = date.today()
    for i in range(days):
        target = today - timedelta(days=i)
        run_pipeline(target, window_days=window_days, top_k=top_k)
