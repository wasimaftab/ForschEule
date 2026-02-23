"""Scoring and ranking papers against the lab profile."""

from __future__ import annotations

import logging
from datetime import date

import numpy as np

from forscheule.config import BOOSTED_PHRASES, LAB_PROFILE, TOP_K
from forscheule.db.repo import Paper
from forscheule.rank.embed import embed_articles, embed_queries

logger = logging.getLogger(__name__)

_RECENCY_WINDOW = 14  # days for linear decay
_KEYWORD_BOOST = 0.05  # per matched phrase
_SHORT_ABSTRACT_PENALTY = 0.15  # penalty if abstract < 100 chars


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _recency_score(published_at: str | None, today: date | None = None) -> float:
    """Linear decay: 1.0 for today, 0.0 for 14+ days ago."""
    if not published_at:
        return 0.5  # unknown date gets neutral score
    today = today or date.today()
    try:
        pub = date.fromisoformat(published_at)
    except ValueError:
        return 0.5
    age_days = (today - pub).days
    return max(0.0, 1.0 - age_days / _RECENCY_WINDOW)


def _keyword_matches(text: str, boosted_phrases: list[str] | None = None) -> list[str]:
    """Find which boosted phrases appear in text."""
    phrases = boosted_phrases if boosted_phrases is not None else BOOSTED_PHRASES
    text_lower = text.lower()
    return [phrase for phrase in phrases if phrase.lower() in text_lower]


def rank_papers(
    papers: list[Paper],
    *,
    lab_profile: str | None = None,
    boosted_phrases: list[str] | None = None,
    top_k: int | None = None,
    reference_date: date | None = None,
) -> list[tuple[Paper, float, list[str]]]:
    """Score and rank papers.

    Accepts optional runtime overrides for lab_profile, boosted_phrases, top_k.
    Falls back to compile-time defaults from config.py.
    *reference_date* anchors the recency score (defaults to today).
    Returns list of (paper, score, matched_terms) sorted desc, trimmed to top_k.
    """
    if not papers:
        return []

    profile = lab_profile or LAB_PROFILE
    phrases = boosted_phrases if boosted_phrases is not None else BOOSTED_PHRASES
    k = top_k if top_k is not None else TOP_K

    # Embed lab profile with query encoder
    profile_emb = embed_queries([profile])[0]

    # Embed all papers with article encoder ([title, abstract] pairs)
    article_pairs = [(p.title, p.abstract) for p in papers]
    paper_embs = embed_articles(article_pairs)

    scored: list[tuple[Paper, float, list[str]]] = []
    ref_date = reference_date or date.today()

    for i, paper in enumerate(papers):
        # Base: cosine similarity
        sim = _cosine_similarity(profile_emb, paper_embs[i])

        # Keyword boost
        combined_text = f"{paper.title} {paper.abstract}"
        matches = _keyword_matches(combined_text, phrases)
        keyword_score = len(matches) * _KEYWORD_BOOST

        # Recency
        recency = _recency_score(paper.published_at, ref_date)

        # Penalty for short abstracts
        penalty = _SHORT_ABSTRACT_PENALTY if len(paper.abstract) < 100 else 0.0

        # Weighted combination
        final = 0.5 * sim + 0.2 * recency + 0.3 * keyword_score - penalty
        final = max(0.0, final)

        scored.append((paper, round(final, 4), matches))

    # Sort descending by score
    scored.sort(key=lambda x: x[1], reverse=True)

    top = scored[:k]
    logger.info(
        "Ranked %d papers; top score=%.4f, bottom of top-%d=%.4f",
        len(scored),
        scored[0][1] if scored else 0,
        k,
        top[-1][1] if top else 0,
    )
    return top
