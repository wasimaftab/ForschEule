"""Deduplication logic for papers."""

from __future__ import annotations

import logging
import re
import unicodedata

from rapidfuzz import fuzz

from forscheule.db.repo import Paper

logger = logging.getLogger(__name__)

_FUZZY_THRESHOLD = 95


def _normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    t = unicodedata.normalize("NFKD", title).lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def deduplicate(papers: list[Paper]) -> list[Paper]:
    """Remove duplicates by DOI, source_id, and fuzzy title matching."""
    seen_ids: set[str] = set()
    seen_dois: set[str] = set()
    norm_titles: list[str] = []
    result: list[Paper] = []

    for p in papers:
        key = f"{p.source}:{p.source_id}"
        if key in seen_ids:
            continue

        # DOI dedup
        if p.doi:
            doi_lower = p.doi.lower().strip()
            if doi_lower in seen_dois:
                logger.debug("Dedup by DOI: %s", p.title[:60])
                continue
            seen_dois.add(doi_lower)

        # Fuzzy title dedup
        norm = _normalize_title(p.title)
        is_dup = False
        for existing_norm in norm_titles:
            if existing_norm == norm:
                is_dup = True
                break
            score = fuzz.token_set_ratio(norm, existing_norm)
            if score >= _FUZZY_THRESHOLD:
                is_dup = True
                logger.debug("Dedup by fuzzy title (%.0f): %s", score, p.title[:60])
                break

        if is_dup:
            continue

        seen_ids.add(key)
        norm_titles.append(norm)
        result.append(p)

    removed = len(papers) - len(result)
    if removed:
        logger.info("Dedup removed %d duplicates, %d remaining", removed, len(result))
    return result
