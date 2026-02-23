"""arXiv fetcher using the export API."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta

from forscheule.db.repo import Paper
from forscheule.sources.http_client import make_session, rate_limited_get

logger = logging.getLogger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom"}

# arXiv rate-limit: ~3s between requests, single connection
_DELAY = 3.5

QUERY = (
    'all:("spatial transcriptomics" OR "spatially resolved transcriptomics" OR "spatial omics")'
    " AND (all:integration OR all:multimodal OR all:alignment OR all:mapping OR all:deconvolution)"
    ' AND (all:transformer OR all:"graph neural network" OR all:"deep learning"'
    ' OR all:"contrastive learning")'
)


def _parse_entry(entry: ET.Element) -> Paper:
    title = (entry.findtext("atom:title", default="", namespaces=_NS) or "").strip()
    title = " ".join(title.split())  # collapse whitespace

    abstract = (entry.findtext("atom:summary", default="", namespaces=_NS) or "").strip()
    abstract = " ".join(abstract.split())

    authors = []
    for author_el in entry.findall("atom:author", _NS):
        name = author_el.findtext("atom:name", default="", namespaces=_NS)
        if name:
            authors.append(name.strip())
    authors_str = "; ".join(authors)

    published = entry.findtext("atom:published", default="", namespaces=_NS)
    pub_date = ""
    if published:
        try:
            dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
            pub_date = dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # arXiv ID from the <id> element (URL form)
    id_url = entry.findtext("atom:id", default="", namespaces=_NS)
    arxiv_id = id_url.rsplit("/abs/", 1)[-1] if "/abs/" in id_url else id_url

    # DOI link if present
    doi = None
    for link in entry.findall("atom:link", _NS):
        if link.get("title") == "doi":
            doi = link.get("href", "").replace("http://dx.doi.org/", "")

    url = id_url if id_url else f"https://arxiv.org/abs/{arxiv_id}"

    return Paper(
        source="arxiv",
        source_id=arxiv_id,
        title=title,
        abstract=abstract,
        authors=authors_str,
        published_at=pub_date,
        url=url,
        doi=doi,
    )


def fetch_arxiv(
    window_days: int = 7, reference_date: date | None = None
) -> list[Paper]:
    """Fetch recent arXiv papers, filtering by publication date.

    If *reference_date* is supplied, the cutoff is computed relative to that
    date instead of today — required for correct backfill behaviour.
    """
    session = make_session()
    anchor = reference_date or date.today()
    cutoff = anchor - timedelta(days=window_days)
    all_papers: list[Paper] = []

    # Paginate: arXiv API returns max 100 per request
    start = 0
    max_results = 100
    while True:
        params = {
            "search_query": QUERY,
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        resp = rate_limited_get(session, ARXIV_API_URL, params=params, delay=_DELAY)
        root = ET.fromstring(resp.text)
        entries = root.findall("atom:entry", _NS)

        if not entries:
            break

        for entry in entries:
            try:
                paper = _parse_entry(entry)
            except Exception:
                logger.exception("Failed to parse arXiv entry")
                continue

            # Filter by date
            if paper.published_at:
                try:
                    pub_d = date.fromisoformat(paper.published_at)
                    if pub_d < cutoff:
                        # Results are sorted by date, so we can stop
                        logger.info("arXiv: reached papers older than cutoff, stopping.")
                        return _deduplicate(all_papers)
                except ValueError:
                    pass

            # Skip entries that are just arXiv "no results" placeholders
            if paper.title and paper.abstract:
                all_papers.append(paper)

        logger.info("arXiv: fetched page starting at %d, got %d entries", start, len(entries))

        if len(entries) < max_results:
            break
        start += max_results

    return _deduplicate(all_papers)


def _deduplicate(papers: list[Paper]) -> list[Paper]:
    """Remove duplicate source_ids within the arXiv batch."""
    seen: set[str] = set()
    out = []
    for p in papers:
        if p.source_id not in seen:
            seen.add(p.source_id)
            out.append(p)
    logger.info("Fetched %d papers from arXiv (after intra-source dedup)", len(out))
    return out
