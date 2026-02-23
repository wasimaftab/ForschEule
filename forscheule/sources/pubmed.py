"""PubMed fetcher using NCBI Entrez E-utilities."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import date, timedelta

from forscheule.config import ENTREZ_API_KEY, ENTREZ_EMAIL
from forscheule.db.repo import Paper
from forscheule.sources.http_client import make_session, rate_limited_get

logger = logging.getLogger(__name__)

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

QUERY = (
    '("spatial transcriptomics"[Title/Abstract] '
    'OR "spatially resolved transcriptomics"[Title/Abstract] '
    'OR "spatial omics"[Title/Abstract]) '
    "AND (integration[Title/Abstract] OR multimodal[Title/Abstract] "
    'OR "multi-modal"[Title/Abstract] OR alignment[Title/Abstract] '
    "OR mapping[Title/Abstract] OR deconvolution[Title/Abstract] "
    "OR imputation[Title/Abstract] OR transformer[Title/Abstract] "
    'OR "graph neural network"[Title/Abstract] '
    'OR "batch correction"[Title/Abstract])'
)

# NCBI guideline: max 3 req/s without key, 10/s with key.
_DELAY = 0.35 if ENTREZ_API_KEY else 1.0


def _base_params() -> dict:
    params: dict = {}
    if ENTREZ_EMAIL:
        params["email"] = ENTREZ_EMAIL
    else:
        logger.warning("No ENTREZ_EMAIL configured for NCBI Entrez")
    if ENTREZ_API_KEY:
        params["api_key"] = ENTREZ_API_KEY
    else:
        logger.warning("No ENTREZ_API_KEY configured for NCBI Entrez")
    return params


def _search_ids(session, min_date: date, max_date: date) -> list[str]:
    params = {
        **_base_params(),
        "db": "pubmed",
        "term": QUERY,
        "datetype": "pdat",
        "mindate": min_date.strftime("%Y/%m/%d"),
        "maxdate": max_date.strftime("%Y/%m/%d"),
        "retmax": 200,
        "retmode": "json",
        "usehistory": "n",
    }
    resp = rate_limited_get(session, ESEARCH_URL, params=params, delay=_DELAY)
    data = resp.json()
    ids = data.get("esearchresult", {}).get("idlist", [])
    logger.info("PubMed search returned %d IDs for %s–%s", len(ids), min_date, max_date)
    return ids


def _fetch_details(session, pmids: list[str]) -> list[Paper]:
    if not pmids:
        return []
    papers: list[Paper] = []
    # Fetch in batches of 50
    batch_size = 50
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        params = {
            **_base_params(),
            "db": "pubmed",
            "id": ",".join(batch),
            "rettype": "xml",
            "retmode": "xml",
        }
        resp = rate_limited_get(session, EFETCH_URL, params=params, delay=_DELAY)
        root = ET.fromstring(resp.text)
        for article in root.findall(".//PubmedArticle"):
            try:
                papers.append(_parse_article(article))
            except Exception:
                logger.exception("Failed to parse PubMed article")
    return papers


def _parse_article(article: ET.Element) -> Paper:
    medline = article.find("MedlineCitation")
    pmid = medline.findtext("PMID", default="")
    art = medline.find("Article")
    title = art.findtext("ArticleTitle", default="") if art is not None else ""

    abstract_parts = []
    if art is not None:
        abs_el = art.find("Abstract")
        if abs_el is not None:
            for at in abs_el.findall("AbstractText"):
                label = at.get("Label", "")
                text = "".join(at.itertext())
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
    abstract = " ".join(abstract_parts)

    # Authors
    authors_list = []
    if art is not None:
        author_list_el = art.find("AuthorList")
        if author_list_el is not None:
            for author in author_list_el.findall("Author"):
                last = author.findtext("LastName", "")
                first = author.findtext("ForeName", "")
                if last:
                    authors_list.append(f"{first} {last}".strip())
    authors_str = "; ".join(authors_list)

    # Published date
    pub_date_el = art.find(".//PubDate") if art is not None else None
    pub_date = ""
    if pub_date_el is not None:
        y = pub_date_el.findtext("Year", "")
        m = pub_date_el.findtext("Month", "01")
        d = pub_date_el.findtext("Day", "01")
        # Month may be text like "Jan"
        month_map = {
            "jan": "01", "feb": "02", "mar": "03", "apr": "04",
            "may": "05", "jun": "06", "jul": "07", "aug": "08",
            "sep": "09", "oct": "10", "nov": "11", "dec": "12",
        }
        m = month_map.get(m.lower()[:3], m.zfill(2)) if m else "01"
        d = d.zfill(2) if d else "01"
        if y:
            pub_date = f"{y}-{m}-{d}"

    # DOI
    doi = None
    article_id_list = article.find(".//ArticleIdList")
    if article_id_list is not None:
        for aid in article_id_list.findall("ArticleId"):
            if aid.get("IdType") == "doi":
                doi = aid.text

    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

    return Paper(
        source="pubmed",
        source_id=pmid,
        title=title,
        abstract=abstract,
        authors=authors_str,
        published_at=pub_date,
        url=url,
        doi=doi,
    )


def fetch_pubmed(
    window_days: int = 7, reference_date: date | None = None
) -> list[Paper]:
    """Fetch recent PubMed papers within the given window.

    If *reference_date* is supplied, the fetch window ends on that date
    instead of today — required for correct backfill behaviour.
    """
    session = make_session()
    anchor = reference_date or date.today()
    min_date = anchor - timedelta(days=window_days)
    pmids = _search_ids(session, min_date, anchor)
    papers = _fetch_details(session, pmids)
    logger.info("Fetched %d papers from PubMed", len(papers))
    return papers
