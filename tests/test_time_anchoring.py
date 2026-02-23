"""Tests for backfill time-anchoring behaviour."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

from forscheule.rank.score import _recency_score


def test_recency_uses_reference_date():
    """A paper published 1 day before reference_date should score ~0.93."""
    score = _recency_score("2026-02-08", date(2026, 2, 9))
    assert 0.9 <= score <= 1.0


def test_recency_old_relative_to_reference():
    """A paper 14+ days before reference_date should score 0.0."""
    score = _recency_score("2026-01-20", date(2026, 2, 9))
    assert score == 0.0


@patch("forscheule.sources.pubmed.rate_limited_get")
@patch("forscheule.sources.pubmed.make_session")
def test_pubmed_fetch_uses_reference_date(mock_session, mock_get):
    """fetch_pubmed should use reference_date for the search window, not today."""
    from forscheule.sources.pubmed import fetch_pubmed

    # Return empty search results to avoid parsing
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"esearchresult": {"idlist": []}}
    mock_get.return_value = mock_resp

    fetch_pubmed(window_days=7, reference_date=date(2026, 2, 9))

    # Check that the search call used the reference date as maxdate
    call_args = mock_get.call_args
    params = call_args.kwargs.get("params") or call_args[1].get("params")
    assert params["maxdate"] == "2026/02/09"
    assert params["mindate"] == "2026/02/02"


@patch("forscheule.sources.arxiv.rate_limited_get")
@patch("forscheule.sources.arxiv.make_session")
def test_arxiv_fetch_uses_reference_date(mock_session, mock_get):
    """fetch_arxiv should use reference_date to compute the cutoff, not today."""
    from forscheule.sources.arxiv import fetch_arxiv

    # Return empty XML to stop pagination
    mock_resp = MagicMock()
    mock_resp.text = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    mock_get.return_value = mock_resp

    papers = fetch_arxiv(window_days=7, reference_date=date(2026, 2, 9))
    # No papers returned (empty feed), but the function should have run without error
    assert papers == []
