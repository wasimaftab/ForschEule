"""Tests for scoring utilities (non-embedding parts) and ranking wiring."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import numpy as np

from forscheule.db.repo import Paper
from forscheule.rank.score import _keyword_matches, _recency_score, rank_papers


def test_recency_today():
    today = date(2024, 1, 15)
    assert _recency_score("2024-01-15", today) == 1.0


def test_recency_old():
    today = date(2024, 1, 15)
    assert _recency_score("2024-01-01", today) == 0.0


def test_recency_midway():
    today = date(2024, 1, 15)
    score = _recency_score("2024-01-08", today)
    assert 0.4 < score < 0.6


def test_recency_none():
    assert _recency_score(None) == 0.5


def test_keyword_matches():
    text = "spatial transcriptomics with graph neural network for deconvolution"
    matches = _keyword_matches(text)
    assert "spatial transcriptomics" in matches
    assert "graph neural network" in matches
    assert "deconvolution" in matches


def test_keyword_no_match():
    matches = _keyword_matches("unrelated biology paper about fish")
    assert matches == []


# ---------------------------------------------------------------------------
# Ranking wiring: verify dual-encoder usage
# ---------------------------------------------------------------------------

_FAKE_DIM = 8


def _make_paper(title: str, abstract: str, pub_date: str = "2026-02-09") -> Paper:
    return Paper(
        source="test",
        source_id=title[:8],
        title=title,
        abstract=abstract,
        published_at=pub_date,
        url="https://example.com",
    )


@patch("forscheule.rank.score.embed_articles")
@patch("forscheule.rank.score.embed_queries")
def test_rank_papers_calls_query_encoder_for_profile(mock_eq, mock_ea):
    """rank_papers must embed the lab profile with the query encoder."""
    mock_eq.return_value = np.random.randn(1, _FAKE_DIM).astype(np.float32)
    mock_ea.return_value = np.random.randn(2, _FAKE_DIM).astype(np.float32)

    papers = [
        _make_paper("Paper A", "Abstract A"),
        _make_paper("Paper B", "Abstract B"),
    ]
    rank_papers(
        papers,
        lab_profile="test profile",
        boosted_phrases=[],
        top_k=2,
        reference_date=date(2026, 2, 9),
    )

    mock_eq.assert_called_once()
    args = mock_eq.call_args[0][0]
    assert args == ["test profile"]


@patch("forscheule.rank.score.embed_articles")
@patch("forscheule.rank.score.embed_queries")
def test_rank_papers_calls_article_encoder_for_papers(mock_eq, mock_ea):
    """rank_papers must embed papers with the article encoder as (title, abstract) pairs."""
    mock_eq.return_value = np.random.randn(1, _FAKE_DIM).astype(np.float32)
    mock_ea.return_value = np.random.randn(2, _FAKE_DIM).astype(np.float32)

    papers = [
        _make_paper("Title X", "Abstract X"),
        _make_paper("Title Y", "Abstract Y"),
    ]
    rank_papers(
        papers,
        lab_profile="test profile",
        boosted_phrases=[],
        top_k=2,
        reference_date=date(2026, 2, 9),
    )

    mock_ea.assert_called_once()
    pairs = mock_ea.call_args[0][0]
    assert pairs == [("Title X", "Abstract X"), ("Title Y", "Abstract Y")]


@patch("forscheule.rank.score.embed_articles")
@patch("forscheule.rank.score.embed_queries")
def test_rank_papers_output_format(mock_eq, mock_ea):
    """rank_papers returns list of (Paper, float, list[str]) sorted descending."""
    mock_eq.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    # Paper A aligns with profile, Paper B does not
    mock_ea.return_value = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32
    )

    papers = [
        _make_paper("Paper A", "spatial transcriptomics study"),
        _make_paper("Paper B", "unrelated topic"),
    ]
    result = rank_papers(
        papers,
        lab_profile="test",
        boosted_phrases=["spatial transcriptomics"],
        top_k=2,
        reference_date=date(2026, 2, 9),
    )

    assert len(result) == 2
    # Each entry is (Paper, score, matched_terms)
    for paper, score, matched in result:
        assert isinstance(paper, Paper)
        assert isinstance(score, float)
        assert isinstance(matched, list)
    # First result should score higher (aligned embedding + keyword match)
    assert result[0][1] >= result[1][1]
    assert result[0][0].title == "Paper A"


@patch("forscheule.rank.score.embed_articles")
@patch("forscheule.rank.score.embed_queries")
def test_rank_papers_empty_list(mock_eq, mock_ea):
    """rank_papers on empty list returns empty without calling encoders."""
    result = rank_papers([], lab_profile="p", boosted_phrases=[], top_k=5)
    assert result == []
    mock_eq.assert_not_called()
    mock_ea.assert_not_called()
