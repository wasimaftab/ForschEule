"""Tests for deduplication logic."""

from __future__ import annotations

from forscheule.db.repo import Paper
from forscheule.rank.dedup import deduplicate


def test_dedup_by_doi():
    papers = [
        Paper(source="pubmed", source_id="1", title="Paper A", doi="10.1234/a"),
        Paper(source="arxiv", source_id="2", title="Paper B", doi="10.1234/a"),
    ]
    result = deduplicate(papers)
    assert len(result) == 1
    assert result[0].source == "pubmed"  # first one kept


def test_dedup_by_fuzzy_title():
    papers = [
        Paper(source="pubmed", source_id="1", title="Spatial Transcriptomics: A New Method"),
        Paper(source="arxiv", source_id="2", title="Spatial transcriptomics a new method"),
    ]
    result = deduplicate(papers)
    assert len(result) == 1


def test_no_dedup_for_different_papers():
    papers = [
        Paper(source="pubmed", source_id="1", title="Spatial Transcriptomics Method"),
        Paper(source="arxiv", source_id="2", title="Single-cell RNA sequencing Analysis"),
    ]
    result = deduplicate(papers)
    assert len(result) == 2


def test_dedup_by_same_source_id():
    papers = [
        Paper(source="arxiv", source_id="2401.00001", title="Paper X"),
        Paper(source="arxiv", source_id="2401.00001", title="Paper X (v2)"),
    ]
    result = deduplicate(papers)
    assert len(result) == 1
