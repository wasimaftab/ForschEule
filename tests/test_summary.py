"""Tests for the summary caching and schema logic."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from forscheule.db.repo import (
    Paper,
    get_paper_summary,
    get_weekly_digest,
    save_paper_summary,
    save_weekly_digest,
    upsert_paper,
)
from forscheule.db.schema import init_db
from forscheule.summary.per_paper import _compute_input_hash
from forscheule.summary.schemas import PER_PAPER_SCHEMA, WEEKLY_DIGEST_SCHEMA


def _tmp_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
    tmp.close()
    return init_db(Path(tmp.name))


def test_input_hash_stable():
    h1 = _compute_input_hash("Title", "Abstract", "gpt-5-mini", "1")
    h2 = _compute_input_hash("Title", "Abstract", "gpt-5-mini", "1")
    assert h1 == h2


def test_input_hash_changes_on_abstract():
    h1 = _compute_input_hash("Title", "Abstract A", "gpt-5-mini", "1")
    h2 = _compute_input_hash("Title", "Abstract B", "gpt-5-mini", "1")
    assert h1 != h2


def test_input_hash_changes_on_model():
    h1 = _compute_input_hash("Title", "Abstract", "gpt-5-mini", "1")
    h2 = _compute_input_hash("Title", "Abstract", "gpt-5.2", "1")
    assert h1 != h2


def test_paper_summary_cache_roundtrip():
    conn = _tmp_db()
    pid = upsert_paper(conn, Paper(source="pubmed", source_id="111", title="T", abstract="A"))
    summary = {"paper_id": "pubmed:111", "one_line_takeaway": "test"}

    assert get_paper_summary(conn, pid, "gpt-5-mini", "1", "hash1") is None

    save_paper_summary(conn, pid, "gpt-5-mini", "1", "hash1", summary)
    cached = get_paper_summary(conn, pid, "gpt-5-mini", "1", "hash1")
    assert cached == summary

    # Different hash => miss
    assert get_paper_summary(conn, pid, "gpt-5-mini", "1", "hash2") is None
    conn.close()


def test_weekly_digest_cache_roundtrip():
    conn = _tmp_db()
    digest = {"themes": ["t1"], "contradictions_or_tensions": []}

    assert get_weekly_digest(conn, "2026-02-09", 7, 10, "sig1", "gpt-5-mini", "gpt-5.2") is None

    save_weekly_digest(conn, "2026-02-09", 7, 10, "sig1", "gpt-5-mini", "gpt-5.2", digest)
    cached = get_weekly_digest(conn, "2026-02-09", 7, 10, "sig1", "gpt-5-mini", "gpt-5.2")
    assert cached == digest

    # Different signature => miss
    assert get_weekly_digest(conn, "2026-02-09", 7, 10, "sig2", "gpt-5-mini", "gpt-5.2") is None
    conn.close()


def test_per_paper_schema_has_required_keys():
    schema = PER_PAPER_SCHEMA["json_schema"]["schema"]
    assert schema["type"] == "object"
    required = schema["required"]
    for key in [
        "paper_id", "one_line_takeaway", "methods", "main_findings",
        "limitations", "relevance_to_lab", "novelty_level", "read_priority",
    ]:
        assert key in required


def test_weekly_digest_schema_has_required_keys():
    schema = WEEKLY_DIGEST_SCHEMA["json_schema"]["schema"]
    assert schema["type"] == "object"
    required = schema["required"]
    for key in [
        "themes", "contradictions_or_tensions", "what_to_read_first",
        "methods_trends", "recommended_next_queries",
    ]:
        assert key in required


def test_summarize_paper_uses_cache():
    """When a cached summary exists, the OpenAI client should NOT be called."""
    conn = _tmp_db()
    pid = upsert_paper(conn, Paper(source="pubmed", source_id="222", title="T2", abstract="A2"))
    summary = {"paper_id": "pubmed:222", "one_line_takeaway": "cached"}
    input_hash = _compute_input_hash("T2", "A2", "gpt-5-mini", "1")
    save_paper_summary(conn, pid, "gpt-5-mini", "1", input_hash, summary)

    mock_client = MagicMock()
    paper = {"title": "T2", "abstract": "A2", "source": "pubmed", "source_id": "222"}

    with patch("forscheule.summary.per_paper.SCHEMA_VERSION", "1"):
        from forscheule.summary.per_paper import summarize_paper

        result = summarize_paper(mock_client, paper, pid, conn, model="gpt-5-mini")

    assert result == summary
    mock_client.chat.completions.create.assert_not_called()
    conn.close()


def test_summarize_paper_force_bypasses_cache():
    """When force=True, the OpenAI client SHOULD be called even if cached."""
    conn = _tmp_db()
    pid = upsert_paper(conn, Paper(source="pubmed", source_id="333", title="T3", abstract="A3"))
    cached_summary = {"paper_id": "pubmed:333", "one_line_takeaway": "old"}
    input_hash = _compute_input_hash("T3", "A3", "gpt-5-mini", "1")
    save_paper_summary(conn, pid, "gpt-5-mini", "1", input_hash, cached_summary)

    new_summary = {"paper_id": "pubmed:333", "one_line_takeaway": "new"}
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(new_summary)
    mock_client.chat.completions.create.return_value = mock_response

    paper = {"title": "T3", "abstract": "A3", "source": "pubmed", "source_id": "333"}

    with patch("forscheule.summary.per_paper.SCHEMA_VERSION", "1"):
        from forscheule.summary.per_paper import summarize_paper

        result = summarize_paper(
            mock_client, paper, pid, conn, model="gpt-5-mini", force=True,
        )

    assert result == new_summary
    mock_client.chat.completions.create.assert_called_once()
    conn.close()
