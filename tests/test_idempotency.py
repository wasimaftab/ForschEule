"""Tests for signature-based idempotency logic."""

from __future__ import annotations

import tempfile
from pathlib import Path

from forscheule.config import compute_pipeline_signature
from forscheule.db.repo import get_pipeline_run_signature, upsert_pipeline_run
from forscheule.db.schema import init_db

_PROFILE = "Test lab profile for spatial transcriptomics"
_PHRASES = ["spatial transcriptomics", "deconvolution"]


def _tmp_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
    tmp.close()
    return init_db(Path(tmp.name))


def test_signature_deterministic():
    """Same inputs produce the same signature."""
    sig1 = compute_pipeline_signature("2026-02-09", 7, 5, _PROFILE, _PHRASES)
    sig2 = compute_pipeline_signature("2026-02-09", 7, 5, _PROFILE, _PHRASES)
    assert sig1 == sig2


def test_signature_changes_on_lab_profile():
    sig1 = compute_pipeline_signature("2026-02-09", 7, 5, _PROFILE, _PHRASES)
    sig2 = compute_pipeline_signature("2026-02-09", 7, 5, "Different profile", _PHRASES)
    assert sig1 != sig2


def test_signature_changes_on_boosted_phrases():
    sig1 = compute_pipeline_signature("2026-02-09", 7, 5, _PROFILE, _PHRASES)
    sig2 = compute_pipeline_signature("2026-02-09", 7, 5, _PROFILE, ["new phrase"])
    assert sig1 != sig2


def test_signature_changes_on_window_days():
    sig1 = compute_pipeline_signature("2026-02-09", 7, 5, _PROFILE, _PHRASES)
    sig2 = compute_pipeline_signature("2026-02-09", 14, 5, _PROFILE, _PHRASES)
    assert sig1 != sig2


def test_signature_changes_on_top_k():
    sig1 = compute_pipeline_signature("2026-02-09", 7, 5, _PROFILE, _PHRASES)
    sig2 = compute_pipeline_signature("2026-02-09", 7, 10, _PROFILE, _PHRASES)
    assert sig1 != sig2


def test_signature_changes_on_date():
    sig1 = compute_pipeline_signature("2026-02-09", 7, 5, _PROFILE, _PHRASES)
    sig2 = compute_pipeline_signature("2026-02-10", 7, 5, _PROFILE, _PHRASES)
    assert sig1 != sig2


def test_pipeline_run_table_crud():
    """upsert and get round-trip for pipeline_runs."""
    conn = _tmp_db()

    # Initially empty
    assert get_pipeline_run_signature(conn, "2026-02-09") is None

    # Insert
    upsert_pipeline_run(conn, "2026-02-09", "abc123", 7, 5)
    assert get_pipeline_run_signature(conn, "2026-02-09") == "abc123"

    # Update
    upsert_pipeline_run(conn, "2026-02-09", "def456", 14, 10)
    assert get_pipeline_run_signature(conn, "2026-02-09") == "def456"

    conn.close()


def test_phrase_order_does_not_change_signature():
    """Boosted phrases in different order should produce the same signature."""
    sig1 = compute_pipeline_signature("2026-02-09", 7, 5, _PROFILE, ["a", "b", "c"])
    sig2 = compute_pipeline_signature("2026-02-09", 7, 5, _PROFILE, ["c", "a", "b"])
    assert sig1 == sig2


# ---------------------------------------------------------------------------
# Embedding model config changes trigger recomputation
# ---------------------------------------------------------------------------

def test_signature_changes_on_query_model():
    """Changing the query model name must produce a different signature."""
    sig1 = compute_pipeline_signature(
        "2026-02-09", 7, 5, _PROFILE, _PHRASES,
        query_model="ncbi/MedCPT-Query-Encoder",
    )
    sig2 = compute_pipeline_signature(
        "2026-02-09", 7, 5, _PROFILE, _PHRASES,
        query_model="some-other/query-model",
    )
    assert sig1 != sig2


def test_signature_changes_on_article_model():
    """Changing the article model name must produce a different signature."""
    sig1 = compute_pipeline_signature(
        "2026-02-09", 7, 5, _PROFILE, _PHRASES,
        article_model="ncbi/MedCPT-Article-Encoder",
    )
    sig2 = compute_pipeline_signature(
        "2026-02-09", 7, 5, _PROFILE, _PHRASES,
        article_model="some-other/article-model",
    )
    assert sig1 != sig2


def test_signature_changes_on_query_max_length():
    """Changing the query max length must produce a different signature."""
    sig1 = compute_pipeline_signature(
        "2026-02-09", 7, 5, _PROFILE, _PHRASES,
        query_max_length=512,
    )
    sig2 = compute_pipeline_signature(
        "2026-02-09", 7, 5, _PROFILE, _PHRASES,
        query_max_length=256,
    )
    assert sig1 != sig2


def test_signature_changes_on_article_max_length():
    """Changing the article max length must produce a different signature."""
    sig1 = compute_pipeline_signature(
        "2026-02-09", 7, 5, _PROFILE, _PHRASES,
        article_max_length=512,
    )
    sig2 = compute_pipeline_signature(
        "2026-02-09", 7, 5, _PROFILE, _PHRASES,
        article_max_length=256,
    )
    assert sig1 != sig2


def test_signature_stable_with_explicit_defaults():
    """Passing default model values explicitly should match omitting them."""
    from forscheule.config import (
        ARTICLE_MAX_LENGTH,
        ARTICLE_MODEL_NAME,
        QUERY_MAX_LENGTH,
        QUERY_MODEL_NAME,
    )

    sig_implicit = compute_pipeline_signature(
        "2026-02-09", 7, 5, _PROFILE, _PHRASES,
    )
    sig_explicit = compute_pipeline_signature(
        "2026-02-09", 7, 5, _PROFILE, _PHRASES,
        query_model=QUERY_MODEL_NAME,
        article_model=ARTICLE_MODEL_NAME,
        query_max_length=QUERY_MAX_LENGTH,
        article_max_length=ARTICLE_MAX_LENGTH,
    )
    assert sig_implicit == sig_explicit
