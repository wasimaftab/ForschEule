"""MedCPT embeddings – dual-encoder for biomedical text.

Uses separate models for queries (lab profile) and articles (title + abstract),
following the MedCPT dual-encoder architecture:
- Query tower:  ``ncbi/MedCPT-Query-Encoder``
- Article tower: ``ncbi/MedCPT-Article-Encoder``
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from forscheule.config import (
    ARTICLE_MAX_LENGTH,
    ARTICLE_MODEL_NAME,
    QUERY_MAX_LENGTH,
    QUERY_MODEL_NAME,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded model singletons
# ---------------------------------------------------------------------------
_query_tokenizer = None
_query_model = None
_article_tokenizer = None
_article_model = None


def _load_query_model():
    global _query_tokenizer, _query_model
    if _query_tokenizer is None:
        logger.info("Loading MedCPT query model: %s", QUERY_MODEL_NAME)
        _query_tokenizer = AutoTokenizer.from_pretrained(QUERY_MODEL_NAME)
        _query_model = AutoModel.from_pretrained(QUERY_MODEL_NAME)
        _query_model.eval()
    return _query_tokenizer, _query_model


def _load_article_model():
    global _article_tokenizer, _article_model
    if _article_tokenizer is None:
        logger.info("Loading MedCPT article model: %s", ARTICLE_MODEL_NAME)
        _article_tokenizer = AutoTokenizer.from_pretrained(ARTICLE_MODEL_NAME)
        _article_model = AutoModel.from_pretrained(ARTICLE_MODEL_NAME)
        _article_model.eval()
    return _article_tokenizer, _article_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_queries(
    texts: list[str],
    batch_size: int = 16,
    max_length: int | None = None,
) -> np.ndarray:
    """Embed query texts (e.g. lab profile) using the query encoder.

    Returns (N, D) float32 array.
    """
    tokenizer, model = _load_query_model()
    ml = max_length or QUERY_MAX_LENGTH
    return _embed_batch(tokenizer, model, texts, batch_size, ml, "query")


def embed_articles(
    items: Sequence[list[str] | tuple[str, str]],
    batch_size: int = 16,
    max_length: int | None = None,
) -> np.ndarray:
    """Embed articles using the article encoder.

    Each item is a ``[title, abstract]`` pair.  The tokenizer encodes
    them as a sentence pair (``text`` / ``text_pair``).

    Returns (N, D) float32 array.
    """
    tokenizer, model = _load_article_model()
    ml = max_length or ARTICLE_MAX_LENGTH
    all_embeddings: list[np.ndarray] = []

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        titles = [pair[0] for pair in batch]
        abstracts = [pair[1] for pair in batch]
        encoded = tokenizer(
            titles,
            abstracts,
            padding=True,
            truncation=True,
            max_length=ml,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**encoded)
            emb = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(emb.cpu().numpy())

    result = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    logger.info("Embedded %d articles -> shape %s", len(items), result.shape)
    return result


# ---------------------------------------------------------------------------
# Backward-compatible wrappers
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str], batch_size: int = 16) -> np.ndarray:
    """Embed plain texts using the query encoder (backward-compatible)."""
    return embed_queries(texts, batch_size=batch_size)


def embed_single(text: str) -> np.ndarray:
    """Embed a single query text, returns (D,) vector."""
    return embed_queries([text])[0]


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _embed_batch(
    tokenizer,
    model,
    texts: list[str],
    batch_size: int,
    max_length: int,
    label: str,
) -> np.ndarray:
    all_embeddings: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**encoded)
            emb = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(emb.cpu().numpy())

    result = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    logger.info("Embedded %d %s texts -> shape %s", len(texts), label, result.shape)
    return result
