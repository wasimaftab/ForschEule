"""Central configuration loaded from environment / .env file."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # loads .env from cwd when present

BASE_DIR = Path(__file__).resolve().parent.parent
_default_db = str(BASE_DIR / "data" / "forscheule_papers.sqlite3")
DB_PATH = Path(os.getenv("FORSCHEULE_DB_PATH", _default_db))
FETCH_WINDOW_DAYS: int = int(os.getenv("FORSCHEULE_FETCH_WINDOW", "7"))

ENTREZ_EMAIL: str | None = os.getenv("ENTREZ_EMAIL")
ENTREZ_API_KEY: str | None = os.getenv("ENTREZ_API_KEY")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

TOP_K = 5  # papers per day

# MedCPT dual-encoder model configuration
QUERY_MODEL_NAME: str = os.getenv(
    "MEDCPT_QUERY_MODEL", "ncbi/MedCPT-Query-Encoder"
)
ARTICLE_MODEL_NAME: str = os.getenv(
    "MEDCPT_ARTICLE_MODEL", "ncbi/MedCPT-Article-Encoder"
)
QUERY_MAX_LENGTH: int = int(os.getenv("MEDCPT_QUERY_MAX_LENGTH", "512"))
ARTICLE_MAX_LENGTH: int = int(os.getenv("MEDCPT_ARTICLE_MAX_LENGTH", "512"))

# Lab profile for relevance scoring (compile-time defaults)
LAB_PROFILE = (
    "Our lab develops computational methods for spatial transcriptomics data analysis, "
    "focusing on integration of multi-modal single-cell and spatial omics datasets. "
    "We use deep learning approaches including graph neural networks, transformers, "
    "variational autoencoders, and contrastive learning for tasks such as spatial domain "
    "identification, cell-cell interaction inference, batch correction, atlas alignment, "
    "imputation, and deconvolution of spatially resolved gene expression data from "
    "platforms like Visium, MERFISH, seqFISH, Slide-seq, Stereo-seq, CosMx, and Xenium."
)

BOOSTED_PHRASES = [
    "spatial transcriptomics",
    "domain adaptation",
    "cell-cell communication",
    "cell-cell interaction",
    "graph neural network",
    "point transformer",
    "contrastive learning",
    "variational autoencoder",
    "atlas alignment",
    "deconvolution",
    "imputation",
    "batch correction",
    "multimodal integration",
    "spatial omics",
]


def get_runtime_settings(conn) -> dict:
    """Load settings from DB, falling back to compile-time defaults.

    Returns dict with keys: lab_profile, boosted_phrases (list), top_k (int).
    """
    from forscheule.db.repo import get_all_settings

    db_settings = get_all_settings(conn)
    lab_profile = db_settings.get("lab_profile", LAB_PROFILE)
    try:
        boosted = json.loads(db_settings.get("boosted_phrases", "null"))
    except (json.JSONDecodeError, TypeError):
        boosted = None
    if not isinstance(boosted, list):
        boosted = BOOSTED_PHRASES

    try:
        top_k = int(db_settings.get("top_k", str(TOP_K)))
    except (ValueError, TypeError):
        top_k = TOP_K

    return {
        "lab_profile": lab_profile,
        "boosted_phrases": boosted,
        "top_k": top_k,
    }


def compute_pipeline_signature(
    target_date: str,
    window_days: int,
    top_k: int,
    lab_profile: str,
    boosted_phrases: list[str],
    *,
    query_model: str | None = None,
    article_model: str | None = None,
    query_max_length: int | None = None,
    article_max_length: int | None = None,
) -> str:
    """Compute a deterministic hash of pipeline configuration.

    Used to detect whether settings have changed since last run for a given date.
    Includes embedding model names and max lengths so that model changes trigger
    recomputation.
    """
    raw = json.dumps(
        {
            "target_date": target_date,
            "window_days": window_days,
            "top_k": top_k,
            "lab_profile_hash": hashlib.sha256(lab_profile.encode()).hexdigest(),
            "boosted_phrases_hash": hashlib.sha256(
                json.dumps(sorted(boosted_phrases)).encode()
            ).hexdigest(),
            "query_model": query_model or QUERY_MODEL_NAME,
            "article_model": article_model or ARTICLE_MODEL_NAME,
            "query_max_length": query_max_length or QUERY_MAX_LENGTH,
            "article_max_length": article_max_length or ARTICLE_MAX_LENGTH,
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def openai_status() -> tuple[bool, str]:
    """Check whether OpenAI summaries can run.

    Returns (ok, reason) where *ok* is True when both the API key is set
    and the ``openai`` package is importable.  *reason* is an empty string
    on success or a user-facing explanation on failure.
    """
    try:
        import openai  # noqa: F401
    except ImportError:
        import sys

        pip_cmd = f"{sys.executable} -m pip install \"openai>=1.30,<2\""
        return False, f"The 'openai' package is not installed. Run:  {pip_cmd}"
    if not OPENAI_API_KEY:
        return False, (
            "OpenAI API key is not configured. "
            "Set OPENAI_API_KEY in your .env file."
        )
    return True, ""


def is_openai_available() -> bool:
    """Convenience wrapper – True when summaries can run."""
    ok, _ = openai_status()
    return ok


def get_openai_client():
    """Create an OpenAI client if API key is available. Returns None if not configured."""
    if not OPENAI_API_KEY:
        return None
    from openai import OpenAI

    return OpenAI(api_key=OPENAI_API_KEY, max_retries=3, timeout=120.0)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not ENTREZ_EMAIL:
        logging.getLogger(__name__).warning(
            "ENTREZ_EMAIL not set – PubMed requests may be rate-limited. "
            "Set it in .env or environment."
        )
