# ForschEule — Project Overview

## What is ForschEule?

ForschEule (German: "Research Owl") is a daily paper recommendation service
designed for research labs working in **Biomedical domains**. Each day it
automatically searches PubMed and arXiv, ranks papers against your lab's
research interests, and surfaces the **top 5 most relevant papers**.

Think of it as a personal research assistant that reads through dozens of new
publications every day and picks the ones your lab should pay attention to.

### Why build this?

The biomedical literature is expanding faster than any lab can reasonably track. Even within a focused area like *spatial transcriptomics*, new papers appear continuously across PubMed and arXiv every week covering techniques like Visium, MERFISH,
Slide-seq, and computational methods for integration, deconvolution, and atlas
alignment. Manually scanning these sources is time-consuming. ForschEule
automates the discovery so researchers can focus on reading the papers that
actually matter to them.

---

## Architecture at a Glance

```
                         ForschEule Architecture
    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    │   DATA SOURCES              PIPELINE                 OUTPUT  │
    │                                                              │
    │   ┌──────────┐     ┌─────────────────────┐    ┌──────────┐  │
    │   │  PubMed  │────>│                     │    │  SQLite   │  │
    │   │ (Entrez) │     │  1. Fetch           │    │    DB     │  │
    │   └──────────┘     │  2. Normalise       │    │          │  │
    │                    │  3. Deduplicate      │───>│ papers   │  │
    │   ┌──────────┐     │  4. Embed (MedCPT   │    │ recs     │  │
    │   │  arXiv   │────>│     dual-encoder)   │    │ settings │  │
    │   │(export)  │     │  5. Score & Rank    │    │ runs     │  │
    │   └──────────┘     │  6. Store top K     │    │ summaries│  │
    │                    └─────────────────────┘    └────┬─────┘  │
    │                                                    │        │
    │   ┌──────────┐     ┌─────────────────────┐         │        │
    │   │ OpenAI   │<───>│     FastAPI          │<────────┘        │
    │   │ (opt.)   │     │  Web console + API   │                  │
    │   └──────────┘     └─────────────────────┘                  │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
```

---

## Key Design Principles

### 1. Fully local and self-contained

Everything runs on a single machine. There are no cloud dependencies for
inference — the MedCPT embedding model runs locally on CPU via PyTorch. The
database is a single SQLite file. No external ranking APIs.

### 2. Signature-based idempotent daily runs

Running the pipeline twice for the same date with the same settings does
nothing the second time. The system computes a deterministic hash (signature)
of the full pipeline configuration — date, window, top_k, lab profile,
boosted phrases, and embedding model names/lengths — and compares it against
the stored signature in `pipeline_runs`. If settings change, the pipeline
automatically recomputes for that date. This means you can safely re-run
after a crash, cron misfire, or manual test without worrying about duplicates.

### 3. Rate-limit respectful

Both PubMed and arXiv have rate limits. ForschEule obeys them:
- **PubMed**: 0.35s delay with API key, 1.0s without (NCBI allows 3 req/s
  without key, 10/s with key)
- **arXiv**: 3.5s delay between requests (their guideline is ~3s)

All HTTP requests use exponential backoff retries for transient errors (429,
500, 502, 503, 504).

### 4. No secrets in logs

The logging system never prints API keys, email addresses, or `.env` contents.
URLs in logs do not contain query-string secrets.

---

## Project File Structure

```
ForschEule/
├── forscheule/                  # Main Python package
│   ├── __init__.py
│   ├── __main__.py              # CLI entrypoint (backfill, run-daily, serve)
│   ├── config.py                # Configuration, lab profile, MedCPT model settings
│   ├── pipeline.py              # Orchestrates the full daily pipeline
│   ├── sources/
│   │   ├── http_client.py       # Shared HTTP session with retries
│   │   ├── pubmed.py            # PubMed Entrez fetcher
│   │   └── arxiv.py             # arXiv export API fetcher
│   ├── rank/
│   │   ├── embed.py             # MedCPT dual-encoder embeddings
│   │   ├── score.py             # Relevance scoring and ranking
│   │   └── dedup.py             # Deduplication (DOI, title, fuzzy match)
│   ├── db/
│   │   ├── schema.py            # SQLite schema creation
│   │   └── repo.py              # CRUD operations (papers, recommendations, settings)
│   ├── api/
│   │   └── app.py               # FastAPI web console, settings, and job triggers
│   └── templates/
│       └── index.html           # Jinja2 web console template
├── tests/                       # Pytest test suite (96 tests)
│   ├── test_api.py              # API endpoint and web console tests
│   ├── test_db.py               # Schema, CRUD, pipeline_runs tests
│   ├── test_dedup.py            # Deduplication tests
│   ├── test_idempotency.py      # Signature-based idempotency tests
│   ├── test_score.py            # Scoring utilities and ranking wiring tests
│   ├── test_summary.py          # OpenAI summary generation tests
│   ├── test_summary_api.py      # Summary job API endpoint tests
│   └── test_time_anchoring.py   # Date-anchored fetch and recency tests
├── deploy/
│   ├── forscheule.service       # systemd service template
│   └── forscheule.timer         # systemd timer template (daily at 07:00)
├── data/
│   └── forscheule_papers.sqlite3  # SQLite database (auto-created)
├── documentation/               # This documentation
├── pyproject.toml               # Package definition and dependencies
├── .env.example                 # Template for environment variables
└── .gitignore
```

---

## Technology Stack

| Component         | Technology                   | Why                                           |
|-------------------|------------------------------|-----------------------------------------------|
| Language          | Python 3.12                  | Lab standard, rich ML ecosystem               |
| Database          | SQLite (WAL mode)            | Zero-config, single-file, fast for this scale  |
| Embeddings        | MedCPT dual-encoder (Query + Article) | Dedicated towers for queries and articles |
| ML runtime        | PyTorch (CPU-only)           | Lightweight install (~200 MB vs 2+ GB CUDA)   |
| HTTP              | requests + urllib3 Retry     | Battle-tested, built-in backoff               |
| Fuzzy matching    | RapidFuzz                    | Fast C-extension for title dedup              |
| API framework     | FastAPI + Jinja2             | Web console, async API, auto-docs             |
| Summaries         | OpenAI API (optional)        | Paper summaries and weekly digest synthesis    |
| Scheduling        | systemd timer                | Reliable, env-var support, journal logging    |

---

## Data Model

```
┌─────────────────────────────┐       ┌──────────────────────────────┐
│          papers              │       │       recommendations         │
├─────────────────────────────┤       ├──────────────────────────────┤
│ id          INTEGER PK       │       │ id           INTEGER PK       │
│ source      TEXT (pubmed|    │       │ date         TEXT              │
│             arxiv)           │◄──────│ paper_id     INTEGER FK       │
│ source_id   TEXT             │       │ score        REAL              │
│ title       TEXT             │       │ rank         INTEGER           │
│ abstract    TEXT             │       │ matched_terms TEXT (JSON)      │
│ authors     TEXT             │       │                                │
│ published_at TEXT            │       │ UNIQUE(date, rank)             │
│ url         TEXT             │       └──────────────────────────────┘
│ doi         TEXT             │
│ embedding   BLOB             │       ┌──────────────────────────────┐
│ created_at  TEXT             │       │       pipeline_runs            │
│                              │       ├──────────────────────────────┤
│ UNIQUE(source, source_id)    │       │ date        TEXT PK            │
└──────────────┬──────────────┘       │ signature   TEXT               │
               │                       │ window_days INTEGER            │
               │                       │ top_k       INTEGER            │
               │                       │ updated_at  TEXT               │
               │                       └──────────────────────────────┘
               │
               │                       ┌──────────────────────────────┐
               │                       │       paper_summaries         │
               ├──────────────────────>├──────────────────────────────┤
               │                       │ id              INTEGER PK    │
               │                       │ paper_id        INTEGER FK    │
               │                       │ model           TEXT           │
               │                       │ schema_version  TEXT           │
               │                       │ input_hash      TEXT           │
               │                       │ summary_json    TEXT           │
               │                       └──────────────────────────────┘
               │
               │                       ┌──────────────────────────────┐
               │                       │       settings                │
               │                       ├──────────────────────────────┤
               │                       │ key   TEXT PK                 │
               │                       │ value TEXT                    │
               │                       └──────────────────────────────┘
               │
               │                       ┌──────────────────────────────┐
               │                       │       weekly_digests          │
               │                       ├──────────────────────────────┤
               │                       │ id                  INT PK    │
               │                       │ date                TEXT      │
               │                       │ window_days         INTEGER   │
               │                       │ top_n               INTEGER   │
               │                       │ paper_set_signature TEXT      │
               │                       │ paper_model         TEXT      │
               │                       │ synthesis_model     TEXT      │
               │                       │ digest_json         TEXT      │
               │                       └──────────────────────────────┘
```

- **papers**: Every paper ever fetched, with metadata and optional embedding blob.
- **recommendations**: Daily top-K picks with scores and matched keyword terms.
- **pipeline_runs**: One row per date, stores the configuration signature so the
  pipeline detects when settings change and recomputes.
- **settings**: Key-value store for runtime settings (lab_profile, boosted_phrases,
  top_k) editable from the web console.
- **paper_summaries**: Cached per-paper OpenAI summaries (keyed by model + input hash).
- **weekly_digests**: Cached weekly digest syntheses across a date range.
