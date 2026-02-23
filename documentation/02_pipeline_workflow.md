# ForschEule — Pipeline Workflow

This document explains what happens when you run `python -m forscheule run-daily`
or `python -m forscheule backfill --days N`. Each step is described with its
purpose, inputs, and outputs.

---

## Pipeline Flow Diagram

```
  ┌───────────────────────────────────────────────────────────────────┐
  │                        PIPELINE START                             │
  │                   (target date, window)                           │
  └───────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │  STEP 0: Signature-based Idempotency Check                        │
  │                                                                   │
  │  Compute a SHA-256 signature of the full config:                  │
  │    (date, window, top_k, lab_profile, boosted_phrases,            │
  │     query_model, article_model, query_max_length,                 │
  │     article_max_length)                                           │
  │                                                                   │
  │  Does `pipeline_runs` have a row for this date                    │
  │  with the SAME signature?                                         │
  │                                                                   │
  │     YES ──> Log "skipping" and EXIT (no work done)                │
  │     DIFFERENT SIG ──> Log "settings changed" and RECOMPUTE        │
  │     NO ROW ──> Continue to Step 1                                 │
  └───────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │  STEP 1: Fetch Papers                                             │
  │                                                                   │
  │  ┌─────────────────────┐    ┌─────────────────────┐               │
  │  │  PubMed (Entrez)    │    │  arXiv (export API)  │              │
  │  │                     │    │                      │              │
  │  │  Search query:      │    │  Search query:       │              │
  │  │  spatial trans-     │    │  spatial trans-      │              │
  │  │  criptomics AND    │    │  criptomics AND     │              │
  │  │  (integration OR   │    │  (integration OR    │              │
  │  │   alignment OR ... │    │   alignment OR ...  │              │
  │  │   transformer ...) │    │   deep learning ..) │              │
  │  │                     │    │                      │              │
  │  │  Date filter:       │    │  Date filter:        │              │
  │  │  last N days        │    │  last N days         │              │
  │  │  (via API params)   │    │  (post-fetch check)  │              │
  │  │                     │    │                      │              │
  │  │  Rate limit:        │    │  Rate limit:         │              │
  │  │  0.35s (w/ key)     │    │  3.5s between        │              │
  │  │  1.0s  (w/o key)    │    │  requests            │              │
  │  └────────┬────────────┘    └────────┬─────────────┘              │
  │           │                          │                            │
  │           └──────────┬───────────────┘                            │
  │                      │                                            │
  │              Combined list of Paper objects                       │
  └───────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │  STEP 2: Deduplicate                                              │
  │                                                                   │
  │  Three layers of dedup applied in order:                          │
  │                                                                   │
  │  1. Source ID  ─── same source + source_id? Skip.                 │
  │  2. DOI        ─── same DOI (case-insensitive)? Skip.            │
  │  3. Fuzzy Title ── RapidFuzz token_set_ratio >= 95? Skip.        │
  │                                                                   │
  │  Title normalisation: lowercase, strip punctuation,               │
  │  collapse whitespace, Unicode NFKD normalisation.                 │
  │                                                                   │
  │  Input:  all fetched papers (may have cross-source duplicates)    │
  │  Output: unique papers only                                       │
  └───────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │  STEP 3: Store Papers in DB                                       │
  │                                                                   │
  │  Each paper is upserted (INSERT ... ON CONFLICT UPDATE).          │
  │  If the paper already exists (same source + source_id),           │
  │  its metadata is updated but existing embeddings are preserved.   │
  └───────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │  STEP 4: Embed & Score & Rank                                     │
  │                                                                   │
  │  ┌────────────────────────────────────────────────────────┐       │
  │  │  MedCPT Dual-Encoder Embedding                          │       │
  │  │                                                        │       │
  │  │  Query tower:  ncbi/MedCPT-Query-Encoder               │       │
  │  │    Input:  LAB_PROFILE text (single string)             │       │
  │  │    Output: 768-dim vector (CLS pooling)                 │       │
  │  │                                                        │       │
  │  │  Article tower: ncbi/MedCPT-Article-Encoder             │       │
  │  │    Input:  [title, abstract] sentence pairs             │       │
  │  │    Output: 768-dim vector per paper (CLS pooling)       │       │
  │  │                                                        │       │
  │  │  Both models are lazy-loaded on first use and cached    │       │
  │  │  in memory for subsequent calls.                        │       │
  │  └────────────────────────────────────────────────────────┘       │
  │                                                                   │
  │  ┌────────────────────────────────────────────────────────┐       │
  │  │  Scoring Formula (per paper)                            │       │
  │  │                                                        │       │
  │  │  final = 0.5 * cosine_similarity                        │       │
  │  │        + 0.2 * recency_score                            │       │
  │  │        + 0.3 * keyword_boost                            │       │
  │  │        - short_abstract_penalty                         │       │
  │  │                                                        │       │
  │  │  cosine_similarity: paper_emb vs lab_profile_emb        │       │
  │  │  recency_score:     linear decay over 14 days           │       │
  │  │  keyword_boost:     0.05 per matched BOOSTED_PHRASE     │       │
  │  │  penalty:           0.15 if abstract < 100 chars        │       │
  │  └────────────────────────────────────────────────────────┘       │
  │                                                                   │
  │  Sort all papers by final score (descending).                     │
  │  Take the top K (default 5, configurable via settings).           │
  └───────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
  ┌───────────────────────────────────────────────────────────────────┐
  │  STEP 5: Store Recommendations & Pipeline Run                     │
  │                                                                   │
  │  DELETE existing recommendations for this date (if any).          │
  │  INSERT the top K with rank, score, and matched_terms.            │
  │  This DELETE+INSERT pattern ensures idempotency —                 │
  │  re-running replaces rather than duplicates.                      │
  │                                                                   │
  │  UPSERT a row in `pipeline_runs` with the date, signature,       │
  │  window_days, and top_k. This records the configuration so that   │
  │  future runs can detect settings changes (Step 0).                │
  └───────────────────────────────────────────────────────────────────┘
```

---

## Scoring Formula Explained

The final score for each paper is a weighted combination of three signals, minus
a penalty:

```
final_score = 0.5 * cosine_sim + 0.2 * recency + 0.3 * keyword_boost - penalty
```

### Cosine Similarity (weight: 50%)

This is the core semantic signal. The lab profile text is embedded by the
**MedCPT Query Encoder** and each paper's title + abstract pair is embedded by
the **MedCPT Article Encoder**. Using separate encoder towers (dual-encoder
architecture) produces embeddings in a shared vector space optimised for
query-to-document matching. Cosine similarity is then computed between the
profile vector and each paper vector.

Values range from -1 to 1, but in practice biomedical texts tend to fall in
the 0.3 to 0.9 range. A paper about "spatial transcriptomics integration using
graph neural networks" will naturally have a high cosine similarity to a lab
profile that describes the same focus area, even if the exact words differ.

### Recency (weight: 20%)

Linear decay over 14 days:

```
recency = max(0, 1 - age_days / 14)
```

- Published today: 1.0
- Published 7 days ago: 0.5
- Published 14+ days ago: 0.0
- Unknown date: 0.5 (neutral)

This ensures newer papers are preferred when relevance is similar.

### Keyword Boost (weight: 30%)

Each matched phrase from `BOOSTED_PHRASES` adds +0.05 to the keyword component.
For example, if a paper's title and abstract contain "spatial transcriptomics",
"graph neural network", and "deconvolution", that is 3 matches = 0.15 keyword
boost.

This gives explicit credit to papers that mention your lab's core terms, on top
of the semantic similarity.

### Short Abstract Penalty

Papers with abstracts shorter than 100 characters receive a -0.15 penalty.
These are often placeholders, corrections, or editorial notes that are not
useful recommendations.

---

## Backfill vs Run-Daily

| Command | What it does |
|---------|-------------|
| `python -m forscheule run-daily` | Runs the pipeline once for **today** |
| `python -m forscheule backfill --days 3` | Runs the pipeline for **today, yesterday, and 2 days ago** |

Both commands are **idempotent** — if a pipeline run for a given date already
exists with the same configuration signature, that date is skipped. If the
settings have changed since the last run (different lab profile, top_k,
embedding model, etc.), the pipeline recomputes automatically.

The `--window` flag controls how far back to look for papers (default: 7 days).
For example, `--window 14` means "search for papers published in the last 14
days" when generating recommendations for a target date.

---

## Deduplication Details

Papers can appear on both PubMed and arXiv, or be indexed with slight title
variations. ForschEule uses three dedup layers:

```
                     ┌──────────────────┐
  Incoming paper ──> │ Same source+id?  │──YES──> SKIP
                     └───────┬──────────┘
                             │ NO
                             ▼
                     ┌──────────────────┐
                     │ Same DOI?        │──YES──> SKIP
                     │ (case-insensitive)│
                     └───────┬──────────┘
                             │ NO
                             ▼
                     ┌──────────────────┐
                     │ Fuzzy title      │──YES──> SKIP
                     │ match >= 95%?    │  (RapidFuzz token_set_ratio)
                     └───────┬──────────┘
                             │ NO
                             ▼
                        KEEP PAPER
```

The fuzzy matching uses `token_set_ratio`, which is robust to word reordering,
extra words, and minor differences. For example, these two titles would be
considered duplicates (score ~98):

- "Spatial Transcriptomics: A New Method for Integration"
- "Spatial transcriptomics a new method for integration"
