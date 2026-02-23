# ForschEule — Tuning Guide

This document explains the two most important configuration levers for
controlling what papers ForschEule recommends: `LAB_PROFILE` and
`BOOSTED_PHRASES`. It also covers how to modify the search queries and scoring
weights if needed.

Configuration lives in `forscheule/config.py` (compile-time defaults) and can
also be changed at runtime via the **web console** (`PUT /settings`) without
editing code or restarting the server.

---

## Why LAB_PROFILE and BOOSTED_PHRASES Matter

ForschEule does not use a simple keyword filter. It uses a **two-layer ranking
system** where `LAB_PROFILE` drives the deep semantic understanding and
`BOOSTED_PHRASES` provides fine-grained control over which specific topics get
extra priority.

```
  ┌─────────────────────────────────────────────────────────────┐
  │                  How papers get scored                       │
  │                                                             │
  │                                                             │
  │   LAB_PROFILE (50% of score)     BOOSTED_PHRASES (30%)      │
  │   ┌───────────────────┐          ┌───────────────────┐      │
  │   │ "Our lab develops │          │ List of exact      │     │
  │   │  computational    │          │ phrases to look    │     │
  │   │  methods for      │          │ for in title +     │     │
  │   │  spatial trans-   │          │ abstract.          │     │
  │   │  criptomics..."  │          │                    │     │
  │   └────────┬──────────┘          │ Each match adds    │     │
  │            │                     │ +0.05 to score.    │     │
  │            ▼                     └────────┬───────────┘     │
  │   Converted to a 768-dim                  │                 │
  │   vector by the MedCPT                   │                 │
  │   Query Encoder. Compared                │                 │
  │   to each paper's vector                 │                 │
  │   (from Article Encoder)                 │                 │
  │   via cosine similarity.                 │                 │
  │            │                              │                 │
  │            ▼                              ▼                 │
  │   Semantic understanding          Explicit keyword          │
  │   of relevance (even if           matching (exact           │
  │   exact words differ)             phrase required)          │
  │                                                             │
  │   Example: profile says           Example: "graph neural    │
  │   "graph neural networks"         network" in the list      │
  │   and paper says "GNN-based       will match papers that    │
  │   spatial analysis" —             literally contain that     │
  │   still high similarity           phrase                    │
  └─────────────────────────────────────────────────────────────┘
```

They serve complementary roles:

| Aspect | LAB_PROFILE | BOOSTED_PHRASES |
|--------|-------------|-----------------|
| **What it does** | Defines the "meaning space" of your research | Gives bonus points to specific terms |
| **How it works** | Embedded by MedCPT Query Encoder into a 768-dim vector; compared via cosine similarity to paper vectors from the Article Encoder | Simple substring matching in title + abstract |
| **Matching style** | Semantic (catches synonyms, related concepts) | Exact phrase (case-insensitive) |
| **Score contribution** | 50% of final score | 30% of final score (0.05 per match) |
| **When to change** | Lab's focus shifts, new project starts | Want to promote/demote specific terms |

---

## Tuning LAB_PROFILE

### Where it is

In `config.py` (compile-time default):

```python
LAB_PROFILE = (
    "Our lab develops computational methods for spatial transcriptomics data analysis, "
    "focusing on integration of multi-modal single-cell and spatial omics datasets. "
    "We use deep learning approaches including graph neural networks, transformers, "
    ...
)
```

Or at runtime via the web console's **Settings** panel / `PUT /settings`
endpoint, which stores the value in the `settings` table and takes effect on
the next pipeline run.

### How to write a good profile

The profile is the single most important text in the system. The MedCPT Query
Encoder converts it into a dense vector that represents "what your lab cares
about." Every paper is then compared to this vector (using the Article Encoder).
A well-written profile dramatically improves recommendation quality.

**Guidelines:**

1. **Be specific, not generic.** Write it the way you would describe your lab's
   focus to a new postdoc, not to a grant agency.

   ```
   Bad:  "We study biology using computational methods."
   Good: "We develop graph neural network methods for integrating spatial
          transcriptomics data with single-cell RNA-seq, focusing on spatial
          domain identification and cell-cell communication inference."
   ```

2. **Mention your core platforms and technologies.** MedCPT understands
   biomedical terms. If your lab works with Visium and MERFISH, say so. If
   you use transformers and contrastive learning, say so. These specific terms
   shift the embedding vector toward the right region of the semantic space.

3. **Keep it to 1-2 paragraphs.** The query encoder has a configurable max
   token length (default 512, set via `MEDCPT_QUERY_MAX_LENGTH`). Very long
   profiles get truncated, and the most important information should be near the
   beginning.

4. **Update it when priorities change.** If the lab starts a new project on
   point cloud methods for spatial data, add that to the profile. The
   recommendations will shift toward those topics.

### Example profiles for different lab types

**Lab focused on spatial method development:**
```python
LAB_PROFILE = (
    "Our lab develops novel computational tools for spatial transcriptomics, "
    "including methods for spatial domain identification, cell type deconvolution, "
    "and integration of Visium, MERFISH, and Stereo-seq data with scRNA-seq references. "
    "We are particularly interested in graph neural networks, attention mechanisms, "
    "and variational inference applied to spatially resolved gene expression."
)
```

**Lab focused on cancer biology using spatial methods:**
```python
LAB_PROFILE = (
    "We study the tumor microenvironment using spatial transcriptomics and "
    "multiplex imaging. Our interests include immune cell infiltration patterns, "
    "cell-cell communication in solid tumors, spatial niches, and integration of "
    "spatial proteomics with transcriptomics data from technologies like "
    "CosMx, Xenium, and CODEX."
)
```

**Lab focused on neuroscience + spatial:**
```python
LAB_PROFILE = (
    "Our group maps cell types and circuit architecture in the brain using "
    "spatial transcriptomics technologies including MERFISH and Slide-seq. "
    "We develop and apply methods for cell segmentation, spatial clustering, "
    "and cross-modal integration with electrophysiology and connectomics data."
)
```

---

## Tuning BOOSTED_PHRASES

### Where it is

In `config.py` (compile-time default):

```python
BOOSTED_PHRASES = [
    "spatial transcriptomics",
    "domain adaptation",
    "cell-cell communication",
    "cell-cell interaction",
    "graph neural network",
    ...
]
```

Or at runtime via the web console's **Settings** panel / `PUT /settings`
(stored as a JSON array in the `settings` table).

### How boosting works

When a paper's title + abstract contain one of these phrases (case-insensitive
substring match), the paper gets **+0.05** added to its keyword component. The
keyword component is weighted at 30% of the final score.

Example scoring impact:

| Matches found | Keyword component | Contribution to final score (x 0.3) |
|---------------|------------------|-------------------------------------|
| 0 phrases     | 0.00             | 0.000                               |
| 1 phrase      | 0.05             | 0.015                               |
| 2 phrases     | 0.10             | 0.030                               |
| 3 phrases     | 0.15             | 0.045                               |
| 5 phrases     | 0.25             | 0.075                               |

A paper matching 3 boosted phrases gets a +0.045 boost to its final score.
Given that top papers typically score between 0.55 and 0.70, this can be enough
to shift rankings.

### How to choose good boosted phrases

1. **Use the exact phrases you want to see.** If you care about "cell-cell
   communication" specifically (not just "communication"), put the full phrase.

2. **Be precise, not broad.** A phrase like "method" would match almost every
   paper and give meaningless boosts. Phrases like "graph neural network" or
   "atlas alignment" are specific enough to be useful.

3. **Include both general and specific terms.** A mix of broad topic terms
   ("spatial transcriptomics") and specific method terms ("point transformer")
   works well.

4. **Monitor matched_terms in the output.** The API response includes a
   `matched_terms` field for each recommendation showing which boosted phrases
   were found. Use this to audit whether your phrases are actually matching.

5. **Limit to 10-25 phrases.** Too few and the boost has no effect. Too many
   and everything gets boosted, reducing differentiation.

### Adding or removing phrases

**Option A — Edit `config.py`** (for permanent defaults):

```python
BOOSTED_PHRASES = [
    "spatial transcriptomics",
    "domain adaptation",
    "cell-cell communication",
    # Add your new terms:
    "foundation model",
    "zero-shot",
    "protein language model",
    # Remove terms that are too broad by deleting or commenting out:
    # "integration",  # too generic, matches everything
]
```

**Option B — Use the web console** (no code changes required):

Navigate to the web console, open the Settings panel, and edit the
"Boosted phrases" JSON array. Changes persist in the database and take effect
on the next pipeline run.

No restart of the database is needed either way. The pipeline's signature-based
idempotency will detect the change and recompute recommendations automatically.

---

## Beyond LAB_PROFILE and BOOSTED_PHRASES

### Changing the search queries

The search queries determine **which papers are fetched** in the first place.
No matter how good your ranking is, it can only rank papers that were fetched.

**PubMed query** (`forscheule/sources/pubmed.py`, line 18):
```python
QUERY = (
    '("spatial transcriptomics"[Title/Abstract] '
    'OR "spatially resolved transcriptomics"[Title/Abstract] '
    'OR "spatial omics"[Title/Abstract]) '
    "AND (integration[Title/Abstract] OR multimodal[Title/Abstract] "
    ...
)
```

**arXiv query** (`forscheule/sources/arxiv.py`, line 20):
```python
QUERY = (
    'all:("spatial transcriptomics" OR "spatially resolved transcriptomics" ...)'
    " AND (all:integration OR all:multimodal OR all:alignment ...)"
    ...
)
```

If you find that relevant papers are missing entirely (not ranked low, but
absent), the fix is to broaden the search query. For example, to also fetch
papers about "spatial proteomics":

```python
# In pubmed.py QUERY, add to the first OR group:
'OR "spatial proteomics"[Title/Abstract] '

# In arxiv.py QUERY, add to the first OR group:
'OR "spatial proteomics"'
```

Be careful not to make the query too broad — fetching 500+ papers per day will
slow down the embedding step.

### Adjusting scoring weights

The scoring formula in `forscheule/rank/score.py` (line 80):

```python
final = 0.5 * sim + 0.2 * recency + 0.3 * keyword_score - penalty
```

You can adjust these weights:

| Change | Effect |
|--------|--------|
| Increase `sim` weight (e.g., 0.7) | Ranking relies more on semantic meaning; good if your profile is well-written |
| Increase `recency` weight (e.g., 0.4) | Newer papers are strongly preferred even if less relevant |
| Increase `keyword_score` weight (e.g., 0.5) | Exact phrase matches dominate; good for very specific interests |
| Decrease `penalty` (e.g., 0.05) | Short-abstract papers are penalised less |

The weights should sum to approximately 1.0 (before penalty) for scores to
stay in a readable range.

### Adjusting the recency window

In `forscheule/rank/score.py`, line 16:

```python
_RECENCY_WINDOW = 14  # days for linear decay
```

- Set to 7 for a faster decay (strongly prefer very recent papers)
- Set to 30 for a slower decay (older papers stay competitive longer)

### Adjusting the keyword boost per phrase

In `forscheule/rank/score.py`, line 17:

```python
_KEYWORD_BOOST = 0.05  # per matched phrase
```

- Increase to 0.10 for stronger per-phrase boosting
- Decrease to 0.02 for subtler boosting

---

## Tuning Workflow Summary

```
  "I'm not seeing papers about [topic X]"
      │
      ├── Are papers about X being fetched at all?
      │       │
      │       ├── NO ──> Edit the QUERY in pubmed.py / arxiv.py
      │       │          to include terms related to X
      │       │
      │       └── YES ─> Are they ranked too low?
      │                    │
      │                    ├── Add X-related terms to BOOSTED_PHRASES
      │                    │
      │                    ├── Mention X in LAB_PROFILE
      │                    │
      │                    └── Increase keyword_score weight in score.py
      │
  "I'm seeing too many irrelevant papers"
      │
      ├── Narrow the search QUERY (more specific AND clauses)
      │
      ├── Remove overly broad terms from BOOSTED_PHRASES
      │
      └── Increase the cosine similarity weight (rely more on semantics)

  "Papers are too old / I only want very recent ones"
      │
      ├── Decrease FETCH_WINDOW_DAYS in .env (default: 7)
      │
      └── Increase recency weight in score.py or decrease _RECENCY_WINDOW
```

---

## Quick Reference: What to Edit

| Goal | File / Method | What to change |
|------|---------------|----------------|
| Change lab research focus | `config.py` or web console | Edit `LAB_PROFILE` text (or `PUT /settings` with `lab_profile`) |
| Boost/demote specific terms | `config.py` or web console | Edit `BOOSTED_PHRASES` list (or `PUT /settings` with `boosted_phrases`) |
| Change how many papers per day | `config.py` or web console | Change `TOP_K` (or `PUT /settings` with `top_k`) |
| Fetch papers on new topics | `sources/pubmed.py`, `sources/arxiv.py` | Edit `QUERY` strings |
| Change fetch window | `.env` | Set `FORSCHEULE_FETCH_WINDOW=14` |
| Change query embedding model | `.env` | Set `MEDCPT_QUERY_MODEL` |
| Change article embedding model | `.env` | Set `MEDCPT_ARTICLE_MODEL` |
| Change max token lengths | `.env` | Set `MEDCPT_QUERY_MAX_LENGTH` / `MEDCPT_ARTICLE_MAX_LENGTH` |
| Adjust score weights | `rank/score.py` | Edit line 98 weights |
| Adjust recency decay | `rank/score.py` | Change `_RECENCY_WINDOW` |
| Adjust fuzzy dedup threshold | `rank/dedup.py` | Change `_FUZZY_THRESHOLD` (default: 95) |
