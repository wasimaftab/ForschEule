# ForschEule — Setup and Usage Guide (Noncommercial / academic use only)
ForschEule (German: “Research Owl”) is a daily paper recommendation service designed for research labs working in **Biomedical** domains. Each day it automatically searches PubMed and arXiv, ranks papers against your lab’s research interests, and surfaces the **top 5** most relevant papers. To know about the project architecture, tech stack, pipeline workflow and to control what papers ForschEule recommends see the docs inside [documentations](https://github.com/wasimaftab/ForschEule/tree/main/documentation) folder. 

## Prerequisites

- Python 3.12+
- ~2.5 GB disk space (for PyTorch CPU + two MedCPT models)
- Internet access (for fetching papers and downloading models on first run)

---

## Installation

### 1. Clone and enter the project

```bash
cd ~/Desktop/ForschEule
```

### 2. Create a virtual environment

```bash
python3.12 -m venv .venv
```

### 3. Install CPU-only PyTorch first (saves disk space)

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

This installs PyTorch without CUDA libraries (~200 MB instead of 2+ GB).

### 4. Install ForschEule and dependencies

```bash
.venv/bin/pip install -e ".[dev]"
```

### 5. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and set your NCBI credentials:

```
ENTREZ_EMAIL=your-email@example.com
ENTREZ_API_KEY=your-ncbi-api-key
```

These are optional but recommended. Without them, PubMed requests are
rate-limited to 3/s instead of 10/s. Get an API key at:
https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/

#### Optional environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FORSCHEULE_DB_PATH` | `./data/forscheule_papers.sqlite3` | SQLite database path |
| `FORSCHEULE_FETCH_WINDOW` | `7` | Default fetch window (days) |
| `MEDCPT_QUERY_MODEL` | `ncbi/MedCPT-Query-Encoder` | HuggingFace query encoder model |
| `MEDCPT_ARTICLE_MODEL` | `ncbi/MedCPT-Article-Encoder` | HuggingFace article encoder model |
| `MEDCPT_QUERY_MAX_LENGTH` | `512` | Max token length for query encoder |
| `MEDCPT_ARTICLE_MAX_LENGTH` | `512` | Max token length for article encoder |
| `OPENAI_API_KEY` | *(none)* | Required for paper summaries and weekly digest |

---

## Usage

### Using Web console

1. Run the following command on a terminal from the project directory to start ForchEule UI server

```bash
.venv/bin/uvicorn forscheule.api.app:app --host 127.0.0.1 --port 8000
```

2. Open your browser and navigate to:

```
http://127.0.0.1:8000/
```

The web console provides a GUI for all common tasks:
- **Results tab**: Load daily recommendations by date, with optional top-K override
- **Weekly Summary tab**: Generate OpenAI-powered weekly digests (requires `OPENAI_API_KEY`)
- **Run Pipeline tab**: Trigger daily runs, backfills, and summary generation jobs
- **Settings tab**: Edit lab profile, boosted phrases, and top-K at runtime

For the 1st time, you must run a pipeline from *Run Pipeline* tab. See the docs inside [documentations](https://github.com/wasimaftab/ForschEule/tree/main/documentation) folder for details. 

### Using Command Line

#### Daily run (typical use)

```bash
.venv/bin/python -m forscheule run-daily
```

This fetches papers from the last 7 days, ranks them using the MedCPT
dual-encoder, and stores the top K (default 5) for today. If a pipeline run
for today already exists with the same configuration signature, it skips
(idempotent). If settings have changed, it recomputes automatically.

#### Backfill past days

```bash
.venv/bin/python -m forscheule backfill --days 7
```

Runs the pipeline for each of the last 7 days (today through 6 days ago).
Dates with matching configuration signatures are skipped.

#### Custom fetch window

```bash
.venv/bin/python -m forscheule run-daily --window 14
```

Look at papers from the last 14 days instead of the default 7.

#### Start the API server

```bash
.venv/bin/python -m forscheule serve
```

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web console (HTML) |
| `GET` | `/daily?date=YYYY-MM-DD&top_k=N` | Retrieve recommendations for a date |
| `GET` | `/settings` | Current runtime settings |
| `PUT` | `/settings` | Update lab profile, boosted phrases, or top_k |
| `POST` | `/jobs/run-daily` | Trigger a daily pipeline run |
| `POST` | `/jobs/backfill` | Trigger a backfill over N days |
| `POST` | `/jobs/summary-weekly` | Generate a weekly digest |
| `GET` | `/weekly-summary?date=&window=&top_n=` | Retrieve a cached weekly digest |
| `GET` | `/jobs/{job_id}` | Check status of a background job |

Example API query:

```bash
curl "http://127.0.0.1:8000/daily?date=2026-02-09"
```

### Interactive API docs

When the server is running, visit:

```
http://127.0.0.1:8000/docs
```

FastAPI generates interactive Swagger documentation automatically.

---

## Query the Database Directly

The SQLite database can be queried directly for ad-hoc analysis:

```bash
# Open the database
sqlite3 ./data/forscheule_papers.sqlite3

# How many papers are stored?
SELECT COUNT(*) FROM papers;

# Which dates have recommendations?
SELECT date, COUNT(*) FROM recommendations GROUP BY date ORDER BY date;

# Show today's top 5
SELECT r.rank, r.score, p.title, p.source
FROM recommendations r
JOIN papers p ON r.paper_id = p.id
WHERE r.date = '2026-02-09'
ORDER BY r.rank;
```

---

## Automated Scheduling with systemd

Template files are in `deploy/`. To set up a daily timer:

### 1. Edit the service file

Update paths in `deploy/forscheule.service`:

```ini
[Service]
WorkingDirectory=/home/youruser/Desktop/ForschEule
EnvironmentFile=/home/youruser/Desktop/ForschEule/.env
ExecStart=/home/youruser/Desktop/ForschEule/.venv/bin/python -m forscheule run-daily
```

### 2. Install and enable

```bash
sudo cp deploy/forscheule.service /etc/systemd/system/
sudo cp deploy/forscheule.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now forscheule.timer
```

### 3. Check status

```bash
# Timer status
systemctl status forscheule.timer

# Last run logs
journalctl -u forscheule.service --since today

# Next scheduled run
systemctl list-timers forscheule.timer
```

The timer runs daily at 07:00. Edit `deploy/forscheule.timer` to change the
schedule:

```ini
[Timer]
OnCalendar=*-*-* 07:00:00   # Change this to your preferred time
```

---

## Running Tests

```bash
.venv/bin/pytest tests/ -v
```

Expected output: 96 tests, all passing. The test suite covers:

- **test_db.py**: Schema creation, paper upsert, recommendation storage,
  settings CRUD, pipeline_runs, distinct signatures
- **test_dedup.py**: DOI dedup, fuzzy title dedup, distinct paper preservation
- **test_score.py**: Recency scoring, keyword matching, dual-encoder ranking
  wiring (mocked), output format validation
- **test_idempotency.py**: Signature determinism, sensitivity to all config
  params (lab profile, boosted phrases, window, top_k, query model, article
  model, max lengths), explicit-defaults stability
- **test_api.py**: API endpoints, web console, settings PUT validation,
  job triggers, TOCTOU atomicity, stale job pruning, JSON body hardening
- **test_summary.py**: OpenAI summary generation and caching
- **test_summary_api.py**: Summary job endpoints, weekly digest API, boolean
  rejection, malformed body handling
- **test_time_anchoring.py**: Date-anchored fetch and recency scoring

### Running the linter

```bash
.venv/bin/ruff check forscheule/ tests/
```

---

## Troubleshooting

### "No papers fetched"

- Check your internet connection.
- If PubMed returns 0 results, the search query may be too narrow for the
  date range. Try `--window 14` to widen the fetch window.
- Check logs for HTTP errors (429 = rate limited, 503 = service unavailable).

### "ENTREZ_EMAIL not set" warning

- This is a warning, not an error. The pipeline still works but PubMed requests
  are slower (1s delay instead of 0.35s).
- Set `ENTREZ_EMAIL` in your `.env` file.

### Disk space issues during install

- Use CPU-only PyTorch: install with `--index-url https://download.pytorch.org/whl/cpu`
- The two MedCPT models are ~400 MB each and are cached in
  `~/.cache/huggingface/`.

### First run is slow

- On the first run, both MedCPT model weights (Query Encoder + Article Encoder)
  are downloaded (~800 MB total). Subsequent runs load from cache and are much
  faster. The article encoder is lazy-loaded only when papers are actually
  ranked.
- Embedding 20 papers takes ~5 seconds on a modern CPU.

### Re-running produces the same recommendations

- This is by design (signature-based idempotency). The pipeline stores a
  configuration signature in `pipeline_runs` and skips if it matches.
- To force a re-run, either change a setting (via web console or config.py)
  — the signature will differ and the pipeline recomputes automatically — or
  manually delete the pipeline run record:

  ```bash
  sqlite3 ./data/forscheule_papers.sqlite3 \
    "DELETE FROM pipeline_runs WHERE date='2026-02-09'"
  ```

  Then re-run the pipeline.
