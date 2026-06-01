# DriveSort

Local-AI Google Drive organiser. Discovers your own folder taxonomy from your
files using unsupervised clustering, lets you review and name categories
interactively, then classifies and moves new files automatically — with
open-set detection for files that don't fit any known category.

**Everything runs locally.** No cloud AI APIs. No subscriptions.

---

## How it works

```
Bootstrap (once)
  All Drive files → Embed (sentence-transformers, 80 MB)
                  → UMAP + HDBSCAN → clusters
                  → LLM names each cluster (Ollama / phi3:mini)
                  → You: Accept / Rename / Merge / Skip
                  → Drive folders created + taxonomy saved

Ongoing scan
  New file → Embed
           → Distance to taxonomy centroids
           ├─ High confidence  → auto-move
           ├─ Medium           → ask you
           └─ Too far (novel)  → log it
                               → LLM: "does this need a new folder?"
                               → Accumulate → re-cluster → new category
```

---

## Setup

### 1. Install dependencies with uv

```bash
uv sync --extra dev
```

This installs all Python dependencies into `.venv/` and sets up pytest for testing.

### 2. Ollama (local LLM for cluster naming)

**Option A: Using mise (recommended)**
```bash
# Install mise from https://mise.jq.rs
mise install      # Installs ollama from mise.toml
mise exec ollama pull phi3:mini   # ~2 GB, fast, works well
```

**Option B: Manual installation**
```bash
# Install from https://ollama.com
ollama pull phi3:mini    # ~2 GB, fast
# or try: ollama pull mistral (4.1 GB, higher quality)
```

### 3. Google Drive credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project → Enable **Google Drive API**
3. Create credentials → **OAuth 2.0 Client ID** → Desktop app
4. Download `credentials.json` → place it at `data/credentials.json`

On first run, a browser window will open for you to authorise access.
The token is cached in `data/token.json` for future runs.

---

## Usage

### Bootstrap (run once)

```bash
drivesort bootstrap
```

This will:
- Fetch all your Drive files
- Embed them locally (~1–5 min depending on Drive size)
- Cluster them and ask Ollama to name each group
- Walk you through each cluster: Accept / Rename / Merge / Skip
- Create folders in your Drive and save the taxonomy to `data/taxonomy.json`

Options:
```bash
drivesort bootstrap --min-cluster-size 5   # coarser clusters
drivesort bootstrap --model mistral        # different Ollama model
```

### Scan for new files

```bash
drivesort scan           # dry-run: shows what would happen
drivesort scan --live    # actually moves files
```

### Check taxonomy

```bash
drivesort status
```

---

## Data files

| Path | Contents |
|---|---|
| `data/credentials.json` | Google OAuth credentials (you provide) |
| `data/token.json` | Cached auth token (auto-generated) |
| `data/taxonomy.json` | Category centroids + folder IDs |
| `data/embedding_cache.json` | Cached file embeddings (speeds up re-runs) |
| `data/novel_files.json` | Accumulated novel files for re-clustering |

**Keep `data/` out of version control** — it contains auth tokens and personal file metadata.

---

## Confidence thresholds

Edit `scanner.py` to adjust:

```python
AUTO_MOVE_THRESHOLD = 0.82   # move without asking
REVIEW_THRESHOLD    = 0.62   # ask the human (below this = novel)
```

And in `taxonomy.py`:

```python
novelty_threshold = 0.42     # cosine distance above which = OOD
```

---

## Architecture

```
drivesort/
  drive.py       — Google Drive API client (auth, list, move, create folder)
  embedder.py    — sentence-transformers wrapper with on-disk cache
  clusterer.py   — UMAP + HDBSCAN + LLM cluster naming
  taxonomy.py    — Persisted centroids, classification, OOD detection
  bootstrap.py   — Interactive terminal UI for first-run review
  scanner.py     — Ongoing scan: auto-move, review, novel detection
  cli.py         — `drivesort` CLI (bootstrap / scan / status)
```

## Dependencies

| Library | Role |
|---|---|
| `sentence-transformers` | Local file embeddings (all-MiniLM-L6-v2) |
| `umap-learn` | Dimensionality reduction before clustering |
| `hdbscan` | Density-based clustering, finds outliers automatically |
| `scikit-learn` | Cosine distance at inference time |
| `ollama` | Local LLM for cluster naming and novel-file suggestions |
| `google-api-python-client` | Drive API |
| `rich` | Terminal UI |
| `typer` | CLI framework |

---

## Development

### Run tests

```bash
uv run pytest tests/
```

### Interactive commands via uv

```bash
uv run drivesort bootstrap
uv run drivesort scan --live
uv run drivesort status
```

Or activate the venv and use `drivesort` directly:

```bash
source .venv/bin/activate
drivesort bootstrap
```
