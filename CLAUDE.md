# DriveSort

Local-AI Google Drive organiser. Discovers folder taxonomy from files using unsupervised clustering, then classifies and moves new files automatically with open-set detection.

## Architecture

```
drivesort/
  drive.py             — Google Drive API client (DriveFile value object + DriveClient wrapper)
  embedder.py          — sentence-transformers (all-MiniLM-L6-v2, 384-dim) with on-disk cache
  content_extractor.py — per-type content enrichment (Drive export, PDF, code, vision LLM captions)
  clusterer.py         — UMAP (384→2D) + HDBSCAN + Ollama LLM cluster naming
  taxonomy.py          — Persisted centroids, classify(), OOD detection, incremental learning
  bootstrap.py         — One-time interactive Rich TUI for reviewing clusters
  scanner.py           — Ongoing scan: auto-move / review / novel detection
  cli.py               — Typer CLI: bootstrap / scan / status
```

## Key invariants

- All embeddings are L2-normalised. Cosine distance = `1 - dot(a, b)` for unit vectors.
- Centroids use running mean (constant size per category). Cannot remove a file from centroid after confirm().
- Taxonomy is the single source of truth after bootstrap. Stored as JSON at `data/taxonomy.json`.
- Embedding cache keyed by `SHA1(id|modified|name)` — changing the model requires deleting both cache and taxonomy.
- `drive.file` OAuth scope (not `drive`) — only accesses files the app creates or opens.

## Confidence thresholds

- `AUTO_MOVE_THRESHOLD = 0.82` (scanner.py) — move without asking
- `REVIEW_THRESHOLD = 0.62` (scanner.py) — ask human; below = novel
- `novelty_threshold = 0.42` (taxonomy.py) — cosine distance above which = OOD

These are intentionally consistent: confidence = 1 - distance.

## Commands

```bash
uv pip install -e .
drivesort serve [--host HOST] [--port PORT] [--reload]
```

Bootstrap, scan, status, and recover have moved to the web UI (`drivesort serve` → http://localhost:7432).

## Development

```bash
uv pip install -e ".[dev]"
pip install pytest
pytest tests/
```

## Known bugs

1. **scanner.py `_handle_novel_files()`**: New single-file categories not added to taxonomy after folder creation. The `taxonomy.add_category()` call is missing.
2. **bootstrap.py merge resolution**: If merge target was skipped/rejected, merged files silently disappear instead of going to Archive.
3. **embedder.py:68**: `hits` and `miss_indices` declared but unused (dead code).

## Data files (all in `data/`, all gitignored)

- `credentials.json` — Google OAuth (user provides)
- `token.json` — cached auth token
- `taxonomy.json` — category centroids + folder IDs
- `embedding_cache.json` — cached file embeddings
- `content_cache.json` — cached extracted text per file (same SHA1 key as embedding cache)
- `novel_files.json` — accumulated novel files for re-clustering
