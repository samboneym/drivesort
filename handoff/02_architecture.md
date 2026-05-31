# 02 — Architecture

## Module map

```
drivesort/
  drive.py       DriveFile value object + DriveClient API wrapper
  embedder.py    Sentence-transformers wrapper with on-disk embedding cache
  clusterer.py   UMAP + HDBSCAN + LLM cluster naming + novel-file suggestions
  taxonomy.py    Persisted centroids, classify(), OOD detection, incremental learning
  bootstrap.py   One-time interactive terminal UI (Rich) — review clusters, create folders
  scanner.py     Ongoing scan — auto-move, review queue, novel handling, re-clustering
  cli.py         Typer CLI: `drivesort bootstrap / scan / status`
```

## Data flow — Bootstrap

```
DriveClient.iter_files()
    │  yields DriveFile objects (no content download)
    ▼
Embedder.embed_files()
    │  filename + extension + snippet → 384-dim float32 vector
    │  cache: data/embedding_cache.json (keyed by SHA1 of id|modified|name)
    ▼
Clusterer.cluster()
    │  UMAP: 384-dim → 2-dim (cosine metric, n_neighbors=15)
    │  HDBSCAN: density clusters in 2D (min_cluster_size=3, EOM selection)
    │  labels: 0,1,2… = cluster id, -1 = outlier
    │  For each cluster → ollama.chat(phi3:mini) → suggested name + description
    ▼
ClusterResult { clusters: list[Cluster], embeddings_2d, outlier_files }
    │
    ▼
bootstrap.run_bootstrap()  ← human reviews each cluster interactively
    │  Accept / Rename / Merge / Skip / Quit
    │  Resolved merges: extend the winning cluster's file list
    │  Outliers → Archive
    │
    ▼
DriveClient.create_folder() for each accepted category
    │
    ▼
Taxonomy.add_category()
    │  centroid = mean(member_embeddings), L2-normalised
    │  Persisted to data/taxonomy.json
    ▼
data/taxonomy.json  ← source of truth from this point forward
```

## Data flow — Ongoing scan

```
DriveClient.iter_files()
    │
    ▼
Embedder.embed_files()  ← cache hit for unchanged files
    │
    ▼
For each file not already in a known folder:
    Taxonomy.classify(embedding)
        │  cosine distance = 1 - dot(emb, centroid)  [both L2-normalised]
        │  best_dist < novelty_threshold (0.42) → known category
        │  best_dist ≥ novelty_threshold → novel/OOD
        ▼
    ScanDecision
        confidence ≥ 0.82  → auto_move
        0.62–0.82          → review (human picks)
        < 0.62 or novel    → novel
            │
            └─ Taxonomy.log_novel_file() → data/novel_files.json
    │
    ▼
After all files processed:
    _maybe_recluster() — if ≥5 novel files accumulated:
        Clusterer.cluster(novel_embeddings)
        Human confirms new categories → Taxonomy.add_category()
        Taxonomy.clear_novel_log()
```

## Data files

| Path | Format | Purpose |
|---|---|---|
| `data/credentials.json` | Google OAuth JSON | User provides; not generated |
| `data/token.json` | Google OAuth token | Auto-generated on first auth |
| `data/embedding_cache.json` | `{sha1_key: [float, ...]}` | Avoids re-embedding unchanged files |
| `data/taxonomy.json` | `{name: CategoryEntry}` | The taxonomy — centroids + folder IDs |
| `data/novel_files.json` | `[{id, name, embedding}]` | Accumulates novel files for re-clustering |

## Key data structures

### DriveFile (drive.py)
Value object, `__slots__` for memory efficiency. Fields:
`id, name, mime_type, extension, parent_id, size_bytes, snippet, created, modified, is_folder`

`text_for_embedding()` → `"{name} {snippet[:400]} {extension}"`

### CategoryEntry (taxonomy.py)
Persisted per category:
```python
name: str
description: str
folder_id: str          # Drive folder ID
centroid: list[float]   # 384-dim, L2-normalised mean embedding
member_count: int
member_ids: list[str]   # Drive file IDs of confirmed members
```

### ClassificationResult (taxonomy.py)
Returned by `Taxonomy.classify()`:
```python
file_id, file_name: str
category: str | None    # None if novel
confidence: float       # 1 - distance, range [0,1]
distance: float         # cosine distance to nearest centroid
is_novel: bool
runner_up: str | None   # second-closest category
runner_up_confidence: float
```

### Cluster (clusterer.py)
Ephemeral — exists only during bootstrap:
```python
id: int                 # HDBSCAN label (-1 = outlier)
files: list[DriveFile]
suggested_name: str     # from LLM
suggested_description: str
llm_confidence: float
accepted_name: str | None   # set by human
merged_into: str | None
rejected: bool
```

## Confidence thresholds (scanner.py)

```python
AUTO_MOVE_THRESHOLD = 0.82   # confidence ≥ this → move without asking
REVIEW_THRESHOLD    = 0.62   # confidence in [0.62, 0.82) → ask human
                             # confidence < 0.62 or is_novel → novel queue
```

## Novelty threshold (taxonomy.py)

```python
novelty_threshold = 0.42  # cosine distance; above this = OOD regardless of confidence
```

Note the relationship: a file can have `confidence = 0.58` (= `distance = 0.42`)
and be marked novel because `distance >= novelty_threshold`. The two thresholds
are intentionally consistent but the novelty check in `Taxonomy.classify()` is
distance-based while the routing in `Scanner._make_decision()` is confidence-based.
They should be kept in sync when tuning.

## Incremental centroid update (taxonomy.py: confirm())

When a file is confirmed (auto-moved or human-approved):
```
new_centroid = (old_centroid * n + new_embedding) / (n + 1)
new_centroid = new_centroid / ||new_centroid||
```
This is a running mean — we never need to store all member embeddings after bootstrap.
The taxonomy gets more accurate as more files are classified correctly.

## OAuth scopes (drive.py)

```python
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",   # list + read metadata
    "https://www.googleapis.com/auth/drive.file",       # move + create folders
]
```

`drive.file` is narrower than `drive` — it only allows access to files the app
itself creates or opens, plus files explicitly opened by the user. This is
intentional: we don't need full Drive write access.

## Component dependencies

```
cli.py
  ├── drive.py
  ├── embedder.py  → drive.py
  ├── clusterer.py → drive.py
  ├── taxonomy.py
  ├── bootstrap.py → clusterer.py, drive.py, taxonomy.py, embedder.py
  └── scanner.py   → drive.py, embedder.py, taxonomy.py, clusterer.py
```

No circular imports. `drive.py` and `taxonomy.py` have no intra-package deps.
