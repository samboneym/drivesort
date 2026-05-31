# 04 — Codebase Reference

Precise contracts for every public class and function.
The source files have docstrings but this adds context the docstrings omit.

---

## drive.py

### `DriveFile`
Value object. Constructed from a raw Drive API response dict.
`__slots__` used for memory efficiency on large Drives.

`text_for_embedding() -> str`
Returns `"{name} {snippet[:400]} {extension}"`.
This is the *only* text used for embedding — keep it consistent.
If you change this, invalidate the entire embedding cache by deleting
`data/embedding_cache.json`.

### `DriveClient`

`__init__(credentials_path, token_path)`
Authenticates on construction. Raises `FileNotFoundError` if credentials.json
is missing. On first run, opens a browser for OAuth. Token cached at
`data/token.json`.

`iter_files(include_folders=False, page_size=200) -> Iterator[DriveFile]`
Pages through all owned files. Folders excluded by default — they have no
content signal and would confuse clustering. Don't include them for embedding.

The Drive API query is: `trashed = false and owner = 'me'`
This intentionally excludes: shared files, files in trash, shortcuts.
Shortcuts are a known gap — they appear as `application/vnd.google-apps.shortcut`
and are currently skipped. They're common in this Drive (e.g., EdgeTX zip
shortcut, Snowrunner links).

`list_folders() -> list[DriveFile]`
Returns all owned folders (not paginated — assumes < 200 folders).
Used in `bootstrap.py`'s `_find_or_create_folder` to avoid creating
duplicate folders.

`move_file(file: DriveFile, target_folder_id: str) -> None`
Uses `addParents` / `removeParents` in a single API call. No-ops if the file
is already in the target folder (guards against double-move bugs).
Raises `HttpError` on failure — callers in `scanner.py` catch this.

`create_folder(name: str, parent_id: str | None = None) -> DriveFile`
Creates at root level if `parent_id` is None. Returns a fully-populated
DriveFile for the new folder.

---

## embedder.py

### `Embedder`

`__init__(model_name='all-MiniLM-L6-v2', cache_path=Path('data/embedding_cache.json'))`
Downloads the model on first run (~80 MB to `~/.cache/huggingface`).
Loads the embedding cache from disk.

`embed_files(files, show_progress=True) -> tuple[list[DriveFile], np.ndarray]`
Returns files in the same order as input, and a float32 matrix of shape
`(N, 384)`. All embeddings are L2-normalised (unit vectors).

Cache key: `SHA1(f"{file.id}|{file.modified}|{file.name}")`.
If a file is renamed or its content changes (modifiedTime updates), it gets
a new cache key and is re-embedded. Old cache entries for deleted files
accumulate harmlessly.

`embed_text(text: str) -> np.ndarray`
Embeds a raw string. Used in `bootstrap.py` to create the Archive centroid
from a description string, and could be used to embed folder descriptions.
Returns shape `(384,)`, L2-normalised.

**Important:** The cache stores raw embeddings as Python lists of floats.
Loading converts them back to float32 numpy arrays. If you change the model,
delete the cache — the new model produces incompatible vectors.

---

## clusterer.py

### `Cluster` (dataclass)
Ephemeral — only lives during bootstrap. Fields set during construction:
`id, files`. Fields set by LLM: `suggested_name, suggested_description,
llm_confidence`. Fields set by human review: `accepted_name, merged_into,
rejected`.

### `ClusterResult` (dataclass)
`clusters: list[Cluster]` — excludes the -1 outlier cluster.
`embeddings_2d: np.ndarray` — 2D UMAP projection, shape (N, 2). Preserved for
future visualisation. Currently unused after bootstrap.
`outlier_files: list[DriveFile]` — HDBSCAN's -1 cluster; files with no group.

### `Clusterer`

`__init__(min_cluster_size=3, umap_n_neighbors=15, novelty_threshold=0.45, ollama_model='phi3:mini')`

Note: `novelty_threshold` here is stored as an instance variable but is NOT
used in `cluster()` — it was originally intended for an inline OOD check
that was moved to `Taxonomy.classify()`. This field is currently dead code
in the context of the Clusterer. The active novelty threshold is in
`Taxonomy.__init__` (default 0.42). Don't add logic here that uses
`self._novelty_threshold` without understanding this.

`cluster(files, embeddings, name_with_llm=True) -> ClusterResult`
Pipeline: UMAP (cosine, 2D) → HDBSCAN (euclidean on 2D, EOM selection) →
LLM naming. The `random_state=42` in UMAP makes results reproducible.

If Ollama is not running, `_name_cluster` catches the exception and falls back
to `"Group N"` as the cluster name. Bootstrap can still proceed without Ollama
— you'll just name everything manually.

`suggest_new_category(file: DriveFile, existing_folders: list[str]) -> dict`
Returns: `{"new_folder": str | None, "rationale": str, "similar_files_hint": str}`
Called from `scanner.py` when a file is novel. `new_folder` is None if the LLM
thinks Archive is appropriate. Callers must handle the None case.

---

## taxonomy.py

### `CategoryEntry` (dataclass)
The persisted representation of one category.
`centroid` is stored as `list[float]` in JSON, loaded as-is, converted to
numpy only in `centroid_array()`. The list representation keeps the JSON
human-readable and editable.

### `ClassificationResult` (dataclass)
Returned by `Taxonomy.classify()`. Both `category` and `runner_up` can be None.
`confidence` and `runner_up_confidence` are always in [0, 1].

### `Taxonomy`

`__init__(path=Path('data/taxonomy.json'), novelty_threshold=0.42)`
Loads from disk on construction. Safe to call when the file doesn't exist.
`is_empty()` returns True in that case.

`add_category(name, description, folder_id, member_embeddings, member_ids)`
Computes centroid as mean of `member_embeddings`, then L2-normalises.
`member_embeddings` must be shape `(N, 384)` float32.
Overwrites any existing category with the same name (idempotent).

`classify(embedding, file_id='', file_name='') -> ClassificationResult`
`embedding` must be L2-normalised (as returned by `Embedder`).
Distance = `1 - dot(embedding, centroid)` — valid for unit vectors.
`is_novel = distance > self._threshold`.
`category` is None when `is_novel` is True.

**Critical invariant:** The embedding passed to `classify()` must be produced
by the same model as the embeddings used to build the taxonomy centroids.
If you change `MODEL_NAME` in `embedder.py`, you must rebuild the taxonomy
from scratch (delete `data/taxonomy.json` and re-run bootstrap).

`confirm(category_name, file_id, embedding)`
Incremental centroid update. Safe to call with an embedding that's already
in `member_ids` — the deduplication check prevents double-counting in the
ID list, but the centroid will drift slightly. Don't call this twice for the
same file.

`log_novel_file(file_id, file_name, embedding)`
Appends to `data/novel_files.json`. Deduplicates by `file_id`. The stored
embedding is the full 384-dim vector as a list of floats. This file grows
until `clear_novel_log()` is called.

`load_novel_files() -> tuple[list[dict], np.ndarray]`
Returns the log records and a (N, 384) float32 matrix. Returns empty
structures if the log doesn't exist or is empty.

---

## bootstrap.py

### `run_bootstrap(result, files, embeddings, drive, taxonomy, embedder)`
The main interactive loop. Call this after `Clusterer.cluster()`.

The `file_emb_map` dict maps `file.id → embedding` for centroid calculation.
It's built from the full `files` + `embeddings` arrays, so it covers both
clustered files and outliers.

**Merge resolution:** Merges are resolved after all clusters are reviewed.
If cluster A says "merge into B" but B was skipped (rejected), the merge
silently does nothing — A's files have nowhere to go and will be missed.
This is a known gap. See `05_known_issues_and_next_steps.md`.

**Folder creation:** `_find_or_create_folder` searches the full folder list
on every call (O(n) where n = number of folders). Acceptable for bootstrap
(called ~10–30 times) but would need caching for repeated use.

Archive is always created, even if no clusters were skipped. The Archive
centroid is seeded from `embedder.embed_text("archive old backup...")` — a
description string, not from actual Archive files. This means the Archive
centroid is synthetic and may be weak initially. It will improve as files
are confirmed into it via `taxonomy.confirm()`.

---

## scanner.py

### `Scanner`

`scan(interactive=True)`
Main entry point. Embeds all files (cache hits for unchanged files).
Skips files whose `parent_id` is a known taxonomy folder ID.

**Gap in `_file_is_organised`:** It checks `parent_id` against folder IDs.
A file in a *subfolder* of a known folder will have its subfolder's ID as
`parent_id`, not the taxonomy folder's ID. It will be treated as unorganised
and potentially moved. This is intentional for now — the system only manages
top-level taxonomy folders, not subfolders within them.

`_make_decision(file, result) -> ScanDecision`
`result.is_novel` is checked first. `result.confidence` thresholds are checked
second. The ordering matters: a file can have low confidence *and* be marked
novel; the novel path takes priority.

`_interactive_review(decisions, embeddings_map)`
Shows each borderline file with its suggested folder and runner-up. Options:
[A]ccept → move immediately. [C]hoose → numbered list of all folders.
[S]kip → leave in place (file will appear again on next scan).

`_maybe_recluster(min_novel=5)`
Uses `types.SimpleNamespace` to create duck-typed file objects that satisfy
`Clusterer.cluster()`'s interface without needing real DriveFile objects.
This works because Clusterer only accesses `.id`, `.name`, `.mime_type`,
`.size_bytes`, `.snippet` on the file objects.

**Known bug in `_handle_novel_files`:** When a new folder is created for a
novel file (user says yes to "Create X and move?"), `drive.create_folder` and
`drive.move_file` are called, but `taxonomy.add_category` is NOT called. The
new folder isn't registered in the taxonomy. On the next scan, files in that
folder will be treated as unorganised again. See `05_known_issues_and_next_steps.md`.

---

## cli.py

Three commands:

`drivesort bootstrap [--min-cluster-size N] [--model MODEL]`
Constructs all components, fetches files, embeds, clusters, runs review.
Prompts before re-running if taxonomy already exists.

`drivesort scan [--live] [--no-interact]`
`--live` disables dry-run. `--no-interact` skips the review queue (only
auto-moves happen). Useful for scheduled/headless runs where you want
auto-moves without prompts.

`drivesort status`
Reads `data/taxonomy.json` and prints a Rich table. Also reports novel file
count. No Drive API call needed.

The `_build_components` helper constructs all five objects. It's called by
both `scan` and `status`. Note that `status` doesn't need `Scanner` or
`Clusterer` — the helper over-constructs slightly. Not worth fixing.
