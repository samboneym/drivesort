# 05 — Known Issues and Next Steps

## Bugs (fix before first live run)

### BUG 1 — New single-file categories not added to taxonomy (scanner.py:263–267)

**Severity:** High — silent data loss

**Location:** `scanner.py`, `_handle_novel_files()`, the block that creates a
new folder for a novel file.

**What happens:** When the user says yes to creating a new folder for a novel
file, `drive.create_folder()` and `drive.move_file()` are called but
`taxonomy.add_category()` is never called. The new folder exists in Drive but
the taxonomy doesn't know about it. On the next scan, files in that folder
have `parent_id` = the new folder's ID, which isn't in any taxonomy entry's
`folder_id`, so they're treated as unorganised and will be classified again.

**Fix:**
```python
# In scanner.py _handle_novel_files(), after drive.move_file():
if not self._dry_run:
    folder = self._drive.create_folder(new_folder)
    self._drive.move_file(d.file, folder.id)
    # Add to taxonomy so the folder is recognised going forward
    emb = embeddings_map.get(d.file.id)
    if emb is not None:
        self._taxonomy.add_category(
            name=new_folder,
            description=suggestion.get("similar_files_hint", ""),
            folder_id=folder.id,
            member_embeddings=emb.reshape(1, -1),
            member_ids=[d.file.id],
        )
        self._taxonomy.save()
```

Note that `_handle_novel_files` doesn't currently receive `embeddings_map` as
a parameter — you'll need to add it to the method signature and the call site
in `scan()`.

---

### BUG 2 — Merge-then-skip leaves files unaccounted for (bootstrap.py:115–119)

**Severity:** Medium — silent data loss during bootstrap

**Location:** `bootstrap.py`, merge resolution loop.

**What happens:** If cluster A says "merge into B" but cluster B was skipped
(`rejected=True`), the merge resolution checks `cluster.merged_into in accepted`
— which is False for a skipped cluster. The merge silently does nothing. Cluster
A's files are neither moved to Archive nor to a folder.

**Fix:** In the merge resolution loop, check if the target was rejected and
route to Archive in that case:
```python
for cluster in result.clusters:
    if cluster.merged_into:
        if cluster.merged_into in accepted:
            target_cluster, target_name = accepted[cluster.merged_into]
            target_cluster.files.extend(cluster.files)
        else:
            # Target was skipped — send to Archive
            console.print(f"[yellow]Merge target '{cluster.merged_into}' not found — sending to Archive[/yellow]")
            archive_cluster = accepted.get("Archive")
            if archive_cluster:
                archive_cluster[0].files.extend(cluster.files)
```

---

### BUG 3 — `hits`, `miss_indices` declared but unused (embedder.py:68)

**Severity:** Low — dead code, no functional impact

**Location:** `embedder.py:68` — `hits, misses, miss_indices = [], [], []`

`hits` and `miss_indices` are declared but never used. The actual logic uses
`misses` and `hit_map`. Safe to delete `hits` and `miss_indices`.

---

## Missing features (in priority order)

### 1. Tests (highest priority)

No tests exist. The `tests/` directory has only an empty `__init__.py`.

**Start here:** `taxonomy.py` is the most critical module and the easiest to
test in isolation (no Drive API, no LLM, no embedder needed).

Suggested test cases for `taxonomy.py`:
- `classify()` returns the correct category when distance is low
- `classify()` returns `is_novel=True` when distance exceeds threshold
- `classify()` returns `is_novel=False` when below threshold
- `classify()` returns `runner_up` correctly
- `classify()` handles empty taxonomy gracefully
- `confirm()` updates centroid correctly (verify with small known vectors)
- `confirm()` re-normalises the centroid after update
- `log_novel_file()` deduplicates by file ID
- `add_category()` normalises the centroid
- `save()` + `__init__()` round-trip produces identical taxonomy

Use `pytest`. No external dependencies needed for taxonomy tests — just numpy.

```python
# Example fixture
import numpy as np
from drivesort.taxonomy import Taxonomy
import tempfile, pathlib

def make_taxonomy(threshold=0.42):
    with tempfile.TemporaryDirectory() as d:
        t = Taxonomy(path=pathlib.Path(d) / "taxonomy.json", novelty_threshold=threshold)
        yield t
```

### 2. Visualisation of bootstrap clusters

`ClusterResult.embeddings_2d` is preserved precisely for this. A scatter plot
of the 2D UMAP projection, coloured by cluster, with file names on hover would
make the bootstrap review session much more intuitive than the terminal table.

Suggested approach: generate a self-contained HTML file using Plotly (which can
be written as a single HTML file with embedded JS). Launch it in the browser
automatically before starting the terminal review.

```python
import plotly.express as px
# result.embeddings_2d: (N, 2)
# labels: list of cluster names per file
fig = px.scatter(x=emb2d[:,0], y=emb2d[:,1], color=labels, hover_name=filenames)
fig.write_html("data/bootstrap_map.html")
```

### 3. Scheduled scanning (cron integration)

`drivesort scan --live --no-interact` already supports headless operation.
What's missing: a way to run it on a schedule and get a summary report.

Suggested: add a `drivesort daemon --interval-hours 6` command that loops,
sleeping between scans, and writes a summary to `data/last_scan_report.json`.
Alternatively, document the cron invocation in README.

### 4. Handling file shortcuts

`application/vnd.google-apps.shortcut` files are in the Drive (e.g., the
EdgeTX shortcut, Snowrunner links). They're currently returned by `iter_files()`
but have no content snippet and no extension. The embedder will embed them on
filename alone.

Shortcuts need special handling: fetch the target file's metadata and embed
based on the target, not the shortcut itself. The Drive API's shortcut target
can be retrieved via the `shortcutDetails` field.

For now, they'll cluster approximately correctly on filename but may end up
as outliers. Not a critical issue.

### 5. Subfolder-aware organisation check

`_file_is_organised()` only checks if a file's direct parent is a taxonomy
folder. Files in subfolders of taxonomy folders (e.g., a file inside
`FPV & RC Hobbies/Rotorflight/`) will be treated as unorganised.

This is intentional conservatism — we don't want to move files that a human
has already sub-organised. But it means files inside *unrecognised* subfolders
(e.g., the old `Car/` folder with JS files from a web scrape) will be
classified on every scan.

Fix: walk the parent chain upward until you hit root or a known folder.

### 6. Embedding cache size management

The cache grows indefinitely. Deleted files leave orphan entries (their SHA1
key is never cleaned up). For a Drive that changes a lot, this could become
large over time.

Fix: periodically prune the cache by intersecting cache keys against current
file IDs. Could be a `drivesort cache --prune` command.

### 7. Bootstrap resume / partial save

If bootstrap is interrupted mid-review (e.g., `[Q]uit`), no taxonomy is saved.
All clustering and LLM naming work is lost.

Fix: save cluster decisions incrementally to a `data/bootstrap_state.json`
checkpoint file. On next run, offer to resume from checkpoint.

---

## Tuning notes (for after first real run)

**If too many files end up in review queue:** Lower `REVIEW_THRESHOLD` in
`scanner.py` (e.g., from 0.62 to 0.55).

**If wrong files are being auto-moved:** Raise `AUTO_MOVE_THRESHOLD` (e.g.,
from 0.82 to 0.88) or raise `novelty_threshold` in `Taxonomy.__init__`.

**If clusters during bootstrap are too fine-grained** (many small clusters
that should be one): Increase `min_cluster_size` (try 5 or 8).

**If clusters are too coarse** (unrelated files in same cluster): Decrease
`min_cluster_size` (try 2) or increase `umap_n_neighbors` (try 30).

**If the embedding cache becomes stale** (wrong classifications after model
change): Delete `data/embedding_cache.json`. Also delete `data/taxonomy.json`
and re-run bootstrap — the centroids are incompatible with a different model.
