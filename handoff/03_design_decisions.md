# 03 — Design Decisions

The reasoning behind every significant choice. Read this before refactoring anything.

---

## Why unsupervised clustering for bootstrap, not a predefined taxonomy

**Rejected alternative:** Define the folders upfront (Career, Finance, etc.) and
classify into them from day one using keyword rules or a zero-shot LLM.

**Why we didn't:** The predefined taxonomy approach requires knowing what
categories exist before you start. It doesn't scale to new content types, and it
encodes the human's current mental model, which is often wrong or incomplete.
The clustering approach lets the *data* tell you what categories naturally exist
in the corpus, then asks the human to name them. The taxonomy that emerges is
grounded in the actual file distribution rather than assumed upfront.

The human's role is naming and curation, not taxonomy design.

---

## Why HDBSCAN over k-means or Gaussian mixture models

**k-means requires specifying k** — you have to know how many categories you
want before clustering. Personal Drive files don't come in a predictable number
of categories.

**Gaussian mixture models** assume spherical or elliptical clusters. File
embedding space doesn't have that property — the FPV cluster and the Finance
cluster may be very different shapes and densities.

**HDBSCAN** finds clusters of arbitrary shape, handles varying density, and
critically: it labels genuine outliers as `-1` rather than forcing every point
into a cluster. Outlier files that don't belong anywhere are a first-class
concept in our system — HDBSCAN gives us this for free.

---

## Why UMAP before HDBSCAN (not HDBSCAN on raw embeddings)

`all-MiniLM-L6-v2` produces 384-dimensional embeddings. HDBSCAN on
high-dimensional spaces suffers from the curse of dimensionality — distance
metrics become unreliable as dimensions grow, and density estimation degrades.

UMAP to 2D first:
- Stabilises the density estimates HDBSCAN relies on
- Makes the clustering significantly more robust on small corpora (~100–1000 files)
- Produces coordinates we can use for visualisation later (the `embeddings_2d`
  field in `ClusterResult` is preserved for this purpose)

Trade-off: UMAP is non-deterministic without `random_state=42`. We fix this for
reproducibility during bootstrap. If you change `random_state`, different
clusters will emerge from the same data.

---

## Why sentence-transformers (all-MiniLM-L6-v2) not a larger model

We embed file metadata (name + snippet), not full documents. The semantic
distinctions we need — "this is a firmware file" vs "this is an insurance
spreadsheet" — are coarse enough that a small, fast model handles them well.

`all-MiniLM-L6-v2` is 80 MB, encodes ~14,000 sentences/second on CPU, and
scores well on semantic similarity benchmarks for short text. A larger model
(e.g., `all-mpnet-base-v2` at 420 MB) would produce marginally better
embeddings but at 4× the model size and ~3× slower inference, for no
meaningful gain on this classification task.

If you find clusters are noisy or categories bleed into each other, try
`all-mpnet-base-v2` — it's a drop-in replacement, just change `MODEL_NAME` in
`embedder.py`.

---

## Why content snippets (not full file downloads) for embedding

The Drive API returns a `contentHints.indexableText` field — a short (~200–400
char) text excerpt — without downloading the file. This means we can embed an
entire Drive without a single file download, which is:

- Fast (no bandwidth cost)
- Private (no file content leaves the API call chain)
- Sufficient (filename + snippet captures the semantic signal we need)

The exception: binary files (`.hex`, `.zip`, `.pdf` without text extraction)
return an empty snippet. For these, the filename and extension are the only
signals. This works well in practice — `chimera.hex` is unambiguously FPV
firmware from the name alone.

---

## Why cosine distance (not Euclidean) for classification

All embeddings are L2-normalised (unit vectors). For unit vectors, cosine
distance = `1 - dot(a, b)`, which is fast and numerically clean.

More importantly, cosine distance measures *angle* between vectors, not
magnitude. This makes it invariant to how "strongly" a file is embedded — a
short filename and a long filename with a snippet will have different vector
magnitudes, but cosine distance correctly captures their semantic similarity.

We normalise embeddings at encode time in `Embedder` (`normalize_embeddings=True`)
and re-normalise centroids after each update in `Taxonomy.confirm()`. If you
ever change the normalisation, the distance computation in `Taxonomy.classify()`
must change too.

---

## Why running mean for centroids (not storing all embeddings)

After bootstrap, each category could be re-represented as:
1. The list of all member embeddings (recompute centroid from scratch)
2. A running mean updated incrementally

We chose option 2. Storing all embeddings would make `taxonomy.json` grow
unboundedly as more files are classified. The running mean:
- Is constant size (384 floats per category)
- Updates in O(1)
- Has a known numerical property: the centroid after n files is the exact mean
  of all n embeddings (not an approximation)

The only downside: you can't remove a file from the centroid. If you
misclassify a file and confirm it, the centroid shifts slightly toward that
file. In practice this doesn't matter — the shift from one file is `1/n` of
the total, and centroids have many members.

---

## Why phi3:mini for the LLM (not mistral or a larger model)

phi3:mini (~2 GB quantised) was chosen for:
- Speed: cluster naming during bootstrap can involve 10–20 LLM calls; a slow
  model makes the interactive session frustrating
- Size: fits comfortably in 8 GB RAM alongside the Python process and embeddings
- Quality: phi3:mini is specifically strong at structured output (JSON) and
  short classification tasks, despite its small size

The model is configurable (`--model` flag on `drivesort bootstrap`). Use
`mistral` or `llama3` if you find naming quality poor.

Ollama was chosen over llama.cpp directly because it handles model management,
GPU offloading, and the HTTP API automatically. The `ollama` Python library is
a thin wrapper around the Ollama local server.

---

## Why the taxonomy lives in JSON (not SQLite or a vector database)

For a personal Drive with ~100–1000 files and ~10–30 categories, JSON is
sufficient. The taxonomy file is:
- Human-readable and editable by hand
- No external dependency
- Trivially backed up (it's just a file)
- Fast to load (a few milliseconds for 30 categories × 384 floats each)

A vector database (FAISS, chromadb) would make sense at 10,000+ categories.
We're nowhere near that scale for a personal Drive.

The embedding cache (`embedding_cache.json`) is also JSON for the same reason.
It's keyed by SHA1 hash so it self-manages — stale entries for deleted files
accumulate but don't cause errors, they just waste a small amount of disk space.

---

## Why the drive.file OAuth scope (not drive)

`drive.file` only grants access to files the application creates or opens — not
the entire Drive. We don't need full Drive write access. Using `drive` would
require a broader consent screen and would be harder to get verified by Google
if we ever distribute this.

The `drive.readonly` scope covers listing and reading file metadata. These two
scopes together give us exactly what we need and nothing more.

---

## Why dry-run is the default for scan

The first time you run `drivesort scan`, it will classify hundreds of files.
Getting a confidence threshold wrong, or having a noisy centroid from a small
bootstrap cluster, could move files to the wrong folder at scale.

Dry-run by default means you always see what *would* happen before it does.
You only pay for mistakes you explicitly authorise with `--live`.

---

## The two-threshold design in scanner.py

Three bands, two thresholds:

```
≥ 0.82  → auto_move   (very confident, no human needed)
0.62–0.82 → review    (uncertain, human confirms)
< 0.62  → novel       (doesn't fit, log for re-clustering)
```

The gap between review and auto-move is intentional. A file at 0.75 confidence
is "probably right" but not "certainly right" — worth a 2-second confirmation.
A file at 0.90 is almost certainly correct and auto-moving it is safe.

These thresholds were chosen conservatively for a first run. After running
against real data, you'll develop intuition for whether they need adjustment.
If you see a lot of correct auto-moves in dry-run, lower the review threshold.
If you see wrong auto-moves in live mode, raise the auto-move threshold.
