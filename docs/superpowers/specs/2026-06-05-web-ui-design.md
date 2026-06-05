# DriveSort Web UI — Design Spec

**Date:** 2026-06-05
**Status:** Approved for implementation planning

---

## Context

DriveSort is a local-AI Google Drive organiser. The current interface is a terminal TUI (`drivesort bootstrap`, `drivesort scan`, `drivesort status`). The core problem driving this redesign:

1. **Clustering produces over-fragmented results** — files that belong together (e.g. books in a series) end up in separate micro-clusters. A visual interface makes it practical to identify and fix these groupings interactively.
2. **Hierarchy is essential** — the current taxonomy supports only two levels (category + parent). Users need arbitrary-depth folder trees (e.g. `Books → Fantasy → Cosmere`).
3. **The TUI is not interactive enough** — merging clusters requires typing exact names; there is no spatial context for placement decisions; taxonomy building across multiple sessions is not supported.
4. **Hard pivot to web** — the CLI commands `bootstrap`, `scan`, `status`, and `recover` are retired. One command (`drivesort serve`) starts a FastAPI server; the browser is the complete interface.

---

## Architecture

### Stack

| Layer | Choice |
|---|---|
| Backend | Python · FastAPI |
| Real-time | WebSocket (`/ws`) — progress events, scan updates |
| Frontend | React 18 + TypeScript · Vite |
| UI components | shadcn/ui + Tailwind CSS |
| Scatter plot | Plotly.js (handles thousands of points, React bindings) |
| Design language | Vibrant dark — deep navy base, saturated accents, subtle gradients (Cursor / Perplexity aesthetic) |

### Process model

```
drivesort serve
│
├── GET /              → React SPA (Vite build output)
├── REST /api/*        → Drive auth, taxonomy CRUD, cache management, stage/commit
├── WS  /ws            → progress stream (embedding %, UMAP coords, scan events)
│
└── Background tasks (FastAPI BackgroundTask)
    ├── EmbedWorker    → content extraction → embedding → writes caches
    ├── ClusterWorker  → UMAP + HDBSCAN → writes cluster cache
    └── ScanWorker     → polls Drive → classifies → adds to review queue
```

A single `drivesort serve` command starts everything. No separate worker processes.

### Cache layers

All caches live in `data/`, keyed by `SHA1(file_id|modified|name)`. On restart with unchanged files, all four layers are warm and the app reaches the review screen in seconds.

| Layer | File | Status |
|---|---|---|
| Content extraction | `content_cache.json` | Exists |
| Embeddings (384-dim) | `embedding_cache.json` | Exists |
| Clustering (UMAP 2D + labels) | `cluster_cache.pkl` | **New** |
| LLM cluster names | `llm_name_cache.json` | **New** |

**Cluster cache key:** `SHA1(sorted(embedding_keys) + umap_params + hdbscan_params)`. Invalidated if any file embedding changes or clustering parameters change. Changing only clustering params does not invalidate the embedding cache.

**Cache invalidation:** exposed via REST API — invalidate by file ID or by Drive folder ID (cascades to all descendants). Surfaced in the Status dashboard UI.

### Taxonomy schema (v2)

The current flat `{name → CategoryEntry}` dict is replaced with an arbitrary-depth tree stored in `data/taxonomy.json`:

```json
{
  "version": 2,
  "nodes": {
    "books": {
      "path": "Books",
      "parent": null,
      "centroid": [...],
      "member_ids": [...],
      "folder_id": "drive-folder-id",
      "description": "..."
    },
    "books/fantasy": {
      "path": "Books/Fantasy",
      "parent": "books",
      "centroid": [...],
      "member_ids": [...],
      "folder_id": "...",
      "description": "..."
    },
    "books/fantasy/cosmere": {
      "path": "Books/Fantasy/Cosmere",
      "parent": "books/fantasy",
      ...
    }
  }
}
```

**Classification** walks the tree top-down: at each node, find the closest child by cosine distance to its centroid. Stops when no child is closer than the novelty threshold. Replaces the hardcoded 2-level routing in `taxonomy.py`.

**Active learning:** every user confirmation (accept placement, correct placement) updates the centroid of the target node via running mean and re-normalises. The model improves with every decision.

### Draft / work-in-progress persistence

Taxonomy building can span multiple sessions. `data/draft.json` captures everything in-progress:

```json
{
  "saved_at": "2026-06-05T10:30:00Z",
  "taxonomy_tree": { ... },
  "staged_changes": [
    { "file_id": "abc", "file_name": "Dune.pdf", "current_path": null, "proposed_path": "Books/Fantasy" }
  ],
  "user_decisions": [
    { "file_id": "abc", "action": "assign", "path": "Books/Fantasy", "timestamp": "..." }
  ]
}
```

- **Auto-saved** after every user action (rename node, merge cluster, assign file).
- **Resume prompt** on restart if draft exists: "Resume work from 10 minutes ago" vs "Start fresh".
- **Cleared** on commit to Drive or explicit "Discard draft".
- **Bootstrap only** — the draft represents the in-progress taxonomy build. The Scan flow has no draft; each file decision in Scan is applied immediately (individual Drive writes, not batched).
- Enables multi-session taxonomy building without any re-analysis cost.

### No Drive writes until commit

The system never modifies Drive during analysis. All changes are staged. Only "Commit to Drive" triggers Drive API writes (folder creation, file moves). This is enforced at the API layer — no Drive mutation endpoints exist outside the commit flow.

---

## App Structure

### Routing

```
/setup/connect     Step 1 — Google Drive OAuth
/setup/analyse     Step 2 — Embedding & clustering progress
/setup/review      Step 3 — Taxonomy builder (main workspace)
/setup/commit      Step 4 — Staged changes review & commit
/scan              Ongoing file review inbox
/status            Dashboard, cache management
```

After initial commit, the app redirects to `/scan` as the default view. The wizard is re-enterable via `/setup/review` if the user wants to rebuild the taxonomy.

### Persistent top bar

A single top bar is always visible across all routes:

- **Left:** DriveSort logo + current route name
- **Centre:** Wizard step indicator (during setup only) — Connect → Analyse → Review → Commit
- **Right:** Drive connection status chip (connected account email, or "Connect Drive" if not authenticated)

Once authenticated, Step 1 collapses permanently into this status chip and is never shown as a full-screen step again.

---

## Screens

### Step 1 — Connect Drive (`/setup/connect`)

Full-screen OAuth entry point, shown only on first run or after disconnect. Contains:
- App name and one-line description
- "Connect Google Drive" button → triggers OAuth flow (local redirect to `localhost:7432/auth/callback`)
- On success: redirects immediately to `/setup/analyse`

After auth, Drive status collapses into the top bar chip. Step 1 is never a full-screen view again.

### Step 2 — Analyse Files (`/setup/analyse`)

Full-screen progress view. Four sequential phases, each shown with status and live stats:

| Phase | Visual |
|---|---|
| Fetch file list | Count of files discovered |
| Content extraction | Progress bar; "N cached, M extracting" |
| Embedding | Progress bar; files/sec; cache hit rate |
| Clustering + naming | Animated scatter plot building live (see below) |

**Live scatter plot during clustering:** as UMAP coords are computed, dots stream in via WebSocket (`{file_id, x, y}`). Initially all grey. When HDBSCAN labels arrive, dots animate to their cluster colour. When LLM naming completes, cluster labels fade in. The user watches their Drive's structure emerge from noise — this is the "sexy visuals" moment and also communicates that analysis is genuinely happening.

Cache awareness is explicit: "1,289 files loaded from cache · 58 new files being analysed". On restart with no changes, the progress screen is near-instant and transitions automatically to `/setup/review`.

### Step 3 — Build Taxonomy (`/setup/review`)

Full-screen, three-column layout. This is the primary workspace and where most time is spent.

```
┌─────────────────┬──────────────────┬────────────────┐
│  UMAP Scatter   │  File List       │  Taxonomy Tree │
│  (spatial map)  │  (selected node) │  (builder)     │
└─────────────────┴──────────────────┴────────────────┘
```

**Left — Scatter plot (Plotly.js):**
- One dot per file, coloured by top-level taxonomy node
- Click a cluster bubble → selects it, populates middle column
- Lasso select → select arbitrary files across cluster boundaries
- Drag one cluster bubble onto another → merge prompt
- Outlier files shown as small grey dots at the periphery
- Nearest-neighbour clusters shown as dotted connection lines when a node is selected

**Middle — File list:**
- Shows all files in the currently selected cluster or taxonomy node
- Columns: file name, type icon, size, current Drive location
- Each file is draggable → drop onto a taxonomy tree node to assign
- "Select all" → drag entire cluster at once
- Populated both by scatter plot clicks and taxonomy tree node clicks

**Right — Taxonomy tree builder:**
- Hierarchical, collapsible, arbitrary depth
- Each node shows: name, file count, action buttons (rename inline, add subfolder, delete)
- Nodes are drop targets — drag files or whole clusters onto a node to assign
- "Drop here → new subfolder" drop zone appears when dragging
- "+ Add folder" button at the bottom
- Unassigned files tracked separately as "Uncategorised (N)"

**Draft status:** persistent "● Draft auto-saved N min ago" indicator in the top bar header. Saving is silent and automatic.

### Step 4 — Review & Commit (`/setup/commit`)

Full-screen, two-column layout.

**Left — Proposed Drive structure (tree diff):**
- The full taxonomy tree, expandable
- Each node shows a badge: `+ new` (folder will be created) or unchanged
- Expand any node to see individual files with `→ Move` badges showing destination
- Files that remain in place show `= unchanged`
- Unassigned files shown under "Archive (+ new)"

**Right — Summary + actions:**
- Totals: folders to create, files to move, files uncategorised
- "Commit to Drive →" — triggers all Drive API writes
- "← Back to edit" — returns to `/setup/review` without losing any work
- "Discard draft" (destructive, requires confirmation)

No Drive changes happen until "Commit to Drive →" is clicked.

### Scan — Review Inbox (`/scan`)

Full-screen, three-column layout (narrower left queue, main suggestion panel, actions right).

**Left — Incoming file queue:**
- List of files awaiting review, sorted by confidence descending
- Each item shows: name, type, confidence badge (colour-coded)
- Novel files (no match) shown in red at the bottom
- Clicking a file loads its suggestion in the centre panel

**Centre — Placement suggestion:**
- File name and metadata at the top
- Taxonomy tree with the predicted path highlighted (the deepest matching node)
- Confidence score shown at the predicted node
- "Similar confirmed files" list below — shows 2-3 files that were previously confirmed to the same node, explaining why the model made this prediction

**Right — Actions:**
- "✓ Accept placement" → moves file to suggested path in Drive (immediate write — scan accepts individual writes unlike bootstrap), updates node centroid
- "Place elsewhere…" → opens tree picker overlay
- "Create new folder" → inline prompt to name a new node, then assigns
- "Skip / Archive" → moves to Archive

Every confirmation updates the target node's centroid (active learning). The model improves over time.

**Scan trigger:** a "Scan for new files" button in the top bar; background polling interval is configurable in the Status dashboard settings panel.

### Status Dashboard (`/status`)

Full-screen dashboard.

**Metrics row:** files organised, total categories, pending review, cache coverage %, last scan timestamp.

**Cache management panel:**
- Four cache layer cards (content, embeddings, clustering, LLM names) each showing entry count, size on disk, and last updated
- "Invalidate file…" → file picker, invalidates that file across all cache layers
- "Invalidate folder…" → Drive folder picker, cascades to all descendants
- "Clear all caches" → destructive, requires typed confirmation

**Scan settings panel:**
- Background polling toggle + interval selector (off / 15 min / 1 hr / 6 hr)
- "Scan for new files" manual trigger (also available in top bar)

**Taxonomy overview:**
- Collapsible tree showing all categories with file counts
- Quick link to `/setup/review` to edit the taxonomy

---

## v0.dev Prompt Strategy

v0.dev is used to scaffold the visual shell. Each major screen gets its own focused prompt. The prompt brief for all screens:

> Dark navy base (#0f1117), vibrant accent colours with subtle gradients on key surfaces, shadcn/ui components, Tailwind CSS, Vite + React 18 + TypeScript, no Next.js, no server components. Modern AI app aesthetic (Cursor / Perplexity).

Screens to scaffold in v0 (in order):
1. App shell — top bar, route layout, wizard step indicator
2. Step 2 — progress view with phase cards and live scatter placeholder
3. Step 3 — three-column taxonomy builder (scatter placeholder, file list, tree)
4. Step 4 — tree diff commit view
5. Scan inbox — three-column review layout
6. Status dashboard — metrics, cache cards

Interactive behaviour (Plotly scatter, drag-and-drop tree, WebSocket) is wired up after v0 scaffolding.

---

## What Is Not In This Spec

- **Plotly interaction implementation** (lasso, drag-to-merge) — implementation detail
- **FastAPI route definitions** — implementation detail
- **WebSocket message schema** — implementation detail
- **OAuth flow implementation** — implementation detail
- **Taxonomy migration from v1 to v2** — implementation detail
- **Clustering parameter tuning** — separate concern (can be addressed independently)
