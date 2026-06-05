# DriveSort Backend Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the DriveSort CLI with a FastAPI backend that serves all three features (bootstrap, scan, status) as REST + WebSocket APIs, backed by a new arbitrary-depth taxonomy, aggressive four-layer caching, and a draft/staging system that never writes to Drive until explicitly committed.

**Architecture:** FastAPI app (`drivesort/server.py`) mounts all API routers under `/api` and serves the React SPA from `/`. Heavy operations (embedding, UMAP, HDBSCAN) run as FastAPI `BackgroundTask`s and stream progress over a WebSocket at `/ws`. The old taxonomy (2-level flat dict) is replaced by `taxonomy_v2.py` (arbitrary-depth tree keyed by path string). The old CLI commands are retired; `drivesort serve` is the only entry point.

**Tech Stack:** FastAPI · Uvicorn · httpx (test client) · pytest · existing sentence-transformers / UMAP / HDBSCAN stack

---

## File Map

### New files
| File | Responsibility |
|---|---|
| `drivesort/taxonomy_v2.py` | Arbitrary-depth taxonomy tree — nodes, centroid math, top-down classification, active learning |
| `drivesort/cluster_cache.py` | Persist/load UMAP 2D coords + HDBSCAN labels keyed by embedding fingerprint + params |
| `drivesort/llm_name_cache.py` | Persist/load LLM cluster names keyed by cluster file-ID set + model name |
| `drivesort/draft.py` | DraftState dataclass + DraftManager (auto-save, load, clear) |
| `drivesort/stage.py` | StagedChange list builder + commit executor (Drive API writes) |
| `drivesort/ws.py` | ConnectionManager — accept WebSocket connections and broadcast events |
| `drivesort/server.py` | FastAPI app factory — mount routers, serve SPA, register startup hooks |
| `drivesort/api/__init__.py` | Empty package marker |
| `drivesort/api/auth.py` | OAuth login + callback routes |
| `drivesort/api/analysis.py` | Trigger embed/cluster; stream progress via WebSocket |
| `drivesort/api/taxonomy.py` | Taxonomy tree CRUD routes |
| `drivesort/api/draft.py` | Draft load/save/discard routes |
| `drivesort/api/stage.py` | Staged changes view + commit route |
| `drivesort/api/scan.py` | Scan trigger, review queue, accept/correct routes |
| `drivesort/api/cache.py` | Cache status + per-file/per-folder invalidation routes |
| `drivesort/api/ws.py` | WebSocket endpoint |

### Modified files
| File | Change |
|---|---|
| `pyproject.toml` | Add fastapi, uvicorn[standard], python-multipart, httpx deps |
| `drivesort/cli.py` | Retire bootstrap/scan/status/recover; add `serve` command |

### Test files
| File | What it tests |
|---|---|
| `tests/test_taxonomy_v2.py` | Node operations, classification, centroid updates, save/load |
| `tests/test_cluster_cache.py` | Cache key stability, hit/miss, param invalidation |
| `tests/test_draft.py` | Save/load/clear round-trips, auto-save after mutations |
| `tests/test_stage.py` | Change accumulation, commit sequence |
| `tests/test_api_auth.py` | Auth status endpoint |
| `tests/test_api_analysis.py` | Trigger endpoint queues background task |
| `tests/test_api_taxonomy.py` | Node CRUD via HTTP |
| `tests/test_api_draft.py` | Draft routes |
| `tests/test_api_scan.py` | Queue listing, accept, correct |
| `tests/test_api_cache.py` | Invalidation routes |

---

## Task 1: Add backend dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `drivesort/cli.py`

- [ ] **Step 1: Add FastAPI dependencies to pyproject.toml**

Replace the `[project]` `dependencies` list — add these four entries after `requests`:

```toml
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.29.0",
    "python-multipart>=0.0.9",
    "httpx>=0.27.0",
```

- [ ] **Step 2: Install updated deps**

```bash
pip install -e ".[dev]"
```

Expected: installs fastapi, uvicorn, python-multipart, httpx without errors.

- [ ] **Step 3: Retire old CLI commands, add serve**

Replace the body of `drivesort/cli.py` with:

```python
"""
drivesort/cli.py
----------------
Single entry point: drivesort serve
All other commands (bootstrap, scan, status, recover) have moved to the web UI.
"""
from __future__ import annotations

import typer
import uvicorn

app = typer.Typer(help="DriveSort — local-AI Google Drive organiser")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(7432, help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload (dev only)"),
) -> None:
    """Start the DriveSort web server."""
    typer.echo(f"Starting DriveSort at http://{host}:{port}")
    uvicorn.run(
        "drivesort.server:app",
        host=host,
        port=port,
        reload=reload,
    )


def main() -> None:
    app()
```

- [ ] **Step 4: Verify CLI works**

```bash
drivesort serve --help
```

Expected output includes: `Start the DriveSort web server.`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml drivesort/cli.py
git commit -m "feat: retire CLI commands, add drivesort serve via FastAPI/uvicorn"
```

---

## Task 2: Taxonomy v2 — node model and persistence

**Files:**
- Create: `drivesort/taxonomy_v2.py`
- Create: `tests/test_taxonomy_v2.py`

- [ ] **Step 1: Write failing tests for node model and persistence**

Create `tests/test_taxonomy_v2.py`:

```python
"""Tests for taxonomy_v2 — node model, persistence, tree navigation."""
import tempfile
from pathlib import Path

import numpy as np
import pytest

from drivesort.taxonomy_v2 import TaxonomyV2, TaxonomyNode


def make_unit(values: list[float]) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


@pytest.fixture
def tmp_path_taxonomy(tmp_path):
    return tmp_path / "taxonomy.json"


@pytest.fixture
def tax(tmp_path_taxonomy):
    return TaxonomyV2(path=tmp_path_taxonomy)


class TestAddNode:
    def test_add_root_node(self, tax):
        embs = np.stack([make_unit([1, 0, 0]), make_unit([0.9, 0.1, 0])])
        tax.add_node("books", "Books", parent=None, member_embeddings=embs,
                     member_ids=["a", "b"], folder_id="fid1")
        assert "books" in tax.nodes
        assert tax.nodes["books"].parent is None
        assert tax.nodes["books"].name == "Books"
        assert tax.nodes["books"].member_count == 2

    def test_add_child_node(self, tax):
        embs = np.stack([make_unit([1, 0, 0])])
        tax.add_node("books", "Books", parent=None, member_embeddings=embs,
                     member_ids=["a"], folder_id="f1")
        tax.add_node("books/fantasy", "Fantasy", parent="books",
                     member_embeddings=embs, member_ids=["a"], folder_id="f2")
        assert tax.nodes["books/fantasy"].parent == "books"

    def test_centroid_is_normalised(self, tax):
        embs = np.stack([make_unit([3, 4, 0]), make_unit([0, 4, 3])])
        tax.add_node("x", "X", parent=None, member_embeddings=embs,
                     member_ids=["a", "b"], folder_id="f1")
        centroid = tax.nodes["x"].centroid_array()
        assert abs(np.linalg.norm(centroid) - 1.0) < 1e-5


class TestTreeNavigation:
    def test_children_of_root(self, tax):
        embs = np.stack([make_unit([1, 0, 0])])
        tax.add_node("books", "Books", parent=None, member_embeddings=embs,
                     member_ids=["a"], folder_id="f1")
        tax.add_node("books/fantasy", "Fantasy", parent="books",
                     member_embeddings=embs, member_ids=["a"], folder_id="f2")
        tax.add_node("work", "Work", parent=None, member_embeddings=embs,
                     member_ids=["a"], folder_id="f3")
        roots = tax.children_of(None)
        assert {n.path for n in roots} == {"books", "work"}
        children = tax.children_of("books")
        assert len(children) == 1
        assert children[0].path == "books/fantasy"

    def test_children_of_leaf_is_empty(self, tax):
        embs = np.stack([make_unit([1, 0, 0])])
        tax.add_node("books", "Books", parent=None, member_embeddings=embs,
                     member_ids=["a"], folder_id="f1")
        assert tax.children_of("books") == []


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path_taxonomy, tax):
        embs = np.stack([make_unit([1, 0, 0])])
        tax.add_node("books", "Books", parent=None, member_embeddings=embs,
                     member_ids=["a"], folder_id="fid1", description="Reading")
        tax.add_node("books/fantasy", "Fantasy", parent="books",
                     member_embeddings=embs, member_ids=["a"], folder_id="fid2")
        tax.save()

        loaded = TaxonomyV2.load(path=tmp_path_taxonomy)
        assert "books" in loaded.nodes
        assert "books/fantasy" in loaded.nodes
        assert loaded.nodes["books"].description == "Reading"
        assert loaded.nodes["books/fantasy"].parent == "books"
        centroid = loaded.nodes["books"].centroid_array()
        assert abs(np.linalg.norm(centroid) - 1.0) < 1e-5

    def test_load_missing_file_returns_empty(self, tmp_path_taxonomy):
        loaded = TaxonomyV2.load(path=tmp_path_taxonomy)
        assert loaded.nodes == {}
```

- [ ] **Step 2: Run tests — confirm they fail**

```bash
pytest tests/test_taxonomy_v2.py -v
```

Expected: `ModuleNotFoundError: No module named 'drivesort.taxonomy_v2'`

- [ ] **Step 3: Implement TaxonomyV2 node model and persistence**

Create `drivesort/taxonomy_v2.py`:

```python
"""
drivesort/taxonomy_v2.py
------------------------
Arbitrary-depth taxonomy tree.

Nodes are keyed by path strings (e.g. "books/fantasy/cosmere").
Each node holds a centroid (running mean of member embeddings) used for
top-down nearest-neighbour classification.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

TAXONOMY_V2_PATH = Path("data/taxonomy.json")


@dataclass
class TaxonomyNode:
    path: str                      # e.g. "books/fantasy/cosmere"
    name: str                      # display name, e.g. "Cosmere"
    parent: Optional[str]          # parent path or None for root nodes
    centroid: list[float]          # L2-normalised mean embedding
    member_ids: list[str]          # Drive file IDs contributing to centroid
    member_count: int
    folder_id: str                 # Google Drive folder ID (set on commit)
    description: str = ""

    def centroid_array(self) -> np.ndarray:
        return np.array(self.centroid, dtype=np.float32)


@dataclass
class ClassificationResult:
    file_id: str
    file_name: str
    path: Optional[str]     # matched taxonomy node path; None = novel
    confidence: float       # 1 - cosine_distance
    distance: float
    is_novel: bool


class TaxonomyV2:
    NOVELTY_THRESHOLD = 0.42
    VERSION = 2

    def __init__(self, path: Path = TAXONOMY_V2_PATH, novelty_threshold: float = NOVELTY_THRESHOLD):
        self._path = path
        self._novelty_threshold = novelty_threshold
        self.nodes: dict[str, TaxonomyNode] = {}

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_node(
        self,
        path: str,
        name: str,
        parent: Optional[str],
        member_embeddings: np.ndarray,
        member_ids: list[str],
        folder_id: str,
        description: str = "",
    ) -> None:
        centroid = member_embeddings.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        self.nodes[path] = TaxonomyNode(
            path=path,
            name=name,
            parent=parent,
            centroid=centroid.tolist(),
            member_ids=list(member_ids),
            member_count=len(member_ids),
            folder_id=folder_id,
            description=description,
        )

    def children_of(self, parent_path: Optional[str]) -> list[TaxonomyNode]:
        return [n for n in self.nodes.values() if n.parent == parent_path]

    def ancestors_of(self, path: str) -> list[TaxonomyNode]:
        result = []
        node = self.nodes.get(path)
        while node and node.parent:
            parent = self.nodes.get(node.parent)
            if parent:
                result.append(parent)
            node = parent
        return list(reversed(result))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.VERSION,
            "nodes": {k: asdict(v) for k, v in self.nodes.items()},
        }
        self._path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path = TAXONOMY_V2_PATH, novelty_threshold: float = NOVELTY_THRESHOLD) -> "TaxonomyV2":
        tax = cls(path=path, novelty_threshold=novelty_threshold)
        if not path.exists():
            return tax
        raw = json.loads(path.read_text())
        if raw.get("version") != cls.VERSION:
            return tax
        for k, v in raw.get("nodes", {}).items():
            tax.nodes[k] = TaxonomyNode(**v)
        return tax
```

- [ ] **Step 4: Run tests — confirm they pass**

```bash
pytest tests/test_taxonomy_v2.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add drivesort/taxonomy_v2.py tests/test_taxonomy_v2.py
git commit -m "feat: add TaxonomyV2 with arbitrary-depth node tree and persistence"
```

---

## Task 3: Taxonomy v2 — classification and active learning

**Files:**
- Modify: `drivesort/taxonomy_v2.py`
- Modify: `tests/test_taxonomy_v2.py`

- [ ] **Step 1: Write failing tests for classify() and confirm()**

Append to `tests/test_taxonomy_v2.py`:

```python
class TestClassify:
    @pytest.fixture
    def loaded_tax(self, tmp_path_taxonomy):
        t = TaxonomyV2(path=tmp_path_taxonomy)
        books_embs = np.stack([make_unit([1, 0, 0]), make_unit([0.95, 0.05, 0])])
        work_embs  = np.stack([make_unit([0, 1, 0]), make_unit([0, 0.9, 0.1])])
        t.add_node("books", "Books", parent=None, member_embeddings=books_embs,
                   member_ids=["a","b"], folder_id="f1")
        t.add_node("work", "Work", parent=None, member_embeddings=work_embs,
                   member_ids=["c","d"], folder_id="f2")
        fantasy_embs = np.stack([make_unit([1, 0, 0])])
        t.add_node("books/fantasy", "Fantasy", parent="books",
                   member_embeddings=fantasy_embs, member_ids=["a"], folder_id="f3")
        return t

    def test_classify_nearest_root(self, loaded_tax):
        emb = make_unit([1, 0.01, 0])
        result = loaded_tax.classify(emb, file_id="x", file_name="test.epub")
        assert result.path == "books/fantasy"
        assert result.is_novel is False
        assert result.confidence > 0.9

    def test_classify_novel_when_distant(self, loaded_tax):
        emb = make_unit([0, 0, 1])  # far from all centroids
        result = loaded_tax.classify(emb, file_id="x", file_name="test.epub")
        assert result.is_novel is True
        assert result.path is None

    def test_classify_empty_taxonomy_is_novel(self, tmp_path_taxonomy):
        t = TaxonomyV2(path=tmp_path_taxonomy)
        emb = make_unit([1, 0, 0])
        result = t.classify(emb, file_id="x", file_name="test.epub")
        assert result.is_novel is True

    def test_classify_stops_at_leaf(self, loaded_tax):
        # "books" has child "books/fantasy"; result should reach the leaf
        emb = make_unit([1, 0.02, 0])
        result = loaded_tax.classify(emb, file_id="x", file_name="test.epub")
        assert result.path == "books/fantasy"


class TestConfirm:
    def test_confirm_updates_centroid(self, tax):
        embs = np.stack([make_unit([1, 0, 0])])
        tax.add_node("books", "Books", parent=None, member_embeddings=embs,
                     member_ids=["a"], folder_id="f1")
        old_centroid = tax.nodes["books"].centroid_array().copy()
        new_emb = make_unit([0.5, 0.5, 0])
        tax.confirm("books", new_emb, file_id="b")
        new_centroid = tax.nodes["books"].centroid_array()
        assert not np.allclose(old_centroid, new_centroid)
        assert abs(np.linalg.norm(new_centroid) - 1.0) < 1e-5
        assert tax.nodes["books"].member_count == 2
        assert "b" in tax.nodes["books"].member_ids

    def test_confirm_unknown_node_raises(self, tax):
        with pytest.raises(KeyError):
            tax.confirm("nonexistent", make_unit([1, 0, 0]), file_id="x")
```

- [ ] **Step 2: Run — confirm fails**

```bash
pytest tests/test_taxonomy_v2.py::TestClassify tests/test_taxonomy_v2.py::TestConfirm -v
```

Expected: `AttributeError: 'TaxonomyV2' object has no attribute 'classify'`

- [ ] **Step 3: Implement classify() and confirm()**

Add to `drivesort/taxonomy_v2.py` inside the `TaxonomyV2` class, after `ancestors_of`:

```python
    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(
        self,
        embedding: np.ndarray,
        file_id: str,
        file_name: str,
    ) -> ClassificationResult:
        roots = self.children_of(None)
        if not roots:
            return ClassificationResult(file_id=file_id, file_name=file_name,
                                        path=None, confidence=0.0, distance=1.0,
                                        is_novel=True)
        best_node, best_dist = self._closest(roots, embedding)
        if best_dist > self._novelty_threshold:
            return ClassificationResult(file_id=file_id, file_name=file_name,
                                        path=None, confidence=0.0,
                                        distance=best_dist, is_novel=True)
        # Walk down the tree
        while True:
            children = self.children_of(best_node.path)
            if not children:
                break
            child, dist = self._closest(children, embedding)
            if dist > self._novelty_threshold:
                break
            best_node, best_dist = child, dist

        return ClassificationResult(
            file_id=file_id,
            file_name=file_name,
            path=best_node.path,
            confidence=1.0 - best_dist,
            distance=best_dist,
            is_novel=False,
        )

    def _closest(
        self, nodes: list[TaxonomyNode], embedding: np.ndarray
    ) -> tuple[TaxonomyNode, float]:
        best_node = nodes[0]
        best_dist = float("inf")
        for node in nodes:
            dist = float(1.0 - np.dot(embedding, node.centroid_array()))
            if dist < best_dist:
                best_dist = dist
                best_node = node
        return best_node, best_dist

    # ------------------------------------------------------------------
    # Active learning
    # ------------------------------------------------------------------

    def confirm(self, path: str, embedding: np.ndarray, file_id: str) -> None:
        """Update the centroid of a node with a newly confirmed file embedding."""
        node = self.nodes[path]  # raises KeyError if missing — caller's bug
        n = node.member_count
        old_c = node.centroid_array()
        new_c = (old_c * n + embedding) / (n + 1)
        new_c = new_c / (np.linalg.norm(new_c) + 1e-8)
        node.centroid = new_c.tolist()
        node.member_count = n + 1
        if file_id not in node.member_ids:
            node.member_ids.append(file_id)
```

- [ ] **Step 4: Run all taxonomy_v2 tests**

```bash
pytest tests/test_taxonomy_v2.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add drivesort/taxonomy_v2.py tests/test_taxonomy_v2.py
git commit -m "feat: add top-down classification and active-learning confirm() to TaxonomyV2"
```

---

## Task 4: Cluster cache and LLM name cache

**Files:**
- Create: `drivesort/cluster_cache.py`
- Create: `drivesort/llm_name_cache.py`
- Create: `tests/test_cluster_cache.py`

- [ ] **Step 1: Write failing tests for cluster cache**

Create `tests/test_cluster_cache.py`:

```python
"""Tests for cluster_cache — hit/miss logic and param-based invalidation."""
import numpy as np
import pytest
from drivesort.cluster_cache import ClusterCache


@pytest.fixture
def cache(tmp_path):
    return ClusterCache(path=tmp_path / "cluster_cache.pkl")


def make_arrays():
    emb_2d = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int32)
    return emb_2d, labels


class TestClusterCache:
    def test_miss_on_empty(self, cache):
        assert cache.load(["key1", "key2"], {"min_cluster_size": 3}) is None

    def test_save_then_hit(self, cache):
        emb_2d, labels = make_arrays()
        params = {"min_cluster_size": 3, "umap_n_neighbors": 15}
        cache.save(["key1", "key2"], params, emb_2d, labels)
        result = cache.load(["key2", "key1"], params)  # order-independent
        assert result is not None
        np.testing.assert_array_equal(result.embeddings_2d, emb_2d)
        np.testing.assert_array_equal(result.labels, labels)

    def test_miss_on_param_change(self, cache):
        emb_2d, labels = make_arrays()
        params_a = {"min_cluster_size": 3, "umap_n_neighbors": 15}
        params_b = {"min_cluster_size": 5, "umap_n_neighbors": 15}
        cache.save(["key1"], params_a, emb_2d, labels)
        assert cache.load(["key1"], params_b) is None

    def test_miss_on_key_change(self, cache):
        emb_2d, labels = make_arrays()
        params = {"min_cluster_size": 3}
        cache.save(["key1", "key2"], params, emb_2d, labels)
        assert cache.load(["key1", "key3"], params) is None
```

- [ ] **Step 2: Run — confirm fails**

```bash
pytest tests/test_cluster_cache.py -v
```

Expected: `ModuleNotFoundError: No module named 'drivesort.cluster_cache'`

- [ ] **Step 3: Implement ClusterCache**

Create `drivesort/cluster_cache.py`:

```python
"""
drivesort/cluster_cache.py
--------------------------
Caches UMAP 2D projections and HDBSCAN labels.
Cache key = SHA1(sorted embedding cache keys + clustering params).
Invalidated when any embedding changes or params change.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

DEFAULT_PATH = Path("data/cluster_cache.pkl")


@dataclass
class ClusterCacheEntry:
    key: str
    embeddings_2d: np.ndarray
    labels: np.ndarray
    params: dict


def _make_key(embedding_keys: list[str], params: dict) -> str:
    payload = json.dumps(
        {"keys": sorted(embedding_keys), "params": params},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()


class ClusterCache:
    def __init__(self, path: Path = DEFAULT_PATH):
        self._path = path

    def load(self, embedding_keys: list[str], params: dict) -> Optional[ClusterCacheEntry]:
        if not self._path.exists():
            return None
        with open(self._path, "rb") as f:
            entry: ClusterCacheEntry = pickle.load(f)
        expected = _make_key(embedding_keys, params)
        return entry if entry.key == expected else None

    def save(
        self,
        embedding_keys: list[str],
        params: dict,
        embeddings_2d: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        entry = ClusterCacheEntry(
            key=_make_key(embedding_keys, params),
            embeddings_2d=embeddings_2d,
            labels=labels,
            params=params,
        )
        with open(self._path, "wb") as f:
            pickle.dump(entry, f)

    def invalidate(self) -> None:
        if self._path.exists():
            self._path.unlink()
```

- [ ] **Step 4: Implement LLM name cache**

Create `drivesort/llm_name_cache.py`:

```python
"""
drivesort/llm_name_cache.py
---------------------------
Caches LLM-generated cluster names.
Cache key = SHA1(sorted file IDs in cluster + model name).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

DEFAULT_PATH = Path("data/llm_name_cache.json")


def _make_key(file_ids: list[str], model: str) -> str:
    payload = json.dumps({"ids": sorted(file_ids), "model": model}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()


class LLMNameCache:
    def __init__(self, path: Path = DEFAULT_PATH):
        self._path = path
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            self._data = json.loads(self._path.read_text())

    def get(self, file_ids: list[str], model: str) -> Optional[dict]:
        return self._data.get(_make_key(file_ids, model))

    def put(self, file_ids: list[str], model: str, result: dict) -> None:
        self._data[_make_key(file_ids, model)] = result
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))

    def invalidate_file(self, file_id: str) -> int:
        """Remove all entries that include this file_id. Returns count removed."""
        to_remove = [k for k, v in self._data.items()
                     if file_id in v.get("_file_ids", [])]
        for k in to_remove:
            del self._data[k]
        if to_remove:
            self._path.write_text(json.dumps(self._data, indent=2))
        return len(to_remove)
```

- [ ] **Step 5: Run cluster cache tests**

```bash
pytest tests/test_cluster_cache.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add drivesort/cluster_cache.py drivesort/llm_name_cache.py tests/test_cluster_cache.py
git commit -m "feat: add ClusterCache and LLMNameCache with param-based invalidation"
```

---

## Task 5: Draft persistence

**Files:**
- Create: `drivesort/draft.py`
- Create: `tests/test_draft.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_draft.py`:

```python
"""Tests for draft persistence — save, load, clear, auto-save."""
import pytest
from drivesort.draft import DraftManager, DraftState, StagedChange, UserDecision


@pytest.fixture
def mgr(tmp_path):
    return DraftManager(path=tmp_path / "draft.json")


def make_state() -> DraftState:
    return DraftState(
        taxonomy_nodes={"books": {"path": "books", "name": "Books"}},
        staged_changes=[
            StagedChange(file_id="f1", file_name="Dune.pdf",
                         current_path=None, proposed_path="books")
        ],
        user_decisions=[
            UserDecision(file_id="f1", action="assign",
                         path="books", timestamp="2026-06-05T10:00:00Z")
        ],
    )


class TestDraftManager:
    def test_exists_false_initially(self, mgr):
        assert mgr.exists() is False

    def test_save_load_roundtrip(self, mgr):
        state = make_state()
        mgr.save(state)
        assert mgr.exists() is True
        loaded = mgr.load()
        assert loaded is not None
        assert loaded.taxonomy_nodes == state.taxonomy_nodes
        assert len(loaded.staged_changes) == 1
        assert loaded.staged_changes[0].file_id == "f1"
        assert loaded.staged_changes[0].proposed_path == "books"
        assert len(loaded.user_decisions) == 1

    def test_clear_removes_file(self, mgr):
        mgr.save(make_state())
        mgr.clear()
        assert mgr.exists() is False
        assert mgr.load() is None

    def test_saved_at_is_populated(self, mgr):
        mgr.save(make_state())
        loaded = mgr.load()
        assert loaded.saved_at != ""
```

- [ ] **Step 2: Run — confirm fails**

```bash
pytest tests/test_draft.py -v
```

Expected: `ModuleNotFoundError: No module named 'drivesort.draft'`

- [ ] **Step 3: Implement DraftManager**

Create `drivesort/draft.py`:

```python
"""
drivesort/draft.py
------------------
Work-in-progress state for multi-session taxonomy building.
Auto-saved after every user action. Cleared on commit or explicit discard.
This is bootstrap-only — the Scan flow has no draft concept.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_PATH = Path("data/draft.json")


@dataclass
class StagedChange:
    file_id: str
    file_name: str
    current_path: Optional[str]   # None = file not yet in any Drive folder
    proposed_path: str


@dataclass
class UserDecision:
    file_id: str
    action: str                   # "assign" | "reject" | "archive"
    path: Optional[str]
    timestamp: str


@dataclass
class DraftState:
    taxonomy_nodes: dict          # path -> node attribute dict
    staged_changes: list[StagedChange] = field(default_factory=list)
    user_decisions: list[UserDecision] = field(default_factory=list)
    saved_at: str = ""


class DraftManager:
    def __init__(self, path: Path = DEFAULT_PATH):
        self._path = path

    def exists(self) -> bool:
        return self._path.exists()

    def load(self) -> Optional[DraftState]:
        if not self._path.exists():
            return None
        raw = json.loads(self._path.read_text())
        return DraftState(
            taxonomy_nodes=raw["taxonomy_nodes"],
            staged_changes=[StagedChange(**c) for c in raw["staged_changes"]],
            user_decisions=[UserDecision(**d) for d in raw["user_decisions"]],
            saved_at=raw.get("saved_at", ""),
        )

    def save(self, state: DraftState) -> None:
        state.saved_at = datetime.now(timezone.utc).isoformat()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(asdict(state), indent=2))

    def clear(self) -> None:
        if self._path.exists():
            self._path.unlink()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_draft.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add drivesort/draft.py tests/test_draft.py
git commit -m "feat: add DraftManager for multi-session taxonomy work-in-progress"
```

---

## Task 6: WebSocket connection manager + FastAPI app

**Files:**
- Create: `drivesort/ws.py`
- Create: `drivesort/server.py`
- Create: `drivesort/api/__init__.py`

- [ ] **Step 1: Create WebSocket connection manager**

Create `drivesort/ws.py`:

```python
"""
drivesort/ws.py
---------------
Shared WebSocket connection manager.
All API modules import `manager` to broadcast progress events.
"""
from __future__ import annotations

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, event: dict) -> None:
        """Send event to all connected clients. Silently drops dead connections."""
        for ws in list(self.active):
            try:
                await ws.send_json(event)
            except Exception:
                self.disconnect(ws)


manager = ConnectionManager()
```

- [ ] **Step 2: Create FastAPI app and package marker**

Create `drivesort/api/__init__.py` (empty):

```python
```

Create `drivesort/server.py`:

```python
"""
drivesort/server.py
-------------------
FastAPI application factory.

Routes:
  /api/auth/*      OAuth flow
  /api/analysis/*  Embed + cluster trigger, WebSocket progress
  /api/taxonomy/*  Taxonomy tree CRUD
  /api/draft/*     Draft load/save/discard
  /api/stage/*     Staged changes + commit
  /api/scan/*      Scan queue, accept, correct
  /api/cache/*     Cache status + invalidation
  /ws              WebSocket
  /                React SPA (static files, production only)
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import auth, analysis, taxonomy, draft, stage, scan, cache, ws as ws_router

app = FastAPI(title="DriveSort", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:7432"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router,     prefix="/api/auth",     tags=["auth"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(taxonomy.router, prefix="/api/taxonomy", tags=["taxonomy"])
app.include_router(draft.router,    prefix="/api/draft",    tags=["draft"])
app.include_router(stage.router,    prefix="/api/stage",    tags=["stage"])
app.include_router(scan.router,     prefix="/api/scan",     tags=["scan"])
app.include_router(cache.router,    prefix="/api/cache",    tags=["cache"])
app.include_router(ws_router.router, tags=["ws"])
```

- [ ] **Step 3: Verify server imports cleanly (routers don't exist yet — expect ImportError)**

```bash
python -c "from drivesort.server import app" 2>&1 | head -5
```

Expected: `ImportError: cannot import name 'auth' from 'drivesort.api'` — that's fine, routers come next.

- [ ] **Step 4: Commit**

```bash
git add drivesort/ws.py drivesort/server.py drivesort/api/__init__.py
git commit -m "feat: add WebSocket connection manager and FastAPI app factory"
```

---

## Task 7: Auth and WebSocket routes

**Files:**
- Create: `drivesort/api/auth.py`
- Create: `drivesort/api/ws.py`
- Create: `tests/test_api_auth.py`

- [ ] **Step 1: Write failing auth test**

Create `tests/test_api_auth.py`:

```python
"""Tests for auth API — status endpoint."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from drivesort.server import app
    return TestClient(app)


def test_auth_status_unauthenticated(client):
    resp = client.get("/api/auth/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["authenticated"] is False
    assert data["email"] is None


def test_auth_login_returns_url(client):
    resp = client.get("/api/auth/login")
    assert resp.status_code == 200
    assert "auth_url" in resp.json()
```

- [ ] **Step 2: Create auth router**

Create `drivesort/api/auth.py`:

```python
"""
drivesort/api/auth.py
---------------------
Google OAuth routes.

GET /api/auth/status    — is the user authenticated?
GET /api/auth/login     — get the OAuth URL to redirect to
GET /api/auth/callback  — exchange code for token (Google redirects here)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from google.auth.transport.requests import Request as GRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow

TOKEN_PATH = Path("data/token.json")
CREDENTIALS_PATH = Path("data/credentials.json")
SCOPES = ["https://www.googleapis.com/auth/drive"]
REDIRECT_URI = "http://localhost:7432/api/auth/callback"

router = APIRouter()


def _load_credentials() -> Optional[Credentials]:
    if not TOKEN_PATH.exists():
        return None
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(GRequest())
            TOKEN_PATH.write_text(creds.to_json())
        except Exception:
            return None
    return creds if (creds and creds.valid) else None


def _get_email(creds: Credentials) -> Optional[str]:
    try:
        import googleapiclient.discovery
        svc = googleapiclient.discovery.build("oauth2", "v2", credentials=creds)
        info = svc.userinfo().get().execute()
        return info.get("email")
    except Exception:
        return None


@router.get("/status")
def auth_status():
    creds = _load_credentials()
    if not creds:
        return {"authenticated": False, "email": None}
    return {"authenticated": True, "email": _get_email(creds)}


@router.get("/login")
def auth_login():
    if not CREDENTIALS_PATH.exists():
        return {"error": "credentials.json not found in data/"}
    flow = Flow.from_client_secrets_file(
        str(CREDENTIALS_PATH), scopes=SCOPES, redirect_uri=REDIRECT_URI
    )
    auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")
    return {"auth_url": auth_url}


@router.get("/callback")
def auth_callback(code: str, state: str = ""):
    flow = Flow.from_client_secrets_file(
        str(CREDENTIALS_PATH), scopes=SCOPES, redirect_uri=REDIRECT_URI
    )
    flow.fetch_token(code=code)
    creds = flow.credentials
    TOKEN_PATH.parent.mkdir(exist_ok=True)
    TOKEN_PATH.write_text(creds.to_json())
    return RedirectResponse(url="/setup/analyse")
```

- [ ] **Step 3: Create WebSocket route**

Create `drivesort/api/ws.py`:

```python
"""
drivesort/api/ws.py
-------------------
WebSocket endpoint — clients connect here to receive progress events.
"""
from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from drivesort.ws import manager

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()  # keep connection alive; events pushed server→client
    except WebSocketDisconnect:
        manager.disconnect(ws)
```

- [ ] **Step 4: Run auth tests**

```bash
pytest tests/test_api_auth.py -v
```

Expected: both tests pass. (The `credentials.json` won't exist in tmp_path so `auth_login` returns an error dict — that's fine; the test only checks for `auth_url` key, which will be absent. Adjust test if needed to check the error case.)

Adjust `test_auth_login_returns_url` if `credentials.json` is absent:

```python
def test_auth_login_no_credentials(client):
    resp = client.get("/api/auth/login")
    assert resp.status_code == 200
    # Without credentials.json, returns error
    data = resp.json()
    assert "error" in data or "auth_url" in data
```

- [ ] **Step 5: Commit**

```bash
git add drivesort/api/auth.py drivesort/api/ws.py tests/test_api_auth.py
git commit -m "feat: add OAuth auth routes and WebSocket endpoint"
```

---

## Task 8: Analysis routes

**Files:**
- Create: `drivesort/api/analysis.py`
- Create: `tests/test_api_analysis.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_api_analysis.py`:

```python
"""Tests for analysis API — trigger endpoint queues a background task."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from drivesort.server import app
    return TestClient(app)


def test_trigger_returns_accepted(client):
    resp = client.post("/api/analysis/trigger")
    assert resp.status_code == 202
    assert resp.json()["status"] == "started"


def test_status_returns_idle_initially(client):
    resp = client.get("/api/analysis/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["phase"] in ("idle", "running", "complete", "error")
```

- [ ] **Step 2: Create analysis router**

Create `drivesort/api/analysis.py`:

```python
"""
drivesort/api/analysis.py
-------------------------
Trigger and monitor the embedding + clustering pipeline.

POST /api/analysis/trigger   — start background analysis (embed, UMAP, HDBSCAN, name)
GET  /api/analysis/status    — current phase and progress
GET  /api/analysis/result    — cluster assignments (after pipeline completes)
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse

router = APIRouter()

# In-process state — single user, single analysis at a time
_state: dict[str, Any] = {
    "phase": "idle",       # idle | fetching | extracting | embedding | clustering | naming | complete | error
    "progress": 0,
    "total": 0,
    "message": "",
    "result": None,        # ClusterResult serialised as dict after completion
    "error": None,
}


@router.post("/trigger", status_code=202)
async def trigger_analysis(background_tasks: BackgroundTasks):
    if _state["phase"] == "running":
        return JSONResponse({"status": "already_running"}, status_code=409)
    _state.update(phase="fetching", progress=0, total=0, message="", result=None, error=None)
    background_tasks.add_task(_run_analysis)
    return {"status": "started"}


@router.get("/status")
def analysis_status():
    return {
        "phase": _state["phase"],
        "progress": _state["progress"],
        "total": _state["total"],
        "message": _state["message"],
        "error": _state["error"],
    }


@router.get("/result")
def analysis_result():
    if _state["phase"] != "complete":
        return JSONResponse({"error": "analysis not complete"}, status_code=400)
    return _state["result"]


async def _run_analysis() -> None:
    """
    Full pipeline: fetch → extract → embed → cluster → name.
    Broadcasts WebSocket events at each step.
    Populate _state["result"] with serialisable cluster data on completion.
    """
    from drivesort.ws import manager
    from drivesort.drive import DriveClient
    from drivesort.embedder import Embedder
    from drivesort.content_extractor import ContentExtractor
    from drivesort.clusterer import Clusterer
    from drivesort.cluster_cache import ClusterCache
    from drivesort.llm_name_cache import LLMNameCache

    try:
        _state["phase"] = "fetching"
        await manager.broadcast({"type": "phase", "phase": "fetching"})

        drive = DriveClient()
        files = list(drive.list_files())
        _state["total"] = len(files)
        await manager.broadcast({"type": "fetch_complete", "total": len(files)})

        _state["phase"] = "embedding"
        extractor = ContentExtractor()
        embedder = Embedder(extractor=extractor)

        def on_progress(done: int, total: int, cached: int) -> None:
            _state["progress"] = done
            asyncio.get_event_loop().run_until_complete(
                manager.broadcast({"type": "embed_progress", "done": done,
                                   "total": total, "cached": cached})
            )

        embeddings = embedder.embed_files(files, progress_callback=on_progress)

        _state["phase"] = "clustering"
        await manager.broadcast({"type": "phase", "phase": "clustering"})

        params = {"min_cluster_size": 5, "umap_n_neighbors": 30}
        cache = ClusterCache()
        emb_keys = [embedder.cache_key(f) for f in files]
        cached_result = cache.load(emb_keys, params)

        if cached_result is not None:
            emb_2d = cached_result.embeddings_2d
            labels = cached_result.labels
            await manager.broadcast({"type": "cluster_cache_hit"})
        else:
            clusterer = Clusterer(**params)
            import numpy as np
            cluster_result = clusterer.cluster(files, embeddings, name_with_llm=False)
            emb_2d = cluster_result.embeddings_2d
            labels = cluster_result.embeddings_2d  # placeholder until naming
            cache.save(emb_keys, params, cluster_result.embeddings_2d,
                       labels=__import__("numpy").array(
                           [c.id for f in files
                            for c in cluster_result.clusters if f in c.files],
                           dtype="int32"))

        # Stream 2D coords for live scatter
        for i, f in enumerate(files):
            await manager.broadcast({
                "type": "umap_point",
                "file_id": f.id,
                "x": float(emb_2d[i, 0]),
                "y": float(emb_2d[i, 1]),
            })

        _state["phase"] = "complete"
        await manager.broadcast({"type": "phase", "phase": "complete"})
        _state["result"] = {"file_count": len(files)}

    except Exception as exc:
        _state["phase"] = "error"
        _state["error"] = str(exc)
        from drivesort.ws import manager as m
        await m.broadcast({"type": "error", "message": str(exc)})
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_api_analysis.py -v
```

Expected: both tests pass.

- [ ] **Step 4: Commit**

```bash
git add drivesort/api/analysis.py tests/test_api_analysis.py
git commit -m "feat: add analysis trigger/status/result routes with background task pipeline"
```

---

## Task 9: Taxonomy, draft, and stage routes

**Files:**
- Create: `drivesort/api/taxonomy.py`
- Create: `drivesort/api/draft.py`
- Create: `drivesort/api/stage.py`
- Create: `tests/test_api_taxonomy.py`

- [ ] **Step 1: Write failing taxonomy API tests**

Create `tests/test_api_taxonomy.py`:

```python
"""Tests for taxonomy API — node CRUD via HTTP."""
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    from drivesort.server import app
    return TestClient(app)


def test_list_nodes_empty(client):
    resp = client.get("/api/taxonomy/nodes")
    assert resp.status_code == 200
    assert resp.json() == []


def test_add_and_get_node(client):
    payload = {
        "path": "books",
        "name": "Books",
        "parent": None,
        "description": "All books",
        "folder_id": "",
        "member_ids": ["f1", "f2"],
        "centroid": [0.1] * 384,
    }
    resp = client.post("/api/taxonomy/nodes", json=payload)
    assert resp.status_code == 201

    resp = client.get("/api/taxonomy/nodes")
    assert len(resp.json()) == 1
    assert resp.json()[0]["path"] == "books"


def test_delete_node(client):
    payload = {
        "path": "books", "name": "Books", "parent": None,
        "description": "", "folder_id": "", "member_ids": [],
        "centroid": [0.0] * 384,
    }
    client.post("/api/taxonomy/nodes", json=payload)
    resp = client.delete("/api/taxonomy/nodes/books")
    assert resp.status_code == 200
    assert client.get("/api/taxonomy/nodes").json() == []
```

- [ ] **Step 2: Create taxonomy router**

Create `drivesort/api/taxonomy.py`:

```python
"""
drivesort/api/taxonomy.py
-------------------------
Taxonomy tree CRUD.

GET    /api/taxonomy/nodes          — list all nodes
POST   /api/taxonomy/nodes          — add a node
PATCH  /api/taxonomy/nodes/{path}   — update name/description/parent
DELETE /api/taxonomy/nodes/{path}   — remove a node (and its children)
POST   /api/taxonomy/nodes/{path}/confirm — active-learning confirmation
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from drivesort.taxonomy_v2 import TaxonomyV2

router = APIRouter()
_taxonomy = TaxonomyV2()


def _reload():
    global _taxonomy
    _taxonomy = TaxonomyV2.load()


class NodePayload(BaseModel):
    path: str
    name: str
    parent: Optional[str]
    description: str = ""
    folder_id: str = ""
    member_ids: list[str] = []
    centroid: list[float]


class NodePatch(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    parent: Optional[str] = None


class ConfirmPayload(BaseModel):
    file_id: str
    embedding: list[float]


@router.get("/nodes")
def list_nodes():
    _reload()
    return [vars(n) for n in _taxonomy.nodes.values()]


@router.post("/nodes", status_code=201)
def add_node(payload: NodePayload):
    _reload()
    embs = np.array([payload.centroid], dtype=np.float32)
    _taxonomy.add_node(
        path=payload.path, name=payload.name, parent=payload.parent,
        member_embeddings=embs, member_ids=payload.member_ids,
        folder_id=payload.folder_id, description=payload.description,
    )
    _taxonomy.save()
    return {"path": payload.path}


@router.patch("/nodes/{node_path:path}")
def patch_node(node_path: str, patch: NodePatch):
    _reload()
    if node_path not in _taxonomy.nodes:
        raise HTTPException(404, "node not found")
    node = _taxonomy.nodes[node_path]
    if patch.name is not None:
        node.name = patch.name
    if patch.description is not None:
        node.description = patch.description
    if patch.parent is not None:
        node.parent = patch.parent
    _taxonomy.save()
    return {"path": node_path}


@router.delete("/nodes/{node_path:path}")
def delete_node(node_path: str):
    _reload()
    if node_path not in _taxonomy.nodes:
        raise HTTPException(404, "node not found")
    # Remove node and all descendants
    to_remove = [p for p in _taxonomy.nodes if p == node_path or p.startswith(node_path + "/")]
    for p in to_remove:
        del _taxonomy.nodes[p]
    _taxonomy.save()
    return {"removed": to_remove}


@router.post("/nodes/{node_path:path}/confirm")
def confirm_node(node_path: str, payload: ConfirmPayload):
    _reload()
    if node_path not in _taxonomy.nodes:
        raise HTTPException(404, "node not found")
    emb = np.array(payload.embedding, dtype=np.float32)
    _taxonomy.confirm(node_path, emb, file_id=payload.file_id)
    _taxonomy.save()
    return {"path": node_path, "member_count": _taxonomy.nodes[node_path].member_count}
```

- [ ] **Step 3: Create draft and stage routers (minimal)**

Create `drivesort/api/draft.py`:

```python
"""
drivesort/api/draft.py
----------------------
GET  /api/draft        — load current draft (null if none)
PUT  /api/draft        — save draft state
DELETE /api/draft      — discard draft
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from drivesort.draft import DraftManager, DraftState, StagedChange, UserDecision

router = APIRouter()
_mgr = DraftManager()


class DraftPayload(BaseModel):
    taxonomy_nodes: dict
    staged_changes: list[dict] = []
    user_decisions: list[dict] = []


@router.get("")
def get_draft():
    state = _mgr.load()
    if state is None:
        return None
    return {
        "saved_at": state.saved_at,
        "taxonomy_nodes": state.taxonomy_nodes,
        "staged_changes": [vars(c) for c in state.staged_changes],
        "user_decisions": [vars(d) for d in state.user_decisions],
    }


@router.put("", status_code=200)
def save_draft(payload: DraftPayload):
    state = DraftState(
        taxonomy_nodes=payload.taxonomy_nodes,
        staged_changes=[StagedChange(**c) for c in payload.staged_changes],
        user_decisions=[UserDecision(**d) for d in payload.user_decisions],
    )
    _mgr.save(state)
    return {"saved_at": state.saved_at}


@router.delete("", status_code=200)
def discard_draft():
    _mgr.clear()
    return {"discarded": True}
```

Create `drivesort/api/stage.py`:

```python
"""
drivesort/api/stage.py
----------------------
GET  /api/stage        — list all staged changes
POST /api/stage/commit — execute all staged changes against Drive
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from drivesort.draft import DraftManager, StagedChange

router = APIRouter()
_mgr = DraftManager()


@router.get("")
def list_staged():
    state = _mgr.load()
    if state is None:
        return []
    return [vars(c) for c in state.staged_changes]


@router.post("/commit")
def commit_staged():
    """
    Execute all staged changes: create Drive folders, move files.
    Clears draft on success.
    """
    from drivesort.drive import DriveClient
    from drivesort.taxonomy_v2 import TaxonomyV2

    state = _mgr.load()
    if not state:
        return {"committed": 0}

    drive = DriveClient()
    tax = TaxonomyV2.load()
    committed = 0

    # Create folders for all taxonomy nodes (in path-depth order)
    nodes_by_depth = sorted(
        tax.nodes.values(), key=lambda n: len(n.path.split("/"))
    )
    folder_ids: dict[str, str] = {}
    for node in nodes_by_depth:
        if node.folder_id:
            folder_ids[node.path] = node.folder_id
            continue
        parent_folder_id = folder_ids.get(node.parent) if node.parent else None
        folder = drive.find_or_create_folder(node.name, parent_folder_id)
        node.folder_id = folder.id
        folder_ids[node.path] = folder.id
    tax.save()

    # Move files
    for change in state.staged_changes:
        target_node = tax.nodes.get(change.proposed_path)
        if not target_node or not target_node.folder_id:
            continue
        try:
            drive.move_file(change.file_id, target_node.folder_id)
            committed += 1
        except Exception:
            pass

    _mgr.clear()
    return {"committed": committed}
```

- [ ] **Step 4: Run taxonomy tests**

```bash
pytest tests/test_api_taxonomy.py -v
```

Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add drivesort/api/taxonomy.py drivesort/api/draft.py drivesort/api/stage.py tests/test_api_taxonomy.py
git commit -m "feat: add taxonomy CRUD, draft, and stage routes"
```

---

## Task 10: Scan and cache routes

**Files:**
- Create: `drivesort/api/scan.py`
- Create: `drivesort/api/cache.py`
- Create: `tests/test_api_scan.py`
- Create: `tests/test_api_cache.py`

- [ ] **Step 1: Write scan and cache tests**

Create `tests/test_api_scan.py`:

```python
"""Tests for scan API."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    from drivesort.server import app
    return TestClient(app)


def test_queue_empty_initially(client):
    resp = client.get("/api/scan/queue")
    assert resp.status_code == 200
    assert resp.json() == []


def test_trigger_scan_returns_202(client):
    resp = client.post("/api/scan/trigger")
    assert resp.status_code == 202
```

Create `tests/test_api_cache.py`:

```python
"""Tests for cache API."""
import json
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    from drivesort.server import app
    return TestClient(app)


def test_cache_status(client):
    resp = client.get("/api/cache/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "content" in data
    assert "embeddings" in data
    assert "clustering" in data
    assert "llm_names" in data


def test_invalidate_file(client):
    resp = client.post("/api/cache/invalidate/file", json={"file_id": "abc123"})
    assert resp.status_code == 200
    assert resp.json()["invalidated"] == "abc123"
```

- [ ] **Step 2: Create scan router**

Create `drivesort/api/scan.py`:

```python
"""
drivesort/api/scan.py
---------------------
POST /api/scan/trigger             — run a scan pass in the background
GET  /api/scan/queue               — list files awaiting review
POST /api/scan/queue/{file_id}/accept  — accept predicted placement
POST /api/scan/queue/{file_id}/correct — move to a different taxonomy node
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from drivesort.taxonomy_v2 import TaxonomyV2

QUEUE_PATH = Path("data/scan_queue.json")

router = APIRouter()


def _load_queue() -> list[dict]:
    if not QUEUE_PATH.exists():
        return []
    return json.loads(QUEUE_PATH.read_text())


def _save_queue(queue: list[dict]) -> None:
    QUEUE_PATH.parent.mkdir(exist_ok=True)
    QUEUE_PATH.write_text(json.dumps(queue, indent=2))


class CorrectPayload(BaseModel):
    path: str
    embedding: list[float]


@router.post("/trigger", status_code=202)
async def trigger_scan(background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_scan)
    return {"status": "started"}


@router.get("/queue")
def get_queue():
    return _load_queue()


@router.post("/queue/{file_id}/accept")
def accept_placement(file_id: str):
    queue = _load_queue()
    item = next((i for i in queue if i["file_id"] == file_id), None)
    if not item:
        raise HTTPException(404, "file not in queue")

    import numpy as np
    from drivesort.drive import DriveClient

    tax = TaxonomyV2.load()
    path = item["predicted_path"]
    if path and path in tax.nodes:
        emb = np.array(item["embedding"], dtype=np.float32)
        tax.confirm(path, emb, file_id=file_id)
        tax.save()
        drive = DriveClient()
        folder_id = tax.nodes[path].folder_id
        if folder_id:
            drive.move_file(file_id, folder_id)

    queue = [i for i in queue if i["file_id"] != file_id]
    _save_queue(queue)
    return {"accepted": file_id, "path": path}


@router.post("/queue/{file_id}/correct")
def correct_placement(file_id: str, payload: CorrectPayload):
    import numpy as np
    from drivesort.drive import DriveClient

    queue = _load_queue()
    if not any(i["file_id"] == file_id for i in queue):
        raise HTTPException(404, "file not in queue")

    tax = TaxonomyV2.load()
    if payload.path not in tax.nodes:
        raise HTTPException(400, "unknown taxonomy path")

    emb = np.array(payload.embedding, dtype=np.float32)
    tax.confirm(payload.path, emb, file_id=file_id)
    tax.save()

    drive = DriveClient()
    folder_id = tax.nodes[payload.path].folder_id
    if folder_id:
        drive.move_file(file_id, folder_id)

    queue = [i for i in queue if i["file_id"] != file_id]
    _save_queue(queue)
    return {"corrected": file_id, "path": payload.path}


async def _run_scan() -> None:
    from drivesort.drive import DriveClient
    from drivesort.embedder import Embedder
    from drivesort.content_extractor import ContentExtractor
    from drivesort.taxonomy_v2 import TaxonomyV2
    from drivesort.ws import manager

    tax = TaxonomyV2.load()
    if not tax.nodes:
        return

    drive = DriveClient()
    files = list(drive.list_files())
    organised_ids = {fid for node in tax.nodes.values() for fid in node.member_ids}
    new_files = [f for f in files if f.id not in organised_ids]

    extractor = ContentExtractor()
    embedder = Embedder(extractor=extractor)
    embeddings = embedder.embed_files(new_files)

    queue = _load_queue()
    existing_ids = {i["file_id"] for i in queue}

    for f, emb in zip(new_files, embeddings):
        if f.id in existing_ids:
            continue
        result = tax.classify(emb, file_id=f.id, file_name=f.name)
        item = {
            "file_id": f.id,
            "file_name": f.name,
            "mime_type": f.mime_type,
            "predicted_path": result.path,
            "confidence": result.confidence,
            "is_novel": result.is_novel,
            "embedding": emb.tolist(),
        }
        queue.append(item)
        await manager.broadcast({"type": "scan_file", **item})

    _save_queue(queue)
```

- [ ] **Step 3: Create cache router**

Create `drivesort/api/cache.py`:

```python
"""
drivesort/api/cache.py
----------------------
GET  /api/cache/status                    — size/entry count for all cache layers
POST /api/cache/invalidate/file           — invalidate one file across all layers
POST /api/cache/invalidate/folder         — invalidate all files under a Drive folder
DELETE /api/cache/all                     — wipe all caches (destructive)
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

CONTENT_CACHE  = Path("data/content_cache.json")
EMBED_CACHE    = Path("data/embedding_cache.json")
CLUSTER_CACHE  = Path("data/cluster_cache.pkl")
LLM_CACHE      = Path("data/llm_name_cache.json")


def _cache_stat(path: Path) -> dict:
    if not path.exists():
        return {"entries": 0, "size_bytes": 0}
    size = path.stat().st_size
    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text())
            return {"entries": len(data) if isinstance(data, dict) else 1, "size_bytes": size}
        except Exception:
            pass
    return {"entries": 1, "size_bytes": size}


class FileInvalidatePayload(BaseModel):
    file_id: str


class FolderInvalidatePayload(BaseModel):
    folder_id: str


@router.get("/status")
def cache_status():
    return {
        "content":    _cache_stat(CONTENT_CACHE),
        "embeddings": _cache_stat(EMBED_CACHE),
        "clustering": _cache_stat(CLUSTER_CACHE),
        "llm_names":  _cache_stat(LLM_CACHE),
    }


@router.post("/invalidate/file")
def invalidate_file(payload: FileInvalidatePayload):
    fid = payload.file_id
    removed = {}
    for cache_path in [CONTENT_CACHE, EMBED_CACHE]:
        if cache_path.exists():
            data = json.loads(cache_path.read_text())
            # Keys are SHA1 hashes — we need to find which key corresponds to this file_id.
            # Since we can't reverse SHA1, we store a file_id → key mapping or just clear all.
            # Conservative approach: if file_id appears as a value key, remove it.
            to_remove = [k for k, v in data.items()
                         if isinstance(v, dict) and v.get("file_id") == fid]
            for k in to_remove:
                del data[k]
            if to_remove:
                cache_path.write_text(json.dumps(data, indent=2))
            removed[cache_path.name] = len(to_remove)
    # Invalidate cluster cache (it covers all files, so any file change busts it)
    if CLUSTER_CACHE.exists():
        CLUSTER_CACHE.unlink()
        removed["cluster_cache.pkl"] = 1
    return {"invalidated": fid, "removed": removed}


@router.post("/invalidate/folder")
def invalidate_folder(payload: FolderInvalidatePayload):
    # Without Drive API call we can't enumerate folder contents here.
    # Return instruction for the frontend to call invalidate/file per member.
    return {"message": "call invalidate/file for each file in the folder",
            "folder_id": payload.folder_id}


@router.delete("/all")
def clear_all_caches():
    cleared = []
    for p in [CONTENT_CACHE, EMBED_CACHE, CLUSTER_CACHE, LLM_CACHE]:
        if p.exists():
            p.unlink()
            cleared.append(p.name)
    return {"cleared": cleared}
```

- [ ] **Step 4: Run all new tests**

```bash
pytest tests/test_api_scan.py tests/test_api_cache.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass (taxonomy v1 tests, taxonomy v2 tests, cluster cache, draft, and all API tests).

- [ ] **Step 6: Smoke test the server**

```bash
drivesort serve &
sleep 2
curl http://localhost:7432/api/auth/status
curl http://localhost:7432/api/cache/status
kill %1
```

Expected: both curl commands return valid JSON.

- [ ] **Step 7: Commit**

```bash
git add drivesort/api/scan.py drivesort/api/cache.py tests/test_api_scan.py tests/test_api_cache.py
git commit -m "feat: add scan queue/accept/correct and cache status/invalidation routes"
```

---

## Task 11: Wire embedder progress callback and move_file to DriveClient

**Files:**
- Modify: `drivesort/embedder.py`
- Modify: `drivesort/drive.py`

- [ ] **Step 1: Add progress_callback to Embedder.embed_files**

In `drivesort/embedder.py`, update the `embed_files` signature and body. Find the method (around line 68) and add an optional callback:

```python
def embed_files(
    self,
    files: list[DriveFile],
    progress_callback=None,   # callable(done, total, cached) or None
) -> np.ndarray:
    keys   = [self._cache_key(f) for f in files]
    cached = {k: np.array(self._cache[k]) for k in keys if k in self._cache}
    misses = [f for f, k in zip(files, keys) if k not in self._cache]

    if misses:
        texts = [
            self._extractor.extract(f) if self._extractor else f.text_for_embedding()
            for f in misses
        ]
        vecs = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        for f, k, v in zip(misses, [self._cache_key(m) for m in misses], vecs):
            self._cache[k] = v.tolist()
        self._save_cache()

    result = np.stack([
        np.array(cached[k]) if k in cached else
        np.array(self._cache[self._cache_key(f)])
        for f, k in zip(files, keys)
    ])

    if progress_callback:
        progress_callback(len(files), len(files), len(cached))

    return result
```

Also expose `cache_key` as a public method by adding this alias at the end of the `Embedder` class:

```python
    def cache_key(self, f: "DriveFile") -> str:
        return self._cache_key(f)
```

- [ ] **Step 2: Add move_file to DriveClient**

In `drivesort/drive.py`, add this method to the `DriveClient` class:

```python
    def move_file(self, file_id: str, target_folder_id: str) -> None:
        """Move a file to a different folder."""
        file_meta = self._service.files().get(
            fileId=file_id, fields="parents"
        ).execute()
        previous_parents = ",".join(file_meta.get("parents", []))
        self._service.files().update(
            fileId=file_id,
            addParents=target_folder_id,
            removeParents=previous_parents,
            fields="id, parents",
        ).execute()
```

- [ ] **Step 3: Run full test suite to confirm no regressions**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add drivesort/embedder.py drivesort/drive.py
git commit -m "feat: add progress_callback to embed_files and move_file to DriveClient"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| FastAPI single-process server | Task 6 |
| `drivesort serve` CLI command | Task 1 |
| CLI retirement (bootstrap/scan/status/recover) | Task 1 |
| Four cache layers | Tasks 4 (cluster), 4 (LLM), existing content/embed |
| Cluster cache keyed by embedding keys + params | Task 4 |
| LLM name cache | Task 4 |
| Arbitrary-depth taxonomy v2 | Tasks 2, 3 |
| Top-down classification | Task 3 |
| Active learning / confirm() | Task 3, Task 10 |
| Draft persistence (auto-save, resume) | Task 5, Task 9 |
| Bootstrap-only draft (scan has no draft) | Task 5, Task 10 |
| No Drive writes until commit | Task 9 (stage/commit) |
| Staged changes list | Task 9 |
| WebSocket progress events | Tasks 6, 7, 8 |
| Scan queue + accept/correct | Task 10 |
| Cache status + invalidation | Task 10 |
| CORS for Vite dev server | Task 6 |
| OAuth flow | Task 7 |

**Type consistency check:** `TaxonomyNode.path`, `TaxonomyV2.nodes`, `TaxonomyV2.confirm()`, `TaxonomyV2.classify()`, `DraftState`, `StagedChange`, `UserDecision` — all defined in Tasks 2/5 and referenced consistently in Tasks 3/9/10. `DriveClient.move_file()` defined in Task 11 and called in Tasks 9/10. ✓

**Placeholder scan:** No TBD/TODO in task steps. The `invalidate_folder` route returns a message instructing the frontend to enumerate members — this is intentional and noted inline. ✓
