"""
drivesort/taxonomy.py
---------------------
The taxonomy is the single source of truth after bootstrapping.
It maps folder names → centroids (mean embeddings of confirmed members).

Responsibilities:
  - Persist the taxonomy to disk
  - Classify new files by nearest centroid
  - Detect out-of-distribution (OOD) files that don't fit any category
  - Update centroids incrementally as new files are confirmed
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

TAXONOMY_PATH = Path("data/taxonomy.json")
OOD_LOG_PATH  = Path("data/novel_files.json")

SUB_CLUSTER_MIN_FILES = 6  # minimum cluster size to offer sub-clustering in bootstrap


@dataclass
class CategoryEntry:
    name: str
    description: str
    folder_id: str            # Google Drive folder ID (set after folder is created)
    centroid: list[float]     # mean embedding of confirmed member files
    member_count: int = 0
    member_ids: list[str] = field(default_factory=list)  # Drive file IDs
    parent_name: Optional[str] = None  # None = top-level category

    def centroid_array(self) -> np.ndarray:
        return np.array(self.centroid, dtype=np.float32)


@dataclass
class ClassificationResult:
    file_id: str
    file_name: str
    category: Optional[str]       # None = novel / OOD
    confidence: float             # 1 - cosine_distance to nearest centroid
    distance: float
    is_novel: bool
    runner_up: Optional[str] = None
    runner_up_confidence: float = 0.0


class Taxonomy:
    """
    Manages the set of known categories and their embedding centroids.

    After bootstrapping, call `classify()` on each new file.
    If `result.is_novel`, surface it to the user for review.
    Call `confirm()` to fold a newly labelled file into a category's centroid.
    """

    def __init__(
        self,
        path: Path = TAXONOMY_PATH,
        novelty_threshold: float = 0.42,
    ) -> None:
        self._path      = path
        self._threshold = novelty_threshold
        self._categories: dict[str, CategoryEntry] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._path.exists():
            raw = json.loads(self._path.read_text())
            self._categories = {}
            for name, entry in raw.items():
                entry.setdefault("parent_name", None)
                self._categories[name] = CategoryEntry(**entry)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(
                {name: asdict(entry) for name, entry in self._categories.items()},
                indent=2,
            )
        )

    # ------------------------------------------------------------------
    # Building the taxonomy
    # ------------------------------------------------------------------

    def add_category(
        self,
        name: str,
        description: str,
        folder_id: str,
        member_embeddings: np.ndarray,
        member_ids: list[str],
        parent_name: Optional[str] = None,
    ) -> None:
        """Add a new category from bootstrapped cluster members."""
        centroid = member_embeddings.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        self._categories[name] = CategoryEntry(
            name=name,
            description=description,
            folder_id=folder_id,
            centroid=centroid.tolist(),
            member_count=len(member_ids),
            member_ids=member_ids,
            parent_name=parent_name,
        )

    def remove_category(self, name: str) -> None:
        self._categories.pop(name, None)

    def rename_category(self, old_name: str, new_name: str) -> None:
        if old_name in self._categories:
            entry = self._categories.pop(old_name)
            entry.name = new_name
            self._categories[new_name] = entry

    @property
    def category_names(self) -> list[str]:
        return list(self._categories.keys())

    @property
    def categories(self) -> dict[str, CategoryEntry]:
        return self._categories

    def is_empty(self) -> bool:
        return len(self._categories) == 0

    def top_level_categories(self) -> dict[str, CategoryEntry]:
        return {n: e for n, e in self._categories.items() if e.parent_name is None}

    def children_of(self, parent_name: str) -> dict[str, CategoryEntry]:
        return {n: e for n, e in self._categories.items() if e.parent_name == parent_name}

    @property
    def all_folder_ids(self) -> set[str]:
        return {e.folder_id for e in self._categories.values()}

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, embedding: np.ndarray, file_id: str = "", file_name: str = "") -> ClassificationResult:
        """
        Two-stage classification:
          Stage 1 — nearest top-level category; OOD check happens here only.
          Stage 2 — if that category has children, find the nearest child and
                    return it (its folder_id is the move target).
        """
        top_level = self.top_level_categories()
        if not top_level:
            return ClassificationResult(
                file_id=file_id, file_name=file_name,
                category=None, confidence=0.0, distance=1.0, is_novel=True,
            )

        # Stage 1: nearest top-level category
        distances = {
            name: float(1.0 - np.dot(embedding, entry.centroid_array()))
            for name, entry in top_level.items()
        }
        sorted_top = sorted(distances.items(), key=lambda x: x[1])
        best_name, best_dist = sorted_top[0]
        runner_up_name = sorted_top[1][0] if len(sorted_top) > 1 else None
        runner_up_conf = max(0.0, 1.0 - sorted_top[1][1]) if runner_up_name else 0.0

        if best_dist > self._threshold:
            return ClassificationResult(
                file_id=file_id, file_name=file_name,
                category=None,
                confidence=max(0.0, 1.0 - best_dist),
                distance=best_dist,
                is_novel=True,
                runner_up=runner_up_name,
                runner_up_confidence=runner_up_conf,
            )

        # Stage 2: route to nearest child if any exist
        children = self.children_of(best_name)
        if children:
            child_distances = {
                name: float(1.0 - np.dot(embedding, entry.centroid_array()))
                for name, entry in children.items()
            }
            best_child_name, best_child_dist = min(child_distances.items(), key=lambda x: x[1])
            return ClassificationResult(
                file_id=file_id, file_name=file_name,
                category=best_child_name,
                confidence=max(0.0, 1.0 - best_child_dist),
                distance=best_child_dist,
                is_novel=False,
                runner_up=runner_up_name,
                runner_up_confidence=runner_up_conf,
            )

        return ClassificationResult(
            file_id=file_id, file_name=file_name,
            category=best_name,
            confidence=max(0.0, 1.0 - best_dist),
            distance=best_dist,
            is_novel=False,
            runner_up=runner_up_name,
            runner_up_confidence=runner_up_conf,
        )

    # ------------------------------------------------------------------
    # Incremental learning
    # ------------------------------------------------------------------

    def confirm(self, category_name: str, file_id: str, embedding: np.ndarray) -> None:
        """
        Update a category's centroid with a newly confirmed file.
        Uses a running mean so we never need to store all embeddings.
        """
        if category_name not in self._categories:
            return

        entry = self._categories[category_name]
        n     = entry.member_count
        old_c = entry.centroid_array()

        # Incremental mean: new_centroid = (n * old + new) / (n + 1)
        new_c = (old_c * n + embedding) / (n + 1)
        new_c = new_c / (np.linalg.norm(new_c) + 1e-8)

        entry.centroid     = new_c.tolist()
        entry.member_count = n + 1
        if file_id not in entry.member_ids:
            entry.member_ids.append(file_id)

    # ------------------------------------------------------------------
    # Novel file log (for periodic re-clustering)
    # ------------------------------------------------------------------

    def log_novel_file(self, file_id: str, file_name: str, embedding: np.ndarray) -> None:
        """Persist novel files so they can be periodically re-clustered."""
        log = []
        if OOD_LOG_PATH.exists():
            log = json.loads(OOD_LOG_PATH.read_text())

        # Avoid duplicates
        if not any(e["id"] == file_id for e in log):
            log.append({
                "id":        file_id,
                "name":      file_name,
                "embedding": embedding.tolist(),
            })
            OOD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            OOD_LOG_PATH.write_text(json.dumps(log, indent=2))

    def load_novel_files(self) -> tuple[list[dict], np.ndarray]:
        """Load accumulated novel files for periodic re-clustering."""
        if not OOD_LOG_PATH.exists():
            return [], np.empty((0,), dtype=np.float32)
        log = json.loads(OOD_LOG_PATH.read_text())
        if not log:
            return [], np.empty((0,), dtype=np.float32)
        embeddings = np.array([e["embedding"] for e in log], dtype=np.float32)
        return log, embeddings

    def clear_novel_log(self) -> None:
        if OOD_LOG_PATH.exists():
            OOD_LOG_PATH.unlink()
