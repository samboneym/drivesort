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
from dataclasses import dataclass, asdict
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
