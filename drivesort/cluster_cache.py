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
