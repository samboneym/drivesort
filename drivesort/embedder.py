"""
drivesort/embedder.py
---------------------
Wraps sentence-transformers.
Keeps an on-disk cache keyed by file ID so re-runs don't re-embed unchanged files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from .drive import DriveFile

if TYPE_CHECKING:
    from .content_extractor import ContentExtractor

MODEL_NAME   = "all-MiniLM-L6-v2"   # 80 MB, fast, good quality
CACHE_PATH   = Path("data/embedding_cache.json")
DIM          = 384                    # all-MiniLM-L6-v2 output dimension


class Embedder:
    """
    Embeds DriveFile objects into fixed-size vectors.

    The embedding is derived from: filename + file extension + content snippet.
    No file download is needed — Drive supplies the snippet via its API.

    Embeddings are cached to disk so repeated scans of a large Drive
    don't re-embed files whose metadata hasn't changed.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        cache_path: Path = CACHE_PATH,
        extractor: "ContentExtractor | None" = None,
    ) -> None:
        self._model      = SentenceTransformer(model_name)
        self._cache_path = cache_path
        self._cache: dict[str, list[float]] = self._load_cache()
        self._extractor  = extractor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_files(
        self,
        files: Iterable[DriveFile],
        show_progress: bool = True,
    ) -> tuple[list[DriveFile], np.ndarray]:
        """
        Embed a collection of files.

        Returns
        -------
        files_out : list[DriveFile]
            Same files, in order.
        matrix : np.ndarray, shape (N, DIM)
            One row per file.
        """
        files_list = list(files)
        vectors: list[np.ndarray] = []

        # Split into cache hits and misses
        misses = []
        hit_map: dict[int, np.ndarray] = {}

        for i, f in enumerate(files_list):
            key = self._cache_key(f)
            if key in self._cache:
                hit_map[i] = np.array(self._cache[key], dtype=np.float32)
            else:
                misses.append((i, f, key))

        # Batch-encode misses
        if misses:
            if self._extractor is not None:
                texts = [self._extractor.extract(f) for _, f, _ in misses]
            else:
                texts = [f.text_for_embedding() for _, f, _ in misses]
            batch  = self._model.encode(
                texts,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            for (i, f, key), vec in zip(misses, batch):
                hit_map[i] = vec
                self._cache[key] = vec.tolist()

            self._save_cache()

        matrix = np.stack([hit_map[i] for i in range(len(files_list))], axis=0)
        return files_list, matrix.astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a raw string (used for folder descriptions at query time)."""
        return self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(f: DriveFile) -> str:
        """
        Key is a hash of (id, modified_time, name) so that renaming
        or editing a file invalidates its cached embedding.
        """
        raw = f"{f.id}|{f.modified}|{f.name}"
        return hashlib.sha1(raw.encode()).hexdigest()

    def _load_cache(self) -> dict[str, list[float]]:
        if self._cache_path.exists():
            return json.loads(self._cache_path.read_text())
        return {}

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(self._cache))
