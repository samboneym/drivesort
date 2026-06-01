"""
drivesort/clusterer.py
----------------------
Takes a matrix of file embeddings and discovers candidate categories.

Pipeline:
  1. UMAP  — reduce to 2D for stable density-based clustering
  2. HDBSCAN — find clusters of arbitrary shape; mark outliers as -1
  3. LLM   — name each cluster from a sample of its member files
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import umap
import hdbscan
import ollama
from rich.console import Console

from .drive import DriveFile

console = Console()

OLLAMA_MODEL = "phi3:mini"   # ~2 GB, fast, works well zero-shot


@dataclass
class Cluster:
    id: int
    files: list[DriveFile]
    suggested_name: str = ""
    suggested_description: str = ""
    llm_confidence: float = 0.0
    # Set by human during review:
    accepted_name: Optional[str] = None
    merged_into: Optional[str] = None
    rejected: bool = False

    @property
    def is_outlier(self) -> bool:
        return self.id == -1

    @property
    def size(self) -> int:
        return len(self.files)


@dataclass
class ClusterResult:
    clusters: list[Cluster]
    embeddings_2d: np.ndarray          # for visualisation
    outlier_files: list[DriveFile]     # HDBSCAN noise points


class Clusterer:
    """
    Discovers latent categories in a Drive file corpus without any labels.

    Parameters
    ----------
    min_cluster_size : int
        Minimum files to form a cluster. Lower = finer-grained, higher = coarser.
        3–5 is a good starting point for personal Drive.
    umap_n_neighbors : int
        Controls the local vs global trade-off in UMAP.
        15 is the default; lower values produce tighter local clusters.
    novelty_threshold : float
        Cosine distance above which a file is considered "novel" at inference time.
    """

    def __init__(
        self,
        min_cluster_size: int = 3,
        umap_n_neighbors: int = 15,
        novelty_threshold: float = 0.45,
        ollama_model: str = OLLAMA_MODEL,
    ) -> None:
        self._min_cluster_size  = min_cluster_size
        self._umap_neighbors    = umap_n_neighbors
        self._novelty_threshold = novelty_threshold
        self._ollama_model      = ollama_model

    # ------------------------------------------------------------------
    # Bootstrap clustering (run once)
    # ------------------------------------------------------------------

    def cluster(
        self,
        files: list[DriveFile],
        embeddings: np.ndarray,
        name_with_llm: bool = True,
    ) -> ClusterResult:
        """
        Cluster files and optionally ask the LLM to name each group.

        Returns a ClusterResult containing one Cluster per discovered group
        plus a list of outlier files that didn't fit any cluster.
        """
        # 1. Reduce dimensionality
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=self._umap_neighbors,
            metric="cosine",
            random_state=42,
        )
        with console.status("[cyan]Reducing dimensions with UMAP…[/cyan]"):
            emb_2d = reducer.fit_transform(embeddings)

        # 2. Cluster in 2D space (HDBSCAN performs better here than on raw high-D)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self._min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        with console.status("[cyan]Finding clusters with HDBSCAN…[/cyan]"):
            labels = clusterer.fit_predict(emb_2d)

        # 3. Group files by label
        cluster_map: dict[int, list[DriveFile]] = {}
        for f, label in zip(files, labels):
            cluster_map.setdefault(int(label), []).append(f)

        outlier_files = cluster_map.pop(-1, [])

        clusters = [
            Cluster(id=cid, files=members)
            for cid, members in sorted(cluster_map.items())
        ]

        # 4. Name clusters with LLM
        if name_with_llm:
            for cluster in clusters:
                with console.status(f"[cyan]Naming cluster {cluster.id + 1}/{len(clusters)} with LLM…[/cyan]"):
                    self._name_cluster(cluster)

        return ClusterResult(
            clusters=clusters,
            embeddings_2d=emb_2d,
            outlier_files=outlier_files,
        )

    # ------------------------------------------------------------------
    # LLM naming
    # ------------------------------------------------------------------

    def _name_cluster(self, cluster: Cluster) -> None:
        """Ask the local LLM to suggest a folder name for a cluster."""
        sample = cluster.files[:8]
        file_list = "\n".join(
            f"- {f.name}  [{f.mime_type.split('/')[-1]}]"
            for f in sample
        )
        extra = f"  … and {cluster.size - 8} more" if cluster.size > 8 else ""

        prompt = f"""You are helping organise a personal Google Drive.
These files were automatically grouped together because they share semantic similarity.

Files in group:
{file_list}{extra}

Suggest a Google Drive folder name for these files.
Rules:
- Short (2-4 words), title case, consistent with personal Drive naming
- If they are clearly different topics, say so in the rationale
- confidence should reflect how coherent the group is (0.0 = random mix, 1.0 = obviously one topic)

Reply ONLY with valid JSON:
{{"name": "...", "description": "one sentence on what belongs here", "confidence": 0.0}}"""

        try:
            resp = ollama.chat(
                model=self._ollama_model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
            )
            result = json.loads(resp["message"]["content"])
            cluster.suggested_name        = result.get("name", f"Group {cluster.id}")
            cluster.suggested_description = result.get("description", "")
            cluster.llm_confidence        = float(result.get("confidence", 0.5))
        except Exception as e:
            # Graceful fallback if Ollama isn't running
            cluster.suggested_name        = f"Group {cluster.id}"
            cluster.suggested_description = f"({len(cluster.files)} files)"
            cluster.llm_confidence        = 0.0

    def suggest_new_category(
        self,
        file: DriveFile,
        existing_folders: list[str],
    ) -> dict:
        """
        Called when a new file doesn't fit any existing category.
        Returns a suggestion for a new folder, or recommends Archive.
        """
        prompt = f"""A new Google Drive file doesn't fit any existing folder.

Existing folders: {json.dumps(existing_folders)}

New file:
- Name: {file.name}
- Type: {file.mime_type}
- Content preview: {file.snippet[:300] if file.snippet else 'none'}

Should this file get a new dedicated folder, or does it belong in Archive?

Reply ONLY with valid JSON:
{{
  "new_folder": "<folder name, or null if Archive is fine>",
  "rationale": "...",
  "similar_files_hint": "what other files would belong in this new folder"
}}"""

        try:
            resp = ollama.chat(
                model=self._ollama_model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
            )
            return json.loads(resp["message"]["content"])
        except Exception:
            return {"new_folder": None, "rationale": "LLM unavailable", "similar_files_hint": ""}
