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
    "result": None,        # cluster result serialised as dict after completion
    "error": None,
}


@router.post("/trigger", status_code=202)
async def trigger_analysis(background_tasks: BackgroundTasks):
    if _state["phase"] in ("fetching", "extracting", "embedding", "clustering", "naming"):
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
    Updates _state throughout for polling via /status.
    """
    from drivesort.ws import manager

    try:
        # --- Fetch ---
        _state["phase"] = "fetching"
        await manager.broadcast({"type": "phase", "phase": "fetching"})

        from drivesort.drive import DriveClient
        drive = DriveClient()
        files = list(drive.list_files())
        _state["total"] = len(files)
        await manager.broadcast({"type": "fetch_complete", "total": len(files)})

        # --- Embed ---
        _state["phase"] = "embedding"
        await manager.broadcast({"type": "phase", "phase": "embedding"})

        from drivesort.content_extractor import ContentExtractor
        from drivesort.embedder import Embedder
        from drivesort.cluster_cache import ClusterCache

        extractor = ContentExtractor()
        embedder = Embedder(extractor=extractor)

        files, embeddings = embedder.embed_files(files)

        # --- Cluster ---
        _state["phase"] = "clustering"
        await manager.broadcast({"type": "phase", "phase": "clustering"})

        params = {"min_cluster_size": 5, "umap_n_neighbors": 30}
        cluster_cache = ClusterCache()
        emb_keys = [embedder._cache_key(f) for f in files]
        cached_entry = cluster_cache.load(emb_keys, params)

        if cached_entry is not None:
            emb_2d = cached_entry.embeddings_2d
            labels = cached_entry.labels
            await manager.broadcast({"type": "cluster_cache_hit"})
        else:
            from drivesort.clusterer import Clusterer
            clusterer = Clusterer(
                min_cluster_size=params["min_cluster_size"],
                umap_n_neighbors=params["umap_n_neighbors"],
            )
            cluster_result = clusterer.cluster(files, embeddings, name_with_llm=False)
            emb_2d = cluster_result.embeddings_2d

            import numpy as np
            # Build labels array: cluster id per file (-1 = outlier)
            file_id_to_label: dict[str, int] = {}
            for cluster in cluster_result.clusters:
                for f in cluster.files:
                    file_id_to_label[f.id] = cluster.id
            labels = np.array(
                [file_id_to_label.get(f.id, -1) for f in files], dtype=np.int32
            )
            cluster_cache.save(emb_keys, params, emb_2d, labels)

        # Stream 2D coords for live scatter plot
        for i, f in enumerate(files):
            await manager.broadcast({
                "type": "umap_point",
                "file_id": f.id,
                "x": float(emb_2d[i, 0]),
                "y": float(emb_2d[i, 1]),
                "label": int(labels[i]),
            })

        _state["phase"] = "complete"
        _state["result"] = {
            "file_count": len(files),
            "cluster_count": int((labels >= 0).sum()),
        }
        await manager.broadcast({"type": "phase", "phase": "complete"})

    except Exception as exc:
        _state["phase"] = "error"
        _state["error"] = str(exc)
        from drivesort.ws import manager as m
        await m.broadcast({"type": "error", "message": str(exc)})
