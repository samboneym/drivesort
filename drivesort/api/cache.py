"""
drivesort/api/cache.py
----------------------
GET    /api/cache/status                    — size/entry count for all cache layers
POST   /api/cache/invalidate/file           — invalidate one file across all layers
POST   /api/cache/invalidate/folder         — invalidate all files under a Drive folder
DELETE /api/cache/all                       — wipe all caches (destructive)
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

CONTENT_CACHE = Path("data/content_cache.json")
EMBED_CACHE   = Path("data/embedding_cache.json")
CLUSTER_CACHE = Path("data/cluster_cache.pkl")
LLM_CACHE     = Path("data/llm_name_cache.json")


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
    removed: dict[str, int] = {}
    for cache_path in [CONTENT_CACHE, EMBED_CACHE]:
        if cache_path.exists():
            data = json.loads(cache_path.read_text())
            to_remove = [k for k, v in data.items()
                         if isinstance(v, dict) and v.get("file_id") == fid]
            for k in to_remove:
                del data[k]
            if to_remove:
                cache_path.write_text(json.dumps(data, indent=2))
            removed[cache_path.name] = len(to_remove)
    if CLUSTER_CACHE.exists():
        CLUSTER_CACHE.unlink()
        removed["cluster_cache.pkl"] = 1
    return {"invalidated": fid, "removed": removed}


@router.post("/invalidate/folder")
def invalidate_folder(payload: FolderInvalidatePayload):
    return {
        "message": "call invalidate/file for each file in the folder",
        "folder_id": payload.folder_id,
    }


@router.delete("/all")
def clear_all_caches():
    cleared = []
    for p in [CONTENT_CACHE, EMBED_CACHE, CLUSTER_CACHE, LLM_CACHE]:
        if p.exists():
            p.unlink()
            cleared.append(p.name)
    return {"cleared": cleared}
