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
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

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
    from drivesort.taxonomy_v2 import TaxonomyV2

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
    from drivesort.taxonomy_v2 import TaxonomyV2

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
