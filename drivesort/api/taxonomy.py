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

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from drivesort.taxonomy_v2 import TaxonomyV2

router = APIRouter()


def _reload() -> TaxonomyV2:
    return TaxonomyV2.load()


class NodePayload(BaseModel):
    path: str
    name: str
    parent: Optional[str] = None
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
    tax = _reload()
    return [asdict(n) for n in tax.nodes.values()]


@router.post("/nodes", status_code=201)
def add_node(payload: NodePayload):
    tax = _reload()
    embs = np.array([payload.centroid], dtype=np.float32)
    tax.add_node(
        path=payload.path,
        name=payload.name,
        parent=payload.parent,
        member_embeddings=embs,
        member_ids=payload.member_ids,
        folder_id=payload.folder_id,
        description=payload.description,
    )
    tax.save()
    return {"path": payload.path}


@router.patch("/nodes/{node_path:path}")
def patch_node(node_path: str, patch: NodePatch):
    tax = _reload()
    if node_path not in tax.nodes:
        raise HTTPException(404, "node not found")
    node = tax.nodes[node_path]
    if patch.name is not None:
        node.name = patch.name
    if patch.description is not None:
        node.description = patch.description
    if patch.parent is not None:
        node.parent = patch.parent
    tax.save()
    return {"path": node_path}


@router.delete("/nodes/{node_path:path}")
def delete_node(node_path: str):
    tax = _reload()
    if node_path not in tax.nodes:
        raise HTTPException(404, "node not found")
    to_remove = [p for p in tax.nodes if p == node_path or p.startswith(node_path + "/")]
    for p in to_remove:
        del tax.nodes[p]
    tax.save()
    return {"removed": to_remove}


@router.post("/nodes/{node_path:path}/confirm")
def confirm_node(node_path: str, payload: ConfirmPayload):
    tax = _reload()
    if node_path not in tax.nodes:
        raise HTTPException(404, "node not found")
    emb = np.array(payload.embedding, dtype=np.float32)
    tax.confirm(node_path, emb, file_id=payload.file_id)
    tax.save()
    return {"path": node_path, "member_count": tax.nodes[node_path].member_count}
