"""
drivesort/api/draft.py
----------------------
GET    /api/draft   — load current draft (null if none)
PUT    /api/draft   — save draft state
DELETE /api/draft   — discard draft
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from drivesort.draft import DraftManager, DraftState, StagedChange, UserDecision

router = APIRouter()


def _mgr() -> DraftManager:
    return DraftManager()


class DraftPayload(BaseModel):
    taxonomy_nodes: dict
    staged_changes: list[dict] = []
    user_decisions: list[dict] = []


@router.get("")
def get_draft():
    state = _mgr().load()
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
    mgr = _mgr()
    state = DraftState(
        taxonomy_nodes=payload.taxonomy_nodes,
        staged_changes=[StagedChange(**c) for c in payload.staged_changes],
        user_decisions=[UserDecision(**d) for d in payload.user_decisions],
    )
    mgr.save(state)
    return {"saved_at": state.saved_at}


@router.delete("", status_code=200)
def discard_draft():
    _mgr().clear()
    return {"discarded": True}
