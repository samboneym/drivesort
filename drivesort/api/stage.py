"""
drivesort/api/stage.py
----------------------
GET  /api/stage        — list all staged changes
POST /api/stage/commit — execute all staged changes against Drive
"""
from __future__ import annotations

from fastapi import APIRouter

from drivesort.draft import DraftManager

router = APIRouter()


@router.get("")
def list_staged():
    state = DraftManager().load()
    if state is None:
        return []
    return [vars(c) for c in state.staged_changes]


@router.post("/commit")
def commit_staged():
    """Create Drive folders + move files + clear draft on success."""
    from drivesort.drive import DriveClient
    from drivesort.taxonomy_v2 import TaxonomyV2

    state = DraftManager().load()
    if not state:
        return {"committed": 0}

    drive = DriveClient()
    tax = TaxonomyV2.load()
    committed = 0

    nodes_by_depth = sorted(
        tax.nodes.values(), key=lambda n: len(n.path.split("/"))
    )
    folder_ids: dict[str, str] = {}
    for node in nodes_by_depth:
        if node.folder_id:
            folder_ids[node.path] = node.folder_id
            continue
        parent_folder_id = folder_ids.get(node.parent) if node.parent else None
        folder = drive.find_or_create_folder(node.name, parent_folder_id)
        node.folder_id = folder.id
        folder_ids[node.path] = folder.id
    tax.save()

    for change in state.staged_changes:
        target_node = tax.nodes.get(change.proposed_path)
        if not target_node or not target_node.folder_id:
            continue
        try:
            drive.move_file(change.file_id, target_node.folder_id)
            committed += 1
        except Exception:
            pass

    DraftManager().clear()
    return {"committed": committed}
