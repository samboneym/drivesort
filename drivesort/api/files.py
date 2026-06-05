"""
drivesort/api/files.py
----------------------
GET /api/files  — list Drive files (id, name, mimeType) for the Review screen
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("")
def list_files():
    try:
        from drivesort.drive import DriveClient
        drive = DriveClient()
        return [
            {"id": f.id, "name": f.name, "mimeType": f.mime_type}
            for f in drive.iter_files()
        ]
    except FileNotFoundError as exc:
        raise HTTPException(401, detail=str(exc)) from exc
