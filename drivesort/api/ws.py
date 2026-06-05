"""
drivesort/api/ws.py
-------------------
WebSocket endpoint — clients connect here to receive progress events.
"""
from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from drivesort.ws import manager

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()  # keep connection alive; events pushed server→client
    except WebSocketDisconnect:
        manager.disconnect(ws)
