"""
drivesort/ws.py
---------------
Shared WebSocket connection manager.
All API modules import `manager` to broadcast progress events.
"""
from __future__ import annotations

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, event: dict) -> None:
        """Send event to all connected clients. Silently drops dead connections."""
        for ws in list(self.active):
            try:
                await ws.send_json(event)
            except Exception:
                self.disconnect(ws)


manager = ConnectionManager()
