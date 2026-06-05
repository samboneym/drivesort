"""
drivesort/draft.py
------------------
Work-in-progress state for multi-session taxonomy building.
Auto-saved after every user action. Cleared on commit or explicit discard.
This is bootstrap-only — the Scan flow has no draft concept.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_PATH = Path("data/draft.json")


@dataclass
class StagedChange:
    file_id: str
    file_name: str
    current_path: Optional[str]   # None = file not yet in any Drive folder
    proposed_path: str


@dataclass
class UserDecision:
    file_id: str
    action: str                   # "assign" | "reject" | "archive"
    path: Optional[str]
    timestamp: str


@dataclass
class DraftState:
    taxonomy_nodes: dict          # path -> node attribute dict
    staged_changes: list[StagedChange] = field(default_factory=list)
    user_decisions: list[UserDecision] = field(default_factory=list)
    saved_at: str = ""


class DraftManager:
    def __init__(self, path: Path = DEFAULT_PATH):
        self._path = path

    def exists(self) -> bool:
        return self._path.exists()

    def load(self) -> Optional[DraftState]:
        if not self._path.exists():
            return None
        raw = json.loads(self._path.read_text())
        return DraftState(
            taxonomy_nodes=raw["taxonomy_nodes"],
            staged_changes=[StagedChange(**c) for c in raw["staged_changes"]],
            user_decisions=[UserDecision(**d) for d in raw["user_decisions"]],
            saved_at=raw.get("saved_at", ""),
        )

    def save(self, state: DraftState) -> None:
        state.saved_at = datetime.now(timezone.utc).isoformat()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(asdict(state), indent=2))

    def clear(self) -> None:
        if self._path.exists():
            self._path.unlink()
