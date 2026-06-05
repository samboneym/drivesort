"""
drivesort/llm_name_cache.py
---------------------------
Caches LLM-generated cluster names.
Cache key = SHA1(sorted file IDs in cluster + model name).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

DEFAULT_PATH = Path("data/llm_name_cache.json")


def _make_key(file_ids: list[str], model: str) -> str:
    payload = json.dumps({"ids": sorted(file_ids), "model": model}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()


class LLMNameCache:
    def __init__(self, path: Path = DEFAULT_PATH):
        self._path = path
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            self._data = json.loads(self._path.read_text())

    def get(self, file_ids: list[str], model: str) -> Optional[dict]:
        return self._data.get(_make_key(file_ids, model))

    def put(self, file_ids: list[str], model: str, result: dict) -> None:
        self._data[_make_key(file_ids, model)] = result
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=2))

    def invalidate_file(self, file_id: str) -> int:
        """Remove all entries that include this file_id. Returns count removed."""
        to_remove = [k for k, v in self._data.items()
                     if file_id in v.get("_file_ids", [])]
        for k in to_remove:
            del self._data[k]
        if to_remove:
            self._path.write_text(json.dumps(self._data, indent=2))
        return len(to_remove)
