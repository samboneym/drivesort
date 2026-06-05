"""Tests for cache API."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    from drivesort.server import app
    return TestClient(app)


def test_cache_status(client):
    resp = client.get("/api/cache/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "content" in data
    assert "embeddings" in data
    assert "clustering" in data
    assert "llm_names" in data


def test_invalidate_file(client):
    resp = client.post("/api/cache/invalidate/file", json={"file_id": "abc123"})
    assert resp.status_code == 200
    assert resp.json()["invalidated"] == "abc123"
