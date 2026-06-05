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


def test_clear_layer(client, tmp_path):
    import json
    f = tmp_path / "data" / "content_cache.json"
    f.write_text(json.dumps({"k": "v"}))
    resp = client.delete("/api/cache/content")
    assert resp.status_code == 200
    assert resp.json() == {"cleared": "content"}
    assert not f.exists()


def test_clear_layer_already_empty(client):
    resp = client.delete("/api/cache/content")
    assert resp.status_code == 200
    assert resp.json() == {"cleared": None}


def test_clear_layer_unknown(client):
    resp = client.delete("/api/cache/not_a_real_layer")
    assert resp.status_code == 404
