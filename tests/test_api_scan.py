"""Tests for scan API."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    from drivesort.server import app
    return TestClient(app)


def test_queue_empty_initially(client):
    resp = client.get("/api/scan/queue")
    assert resp.status_code == 200
    assert resp.json() == []


def test_trigger_scan_returns_202(client):
    resp = client.post("/api/scan/trigger")
    assert resp.status_code == 202


def test_stats_empty(client):
    resp = client.get("/api/scan/stats")
    assert resp.status_code == 200
    assert resp.json() == {"queued": 0, "accepted": 0, "corrected": 0}


def test_stats_with_queue(client, tmp_path):
    import json
    (tmp_path / "data" / "scan_queue.json").write_text(
        json.dumps([{"file_id": "1"}, {"file_id": "2"}])
    )
    resp = client.get("/api/scan/stats")
    assert resp.json()["queued"] == 2


def test_stats_with_stats_file(client, tmp_path):
    import json
    (tmp_path / "data" / "scan_stats.json").write_text(
        json.dumps({"accepted": 5, "corrected": 2})
    )
    data = client.get("/api/scan/stats").json()
    assert data["accepted"] == 5
    assert data["corrected"] == 2
