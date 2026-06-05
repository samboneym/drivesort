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
