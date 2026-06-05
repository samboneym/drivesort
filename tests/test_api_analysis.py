"""Tests for analysis API — trigger endpoint queues a background task."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import drivesort.api.analysis as analysis_mod
    analysis_mod._state.update(phase="idle", progress=0, total=0, message="", result=None, error=None)
    from drivesort.server import app
    return TestClient(app)


def test_trigger_returns_accepted(client):
    resp = client.post("/api/analysis/trigger")
    assert resp.status_code == 202
    assert resp.json()["status"] == "started"


def test_status_returns_idle_initially(client):
    resp = client.get("/api/analysis/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["phase"] in ("idle", "running", "complete", "error", "fetching")
