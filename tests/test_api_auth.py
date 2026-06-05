"""Tests for auth API — status endpoint."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from drivesort.server import app
    return TestClient(app)


def test_auth_status_unauthenticated(client):
    resp = client.get("/api/auth/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["authenticated"] is False
    assert data["email"] is None


def test_auth_login_no_credentials(client):
    resp = client.get("/api/auth/login")
    assert resp.status_code == 200
    # Without credentials.json, returns error
    data = resp.json()
    assert "error" in data or "auth_url" in data
