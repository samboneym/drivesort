"""Tests for taxonomy API — node CRUD via HTTP."""
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    from drivesort.server import app
    return TestClient(app)


def test_list_nodes_empty(client):
    resp = client.get("/api/taxonomy/nodes")
    assert resp.status_code == 200
    assert resp.json() == []


def test_add_and_get_node(client):
    payload = {
        "path": "books",
        "name": "Books",
        "parent": None,
        "description": "All books",
        "folder_id": "",
        "member_ids": ["f1", "f2"],
        "centroid": [0.1] * 384,
    }
    resp = client.post("/api/taxonomy/nodes", json=payload)
    assert resp.status_code == 201

    resp = client.get("/api/taxonomy/nodes")
    assert len(resp.json()) == 1
    assert resp.json()[0]["path"] == "books"


def test_delete_node(client):
    payload = {
        "path": "books", "name": "Books", "parent": None,
        "description": "", "folder_id": "", "member_ids": [],
        "centroid": [0.0] * 384,
    }
    client.post("/api/taxonomy/nodes", json=payload)
    resp = client.delete("/api/taxonomy/nodes/books")
    assert resp.status_code == 200
    assert client.get("/api/taxonomy/nodes").json() == []
