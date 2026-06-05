"""Tests for files API."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    from drivesort.server import app
    return TestClient(app)


def test_list_files(client):
    mock_file = MagicMock()
    mock_file.id = "file123"
    mock_file.name = "report.pdf"
    mock_file.mime_type = "application/pdf"
    mock_drive = MagicMock()
    mock_drive.iter_files.return_value = [mock_file]
    with patch("drivesort.drive.DriveClient", return_value=mock_drive):
        resp = client.get("/api/files")
    assert resp.status_code == 200
    assert resp.json() == [{"id": "file123", "name": "report.pdf", "mimeType": "application/pdf"}]


def test_list_files_unauthenticated(client):
    with patch("drivesort.drive.DriveClient", side_effect=FileNotFoundError("no credentials")):
        resp = client.get("/api/files")
    assert resp.status_code == 401
