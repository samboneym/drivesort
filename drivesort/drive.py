"""
drivesort/drive.py
------------------
Google Drive API client.
Handles OAuth, file listing, and moving files between folders.
All Drive I/O is isolated here so the rest of the codebase is Drive-agnostic.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Iterator

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Only need read + file-organisation scopes (no content write)
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
]

TOKEN_PATH = Path("data/token.json")
CREDENTIALS_PATH = Path("data/credentials.json")


class DriveFile:
    """Lightweight value object representing one Drive file or folder."""

    __slots__ = (
        "id", "name", "mime_type", "extension", "parent_id",
        "size_bytes", "snippet", "created", "modified", "is_folder",
    )

    def __init__(self, raw: dict) -> None:
        self.id          = raw["id"]
        self.name        = raw.get("name", "")
        self.mime_type   = raw.get("mimeType", "")
        self.extension   = raw.get("fileExtension", "").lower()
        self.parent_id   = (raw.get("parents") or [None])[0]
        self.size_bytes  = int(raw.get("size", 0))
        self.snippet     = raw.get("contentHints", {}).get("indexableText", "")
        self.created     = raw.get("createdTime", "")
        self.modified    = raw.get("modifiedTime", "")
        self.is_folder   = self.mime_type == "application/vnd.google-apps.folder"

    def text_for_embedding(self) -> str:
        """Concatenate the signals available without downloading the file."""
        parts = [self.name]
        if self.snippet:
            parts.append(self.snippet[:400])
        if self.extension:
            parts.append(self.extension)
        return " ".join(parts)

    def __repr__(self) -> str:
        kind = "folder" if self.is_folder else self.mime_type.split("/")[-1]
        return f"<DriveFile {self.name!r} ({kind})>"


class DriveClient:
    """
    Thin wrapper around the Drive v3 REST API.

    Usage
    -----
    client = DriveClient()
    for f in client.iter_files():
        print(f.name)
    """

    _LIST_FIELDS = (
        "nextPageToken,"
        "files(id,name,mimeType,fileExtension,parents,size,"
        "contentHints/indexableText,createdTime,modifiedTime)"
    )

    def __init__(
        self,
        credentials_path: Path = CREDENTIALS_PATH,
        token_path: Path = TOKEN_PATH,
    ) -> None:
        self._service = self._authenticate(credentials_path, token_path)

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    @staticmethod
    def _authenticate(creds_path: Path, token_path: Path):
        creds = None

        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not creds_path.exists():
                    raise FileNotFoundError(
                        f"credentials.json not found at {creds_path}.\n"
                        "Download it from Google Cloud Console → APIs & Services → Credentials."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
                creds = flow.run_local_server(port=0)

            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(creds.to_json())

        return build("drive", "v3", credentials=creds)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def iter_files(
        self,
        include_folders: bool = False,
        page_size: int = 200,
    ) -> Iterator[DriveFile]:
        """
        Yield every file the authenticated user owns.
        Folders are excluded by default (they contribute no content signal).
        """
        query = "trashed = false and owner = 'me'"
        if not include_folders:
            query += " and mimeType != 'application/vnd.google-apps.folder'"

        token = None
        while True:
            resp = (
                self._service.files()
                .list(
                    q=query,
                    pageSize=page_size,
                    fields=self._LIST_FIELDS,
                    pageToken=token,
                )
                .execute()
            )
            for raw in resp.get("files", []):
                yield DriveFile(raw)

            token = resp.get("nextPageToken")
            if not token:
                break

    def list_folders(self) -> list[DriveFile]:
        """Return all top-level folders owned by the user."""
        resp = (
            self._service.files()
            .list(
                q="mimeType = 'application/vnd.google-apps.folder' "
                  "and trashed = false and owner = 'me'",
                fields="files(id,name,mimeType,parents)",
                pageSize=200,
            )
            .execute()
        )
        return [DriveFile(r) for r in resp.get("files", [])]

    def get_file(self, file_id: str) -> DriveFile:
        raw = (
            self._service.files()
            .get(fileId=file_id, fields=self._LIST_FIELDS.split(",", 1)[1])
            .execute()
        )
        return DriveFile(raw)

    # ------------------------------------------------------------------
    # Writing (moves only — we never delete or rename)
    # ------------------------------------------------------------------

    def move_file(self, file: DriveFile, target_folder_id: str) -> None:
        """
        Move a file to a different folder.
        Raises HttpError on failure.
        """
        if file.parent_id == target_folder_id:
            return  # already there

        self._service.files().update(
            fileId=file.id,
            addParents=target_folder_id,
            removeParents=file.parent_id or "",
            fields="id, parents",
        ).execute()

    def create_folder(self, name: str, parent_id: str | None = None) -> DriveFile:
        """Create a new folder and return it as a DriveFile."""
        meta = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parent_id:
            meta["parents"] = [parent_id]

        raw = self._service.files().create(body=meta, fields="id,name,mimeType,parents").execute()
        return DriveFile(raw)
