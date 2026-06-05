"""
drivesort/api/auth.py
---------------------
Google OAuth routes.

GET /api/auth/status    — is the user authenticated?
GET /api/auth/login     — get the OAuth URL to redirect to
GET /api/auth/callback  — exchange code for token (Google redirects here)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import RedirectResponse
from google.auth.transport.requests import Request as GRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow

TOKEN_PATH = Path("data/token.json")
CREDENTIALS_PATH = Path("data/credentials.json")
SCOPES = ["https://www.googleapis.com/auth/drive"]
REDIRECT_URI = "http://localhost:7432/api/auth/callback"

router = APIRouter()


def _load_credentials() -> Optional[Credentials]:
    if not TOKEN_PATH.exists():
        return None
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(GRequest())
            TOKEN_PATH.write_text(creds.to_json())
        except Exception:
            return None
    return creds if (creds and creds.valid) else None


def _get_email(creds: Credentials) -> Optional[str]:
    try:
        import googleapiclient.discovery
        svc = googleapiclient.discovery.build("oauth2", "v2", credentials=creds)
        info = svc.userinfo().get().execute()
        return info.get("email")
    except Exception:
        return None


@router.get("/status")
def auth_status():
    creds = _load_credentials()
    if not creds:
        return {"authenticated": False, "email": None}
    return {"authenticated": True, "email": _get_email(creds)}


@router.get("/login")
def auth_login():
    if not CREDENTIALS_PATH.exists():
        return {"error": "credentials.json not found in data/"}
    flow = Flow.from_client_secrets_file(
        str(CREDENTIALS_PATH), scopes=SCOPES, redirect_uri=REDIRECT_URI
    )
    auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")
    return {"auth_url": auth_url}


@router.get("/callback")
def auth_callback(code: str, state: str = ""):
    flow = Flow.from_client_secrets_file(
        str(CREDENTIALS_PATH), scopes=SCOPES, redirect_uri=REDIRECT_URI
    )
    flow.fetch_token(code=code)
    creds = flow.credentials
    TOKEN_PATH.parent.mkdir(exist_ok=True)
    TOKEN_PATH.write_text(creds.to_json())
    return RedirectResponse(url="/setup/analyse")
