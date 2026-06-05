"""
drivesort/server.py
-------------------
FastAPI application factory.

Routes:
  /api/auth/*      OAuth flow
  /api/analysis/*  Embed + cluster trigger, WebSocket progress
  /api/taxonomy/*  Taxonomy tree CRUD
  /api/draft/*     Draft load/save/discard
  /api/stage/*     Staged changes + commit
  /api/scan/*      Scan queue, accept, correct
  /api/cache/*     Cache status + invalidation
  /ws              WebSocket
  /                React SPA (static files, production only)
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import auth, analysis, taxonomy, draft, stage, scan, cache, ws as ws_router

app = FastAPI(title="DriveSort", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:7432"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router,     prefix="/api/auth",     tags=["auth"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(taxonomy.router, prefix="/api/taxonomy", tags=["taxonomy"])
app.include_router(draft.router,    prefix="/api/draft",    tags=["draft"])
app.include_router(stage.router,    prefix="/api/stage",    tags=["stage"])
app.include_router(scan.router,     prefix="/api/scan",     tags=["scan"])
app.include_router(cache.router,    prefix="/api/cache",    tags=["cache"])
app.include_router(ws_router.router, tags=["ws"])
