"""
drivesort/cli.py
----------------
Single entry point: drivesort serve
All other commands (bootstrap, scan, status, recover) have moved to the web UI.
"""
from __future__ import annotations

import typer
import uvicorn

app = typer.Typer(help="DriveSort — local-AI Google Drive organiser")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(7432, help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload (dev only)"),
) -> None:
    """Start the DriveSort web server."""
    typer.echo(f"Starting DriveSort at http://{host}:{port}")
    uvicorn.run(
        "drivesort.server:app",
        host=host,
        port=port,
        reload=reload,
    )


def main() -> None:
    app()
