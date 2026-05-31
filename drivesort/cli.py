"""
drivesort/cli.py
----------------
Command-line interface.

Commands
--------
  drivesort bootstrap   — First run: cluster Drive, human reviews, taxonomy created
  drivesort scan        — Classify new/unorganised files (dry-run by default)
  drivesort scan --live — Actually move files
  drivesort status      — Show current taxonomy
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from .drive import DriveClient
from .embedder import Embedder
from .clusterer import Clusterer
from .taxonomy import Taxonomy
from .bootstrap import run_bootstrap
from .scanner import Scanner

app     = typer.Typer(help="DriveSort — local-AI Google Drive organiser")
console = Console()


def _build_components(dry_run: bool = True):
    drive    = DriveClient()
    embedder = Embedder()
    taxonomy = Taxonomy()
    clusterer = Clusterer()
    scanner  = Scanner(drive, embedder, taxonomy, clusterer, dry_run=dry_run)
    return drive, embedder, taxonomy, clusterer, scanner


@app.command()
def bootstrap(
    min_cluster_size: int = typer.Option(3, help="Minimum files to form a cluster"),
    model: str = typer.Option("phi3:mini", help="Ollama model for cluster naming"),
):
    """
    One-time bootstrap: discover categories from your Drive, review them,
    and build the initial taxonomy index.
    """
    console.print("[bold magenta]DriveSort Bootstrap[/bold magenta]\n")

    drive    = DriveClient()
    embedder = Embedder()
    taxonomy = Taxonomy()
    clusterer = Clusterer(min_cluster_size=min_cluster_size, ollama_model=model)

    if not taxonomy.is_empty():
        if not typer.confirm("Taxonomy already exists. Rebuild from scratch?"):
            raise typer.Exit()

    # 1. Fetch all files
    with console.status("Fetching files from Drive…"):
        files = list(drive.iter_files(include_folders=False))
    console.print(f"[green]✓[/green] Found {len(files)} files\n")

    # 2. Embed
    console.print("Embedding files (first run downloads ~80 MB model)…")
    files, embeddings = embedder.embed_files(files, show_progress=True)
    console.print(f"[green]✓[/green] Embedded {len(files)} files\n")

    # 3. Cluster
    with console.status("Clustering…"):
        result = clusterer.cluster(files, embeddings, name_with_llm=True)
    console.print(f"[green]✓[/green] Found {len(result.clusters)} clusters, {len(result.outlier_files)} outliers\n")

    # 4. Interactive review → creates folders + saves taxonomy
    run_bootstrap(result, files, embeddings, drive, taxonomy, embedder)


@app.command()
def scan(
    live: bool = typer.Option(False, "--live", help="Actually move files (default is dry-run)"),
    no_interact: bool = typer.Option(False, "--no-interact", help="Skip interactive review; only auto-moves"),
):
    """
    Classify and optionally move unorganised files.
    Default is dry-run — pass --live to make real changes.
    """
    _, embedder, taxonomy, clusterer, scanner = _build_components(dry_run=not live)
    drive = DriveClient()
    scanner = Scanner(drive, embedder, taxonomy, clusterer, dry_run=not live)
    scanner.scan(interactive=not no_interact)


@app.command()
def status():
    """Show the current taxonomy: categories, file counts, folder IDs."""
    taxonomy = Taxonomy()

    if taxonomy.is_empty():
        console.print("[yellow]No taxonomy yet. Run `drivesort bootstrap` first.[/yellow]")
        raise typer.Exit()

    t = Table(title="DriveSort Taxonomy", box=box.ROUNDED, header_style="bold cyan")
    t.add_column("Category",    style="bold white", min_width=24)
    t.add_column("Files",       justify="right", width=7)
    t.add_column("Description", style="dim", max_width=45)
    t.add_column("Folder ID",   style="dim", max_width=30)

    for name, entry in taxonomy.categories.items():
        t.add_row(name, str(entry.member_count), entry.description, entry.folder_id)

    console.print(t)

    novel_records, _ = taxonomy.load_novel_files()
    if novel_records:
        console.print(f"\n[yellow]{len(novel_records)} novel files accumulated[/yellow] — run `scan` to re-cluster them.")


def main():
    app()
