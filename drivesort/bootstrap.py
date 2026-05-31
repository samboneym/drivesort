"""
drivesort/bootstrap.py
----------------------
Interactive terminal UI for the one-time bootstrap session.

Presents each discovered cluster to the human for:
  [A]ccept suggested name
  [R]ename to a custom name
  [M]erge into another cluster
  [S]kip → send files to Archive
  [Q]uit and save progress

After review, creates folders in Drive and builds the initial Taxonomy.
"""

from __future__ import annotations

from typing import Optional

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import box

from .clusterer import Cluster, ClusterResult
from .drive import DriveClient, DriveFile
from .taxonomy import Taxonomy
from .embedder import Embedder

console = Console()


def _file_table(files: list[DriveFile], max_rows: int = 8) -> Table:
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", padding=(0, 1))
    t.add_column("File", style="white", no_wrap=False, max_width=52)
    t.add_column("Type", style="dim", width=18)
    t.add_column("Size", style="dim", width=8, justify="right")

    for f in files[:max_rows]:
        mime_short = f.mime_type.split(".")[-1].replace("vnd.google-apps.", "")
        size = f"{f.size_bytes // 1024} KB" if f.size_bytes else "—"
        t.add_row(f.name, mime_short, size)

    if len(files) > max_rows:
        t.add_row(f"[dim]… and {len(files) - max_rows} more[/dim]", "", "")

    return t


def run_bootstrap(
    result: ClusterResult,
    files: list[DriveFile],
    embeddings,          # np.ndarray
    drive: DriveClient,
    taxonomy: Taxonomy,
    embedder: Embedder,
) -> None:
    """
    Walk the human through each cluster interactively.
    Builds the taxonomy and creates Drive folders when done.
    """
    console.rule("[bold magenta]DriveSort Bootstrap — Category Review[/bold magenta]")
    console.print(
        f"\nFound [bold]{len(result.clusters)}[/bold] candidate categories "
        f"and [bold]{len(result.outlier_files)}[/bold] unclustered files.\n"
    )

    # Map filename → embedding for later centroid calculation
    file_emb_map = {f.id: embeddings[i] for i, f in enumerate(files)}

    # Track decisions for merge resolution
    accepted: dict[str, tuple[Cluster, str]] = {}  # name → (cluster, final_name)

    # ------------------------------------------------------------------
    # Review each cluster
    # ------------------------------------------------------------------
    for i, cluster in enumerate(result.clusters):
        _show_cluster_header(i + 1, len(result.clusters), cluster)
        console.print(_file_table(cluster.files))

        choice = _prompt_action()

        if choice == "a":
            name = cluster.suggested_name
            cluster.accepted_name = name
            accepted[name] = (cluster, name)
            console.print(f"[green]✓ Accepted:[/green] [bold]{name}[/bold]\n")

        elif choice == "r":
            name = Prompt.ask("  Folder name").strip()
            desc = Prompt.ask("  Description (Enter to keep suggestion)", default=cluster.suggested_description).strip()
            cluster.accepted_name = name
            cluster.suggested_description = desc
            accepted[name] = (cluster, name)
            console.print(f"[green]✓ Renamed to:[/green] [bold]{name}[/bold]\n")

        elif choice == "m":
            target = Prompt.ask("  Merge into which folder (type its name)").strip()
            cluster.merged_into = target
            console.print(f"[yellow]↗ Will merge into:[/yellow] [bold]{target}[/bold]\n")

        elif choice == "s":
            cluster.rejected = True
            console.print("[dim]→ Skipped — files will go to Archive[/dim]\n")

        elif choice == "q":
            console.print("[yellow]Saving progress and stopping early.[/yellow]")
            break

    # ------------------------------------------------------------------
    # Resolve merges
    # ------------------------------------------------------------------
    for cluster in result.clusters:
        if cluster.merged_into and cluster.merged_into in accepted:
            target_cluster, target_name = accepted[cluster.merged_into]
            target_cluster.files.extend(cluster.files)
            console.print(f"[cyan]Merged[/cyan] '{cluster.suggested_name}' → '{target_name}'")

    # ------------------------------------------------------------------
    # Handle outliers
    # ------------------------------------------------------------------
    if result.outlier_files:
        console.rule("[dim]Unclustered Files[/dim]")
        console.print(f"\n[dim]{len(result.outlier_files)} files didn't cluster. They'll go to Archive.[/dim]")
        console.print(_file_table(result.outlier_files, max_rows=6))

    # ------------------------------------------------------------------
    # Create Drive folders + build taxonomy
    # ------------------------------------------------------------------
    if not Confirm.ask("\n[bold]Create folders in Drive and save taxonomy?[/bold]"):
        console.print("[yellow]Aborted — no changes made.[/yellow]")
        return

    console.rule("Creating folders")

    # Ensure Archive exists
    archive_folder = _find_or_create_folder(drive, "Archive", None)

    for name, (cluster, final_name) in accepted.items():
        if cluster.rejected or cluster.merged_into:
            continue

        console.print(f"  Creating [cyan]{final_name}[/cyan]…", end=" ")
        folder = _find_or_create_folder(drive, final_name, None)
        console.print("[green]✓[/green]")

        # Collect member embeddings
        member_embs = [
            file_emb_map[f.id]
            for f in cluster.files
            if f.id in file_emb_map
        ]
        member_ids = [f.id for f in cluster.files]

        import numpy as np
        if member_embs:
            emb_matrix = np.stack(member_embs)
            taxonomy.add_category(
                name=final_name,
                description=cluster.suggested_description,
                folder_id=folder.id,
                member_embeddings=emb_matrix,
                member_ids=member_ids,
            )

    # Add Archive to taxonomy if not already present
    if "Archive" not in taxonomy.category_names:
        archive_emb = embedder.embed_text("archive old backup historical reference dormant")
        import numpy as np
        taxonomy.add_category(
            name="Archive",
            description="Old, dormant, or historical files",
            folder_id=archive_folder.id,
            member_embeddings=archive_emb.reshape(1, -1),
            member_ids=[],
        )

    taxonomy.save()
    console.print(f"\n[bold green]✓ Taxonomy saved with {len(taxonomy.category_names)} categories.[/bold green]")
    console.print("\nNext step: run [bold]drivesort scan[/bold] to classify and move your files.")


# ------------------------------------------------------------------
# UI helpers
# ------------------------------------------------------------------

def _show_cluster_header(current: int, total: int, cluster: Cluster) -> None:
    conf_color = "green" if cluster.llm_confidence > 0.75 else "yellow" if cluster.llm_confidence > 0.5 else "red"
    conf_str   = f"[{conf_color}]{cluster.llm_confidence:.0%} confidence[/{conf_color}]"

    panel = Panel(
        f"[bold white]{cluster.suggested_name}[/bold white]\n"
        f"[dim]{cluster.suggested_description}[/dim]\n\n"
        f"{conf_str}  ·  {cluster.size} files",
        title=f"[dim]Cluster {current}/{total}[/dim]",
        border_style="bright_blue",
    )
    console.print(panel)


def _prompt_action() -> str:
    while True:
        raw = Prompt.ask(
            "  [bold]\\[A][/bold]ccept  [bold]\\[R][/bold]ename  [bold]\\[M][/bold]erge  [bold]\\[S][/bold]kip  [bold]\\[Q][/bold]uit",
            default="a",
        ).strip().lower()
        if raw in ("a", "r", "m", "s", "q"):
            return raw
        console.print("[red]Please enter A, R, M, S, or Q[/red]")


def _find_or_create_folder(drive: DriveClient, name: str, parent_id: Optional[str]) -> DriveFile:
    """Return existing folder with this name, or create it."""
    existing = drive.list_folders()
    for folder in existing:
        if folder.name == name:
            return folder
    return drive.create_folder(name, parent_id)
