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

import numpy as np
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import box

from .clusterer import Cluster, ClusterResult, Clusterer
from .drive import DriveClient, DriveFile
from .taxonomy import Taxonomy, SUB_CLUSTER_MIN_FILES
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
    embeddings: np.ndarray,
    drive: DriveClient,
    taxonomy: Taxonomy,
    embedder: Embedder,
    clusterer: Optional[Clusterer] = None,
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

    # Map file id → embedding for later centroid calculation
    file_emb_map = {f.id: embeddings[i] for i, f in enumerate(files)}

    # name → (cluster, final_name, sub_result_or_None)
    accepted: dict[str, tuple[Cluster, str, Optional[ClusterResult]]] = {}

    archive_overflow: list[DriveFile] = []

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
            sub = _maybe_offer_sub_clustering(cluster, file_emb_map, clusterer)
            accepted[name] = (cluster, name, sub)
            console.print(f"[green]✓ Accepted:[/green] [bold]{name}[/bold]\n")

        elif choice == "r":
            name = Prompt.ask("  Folder name").strip()
            desc = Prompt.ask(
                "  Description (Enter to keep suggestion)",
                default=cluster.suggested_description,
            ).strip()
            cluster.accepted_name = name
            cluster.suggested_description = desc
            sub = _maybe_offer_sub_clustering(cluster, file_emb_map, clusterer)
            accepted[name] = (cluster, name, sub)
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
        if cluster.merged_into:
            if cluster.merged_into in accepted:
                target_cluster, target_name, _ = accepted[cluster.merged_into]
                target_cluster.files.extend(cluster.files)
                console.print(
                    f"[cyan]Merged[/cyan] '{cluster.suggested_name}' → '{target_name}'"
                )
            else:
                console.print(
                    f"[yellow]Merge target '{cluster.merged_into}' not found"
                    " — sending to Archive[/yellow]"
                )
                archive_overflow.extend(cluster.files)

    # ------------------------------------------------------------------
    # Handle outliers
    # ------------------------------------------------------------------
    if result.outlier_files:
        console.rule("[dim]Unclustered Files[/dim]")
        console.print(f"\n[dim]{len(result.outlier_files)} files didn't cluster. They'll go to Archive.[/dim]")
        t = _file_table(result.outlier_files, max_rows=len(result.outlier_files))
        if len(result.outlier_files) > 20:
            with console.pager():
                console.print(t)
        else:
            console.print(t)

    # ------------------------------------------------------------------
    # Create Drive folders + build taxonomy
    # ------------------------------------------------------------------
    if not Confirm.ask("\n[bold]Create folders in Drive and save taxonomy?[/bold]"):
        console.print("[yellow]Aborted — no changes made.[/yellow]")
        return

    console.rule("Creating folders")

    # Ensure Archive exists
    archive_folder = _find_or_create_folder(drive, "Archive", None)

    for name, (cluster, final_name, sub_result) in accepted.items():
        if cluster.rejected or cluster.merged_into:
            continue

        console.print(f"  Creating [cyan]{final_name}[/cyan]…", end=" ")
        folder = _find_or_create_folder(drive, final_name, None)
        console.print("[green]✓[/green]")

        member_embs = [
            file_emb_map[f.id]
            for f in cluster.files
            if f.id in file_emb_map
        ]
        member_ids = [f.id for f in cluster.files]

        if member_embs:
            taxonomy.add_category(
                name=final_name,
                description=cluster.suggested_description,
                folder_id=folder.id,
                member_embeddings=np.stack(member_embs),
                member_ids=member_ids,
                parent_name=None,
            )

        if sub_result is not None:
            for sc in sub_result.clusters:
                sub_name = sc.accepted_name or sc.suggested_name
                console.print(f"    Creating sub-folder [cyan]{sub_name}[/cyan]…", end=" ")
                sub_folder = _find_or_create_folder(drive, sub_name, folder.id)
                console.print("[green]✓[/green]")

                sub_embs = [
                    file_emb_map[f.id]
                    for f in sc.files
                    if f.id in file_emb_map
                ]
                sub_ids = [f.id for f in sc.files]

                if sub_embs:
                    taxonomy.add_category(
                        name=sub_name,
                        description=sc.suggested_description,
                        folder_id=sub_folder.id,
                        member_embeddings=np.stack(sub_embs),
                        member_ids=sub_ids,
                        parent_name=final_name,
                    )

    # Add Archive to taxonomy if not already present
    if "Archive" not in taxonomy.category_names:
        base_emb = embedder.embed_text("archive old backup historical reference dormant")
        overflow_embs = [file_emb_map[f.id] for f in archive_overflow if f.id in file_emb_map]
        emb_matrix = (
            np.vstack([base_emb.reshape(1, -1)] + [e.reshape(1, -1) for e in overflow_embs])
            if overflow_embs else base_emb.reshape(1, -1)
        )
        taxonomy.add_category(
            name="Archive",
            description="Old, dormant, or historical files",
            folder_id=archive_folder.id,
            member_embeddings=emb_matrix,
            member_ids=[f.id for f in archive_overflow],
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


def _find_or_create_folder(
    drive: DriveClient, name: str, parent_id: Optional[str]
) -> DriveFile:
    return drive.find_or_create_folder(name, parent_id)


def _maybe_offer_sub_clustering(
    cluster: Cluster,
    file_emb_map: dict[str, np.ndarray],
    clusterer: Optional[Clusterer],
) -> Optional[ClusterResult]:
    """
    If the cluster is large enough and the user opts in, run a sub-clustering
    pass and let the user name each sub-cluster interactively.

    Returns a ClusterResult (with only accepted sub-clusters) or None.
    """
    if clusterer is None or cluster.size < SUB_CLUSTER_MIN_FILES:
        return None

    if not Confirm.ask(
        f"  Create sub-folders inside [bold]{cluster.accepted_name}[/bold]?",
        default=False,
    ):
        return None

    cluster_embs = np.stack([
        file_emb_map[f.id] for f in cluster.files if f.id in file_emb_map
    ])

    sub_result = clusterer.sub_cluster(files=cluster.files, embeddings=cluster_embs)

    if sub_result is None:
        console.print(
            "[yellow]  Sub-clustering produced fewer than 2 groups"
            " — keeping as a single folder.[/yellow]"
        )
        return None

    console.print(
        f"\n  Found [bold]{len(sub_result.clusters)}[/bold] sub-clusters "
        f"({len(sub_result.outlier_files)} files will stay in the parent folder):\n"
    )

    accepted_subs: list[Cluster] = []
    for j, sc in enumerate(sub_result.clusters):
        console.print(
            f"    [dim]Sub-cluster {j + 1}/{len(sub_result.clusters)}[/dim]  "
            f"[bold]{sc.suggested_name}[/bold]  ({sc.size} files)"
        )
        console.print(_file_table(sc.files, max_rows=4))

        sub_choice = Prompt.ask(
            "    [A]ccept  [R]ename  [S]kip", default="a"
        ).strip().lower()

        if sub_choice == "a":
            sc.accepted_name = sc.suggested_name
            accepted_subs.append(sc)
            console.print(f"    [green]✓ Sub-folder:[/green] {sc.suggested_name}\n")
        elif sub_choice == "r":
            sub_name = Prompt.ask("    Sub-folder name").strip()
            sc.accepted_name = sub_name
            accepted_subs.append(sc)
            console.print(f"    [green]✓ Sub-folder renamed to:[/green] {sub_name}\n")
        else:
            console.print("    [dim]Skipped — files will stay in parent folder[/dim]\n")

    if len(accepted_subs) < 2:
        console.print(
            "[yellow]  Fewer than 2 sub-clusters accepted"
            " — keeping as a single folder.[/yellow]"
        )
        return None

    sub_result.clusters[:] = accepted_subs
    return sub_result
