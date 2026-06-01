"""
drivesort/scanner.py
--------------------
Ongoing file scanner.  Run periodically (cron / manual) after bootstrap.

For each unorganised file:
  1. Embed it
  2. Classify against taxonomy centroids
  3. If confident → propose auto-move
  4. If uncertain → flag for human review
  5. If novel → log it and ask LLM for new-category suggestion
     Periodically re-cluster novel files to detect emerging categories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import box

from .drive import DriveClient, DriveFile
from .embedder import Embedder
from .taxonomy import Taxonomy, ClassificationResult
from .clusterer import Clusterer

console = Console()

# Confidence bands
AUTO_MOVE_THRESHOLD  = 0.82   # move without asking
REVIEW_THRESHOLD     = 0.62   # ask the human
# below REVIEW_THRESHOLD → novel


@dataclass
class ScanDecision:
    file: DriveFile
    result: ClassificationResult
    action: str        # "auto_move" | "review" | "novel"
    target_folder_id: Optional[str] = None
    target_folder_name: Optional[str] = None


class Scanner:
    def __init__(
        self,
        drive: DriveClient,
        embedder: Embedder,
        taxonomy: Taxonomy,
        clusterer: Clusterer,
        dry_run: bool = True,
    ) -> None:
        self._drive    = drive
        self._embedder = embedder
        self._taxonomy = taxonomy
        self._cluster  = clusterer
        self._dry_run  = dry_run

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def scan(self, interactive: bool = True) -> None:
        """
        Scan all Drive files and classify them.

        In dry_run mode, only print what would happen — no moves made.
        In interactive mode, present review items to the human in the terminal.
        """
        if self._taxonomy.is_empty():
            console.print("[red]No taxonomy found. Run `drivesort bootstrap` first.[/red]")
            return

        console.rule("[bold magenta]DriveSort Scan[/bold magenta]")
        mode = "[dim](dry run)[/dim]" if self._dry_run else "[yellow](LIVE — will move files)[/yellow]"
        console.print(f"Scanning Drive…  {mode}\n")

        files        = list(self._drive.iter_files(exclude_orphans=True))
        auto_moves:  list[ScanDecision] = []
        for_review:  list[ScanDecision] = []
        novel_files: list[ScanDecision] = []

        with console.status("Embedding files…"):
            _, embeddings = self._embedder.embed_files(files, show_progress=False)

        for file, emb in zip(files, embeddings):
            # Skip files already in a known folder
            if self._file_is_organised(file):
                continue

            result  = self._taxonomy.classify(emb, file.id, file.name)
            decision = self._make_decision(file, result)

            if decision.action == "auto_move":
                auto_moves.append(decision)
            elif decision.action == "review":
                for_review.append(decision)
            else:
                novel_files.append(decision)
                # Log for periodic re-clustering
                self._taxonomy.log_novel_file(file.id, file.name, emb)

        # ------------------------------------------------------------------
        # Auto-moves
        # ------------------------------------------------------------------
        if auto_moves:
            self._show_auto_moves(auto_moves)
            if not self._dry_run:
                if interactive and not Confirm.ask(f"\nApply {len(auto_moves)} auto-moves?"):
                    console.print("[yellow]Skipped.[/yellow]")
                else:
                    self._apply_moves(auto_moves, update_centroids=True, embeddings_map={
                        f.id: embeddings[i] for i, f in enumerate(files)
                    })

        # ------------------------------------------------------------------
        # Human review items
        # ------------------------------------------------------------------
        if for_review:
            self._interactive_review(for_review, embeddings_map={
                f.id: embeddings[i] for i, f in enumerate(files)
            })

        # ------------------------------------------------------------------
        # Novel files
        # ------------------------------------------------------------------
        if novel_files:
            self._handle_novel_files(novel_files, embeddings_map={
                f.id: embeddings[i] for i, f in enumerate(files)
            })

        # ------------------------------------------------------------------
        # Periodic re-cluster novel accumulation
        # ------------------------------------------------------------------
        self._maybe_recluster()

        console.print("\n[bold green]Scan complete.[/bold green]")
        self._taxonomy.save()

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _make_decision(self, file: DriveFile, result: ClassificationResult) -> ScanDecision:
        if result.is_novel or result.confidence < REVIEW_THRESHOLD:
            return ScanDecision(file=file, result=result, action="novel")

        category = self._taxonomy.categories[result.category]

        if result.confidence >= AUTO_MOVE_THRESHOLD:
            return ScanDecision(
                file=file, result=result, action="auto_move",
                target_folder_id=category.folder_id,
                target_folder_name=result.category,
            )

        return ScanDecision(
            file=file, result=result, action="review",
            target_folder_id=category.folder_id,
            target_folder_name=result.category,
        )

    def _file_is_organised(self, file: DriveFile) -> bool:
        """True if the file is already in one of the taxonomy folders (including sub-folders)."""
        return file.parent_id in self._taxonomy.all_folder_ids

    def _display_name(self, category_name: str) -> str:
        """Return 'Parent / Child' for sub-categories, plain name for top-level."""
        entry = self._taxonomy.categories.get(category_name)
        if entry and entry.parent_name:
            return f"{entry.parent_name} / {category_name}"
        return category_name

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _show_auto_moves(self, decisions: list[ScanDecision]) -> None:
        t = Table(title="Auto-moves (high confidence)", box=box.SIMPLE,
                  header_style="bold cyan", show_lines=False)
        t.add_column("File", max_width=45)
        t.add_column("→ Folder", style="green", width=26)
        t.add_column("Conf", justify="right", width=6)

        for d in decisions:
            t.add_row(
                d.file.name,
                self._display_name(d.target_folder_name),
                f"{d.result.confidence:.0%}",
            )

        if len(decisions) > 20:
            with console.pager():
                console.print(t)
        else:
            console.print(t)

    def _interactive_review(
        self,
        decisions: list[ScanDecision],
        embeddings_map: dict[str, np.ndarray],
    ) -> None:
        console.rule("[cyan]Files for Review[/cyan]")
        folder_names = self._taxonomy.category_names

        for d in decisions:
            display = self._display_name(d.target_folder_name)
            console.print(
                f"\n  [bold]{d.file.name}[/bold]  "
                f"[dim]{d.file.mime_type.split('.')[-1]}[/dim]\n"
                f"  Suggested: [green]{display}[/green]  "
                f"({d.result.confidence:.0%})  "
                f"Runner-up: [dim]{d.result.runner_up}[/dim]"
                f" ({d.result.runner_up_confidence:.0%})"
            )
            if d.file.snippet:
                console.print(f"  [dim]{d.file.snippet[:120]}…[/dim]")

            action = Prompt.ask(
                "  [A]ccept  [C]hoose folder  [S]kip",
                default="a",
            ).strip().lower()

            if action == "a":
                if not self._dry_run:
                    self._drive.move_file(d.file, d.target_folder_id)
                    if d.file.id in embeddings_map:
                        self._taxonomy.confirm(
                            d.target_folder_name, d.file.id, embeddings_map[d.file.id]
                        )
                console.print(f"  [green]✓ → {display}[/green]")

            elif action == "c":
                for i, name in enumerate(folder_names):
                    label = self._display_name(name)
                    console.print(f"  [{i}] {label}")
                idx_str = Prompt.ask("  Choose number")
                try:
                    chosen = folder_names[int(idx_str)]
                    folder_id = self._taxonomy.categories[chosen].folder_id
                    if not self._dry_run:
                        self._drive.move_file(d.file, folder_id)
                        if d.file.id in embeddings_map:
                            self._taxonomy.confirm(
                                chosen, d.file.id, embeddings_map[d.file.id]
                            )
                    console.print(f"  [green]✓ → {self._display_name(chosen)}[/green]")
                except (ValueError, IndexError):
                    console.print("  [yellow]Invalid — skipped[/yellow]")

            else:
                console.print("  [dim]Skipped[/dim]")

    def _handle_novel_files(self, decisions: list[ScanDecision], embeddings_map: dict[str, np.ndarray]) -> None:
        console.rule("[yellow]Novel Files — Don't Fit Any Category[/yellow]")
        console.print(f"[dim]{len(decisions)} files logged for re-clustering.[/dim]\n")

        t = Table(box=box.SIMPLE, header_style="bold yellow")
        t.add_column("File", max_width=50)
        t.add_column("Nearest (weak)", style="dim", width=26)
        t.add_column("Dist", justify="right", width=6)

        for d in decisions:
            nearest = d.result.runner_up or "—"
            t.add_row(d.file.name, nearest, f"{d.result.distance:.2f}")

        if len(decisions) > 20:
            with console.pager():
                console.print(t)
        else:
            console.print(t)

        # Offer LLM suggestion for each
        if Confirm.ask("\nAsk LLM for new-category suggestions on these files?", default=False):
            for d in decisions:
                suggestion = self._cluster.suggest_new_category(
                    d.file, self._taxonomy.category_names
                )
                new_folder = suggestion.get("new_folder")
                rationale  = suggestion.get("rationale", "")
                if new_folder:
                    console.print(f"  [cyan]{d.file.name}[/cyan] → suggested new folder: [bold]{new_folder}[/bold]")
                    console.print(f"  [dim]{rationale}[/dim]")
                    if Confirm.ask(f"  Create '{new_folder}' and move this file?", default=False):
                        if not self._dry_run:
                            folder = self._drive.create_folder(new_folder)
                            self._drive.move_file(d.file, folder.id)
                            emb = embeddings_map.get(d.file.id)
                            if emb is not None:
                                self._taxonomy.add_category(
                                    name=new_folder,
                                    description=suggestion.get("similar_files_hint", ""),
                                    folder_id=folder.id,
                                    member_embeddings=emb.reshape(1, -1),
                                    member_ids=[d.file.id],
                                )
                            console.print(f"  [green]✓ Created '{new_folder}' and moved file.[/green]")
                else:
                    console.print(f"  [dim]{d.file.name}[/dim] → LLM says Archive is fine  [dim]({rationale})[/dim]")

    # ------------------------------------------------------------------
    # Periodic re-clustering of novel files
    # ------------------------------------------------------------------

    def _maybe_recluster(self, min_novel: int = 5) -> None:
        """
        If enough novel files have accumulated, re-cluster them to detect
        emerging categories. Surfaces results for human confirmation.
        """
        novel_records, novel_embeddings = self._taxonomy.load_novel_files()

        if len(novel_records) < min_novel:
            return

        console.rule(f"[magenta]Re-clustering {len(novel_records)} novel files[/magenta]")

        # Build temporary DriveFile-like objects for the clusterer
        from .drive import DriveFile as DF
        import types

        temp_files = []
        for rec in novel_records:
            f = types.SimpleNamespace(
                id=rec["id"], name=rec["name"],
                mime_type="unknown", size_bytes=0, snippet="",
            )
            temp_files.append(f)

        result = self._cluster.cluster(
            temp_files, novel_embeddings, name_with_llm=True
        )

        if not result.clusters:
            console.print("[dim]No new clusters detected yet.[/dim]")
            return

        console.print(f"\nDetected [bold]{len(result.clusters)}[/bold] potential new categories:\n")
        for cluster in result.clusters:
            console.print(
                f"  [bold cyan]{cluster.suggested_name}[/bold cyan] "
                f"({cluster.size} files, {cluster.llm_confidence:.0%} confidence)"
            )
            for f in cluster.files[:4]:
                console.print(f"    • {f.name}")

        if Confirm.ask("\nCreate these as new Drive folders?", default=False):
            for cluster in result.clusters:
                name = Prompt.ask(f"  Name for this cluster", default=cluster.suggested_name).strip()
                if not self._dry_run:
                    folder = self._drive.create_folder(name)
                    emb_matrix = np.stack([
                        np.array(novel_records[i]["embedding"])
                        for i, rec in enumerate(novel_records)
                        if rec["id"] in {f.id for f in cluster.files}
                    ])
                    self._taxonomy.add_category(
                        name=name,
                        description=cluster.suggested_description,
                        folder_id=folder.id,
                        member_embeddings=emb_matrix,
                        member_ids=[f.id for f in cluster.files],
                    )
                console.print(f"  [green]✓ Created '{name}'[/green]")

            self._taxonomy.clear_novel_log()
            self._taxonomy.save()

    # ------------------------------------------------------------------
    # Applying moves
    # ------------------------------------------------------------------

    def _apply_moves(
        self,
        decisions: list[ScanDecision],
        update_centroids: bool,
        embeddings_map: dict[str, np.ndarray],
    ) -> None:
        for d in decisions:
            try:
                self._drive.move_file(d.file, d.target_folder_id)
                if update_centroids and d.file.id in embeddings_map:
                    self._taxonomy.confirm(
                        d.target_folder_name,
                        d.file.id,
                        embeddings_map[d.file.id],
                    )
                console.print(f"  [green]✓[/green] {d.file.name} → {d.target_folder_name}")
            except Exception as e:
                console.print(f"  [red]✗[/red] {d.file.name}: {e}")
