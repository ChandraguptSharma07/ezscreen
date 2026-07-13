from __future__ import annotations

import csv
import threading
from datetime import datetime, timezone
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Select,
    Static,
)

from ezscreen.results.run_layout import resolve_output_dir
from ezscreen.tui.widgets.breadcrumb import Breadcrumb


class MDSimulationScreen(Screen):
    """Browse completed runs, inspect their leaderboard, and launch an MD job."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def __init__(self) -> None:
        super().__init__()
        self._run_rows: list[dict[str, str]] = []
        self._all_hits: list[dict[str, str]] = []
        self._visible_hits: list[dict[str, str]] = []
        self._selected_run_id: str = ""
        self._selected_ligand_id: str | None = None
        self._score_col: str = "score"

    def compose(self) -> ComposeResult:
        yield Header()
        yield Breadcrumb(["Home", "MD Simulation"])
        with Horizontal(id="md-main"):
            with Vertical(id="md-browser"):
                yield Label("Completed runs", classes="form-section")
                yield Input(placeholder="Filter by run ID or date", id="md-run-filter")
                yield DataTable(id="md-runs", cursor_type="row")
                yield Static("", id="md-run-summary", classes="form-status")
            with VerticalScroll(id="md-detail"):
                yield Label("Selected run leaderboard", classes="form-section")
                yield Input(placeholder="Filter ligands in selected run", id="md-hit-filter")
                yield DataTable(id="md-leaderboard", cursor_type="row")
                yield Static("Select a run to load its leaderboard.", id="md-hit-summary", classes="form-status")

                yield Label("MD job", classes="form-section")
                yield Static("", id="md-account-summary", classes="form-status")
                yield Static("Run: -\nLigand: top-ranked hit", id="md-selection", classes="form-status")
                yield Static("", id="md-jobs-summary", classes="form-status")

                yield Label("Pose source", classes="form-label")
                yield Select(
                    [("UniDock pose", "unidock"), ("GNINA pose", "gnina")],
                    id="md-pose-source",
                    allow_blank=False,
                    value="unidock",
                )

                yield Label("Simulation preset", classes="form-label")
                yield Select(
                    [
                        ("Setup check - minimize only", "setup"),
                        ("Smoke - 10 ps", "smoke"),
                        ("Quick - 100 ps", "quick"),
                        ("Standard - 1 ns", "standard"),
                    ],
                    id="md-preset",
                    allow_blank=False,
                    value="smoke",
                )

                yield Label("Kaggle GPU", classes="form-label")
                yield Select(
                    [("T4 x 2", "nvidiaTeslaT4"), ("P100", "nvidiaTeslaP100")],
                    id="md-gpu",
                    allow_blank=False,
                    value="nvidiaTeslaT4",
                )

                with Horizontal(classes="form-row form-actions", id="md-action-row"):
                    yield Button("Run MD", id="btn-md-run", variant="primary")
                    yield Button("View MD Results", id="btn-md-results", variant="default")
                    yield Button("All MD Results", id="btn-md-results-all", variant="default")

                yield Static("", id="md-status", classes="form-status")
                yield Label("Progress", classes="form-section")
                yield ProgressBar(total=100, show_eta=False, id="md-progress")
                yield Static("Waiting for input.", id="md-progress-text", classes="form-status")
                yield Label("Log", classes="form-section")
                yield RichLog(id="md-log", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_account_summary()
        self._refresh_runs()
        run_id = getattr(self.app.nav, "selected_run_id", None) or self._latest_complete_run_id()
        if run_id:
            self._select_run(run_id)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "md-run-filter":
            self._refresh_runs(event.value)
            return
        if event.input.id == "md-hit-filter":
            self._apply_hit_filter()

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key is None:
            return
        if event.data_table.id == "md-runs":
            self._select_run(str(event.row_key.value))
            return
        if event.data_table.id == "md-leaderboard":
            self._select_hit(int(str(event.row_key.value)))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-md-run":
            self._start()
        elif event.button.id == "btn-md-results":
            self._open_md_results()
        elif event.button.id == "btn-md-results-all":
            self._open_all_md_results()

    def _refresh_account_summary(self) -> None:
        from ezscreen import auth

        label = "Using the required local Kaggle credential saved for this ezscreen account."
        try:
            path = auth.get_kaggle_json_path()
            if path and path.exists():
                username = auth.validate_kaggle_json(path).get("username", "")
                if username:
                    label = f"Using signed-in Kaggle account: {username}"
        except Exception:
            pass
        self.query_one("#md-account-summary", Static).update(label)

    def _refresh_runs(self, filter_text: str = "") -> None:
        rows = self._load_runs()
        needle = filter_text.strip().lower()
        if needle:
            rows = [
                row
                for row in rows
                if needle in row["run_id"].lower() or needle in row["created_at"].lower()
            ]

        rows.sort(key=lambda row: row["created_at"], reverse=True)
        self._run_rows = rows

        table = self.query_one("#md-runs", DataTable)
        table.clear(columns=True)
        table.add_columns("Run ID", "Created", "Hits")
        for row in rows[:500]:
            table.add_row(row["run_id"], row["created_at"], row["hit_count"], key=row["run_id"])

        summary = f"{len(rows)} completed runs with leaderboard data"
        if self._selected_run_id and any(row["run_id"] == self._selected_run_id for row in rows):
            summary += f" | selected: {self._selected_run_id}"
        self.query_one("#md-run-summary", Static).update(summary)

    def _load_runs(self) -> list[dict[str, str]]:
        base = Path.home() / ".ezscreen" / "runs"
        runs: dict[str, dict[str, str]] = {}

        def record(run_id: str, created_at: str) -> None:
            run_dir = base / run_id
            scores_csv = resolve_output_dir(run_dir) / "scores.csv"
            if not scores_csv.exists():
                return
            if run_id not in runs:
                runs[run_id] = {
                    "run_id": run_id,
                    "created_at": created_at,
                    "hit_count": str(self._count_hits(scores_csv)),
                }

        try:
            from ezscreen import checkpoint

            checkpoint.init_db()
            for row in checkpoint.list_runs():
                if str(row.get("status", "")).lower() != "complete":
                    continue
                run_id = str(row.get("run_id", "")).strip()
                if not run_id:
                    continue
                created_at = str(row.get("created_at", "")).strip()
                record(run_id, created_at)
        except Exception:
            pass

        if base.exists():
            for run_dir in base.iterdir():
                if not run_dir.is_dir() or run_dir.name in runs:
                    continue
                scores_csv = resolve_output_dir(run_dir) / "scores.csv"
                if not scores_csv.exists():
                    continue
                created = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc)
                runs[run_dir.name] = {
                    "run_id": run_dir.name,
                    "created_at": created.isoformat(timespec="seconds"),
                    "hit_count": str(self._count_hits(scores_csv)),
                }

        return list(runs.values())

    @staticmethod
    def _count_hits(scores_csv: Path) -> int:
        with scores_csv.open(newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))

    def _latest_complete_run_id(self) -> str:
        rows = self._load_runs()
        rows.sort(key=lambda row: row["created_at"], reverse=True)
        return rows[0]["run_id"] if rows else ""

    def _select_run(self, run_id: str) -> None:
        self._selected_run_id = run_id
        self.app.nav.selected_run_id = run_id
        self._selected_ligand_id = None
        self._update_selection_summary()
        self._refresh_md_job_summary()
        self._focus_run(run_id)
        self.query_one("#md-hit-summary", Static).update(f"Loading leaderboard for {run_id}...")
        self.query_one("#md-status", Static).update("")
        threading.Thread(target=self._load_hits_worker, args=(run_id,), daemon=True).start()

    def _focus_run(self, run_id: str) -> None:
        table = self.query_one("#md-runs", DataTable)
        for idx, row in enumerate(table.rows):
            if str(row.value) == run_id:
                table.move_cursor(row=idx)
                break

    def _load_hits_worker(self, run_id: str) -> None:
        scores_csv = resolve_output_dir(Path.home() / ".ezscreen" / "runs" / run_id) / "scores.csv"
        if not scores_csv.exists():
            self.app.call_from_thread(self._populate_hits_error, f"scores.csv not found for {run_id}")
            return
        with scores_csv.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.app.call_from_thread(self._set_hits, run_id, rows)

    def _set_hits(self, run_id: str, rows: list[dict[str, str]]) -> None:
        if run_id != self._selected_run_id:
            return
        self._all_hits = rows
        self._apply_hit_filter()

    def _apply_hit_filter(self) -> None:
        needle = self.query_one("#md-hit-filter", Input).value.strip().lower()
        rows = self._all_hits
        if needle:
            rows = [row for row in rows if any(needle in str(value).lower() for value in row.values())]
        self._visible_hits = rows
        self._populate_hits_table(rows)

    def _populate_hits_table(self, rows: list[dict[str, str]]) -> None:
        table = self.query_one("#md-leaderboard", DataTable)
        table.clear(columns=True)
        table.add_columns("#", "Ligand", "Name", "Score", "Variant")

        self._score_col = self._detect_score_col(rows)
        for idx, row in enumerate(rows[:500], start=1):
            table.add_row(
                str(idx),
                row.get("ligand", ""),
                row.get("name", ""),
                row.get(self._score_col, ""),
                row.get("receptor_variant", ""),
                key=str(idx - 1),
            )

        if not rows:
            self.query_one("#md-hit-summary", Static).update("No leaderboard rows match the current filter.")
            self._selected_ligand_id = None
            self._update_selection_summary()
            return

        self.query_one("#md-hit-summary", Static).update(
            f"{len(rows)} ligands available for {self._selected_run_id}"
        )
        self._select_hit(0)
        table.move_cursor(row=0)

    @staticmethod
    def _detect_score_col(rows: list[dict[str, str]]) -> str:
        if not rows:
            return "score"
        headers = list(rows[0].keys())
        for header in headers:
            lowered = header.lower()
            if "affinity" in lowered or "score" in lowered:
                return header
        return headers[-1] if headers else "score"

    def _select_hit(self, index: int) -> None:
        if index < 0 or index >= len(self._visible_hits):
            return
        row = self._visible_hits[index]
        self._selected_ligand_id = row.get("ligand") or row.get("name") or None
        score = row.get(self._score_col, "")
        status = f"Selected {self._selected_ligand_id or 'top-ranked hit'}"
        if score:
            status += f" | {self._score_col}: {score}"
        self.query_one("#md-status", Static).update(status)
        self._update_selection_summary()

    def _update_selection_summary(self) -> None:
        run_label = self._selected_run_id or "-"
        ligand_label = self._selected_ligand_id or "top-ranked hit"
        self.query_one("#md-selection", Static).update(f"Run: {run_label}\nLigand: {ligand_label}")

    def _populate_hits_error(self, msg: str) -> None:
        self._all_hits = []
        self._visible_hits = []
        table = self.query_one("#md-leaderboard", DataTable)
        table.clear(columns=True)
        table.add_column("Info")
        table.add_row(msg)
        self.query_one("#md-hit-summary", Static).update(msg)
        self._selected_ligand_id = None
        self._update_selection_summary()

    def _set_progress(self, msg: str, pct: float) -> None:
        bar = self.query_one("#md-progress", ProgressBar)
        bar.update(progress=max(0, min(100, pct * 100)))
        self.query_one("#md-progress-text", Static).update(msg)

    def _refresh_md_job_summary(self) -> None:
        from ezscreen import checkpoint

        try:
            checkpoint.init_db()
            rows = checkpoint.list_md_runs(self._selected_run_id)
        except Exception:
            rows = []
        if not rows:
            self.query_one("#md-jobs-summary", Static).update("No MD jobs recorded for this run yet.")
            return
        complete = sum(1 for row in rows if row.get("status") == "complete")
        running = sum(1 for row in rows if row.get("status") == "running")
        failed = sum(1 for row in rows if row.get("status") == "failed")
        self.query_one("#md-jobs-summary", Static).update(
            f"MD jobs for this run: {len(rows)} total | {complete} complete | {running} running | {failed} failed"
        )

    def _start(self) -> None:
        run_id = self._selected_run_id
        ligand_id = self._selected_ligand_id or (self._visible_hits[0].get("ligand") if self._visible_hits else None)
        preset = str(self.query_one("#md-preset", Select).value)
        pose_source = str(self.query_one("#md-pose-source", Select).value)
        accelerator = str(self.query_one("#md-gpu", Select).value)

        if not run_id:
            self._set_status("[#f85149]Select a completed run first.[/#f85149]")
            return
        if not ligand_id:
            self._set_status("[#f85149]Select a ligand from the leaderboard first.[/#f85149]")
            return

        self.query_one("#btn-md-run", Button).disabled = True
        self.query_one("#md-log", RichLog).clear()
        self.query_one("#md-progress", ProgressBar).update(progress=0)
        self._set_progress("Preparing MD job", 0.0)
        self._set_status("[#e3b341]Submitting to Kaggle...[/#e3b341]")
        self._log("[#79c0ff]Preparing OpenMM MD job...[/#79c0ff]")
        self.run_worker(lambda: self._do_run(run_id, ligand_id, preset, pose_source, accelerator), thread=True)

    def _do_run(self, run_id: str, ligand_id: str, preset: str, pose_source: str, accelerator: str) -> None:
        try:
            from ezscreen.backends.kaggle.md_runner import run_md_simulation

            result = run_md_simulation(
                run_id=run_id,
                ligand_id=ligand_id,
                preset=preset,
                pose_source=pose_source,
                accelerator=accelerator,
                progress_cb=lambda msg, pct: self.app.call_from_thread(self._set_progress, msg, pct),
            )
            self.app.call_from_thread(self._finish, result)
        except Exception as exc:
            self.app.call_from_thread(self._error, str(exc))

    def _finish(self, result: dict) -> None:
        if result.get("status") == "complete":
            out_dir = result.get("output_dir")
            self._set_status("[#3fb950]Done[/#3fb950]")
            self._set_progress("MD complete", 1.0)
            self._log(f"[#3fb950]Output -> {out_dir}[/#3fb950]")
            self._refresh_md_job_summary()
            self._open_md_results(result.get("md_run_id"))
        else:
            self._error(str(result.get("error") or result.get("status") or "unknown error"))
            return
        self.query_one("#btn-md-run", Button).disabled = False

    def _open_md_results(self, md_run_id: str | None = None) -> None:
        from ezscreen.tui.screens.md_results import MDResultsScreen

        self.app.push_screen(MDResultsScreen(self._selected_run_id, md_run_id))

    def _open_all_md_results(self) -> None:
        from ezscreen.tui.screens.md_results import MDResultsScreen

        self.app.push_screen(MDResultsScreen(None))

    def _error(self, msg: str) -> None:
        self._set_status(f"[#f85149]Error: {msg}[/#f85149]")
        self._set_progress("MD failed", 0.0)
        self._log(f"[#f85149]{msg}[/#f85149]")
        self._refresh_md_job_summary()
        self.query_one("#btn-md-run", Button).disabled = False

    def _log(self, msg: str) -> None:
        self.query_one("#md-log", RichLog).write(msg)

    def _set_status(self, msg: str) -> None:
        self.query_one("#md-status", Static).update(msg)
