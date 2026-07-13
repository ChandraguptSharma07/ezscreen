from __future__ import annotations

import json
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, Static

from ezscreen.results.md_viewer import (
    generate_structure_viewer,
    generate_trajectory_viewer,
)
from ezscreen.tui.widgets.breadcrumb import Breadcrumb

_STATUS_STYLE: dict[str, str] = {
    "complete": "bold #3fb950",
    "running": "bold #79c0ff",
    "failed": "bold #f85149",
}


class MDResultsScreen(Screen):
    """Browse completed and in-flight MD jobs for a screening run."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def __init__(self, run_id: str | None = None, md_run_id: str | None = None) -> None:
        super().__init__()
        self._run_id = run_id
        self._initial_md_run_id = md_run_id
        self._rows: list[dict] = []
        self._selected_md_run_id: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        label = self._run_id or "All"
        yield Breadcrumb(["Home", f"MD Results - {label}"])
        with Horizontal(id="md-results-main"):
            with Vertical(id="md-results-list"):
                yield Label("MD jobs", classes="section-title")
                yield Input(placeholder="Filter by ligand, status, or MD run ID", id="md-results-filter")
                yield DataTable(id="md-results-table", cursor_type="row")
                yield Static("", id="md-results-summary", classes="form-status")
            with VerticalScroll(id="md-results-detail"):
                yield Label("MD summary", classes="section-title")
                yield Static("Select an MD job to review its output.", id="md-results-card")
                yield Label("Time-series metrics", classes="section-title")
                yield Static("", id="md-metrics-card")
                yield Label("Actions", classes="section-title")
                with Vertical(id="md-results-actions"):
                    with Horizontal(classes="md-results-action-row"):
                        yield Button("Open MD report", id="btn-md-report")
                        yield Button("Open output folder", id="btn-md-folder")
                    with Horizontal(classes="md-results-action-row"):
                        yield Button("Open minimized structure", id="btn-md-min")
                        yield Button("Open final structure", id="btn-md-final")
                    with Horizontal(classes="md-results-action-row"):
                        yield Button("Open trajectory", id="btn-md-traj")
                yield Static("", id="md-results-files", classes="form-status")
        yield Footer()

    def on_mount(self) -> None:
        if not self._run_id:
            self._run_id = getattr(self.app.nav, "selected_run_id", None)
        self._set_empty_detail()
        self._populate_table()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "md-results-filter":
            self._populate_table(event.value)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id != "md-results-table" or event.row_key is None:
            return
        self._select_md_run(str(event.row_key.value))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if not self._selected_md_run_id:
            return
        row = next((item for item in self._rows if item["md_run_id"] == self._selected_md_run_id), None)
        if not row:
            return
        out_dir = Path(row.get("output_dir") or "")
        target = self._resolve_action_target(event.button.id or "", out_dir)
        if target and target.exists():
            self._open_path(target)

    def _populate_table(self, filter_text: str = "") -> None:
        from ezscreen import checkpoint

        checkpoint.init_db()
        rows = checkpoint.list_md_runs(self._run_id)
        needle = filter_text.strip().lower()
        if needle:
            rows = [
                row for row in rows
                if needle in str(row.get("md_run_id", "")).lower()
                or needle in str(row.get("ligand_id", "")).lower()
                or needle in str(row.get("status", "")).lower()
                or needle in str(row.get("preset", "")).lower()
            ]
        self._rows = rows

        table = self.query_one("#md-results-table", DataTable)
        table.clear(columns=True)
        table.add_columns("MD run", "Ligand", "Status", "Preset", "Created")
        if not rows:
            table.add_row("-", "-", "-", "-", "no MD jobs yet")
            self.query_one("#md-results-summary", Static).update("No MD jobs recorded for this run.")
            return
        for row in rows:
            style = _STATUS_STYLE.get(str(row.get("status", "")), "white")
            table.add_row(
                row.get("md_run_id", ""),
                row.get("ligand_id", ""),
                Text(str(row.get("status", "")), style=style),
                row.get("preset", ""),
                str(row.get("created_at", ""))[:19],
                key=row.get("md_run_id", ""),
            )
        self.query_one("#md-results-summary", Static).update(f"{len(rows)} MD job(s)")
        target_md_run_id = self._initial_md_run_id if any(row.get("md_run_id") == self._initial_md_run_id for row in rows) else rows[0].get("md_run_id", "")
        self._select_md_run(str(target_md_run_id))
        for idx, row in enumerate(rows):
            if row.get("md_run_id") == target_md_run_id:
                table.move_cursor(row=idx)
                break

    def _select_md_run(self, md_run_id: str) -> None:
        row = next((item for item in self._rows if item["md_run_id"] == md_run_id), None)
        if not row:
            return
        self._selected_md_run_id = md_run_id
        self._update_detail(row)

    def _update_detail(self, row: dict) -> None:
        out_dir = Path(row.get("output_dir") or "")
        self._ensure_report_artifacts(out_dir)
        summary = self._load_json(out_dir / "md_summary.json")
        metrics = self._load_json(out_dir / "md_metrics.json")
        files = [
            name for name in ["md_report.html", "complex_initial.pdb", "solvated.pdb", "minimized.pdb", "final.pdb", "trajectory.dcd", "state.csv"]
            if (out_dir / name).exists()
        ]

        detail = Text()
        detail.append(f"{row.get('md_run_id', '')}\n\n", style="bold #79c0ff")
        self._append_row(detail, "Run", row.get("run_id", ""))
        self._append_row(detail, "Ligand", row.get("ligand_id", ""))
        self._append_row(detail, "Status", row.get("status", ""), _STATUS_STYLE.get(str(row.get("status", "")), "white"))
        self._append_row(detail, "Preset", row.get("preset", ""))
        self._append_row(detail, "Pose", row.get("pose_source", ""))
        self._append_row(detail, "GPU", row.get("accelerator", ""))
        self._append_row(detail, "Platform", summary.get("platform", ""))
        self._append_row(detail, "Sim ps", summary.get("simulated_ps", ""))
        self._append_row(detail, "Pot E", summary.get("potential_energy_kj_mol", ""))
        self._append_row(detail, "Kin E", summary.get("kinetic_energy_kj_mol", ""))
        self._append_row(detail, "Overlap waters", summary.get("deleted_overlap_waters", ""))
        self._append_row(detail, "Chain splits", summary.get("protein_chain_splits", ""))
        if row.get("error_message"):
            self._append_row(detail, "Error", row.get("error_message", ""), "#f85149")
        self.query_one("#md-results-card", Static).update(detail)

        metrics_text = Text()
        if metrics:
            metrics_text.append(f"Frames: {metrics.get('frames', 0)}\n\n", style="#f0f6fc")
            for key, label in [("temperature", "Temperature"), ("potentialEnergy", "Potential E"), ("totalEnergy", "Total E")]:
                block = metrics.get(key)
                if not isinstance(block, dict):
                    continue
                metrics_text.append(f"{label}\n", style="bold #79c0ff")
                metrics_text.append(
                    f"  first {float(block.get('first', 0.0)):.3f}  last {float(block.get('last', 0.0)):.3f}\n  min {float(block.get('min', 0.0)):.3f}    max {float(block.get('max', 0.0)):.3f}\n\n",
                    style="#f0f6fc",
                )
        else:
            metrics_text.append("No time-series metrics were generated for this MD job.", style="#6e7681")
        self.query_one("#md-metrics-card", Static).update(metrics_text)
        self.query_one("#md-results-files", Static).update("Files: " + (", ".join(files) if files else "none yet"))
        self._refresh_action_buttons(out_dir)

    def _ensure_report_artifacts(self, out_dir: Path) -> None:
        if not out_dir.exists():
            return
        summary_path = out_dir / "md_summary.json"
        report_path = out_dir / "md_report.html"
        metrics_path = out_dir / "md_metrics.json"
        if not summary_path.exists():
            return
        should_regenerate = not report_path.exists() or not metrics_path.exists()
        if not should_regenerate and metrics_path.exists() and (out_dir / "state.csv").exists():
            metrics = self._load_json(metrics_path)
            metric_keys = {"temperature", "potentialEnergy", "kineticEnergy", "totalEnergy"}
            should_regenerate = not any(key in metrics for key in metric_keys)
        if not should_regenerate:
            return
        try:
            from ezscreen.backends.kaggle.md_runner import _write_md_report

            summary = self._load_json(summary_path)
            if summary:
                _write_md_report(out_dir, summary)
        except Exception:
            pass

    def _refresh_action_buttons(self, out_dir: Path) -> None:
        mapping = {
            "#btn-md-report": out_dir / "md_report.html",
            "#btn-md-folder": out_dir,
            "#btn-md-min": out_dir / "minimized.pdb",
            "#btn-md-final": out_dir / "final.pdb",
            "#btn-md-traj": out_dir / "md_report.html",
        }
        for selector, path in mapping.items():
            self.query_one(selector, Button).disabled = not path.exists()

    def _set_empty_detail(self) -> None:
        for selector in ["#btn-md-report", "#btn-md-folder", "#btn-md-min", "#btn-md-final", "#btn-md-traj"]:
            self.query_one(selector, Button).disabled = True

    @staticmethod
    def _open_path(path: Path) -> None:
        try:
            if path.suffix.lower() in {".html", ".htm"}:
                webbrowser.open(path.as_uri())
                return
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
                return
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
                return
            subprocess.Popen(["xdg-open", str(path)])
        except Exception:
            webbrowser.open(path.as_uri())

    @staticmethod
    def _load_json(path: Path) -> dict:
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}

    @staticmethod
    def _append_row(text: Text, label: str, value: object, style: str = "#f0f6fc") -> None:
        text.append(f"{label:<15}", style="#6e7681")
        text.append(f"{value}\n", style=style)

    @staticmethod
    def _resolve_action_target(action_id: str, out_dir: Path) -> Path | None:
        if action_id == "btn-md-report":
            return out_dir / "md_report.html"
        if action_id == "btn-md-folder":
            return out_dir
        if action_id == "btn-md-min":
            source = out_dir / "minimized.pdb"
            return generate_structure_viewer(out_dir, source.name) if source.exists() else None
        if action_id == "btn-md-final":
            source = out_dir / "final.pdb"
            return generate_structure_viewer(out_dir, source.name) if source.exists() else None
        if action_id == "btn-md-traj":
            report = out_dir / "md_report.html"
            if report.exists():
                return generate_trajectory_viewer(out_dir)
        return None
