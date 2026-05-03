from __future__ import annotations

import csv
import json
import webbrowser
from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, Static

from ezscreen.tui.widgets.breadcrumb import Breadcrumb

_SKIP_COLS = {"rmsd_lb", "rmsd_ub"}


class ResultsScreen(Screen):
    """Top hits table with compound detail panel and 3D viewer launch."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("o", "open_viewer", "Open 3D Viewer"),
    ]

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self._run_id    = run_id
        self._output    = Path.home() / ".ezscreen" / "runs" / run_id / "output"
        self._rows:  list[dict] = []
        self._headers: list[str] = []
        self._score_col: str = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Breadcrumb(["Home", f"Results — {self._run_id}"])
        with Horizontal(id="results-main"):
            with Vertical(id="hits-panel"):
                yield Label(f"Top Hits  ({self._run_id})", classes="section-title")
                yield DataTable(id="hits-table", cursor_type="row")
            with Vertical(id="compound-detail"):
                yield Label("Selected Compound", classes="section-title")
                yield Static(
                    "[#6e7681]Select a hit to see details.[/#6e7681]",
                    id="compound-info",
                )
                yield Button("Open 3D Viewer",        id="btn-3d",        variant="default")
                yield Button("Open Report",            id="btn-report",    variant="default")
                yield Button("Cluster Hits",           id="btn-cluster",   variant="default")
                yield Button("Analyse Interactions",   id="btn-plip",      variant="default")
                yield Static("", id="cluster-result")
                yield Static("", id="plip-result")
                yield Label("Validate Setup", classes="section-title", id="validate-label")
                yield Input(placeholder="Path to known actives (.smi)", id="actives-input")
                yield Button(
                    "Run Enrichment Benchmark", id="btn-validate", variant="primary"
                )
                yield Static("", id="benchmark-result")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#btn-3d").display      = False
        self.query_one("#btn-report").display  = False
        self.query_one("#btn-cluster").display = False
        self.query_one("#btn-plip").display    = False
        table = self.query_one("#hits-table", DataTable)
        table.add_column("Info")
        table.add_row("Loading results…")
        import threading
        threading.Thread(target=self._load_hits_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _load_hits_worker(self) -> None:
        """Read and process CSV in a background thread, then populate the table."""
        scores_csv = self._output / "scores.csv"

        if not scores_csv.exists():
            self.app.call_from_thread(self._populate_error, f"No results found in {self._output}")
            return

        with scores_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            self.app.call_from_thread(self._populate_error, "scores.csv is empty")
            return

        all_cols    = list(rows[0].keys())
        headers     = [h for h in all_cols if h not in _SKIP_COLS]
        score_col   = next(
            (h for h in headers if "score" in h.lower() or "affinity" in h.lower()),
            headers[-1],
        )

        self._rows      = rows
        self._headers   = headers
        self._score_col = score_col

        self.app.call_from_thread(self._populate_table, rows[:200], headers, score_col)
        self.app.call_from_thread(self._refresh_report_button)
        self.app.call_from_thread(self._refresh_plip_button)

    def _populate_error(self, msg: str) -> None:
        table = self.query_one("#hits-table", DataTable)
        table.clear(columns=True)
        table.add_column("Info")
        table.add_row(msg)

    def _populate_table(self, rows: list[dict], headers: list[str], score_col: str) -> None:
        table = self.query_one("#hits-table", DataTable)
        table.clear(columns=True)
        table.add_column("#", width=4)
        for h in headers:
            table.add_column(h.replace("_", " ").title())

        for i, row in enumerate(rows, 1):
            cells: list = [str(i)]
            for h in headers:
                val = row.get(h, "")
                if h == score_col:
                    cells.append(Text(val, style="bold #79c0ff"))
                elif h == "LE" and val:
                    try:
                        style = "#e3b341" if float(val) > 0.5 else "#8b949e"
                    except ValueError:
                        style = "#8b949e"
                    cells.append(Text(val, style=style))
                elif i <= 3:
                    cells.append(Text(val, style="#3fb950"))
                else:
                    cells.append(val)
            table.add_row(*cells, key=str(i - 1))

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        idx = int(event.row_key.value) if event.row_key else None
        if idx is None or idx >= len(self._rows):
            return

        row    = self._rows[idx]
        name   = row.get("name") or row.get("ligand", "—")
        score  = row.get(self._score_col, "—") if self._score_col else "—"
        smiles = row.get("smiles", "")

        lines = [
            f"[bold #f0f6fc]{name}[/bold #f0f6fc]",
            "",
            f"[#6e7681]Score:[/#6e7681]  [bold #79c0ff]{score} kcal/mol[/bold #79c0ff]",
        ]
        if smiles:
            truncated = smiles if len(smiles) <= 38 else smiles[:35] + "..."
            lines += ["", "[#6e7681]SMILES:[/#6e7681]", f"[#8b949e]{truncated}[/#8b949e]"]

        self.query_one("#compound-info", Static).update("\n".join(lines))
        self.query_one("#btn-3d").display = self._viewer_html() is not None

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-3d":
            self.action_open_viewer()
        elif event.button.id == "btn-report":
            self._open_report()
        elif event.button.id == "btn-cluster":
            self._run_clustering()
        elif event.button.id == "btn-plip":
            self._handle_plip()
        elif event.button.id == "btn-validate":
            self._run_benchmark()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_open_viewer(self) -> None:
        html = self._viewer_html()
        if html:
            webbrowser.open(html.as_uri())
            self.app.notify("Opened 3D viewer in browser.", timeout=3)
        else:
            self.app.notify("No 3D viewer HTML found for this run.", timeout=4)

    def _refresh_report_button(self) -> None:
        has_results = (self._output / "scores.csv").exists()
        self.query_one("#btn-report").display  = has_results
        self.query_one("#btn-cluster").display = has_results

    def _report_path(self) -> Path:
        return self._output / "results_report.html"

    def _open_report(self) -> None:
        scores_csv = self._output / "scores.csv"
        if not scores_csv.exists():
            self.app.notify("No docking results found for this run.", severity="error", timeout=5)
            return

        report = self._report_path()
        if report.exists():
            webbrowser.open(report.as_uri())
            self.app.notify("Report opened in browser.", timeout=3)
            return

        self.app.notify("Generating report...", timeout=3)
        self.query_one("#btn-report").disabled = True

        def _worker() -> None:
            from ezscreen.results.report_html import write_results_report
            try:
                write_results_report(scores_csv, report, run_id=self._run_id)
                self.app.call_from_thread(self._on_report_ready, report)
            except Exception as exc:
                self.app.call_from_thread(
                    self.app.notify,
                    f"Report generation failed: {exc}",
                    severity="error",
                    timeout=8,
                )
                self.app.call_from_thread(
                    setattr, self.query_one("#btn-report"), "disabled", False
                )

        import threading
        threading.Thread(target=_worker, daemon=True).start()

    def _on_report_ready(self, report: Path) -> None:
        self.query_one("#btn-report").disabled = False
        webbrowser.open(report.as_uri())
        self.app.notify("Report opened in browser.", timeout=3)

    def _run_clustering(self) -> None:
        if not self._rows:
            return
        self.query_one("#cluster-result", Static).update(
            "[#e3b341]Clustering...[/#e3b341]"
        )
        self.query_one("#btn-cluster").disabled = True

        rows       = self._rows
        score_col  = self._score_col
        output_dir = self._output

        def _worker() -> None:
            from ezscreen.results.clustering import (
                cluster_hits,
                export_centroids,
            )
            try:
                result = cluster_hits(rows, score_col)
                centroids_path = output_dir / "centroids.smi"
                export_centroids(rows, result, centroids_path)
                self.app.call_from_thread(self._show_cluster_result, result, centroids_path)
            except Exception as exc:
                self.app.call_from_thread(
                    self.query_one("#cluster-result", Static).update,
                    f"[#f85149]Clustering failed: {exc}[/#f85149]",
                )
                self.app.call_from_thread(
                    setattr, self.query_one("#btn-cluster"), "disabled", False
                )

        import threading
        threading.Thread(target=_worker, daemon=True).start()

    def _show_cluster_result(self, result, centroids_path: Path) -> None:
        self.query_one("#btn-cluster").disabled = False
        if result.n_clusters == 0:
            self.query_one("#cluster-result", Static).update(
                "[#e3b341]No SMILES data available for clustering.[/#e3b341]"
            )
            return
        lines = [
            "[bold #3fb950]Clustering complete[/bold #3fb950]",
            "",
            f"[#6e7681]Clusters:[/#6e7681]  [bold #79c0ff]{result.n_clusters}[/bold #79c0ff]",
            f"[#6e7681]Largest:[/#6e7681]   {max(result.sizes)} compounds",
            f"[#6e7681]Singletons:[/#6e7681] {result.sizes.count(1)}",
            f"[#6e7681]Centroids → {centroids_path.name}[/#6e7681]",
        ]
        self.query_one("#cluster-result", Static).update("\n".join(lines))
        self.app.notify(f"{result.n_clusters} clusters. Centroids saved.", timeout=4)

    def _refresh_plip_button(self) -> None:
        has_results = (self._output / "scores.csv").exists()
        btn = self.query_one("#btn-plip")
        btn.display = has_results
        if (self._output / "interactions_top_n.json").exists():
            btn.label = "Open Interaction Viewer"
        else:
            btn.label = "Analyse Interactions"

    def _handle_plip(self) -> None:
        interactions_json = self._output / "interactions_top_n.json"
        work_dir = self._output.parent

        # Check receptor PDB availability
        resume_json = work_dir / "resume.json"
        has_receptor_pdb = False
        if resume_json.exists():
            info = json.loads(resume_json.read_text())
            p = info.get("receptor_pdb")
            has_receptor_pdb = bool(p and Path(p).exists())
        if not has_receptor_pdb:
            fallback = work_dir / "receptor" / "receptor_prep.pdb"
            has_receptor_pdb = fallback.exists()

        if not has_receptor_pdb:
            self.query_one("#plip-result", Static).update(
                "[#f85149]Interaction analysis unavailable — receptor PDB not saved for this run (pre-dates v1.9.0)[/#f85149]"
            )
            return

        if interactions_json.exists():
            self._open_interaction_viewer(work_dir)
            return

        self.query_one("#btn-plip").disabled = True
        self.query_one("#btn-plip").label = "Analysing..."
        self.query_one("#plip-result", Static).update("[#e3b341]Running PLIP on Kaggle...[/#e3b341]")

        run_id = self._run_id

        def _worker() -> None:
            from ezscreen.backends.kaggle.plip_runner import run_plip_analysis
            try:
                result = run_plip_analysis(run_id, work_dir)
                self.app.call_from_thread(self._on_plip_done, result, work_dir)
            except Exception as exc:
                self.app.call_from_thread(self._on_plip_error, str(exc))

        import threading
        threading.Thread(target=_worker, daemon=True).start()

    def _on_plip_done(self, result: dict, work_dir: Path) -> None:
        btn = self.query_one("#btn-plip")
        btn.disabled = False
        if result["status"] == "complete":
            btn.label = "Open Interaction Viewer"
            self.query_one("#plip-result", Static).update("[#3fb950]PLIP complete[/#3fb950]")
            self._open_interaction_viewer(work_dir)
        else:
            btn.label = "Analyse Interactions"
            self.query_one("#plip-result", Static).update(
                f"[#f85149]PLIP failed: {result.get('error', 'unknown error')}[/#f85149]"
            )

    def _on_plip_error(self, msg: str) -> None:
        self.query_one("#btn-plip").disabled = False
        self.query_one("#btn-plip").label = "Analyse Interactions"
        self.query_one("#plip-result", Static).update(f"[#f85149]PLIP error: {msg}[/#f85149]")

    def _open_interaction_viewer(self, work_dir: Path) -> None:
        from ezscreen.results.pose_inspector import generate_viewer
        try:
            html_path = generate_viewer(work_dir)
            webbrowser.open(html_path.as_uri())
            self.app.notify("Interaction viewer opened in browser.", timeout=3)
        except Exception as exc:
            self.app.notify(f"Viewer generation failed: {exc}", severity="error", timeout=8)

    def _run_benchmark(self) -> None:
        actives_str = self.query_one("#actives-input", Input).value.strip()
        if not actives_str:
            self.app.notify("Enter the path to your known actives file.", timeout=4)
            return

        actives_path = Path(actives_str).expanduser()
        if not actives_path.exists():
            self.app.notify(f"File not found: {actives_path}", severity="error", timeout=5)
            return

        scores_csv = self._output / "scores.csv"
        if not scores_csv.exists():
            self.app.notify("No docking results found for this run.", severity="error", timeout=5)
            return

        self.query_one("#benchmark-result", Static).update(
            "[#e3b341]Running benchmark...[/#e3b341]"
        )

        def _worker() -> None:
            from ezscreen.benchmark.runner import run_benchmark
            from ezscreen.results.report_html import write_benchmark_report

            try:
                result = run_benchmark(actives_path, scores_csv)
                report_path = self._output / "benchmark_report.html"
                write_benchmark_report(result, report_path)
                self.call_from_thread(self._show_benchmark_result, result, report_path)
            except Exception as exc:
                self.call_from_thread(
                    self.app.notify,
                    f"Benchmark failed: {exc}",
                    severity="error",
                    timeout=8,
                )

        import threading
        threading.Thread(target=_worker, daemon=True).start()

    def _show_benchmark_result(self, result, report_path: Path) -> None:

        lines = [
            "[bold #3fb950]Benchmark complete[/bold #3fb950]",
            "",
            f"[#6e7681]EF 1%:[/#6e7681]  [bold #79c0ff]{result.ef1:.2f}x[/bold #79c0ff]",
            f"[#6e7681]EF 5%:[/#6e7681]  [bold #79c0ff]{result.ef5:.2f}x[/bold #79c0ff]",
            f"[#6e7681]AUC-ROC:[/#6e7681] [bold #79c0ff]{result.auc_roc:.3f}[/bold #79c0ff]",
            f"[#6e7681]Actives matched:[/#6e7681] {result.n_actives} / {result.total_screened}",
        ]
        self.query_one("#benchmark-result", Static).update("\n".join(lines))

        if report_path.exists():
            webbrowser.open(report_path.as_uri())
            self.app.notify("Report opened in browser.", timeout=3)

    def _viewer_html(self) -> Path | None:
        p = self._output / "viewer.html"
        return p if p.exists() and p.stat().st_size > 0 else None
