from __future__ import annotations

import csv
import webbrowser
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def invoke(results_dir: Path, top_n: int = 25) -> None:
    """Open the results viewer — Rich table + self-contained py3Dmol HTML."""
    scores_csv = results_dir / "scores.csv"
    poses_sdf  = results_dir / "poses.sdf"

    if not scores_csv.exists():
        console.print(f"[red]scores.csv not found in {results_dir}[/red]")
        return

    with scores_csv.open() as f:
        rows = list(csv.DictReader(f))

    if not rows:
        console.print("[dim]No results in scores.csv[/dim]")
        return

    # Detect which column is the score
    headers    = list(rows[0].keys())
    score_col  = next((h for h in headers if "score" in h.lower() or "affinity" in h.lower()), headers[-1])

    # Rich table
    t = Table(title=f"Top {min(top_n, len(rows))} docking hits", show_lines=False, expand=False)
    t.add_column("Rank", justify="right", style="dim", no_wrap=True)
    for h in headers:
        t.add_column(h, no_wrap=True)

    for i, row in enumerate(rows[:top_n], 1):
        style = "bold green" if i <= 3 else ("green" if i <= 10 else "")
        t.add_row(str(i), *[row[h] for h in headers], style=style)

    console.print(t)
    console.print(f"\n  [dim]{len(rows)} total poses  ·  sorted by {score_col}[/dim]")

    # 3D HTML viewer
    if poses_sdf.exists():
        html_path = results_dir / "viewer.html"
        _write_viewer(poses_sdf, rows[:top_n], html_path, score_col, headers)
        console.print(f"  3D viewer → [bold]{html_path}[/bold]")
        webbrowser.open(html_path.as_uri())


def _write_viewer(
    poses_sdf: Path,
    rows: list[dict],
    out: Path,
    score_col: str,
    headers: list[str],
) -> None:
    sdf_js = poses_sdf.read_text().replace("\\", "\\\\").replace("`", "\\`")
    id_col = headers[0]

    hit_items = "\n".join(
        f'<div class="hit" onclick="showPose({i})">'
        f'<span class="rank">#{i+1}</span> '
        f'<span class="name">{row.get(id_col, "")}</span> '
        f'<span class="score">{row.get(score_col, "")}</span>'
        f'</div>'
        for i, row in enumerate(rows)
    )

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>ezscreen — Results Viewer</title>
<script src="https://3dmol.org/build/3Dmol-min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0c0c14; color: #c9d1d9; font-family: 'Segoe UI', monospace; display: flex; height: 100vh; overflow: hidden; }}
  #viewer {{ flex: 1; }}
  #panel {{ width: 300px; background: #10101a; border-left: 1px solid #21262d; display: flex; flex-direction: column; }}
  #panel h2 {{ padding: 12px 16px; font-size: 13px; color: #58a6ff; border-bottom: 1px solid #21262d; letter-spacing: 1px; text-transform: uppercase; }}
  #hits {{ overflow-y: auto; flex: 1; padding: 8px; }}
  .hit {{ padding: 8px 10px; cursor: pointer; border-radius: 6px; border-left: 3px solid transparent; margin-bottom: 4px; transition: all .15s; }}
  .hit:hover {{ background: #161b22; border-color: #58a6ff; }}
  .hit.active {{ background: #0d1117; border-color: #3fb950; }}
  .rank {{ color: #8b949e; font-size: 11px; margin-right: 6px; }}
  .name {{ font-weight: 600; font-size: 12px; }}
  .score {{ float: right; color: #3fb950; font-size: 12px; font-weight: bold; }}
</style></head>
<body>
<div id="viewer"></div>
<div id="panel">
  <h2>Top Hits ({len(rows)})</h2>
  <div id="hits">{hit_items}</div>
</div>
<script>
const viewer = $3Dmol.createViewer('viewer', {{backgroundColor: '0x0c0c14'}});
const sdf = `{sdf_js}`;
viewer.addModelsAsFrames(sdf, 'sdf');
viewer.setStyle({{}}, {{stick: {{radius: 0.15}}, sphere: {{radius: 0.35}}}});
viewer.zoomTo(); viewer.render();
function showPose(i) {{
  document.querySelectorAll('.hit').forEach((el, j) => el.classList.toggle('active', i === j));
  viewer.setFrame(i); viewer.render();
}}
document.querySelectorAll('.hit')[0]?.classList.add('active');
</script></body></html>"""
    out.write_text(html)
