from __future__ import annotations

import base64
import csv
import io
from pathlib import Path

from ezscreen.benchmark.metrics import BenchmarkResult

# ---------------------------------------------------------------------------
# Shared CSS
# ---------------------------------------------------------------------------

_CSS = """
  body { font-family: sans-serif; max-width: 1000px; margin: 40px auto; color: #222; }
  h1   { color: #1a3a5c; }
  h2   { color: #2c5f8a; border-bottom: 1px solid #ccc; padding-bottom: 4px; margin-top: 36px; }
  .metrics { display: flex; gap: 24px; flex-wrap: wrap; margin: 16px 0; }
  .badge {
    background: #1a3a5c; color: #fff;
    border-radius: 8px; padding: 16px 24px;
    text-align: center; min-width: 120px;
  }
  .badge .value { font-size: 2em; font-weight: bold; }
  .badge .label { font-size: 0.85em; opacity: 0.85; margin-top: 4px; }
  .note  { color: #666; font-size: 0.9em; }
  img    { max-width: 100%; margin-top: 12px; }
  table  { border-collapse: collapse; width: 100%; margin-top: 8px; }
  td, th { border: 1px solid #ddd; padding: 8px 12px; text-align: left; font-size: 0.9em; }
  th     { background: #f4f4f4; font-weight: 600; }
  .plots { display: flex; gap: 16px; flex-wrap: wrap; }
  .plots img { max-width: 48%; }
  .structs { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 12px; }
  .struct-card {
    border: 1px solid #ddd; border-radius: 6px; padding: 10px;
    text-align: center; width: 180px; font-size: 0.82em;
  }
  .struct-card svg, .struct-card img { width: 160px; height: 120px; }
  .rank { font-weight: bold; color: #1a3a5c; font-size: 1.1em; }
  .score { color: #2c5f8a; }
"""

# ---------------------------------------------------------------------------
# Benchmark report (unchanged public API)
# ---------------------------------------------------------------------------

_BENCHMARK_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>ezscreen benchmark report</title>
<style>{css}</style></head>
<body>
<h1>Docking Benchmark Report</h1>

<h2>Enrichment Summary</h2>
<div class="metrics">
  <div class="badge"><div class="value">{ef1:.2f}x</div><div class="label">EF 1%</div></div>
  <div class="badge"><div class="value">{ef5:.2f}x</div><div class="label">EF 5%</div></div>
  <div class="badge"><div class="value">{auc:.3f}</div><div class="label">AUC-ROC</div></div>
</div>

<table>
  <tr><th>Actives</th><th>Decoys</th><th>Total screened</th></tr>
  <tr><td>{n_actives}</td><td>{n_decoys}</td><td>{total}</td></tr>
</table>

<p class="note">
  EF = enrichment factor (how many times more actives appear in the top X% vs random).
  AUC-ROC = area under the receiver operating characteristic curve (0.5 = random, 1.0 = perfect).
</p>

<h2>ROC Curve</h2>
<img src="data:image/png;base64,{roc_b64}" alt="ROC curve">

</body></html>
"""


def _roc_plot_b64(result: BenchmarkResult) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tpr = [0.0]
    fpr = [0.0]
    tp = fp = 0
    for label in result.ranked_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / result.n_actives if result.n_actives else 0)
        fpr.append(fp / result.n_decoys  if result.n_decoys  else 0)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#2c5f8a", linewidth=2, label=f"AUC = {result.auc_roc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#aaa", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def write_benchmark_report(result: BenchmarkResult, output_path: Path) -> Path:
    html = _BENCHMARK_TEMPLATE.format(
        css=_CSS,
        ef1=result.ef1,
        ef5=result.ef5,
        auc=result.auc_roc,
        n_actives=result.n_actives,
        n_decoys=result.n_decoys,
        total=result.total_screened,
        roc_b64=_roc_plot_b64(result),
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Docking results report
# ---------------------------------------------------------------------------

def _detect_score_col(headers: list[str]) -> str:
    for h in headers:
        if "score" in h.lower() or "affinity" in h.lower():
            return h
    return headers[-1]


def _compute_props(smiles_list: list[str]) -> list[dict]:
    try:
        from rdkit import Chem
        from rdkit.Chem.Descriptors import MolLogP, MolWt
    except ImportError:
        return [{} for _ in smiles_list]

    out = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                out.append({"mw": MolWt(mol), "logp": MolLogP(mol)})
            else:
                out.append({})
        except Exception:
            out.append({})
    return out


def _score_histogram_b64(scores: list[float], score_col: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(scores, bins=40, color="#2c5f8a", edgecolor="#1a3a5c", alpha=0.85)
    ax.set_xlabel(score_col.replace("_", " ").title())
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _scatter_b64(
    scores: list[float],
    prop_vals: list[float],
    prop_name: str,
    score_col: str,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paired = [(s, p) for s, p in zip(scores, prop_vals) if p is not None]
    if not paired:
        return ""
    xs, ys = zip(*paired)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(ys, xs, alpha=0.4, s=12, color="#2c5f8a")
    ax.set_xlabel(prop_name)
    ax.set_ylabel(score_col.replace("_", " ").title())
    ax.set_title(f"Score vs {prop_name}")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _mol_svg(smiles: str, width: int = 160, height: int = 120) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return ""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.drawOptions().addStereoAnnotation = False
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        # strip XML declaration so it embeds cleanly
        return svg[svg.find("<svg"):]
    except Exception:
        return ""


def _struct_cards_html(rows: list[dict], score_col: str, n: int = 10) -> str:
    cards = []
    for i, row in enumerate(rows[:n], 1):
        smi   = row.get("smiles", "")
        name  = row.get("name", f"#{i}")
        score = row.get(score_col, "—")
        svg   = _mol_svg(smi) if smi else ""
        struct_html = svg if svg else "<div style='height:120px;line-height:120px;color:#aaa'>no structure</div>"
        cards.append(
            f'<div class="struct-card">'
            f'<div class="rank">#{i}</div>'
            f'{struct_html}'
            f'<div>{name}</div>'
            f'<div class="score">{score}</div>'
            f'</div>'
        )
    return "\n".join(cards)


def _metadata_table_html(run_id: str, n_total: int, top_score: str, metadata: dict) -> str:
    rows = [
        ("Run ID",          run_id or "—"),
        ("Total compounds", f"{n_total:,}"),
        ("Top score",       top_score),
    ]
    for k, v in metadata.items():
        rows.append((k.replace("_", " ").title(), str(v)))

    trs = "\n".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<table>{trs}</table>"


def _cluster_section_html(
    rows: list[dict],
    score_col: str,
    tanimoto_cutoff: float = 0.4,
) -> str:
    from ezscreen.results.clustering import cluster_hits

    try:
        result = cluster_hits(rows, score_col, tanimoto_cutoff)
    except Exception:
        return ""

    cards = []
    for rank, (ci, size) in enumerate(
        sorted(zip(result.centroid_indices, result.sizes), key=lambda x: -x[1])[:20], 1
    ):
        row   = rows[ci]
        smi   = row.get("smiles", "")
        name  = row.get("name", f"#{ci}")
        score = row.get(score_col, "—")
        svg   = _mol_svg(smi) if smi else ""
        struct = svg if svg else "<div style='height:120px'></div>"
        cards.append(
            f'<div class="struct-card">'
            f'<div class="rank">Cluster {rank}</div>'
            f'{struct}'
            f'<div>{name}</div>'
            f'<div class="score">{score}</div>'
            f'<div style="color:#888;font-size:0.8em">{size} compounds</div>'
            f'</div>'
        )

    n = result.n_clusters
    singletons = result.sizes.count(1)
    summary = (
        f"<p>{n} clusters &nbsp;·&nbsp; largest: {max(result.sizes)} &nbsp;·&nbsp; "
        f"singletons: {singletons} &nbsp;·&nbsp; cutoff: {tanimoto_cutoff} Tc</p>"
    )
    return summary + '<div class="structs">' + "\n".join(cards) + "</div>"


def _interactions_heatmap_html(
    interactions: dict[str, dict[str, dict[str, int]]],
) -> str:
    # collect all residues and interaction types
    residues: list[str] = []
    itypes: list[str] = []
    for contacts in interactions.values():
        for res, imap in contacts.items():
            if res not in residues:
                residues.append(res)
            for it in imap:
                if it not in itypes:
                    itypes.append(it)

    if not residues:
        return "<p>No interactions found.</p>"

    # colour per interaction type
    _ITYPE_COLOURS = {
        "HBDonor":     "#4e91d0",
        "HBAcceptor":  "#3fb950",
        "Hydrophobic": "#e3b341",
        "Cationic":    "#f85149",
        "Anionic":     "#bc8cff",
        "PiStacking":  "#79c0ff",
        "PiCation":    "#ff9800",
    }

    header = "<tr><th>Compound</th>" + "".join(f"<th>{r}</th>" for r in residues[:30]) + "</tr>"
    rows_html = []
    for name, contacts in list(interactions.items())[:20]:
        cells = [f"<td><b>{name}</b></td>"]
        for res in residues[:30]:
            imap = contacts.get(res, {})
            if imap:
                dominant = max(imap, key=imap.get)
                colour = _ITYPE_COLOURS.get(dominant, "#8b949e")
                title = ", ".join(f"{k}:{v}" for k, v in imap.items())
                cells.append(
                    f'<td style="background:{colour};color:#fff;text-align:center" title="{title}">'
                    f'{dominant[:3]}</td>'
                )
            else:
                cells.append("<td></td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    legend = "".join(
        f'<span style="background:{c};color:#fff;padding:2px 8px;margin:2px;border-radius:4px;'
        f'font-size:0.8em">{t}</span>'
        for t, c in _ITYPE_COLOURS.items()
    )

    return (
        f'<p style="font-size:0.85em;color:#666">Top 30 residues · top 20 compounds</p>'
        f"<div style='margin-bottom:8px'>{legend}</div>"
        f"<div style='overflow-x:auto'><table>{header}{''.join(rows_html)}</table></div>"
    )


def write_results_report(
    scores_csv: Path,
    output_path: Path,
    run_id: str = "",
    metadata: dict | None = None,
    cluster: bool = True,
    interactions: dict | None = None,
) -> Path:
    metadata = metadata or {}

    with scores_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        output_path.write_text("<html><body>No results.</body></html>", encoding="utf-8")
        return output_path

    headers   = list(rows[0].keys())
    score_col = _detect_score_col(headers)

    scores: list[float] = []
    for r in rows:
        try:
            scores.append(float(r[score_col]))
        except (ValueError, KeyError):
            pass

    top_score = f"{min(scores):.3f}" if scores else "—"
    meta_html = _metadata_table_html(run_id, len(rows), top_score, metadata)

    # property computation for scatter plots
    smiles_list = [r.get("smiles", "") for r in rows]
    props       = _compute_props(smiles_list)
    mw_vals     = [p.get("mw") for p in props]
    logp_vals   = [p.get("logp") for p in props]

    hist_b64    = _score_histogram_b64(scores, score_col) if scores else ""
    mw_b64      = _scatter_b64(scores, mw_vals,   "MW (Da)",   score_col)
    logp_b64    = _scatter_b64(scores, logp_vals,  "LogP",      score_col)
    structs_html = _struct_cards_html(rows, score_col)

    plots_html = ""
    for b64, alt in [(hist_b64, "Score histogram"), (mw_b64, "Score vs MW"), (logp_b64, "Score vs LogP")]:
        if b64:
            plots_html += f'<img src="data:image/png;base64,{b64}" alt="{alt}">\n'

    cluster_html = _cluster_section_html(rows, score_col) if cluster else ""
    interactions_html = _interactions_heatmap_html(interactions) if interactions else ""

    cluster_section = (
        f"\n<h2>Scaffold Clusters</h2>\n{cluster_html}" if cluster_html else ""
    )
    interactions_section = (
        f"\n<h2>Interaction Fingerprints</h2>\n{interactions_html}" if interactions_html else ""
    )

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>ezscreen results — {run_id}</title>
<style>{_CSS}</style></head>
<body>
<h1>Docking Results Report</h1>

<h2>Run Summary</h2>
{meta_html}

<h2>Score Distribution &amp; Property Scatter</h2>
<div class="plots">
{plots_html}
</div>

<h2>Top 10 Compounds</h2>
<div class="structs">
{structs_html}
</div>
{cluster_section}
{interactions_section}
</body></html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path
