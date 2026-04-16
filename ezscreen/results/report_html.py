from __future__ import annotations

import base64
import io
from pathlib import Path

from ezscreen.benchmark.metrics import BenchmarkResult

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ezscreen benchmark report</title>
<style>
  body {{ font-family: sans-serif; max-width: 900px; margin: 40px auto; color: #222; }}
  h1 {{ color: #1a3a5c; }}
  h2 {{ color: #2c5f8a; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
  .metrics {{ display: flex; gap: 24px; flex-wrap: wrap; margin: 16px 0; }}
  .badge {{
    background: #1a3a5c; color: #fff;
    border-radius: 8px; padding: 16px 24px;
    text-align: center; min-width: 120px;
  }}
  .badge .value {{ font-size: 2em; font-weight: bold; }}
  .badge .label {{ font-size: 0.85em; opacity: 0.85; margin-top: 4px; }}
  .note {{ color: #666; font-size: 0.9em; }}
  img {{ max-width: 100%; margin-top: 12px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td, th {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #f4f4f4; }}
</style>
</head>
<body>
<h1>Docking Benchmark Report</h1>

<h2>Enrichment Summary</h2>
<div class="metrics">
  <div class="badge">
    <div class="value">{ef1:.2f}x</div>
    <div class="label">EF 1%</div>
  </div>
  <div class="badge">
    <div class="value">{ef5:.2f}x</div>
    <div class="label">EF 5%</div>
  </div>
  <div class="badge">
    <div class="value">{auc:.3f}</div>
    <div class="label">AUC-ROC</div>
  </div>
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

</body>
</html>
"""


def _roc_plot_b64(result: BenchmarkResult) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_actives = result.n_actives
    n_decoys  = result.n_decoys

    tpr = [0.0]
    fpr = [0.0]
    tp = fp = 0
    for label in result.ranked_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / n_actives if n_actives else 0)
        fpr.append(fp / n_decoys  if n_decoys  else 0)

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
    roc_b64 = _roc_plot_b64(result)
    html = _HTML_TEMPLATE.format(
        ef1=result.ef1,
        ef5=result.ef5,
        auc=result.auc_roc,
        n_actives=result.n_actives,
        n_decoys=result.n_decoys,
        total=result.total_screened,
        roc_b64=roc_b64,
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path
