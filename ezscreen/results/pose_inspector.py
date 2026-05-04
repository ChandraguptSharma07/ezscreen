from __future__ import annotations

import json
from pathlib import Path

_INTERACTION_COLORS = {
    "hbond":       "#3b82f6",
    "hydrophobic": "#f97316",
    "pi_stack":    "#22c55e",
    "pi_cation":   "#a855f7",
    "salt_bridge": "#ef4444",
    "halogen":     "#14b8a6",
}

_3DMOL_CDN = "https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"


def _build_html(compounds: list[dict], receptor_pdb_text: str) -> str:
    receptor_escaped = receptor_pdb_text.replace("`", "\\`").replace("\\", "\\\\")
    compounds_json   = json.dumps(compounds)
    colors_json      = json.dumps(_INTERACTION_COLORS)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Interaction Viewer</title>
<script src="{_3DMOL_CDN}"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #0d1117; color: #c9d1d9;
         display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}
  #banner {{ background: #161b22; border-bottom: 1px solid #30363d; padding: 6px 16px;
             font-size: 12px; color: #8b949e; text-align: center; }}
  #main {{ display: flex; flex: 1; overflow: hidden; }}
  #viewer {{ flex: 1; position: relative; }}
  #sidebar {{ width: 320px; background: #161b22; border-left: 1px solid #30363d;
              display: flex; flex-direction: column; overflow: hidden; }}
  #controls {{ padding: 12px; border-bottom: 1px solid #30363d; }}
  #controls select {{ width: 100%; background: #21262d; color: #c9d1d9;
                      border: 1px solid #30363d; border-radius: 6px; padding: 6px; font-size: 13px; }}
  #toggles {{ padding: 10px 12px; border-bottom: 1px solid #30363d; }}
  .toggle-row {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; font-size: 12px; cursor: pointer; }}
  .toggle-row input {{ width: 14px; height: 14px; cursor: pointer; }}
  .color-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  #ilist {{ flex: 1; overflow-y: auto; padding: 8px; }}
  .i-row {{ background: #21262d; border-radius: 6px; margin: 4px 0; padding: 8px; font-size: 12px; }}
  .i-type {{ font-weight: 600; text-transform: capitalize; }}
  .i-detail {{ color: #8b949e; margin-top: 2px; }}
  #no-compound {{ padding: 16px; color: #8b949e; font-size: 13px; text-align: center; }}
  h3 {{ font-size: 12px; font-weight: 600; color: #8b949e; margin-bottom: 8px; letter-spacing: .05em; text-transform: uppercase; }}
</style>
</head>
<body>

<div id="banner">⚠ Predicted pose — not experimentally validated</div>

<div id="main">
  <div id="viewer"></div>
  <div id="sidebar">
    <div id="controls">
      <h3>Compound</h3>
      <select id="compound-select" onchange="selectCompound(this.value)">
        <option value="">— select compound —</option>
      </select>
    </div>
    <div id="toggles">
      <h3>Interaction types</h3>
      <!-- populated by JS -->
    </div>
    <div id="ilist"><div id="no-compound">Select a compound above</div></div>
  </div>
</div>

<script>
const COMPOUNDS   = {compounds_json};
const COLORS      = {colors_json};
const RECEPTOR    = `{receptor_escaped}`;

const viewer = $3Dmol.createViewer("viewer", {{ backgroundColor: "#0d1117" }});

// Load receptor once
viewer.addModel(RECEPTOR, "pdb");
viewer.setStyle({{ model: 0 }}, {{ cartoon: {{ color: "spectrum", opacity: 0.85 }} }});
viewer.addSurface($3Dmol.SurfaceType.MS, {{ opacity: 0.25, color: "#556677" }}, {{ model: 0 }});
viewer.zoomTo({{ model: 0 }});
viewer.render();

let ligandModel   = null;
let activeToggles = Object.fromEntries(Object.keys(COLORS).map(k => [k, true]));
let currentShapes = [];
let currentData   = null;

// Build compound dropdown
const sel = document.getElementById("compound-select");
COMPOUNDS.forEach(c => {{
  const opt = document.createElement("option");
  opt.value = c.lig_id;
  opt.textContent = `#${{c.rank}} ${{c.name || c.lig_id}} (${{c.score}} kcal/mol)`;
  sel.appendChild(opt);
}});

// Build toggles
const togglesDiv = document.getElementById("toggles");
Object.entries(COLORS).forEach(([type, color]) => {{
  const row = document.createElement("label");
  row.className = "toggle-row";
  row.innerHTML = `<input type="checkbox" checked onchange="toggleType('${{type}}', this.checked)">
    <span class="color-dot" style="background:${{color}}"></span>
    <span>${{type.replace("_", " ")}}</span>`;
  togglesDiv.appendChild(row);
}});

function clearLigand() {{
  if (ligandModel !== null) {{ viewer.removeModel(ligandModel); ligandModel = null; }}
  currentShapes.forEach(s => viewer.removeShape(s));
  currentShapes = [];
  viewer.setStyle({{ model: 0 }}, {{ cartoon: {{ color: "spectrum", opacity: 0.85 }} }});
}}

function drawInteractions(interactions) {{
  interactions.forEach(ix => {{
    if (!activeToggles[ix.type]) return;
    const color = COLORS[ix.type] || "#ffffff";
    const s = viewer.addCylinder({{
      start: {{ x: ix.protein_coords[0], y: ix.protein_coords[1], z: ix.protein_coords[2] }},
      end:   {{ x: ix.ligand_coords[0],  y: ix.ligand_coords[1],  z: ix.ligand_coords[2]  }},
      radius: 0.12, color: color, opacity: 0.85, dashed: true,
    }});
    currentShapes.push(s);
  }});
  viewer.render();
}}

function renderSidebar(compound) {{
  const list = document.getElementById("ilist");
  list.innerHTML = "";
  if (!compound || compound.plip_failed) {{
    list.innerHTML = `<div id="no-compound" style="color:#f85149">${{
      compound ? (compound.plip_error || "PLIP analysis failed") : "No data"
    }}</div>`;
    return;
  }}
  const active = compound.interactions.filter(ix => activeToggles[ix.type]);
  if (!active.length) {{
    list.innerHTML = `<div id="no-compound">No visible interactions (check toggles)</div>`;
    return;
  }}
  active.forEach(ix => {{
    const div = document.createElement("div");
    div.className = "i-row";
    const color = COLORS[ix.type] || "#ffffff";
    div.innerHTML = `<div class="i-type" style="color:${{color}}">${{ix.type.replace("_"," ")}}</div>
      <div class="i-detail">${{ix.residue_name}} ${{ix.residue_number}} (${{ix.chain}})
        · ${{ix.distance.toFixed(2)}} Å</div>`;
    list.appendChild(div);
  }});
}}

function selectCompound(lig_id) {{
  clearLigand();
  if (!lig_id) return;
  const compound = COMPOUNDS.find(c => c.lig_id === lig_id);
  if (!compound) return;
  currentData = compound;

  if (compound.sdf_b64) {{
    const sdf = atob(compound.sdf_b64);
    ligandModel = viewer.addModel(sdf, "sdf");
    viewer.setStyle({{ model: ligandModel }}, {{ stick: {{ colorscheme: "default", radius: 0.2 }} }});
  }}

  // Show binding site residue sidechains as thin sticks (whiteCarbon = standard for publications)
  const bsResidues = [...new Set((compound.interactions || []).map(ix => ix.residue_number))];
  if (bsResidues.length) {{
    viewer.addStyle({{ model: 0, resi: bsResidues }}, {{ stick: {{ colorscheme: "whiteCarbon", radius: 0.12 }} }});
  }}

  drawInteractions(compound.interactions || []);
  renderSidebar(compound);
  // Center on ligand but keep the whole protein in the viewport
  viewer.center({{ model: ligandModel }});
  viewer.zoom(1.8);
  viewer.render();
}}

function toggleType(type, checked) {{
  activeToggles[type] = checked;
  if (!currentData) return;
  currentShapes.forEach(s => viewer.removeShape(s));
  currentShapes = [];
  drawInteractions(currentData.interactions || []);
  renderSidebar(currentData);
}}

viewer.render();
</script>
</body>
</html>
"""


def generate_viewer(work_dir: Path) -> Path:
    output_dir = work_dir / "output"
    viewer_path = output_dir / "interaction_viewer.html"

    if viewer_path.exists() and viewer_path.stat().st_size > 0:
        return viewer_path

    interactions_json = output_dir / "interactions_top_n.json"
    if not interactions_json.exists():
        raise FileNotFoundError("interactions_top_n.json not found — run PLIP analysis first")

    receptor_pdb: Path | None = None
    resume_json = work_dir / "resume.json"
    if resume_json.exists():
        info = json.loads(resume_json.read_text())
        if info.get("receptor_pdb"):
            p = Path(info["receptor_pdb"])
            if p.exists():
                receptor_pdb = p

    if receptor_pdb is None:
        candidate = work_dir / "receptor" / "receptor_prep.pdb"
        if candidate.exists():
            receptor_pdb = candidate

    receptor_text = receptor_pdb.read_text() if receptor_pdb else ""

    compounds = json.loads(interactions_json.read_text())
    html = _build_html(compounds, receptor_text)
    viewer_path.write_text(html, encoding="utf-8")
    return viewer_path
