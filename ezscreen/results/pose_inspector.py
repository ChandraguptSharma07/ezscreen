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
    # backslashes first, then backticks — order matters
    receptor_escaped = receptor_pdb_text.replace("\\", "\\\\").replace("`", "\\`")
    compounds_json   = json.dumps(compounds)
    colors_json      = json.dumps(_INTERACTION_COLORS)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SwiftScreen — Interaction Viewer</title>
<script src="{_3DMOL_CDN}"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
  font-family: system-ui, sans-serif;
  background: #0d1117;
  color: #c9d1d9;
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
  transition: background 0.2s, color 0.2s;
}}

body.light {{
  background: #f6f8fa;
  color: #24292f;
}}

#banner {{
  background: #161b22;
  border-bottom: 1px solid #30363d;
  padding: 4px 16px;
  font-size: 11px;
  color: #8b949e;
  text-align: center;
  flex-shrink: 0;
}}

body.light #banner {{ background: #eaeef2; border-color: #d0d7de; }}

#toolbar {{
  background: #161b22;
  border-bottom: 1px solid #30363d;
  padding: 6px 12px;
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  flex-shrink: 0;
}}

body.light #toolbar {{ background: #eaeef2; border-color: #d0d7de; }}

.tb-sep {{ width: 1px; height: 20px; background: #30363d; margin: 0 3px; flex-shrink: 0; }}
body.light .tb-sep {{ background: #d0d7de; }}

.tb-label {{ font-size: 11px; color: #8b949e; white-space: nowrap; }}

.tb-btn {{
  background: #21262d;
  color: #c9d1d9;
  border: 1px solid #30363d;
  border-radius: 6px;
  padding: 4px 10px;
  font-size: 12px;
  cursor: pointer;
  white-space: nowrap;
  transition: background 0.12s;
}}

.tb-btn:hover {{ background: #30363d; }}
.tb-btn:disabled {{ opacity: 0.45; cursor: not-allowed; }}

.tb-btn.active {{
  background: #1f6feb;
  border-color: #388bfd;
  color: #fff;
}}

body.light .tb-btn {{ background: #fff; color: #24292f; border-color: #d0d7de; }}
body.light .tb-btn:hover {{ background: #f3f4f6; }}
body.light .tb-btn.active {{ background: #0969da; border-color: #0969da; color: #fff; }}

#main {{ display: flex; flex: 1; overflow: hidden; }}

#viewer {{ flex: 1; position: relative; }}

#sidebar {{
  width: 340px;
  background: #161b22;
  border-left: 1px solid #30363d;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}}

body.light #sidebar {{ background: #f6f8fa; border-color: #d0d7de; }}

#controls {{
  padding: 10px 12px;
  border-bottom: 1px solid #30363d;
  flex-shrink: 0;
}}

body.light #controls {{ border-color: #d0d7de; }}

#controls select {{
  width: 100%;
  background: #21262d;
  color: #c9d1d9;
  border: 1px solid #30363d;
  border-radius: 6px;
  padding: 6px;
  font-size: 12px;
}}

body.light #controls select {{
  background: #fff;
  color: #24292f;
  border-color: #d0d7de;
}}

#toggles {{
  padding: 8px 12px;
  border-bottom: 1px solid #30363d;
  flex-shrink: 0;
}}

body.light #toggles {{ border-color: #d0d7de; }}

.toggle-row {{
  display: flex;
  align-items: center;
  gap: 7px;
  margin: 3px 0;
  font-size: 12px;
  cursor: pointer;
}}

.toggle-row input {{ width: 13px; height: 13px; cursor: pointer; }}
.color-dot {{ width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }}

/* 2D diagram panel */
#diagram-panel {{
  border-bottom: 1px solid #30363d;
  padding: 8px 12px 10px;
  flex-shrink: 0;
}}

body.light #diagram-panel {{ border-color: #d0d7de; }}

#diagram-panel .panel-header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 6px;
}}

.svg-export-btn {{
  font-size: 11px;
  color: #8b949e;
  background: none;
  border: none;
  cursor: pointer;
  text-decoration: underline;
  padding: 0;
}}

.svg-export-btn:hover {{ color: #c9d1d9; }}
body.light .svg-export-btn:hover {{ color: #24292f; }}

#diagram-container {{
  width: 100%;
  display: flex;
  justify-content: center;
}}

.diagram-placeholder {{
  font-size: 12px;
  color: #8b949e;
  text-align: center;
  padding: 10px 0;
  width: 100%;
}}

#ilist {{
  flex: 1;
  overflow-y: auto;
  padding: 8px;
  min-height: 0;
}}

.i-row {{
  background: #21262d;
  border-radius: 6px;
  margin: 3px 0;
  padding: 7px 8px;
  font-size: 12px;
}}

body.light .i-row {{ background: #fff; border: 1px solid #d0d7de; }}

.i-type {{ font-weight: 600; text-transform: capitalize; }}
.i-detail {{ color: #8b949e; margin-top: 2px; font-size: 11px; }}

#no-compound {{
  padding: 16px;
  color: #8b949e;
  font-size: 13px;
  text-align: center;
}}

h3 {{
  font-size: 11px;
  font-weight: 600;
  color: #8b949e;
  letter-spacing: .05em;
  text-transform: uppercase;
}}

/* Record progress overlay */
#record-overlay {{
  display: none;
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background: rgba(0,0,0,0.78);
  color: #fff;
  padding: 12px 24px;
  border-radius: 8px;
  font-size: 13px;
  z-index: 200;
  pointer-events: none;
  text-align: center;
}}
</style>
</head>
<body>

<div id="banner">
  Predicted pose — not experimentally validated &nbsp;&middot;&nbsp;
  Pipeline: UniDock &rarr; PLIP &rarr; SwiftScreen &nbsp;&middot;&nbsp;
  Cite: UniDock (Yu et al. 2023) &nbsp;&middot;&nbsp; PLIP (Salentin et al. 2015, NAR)
</div>

<div id="toolbar">
  <span class="tb-label">Display</span>
  <button class="tb-btn" id="btn-bg"     onclick="toggleBackground()" title="Switch light/dark background">Dark BG</button>
  <button class="tb-btn" id="btn-labels" onclick="toggleResLabels()"  title="Show residue name labels on contact residues in 3D">Res Labels</button>
  <button class="tb-btn" id="btn-dist"   onclick="toggleDistLabels()"  title="Show distance labels on interaction lines">Distances</button>
  <button class="tb-btn" id="btn-hydro"  onclick="toggleHydrophob()"   title="Colour binding-pocket surface by Kyte-Doolittle hydrophobicity">Hydrophobicity</button>

  <div class="tb-sep"></div>

  <span class="tb-label">Export</span>
  <button class="tb-btn" onclick="exportPNG()"   title="Save current 3D view as PNG">Export PNG</button>
  <button class="tb-btn" id="btn-video" onclick="exportVideo()" title="Record 360° rotation as WebM video (open in any video player)">Export 360&deg; Video</button>
</div>

<div id="main">
  <div id="viewer">
    <div id="record-overlay">Recording&hellip; <span id="record-pct">0%</span><br><small>Please wait</small></div>
  </div>

  <div id="sidebar">
    <div id="controls">
      <h3 style="margin-bottom:6px">Compound</h3>
      <select id="compound-select" onchange="selectCompound(this.value)">
        <option value="">— select compound —</option>
      </select>
    </div>

    <div id="toggles">
      <h3>Interaction Types</h3>
      <!-- populated by JS -->
    </div>

    <div id="diagram-panel">
      <div class="panel-header">
        <h3>2D Interaction Map</h3>
        <button class="svg-export-btn" onclick="exportSVG()" title="Download as SVG for Illustrator / Inkscape">&darr; SVG</button>
      </div>
      <div id="diagram-container">
        <p class="diagram-placeholder">Select a compound above</p>
      </div>
    </div>

    <div id="ilist">
      <div id="no-compound">Select a compound above</div>
    </div>
  </div>
</div>

<script>
const COMPOUNDS = {compounds_json};
const COLORS    = {colors_json};
const RECEPTOR  = `{receptor_escaped}`;

// Kyte-Doolittle scale (standard hydrophobicity)
const KD = {{
  ILE:4.5, VAL:4.2, LEU:3.8, PHE:2.8, CYS:2.5, MET:1.9, ALA:1.8,
  GLY:-0.4, THR:-0.7, SER:-0.8, TRP:-0.9, TYR:-1.3, PRO:-1.6,
  HIS:-3.2, GLU:-3.5, GLN:-3.5, ASP:-3.5, ASN:-3.5, LYS:-3.9, ARG:-4.5,
}};

function kdHex(resn) {{
  const score = KD[resn] ?? 0;
  const t = Math.max(0, Math.min(1, (score + 4.5) / 9.0));
  let r, g, b;
  if (t < 0.5) {{
    const s = t * 2;
    r = Math.round(68  + (255 - 68)  * s);
    g = Math.round(136 + (255 - 136) * s);
    b = Math.round(204 + (255 - 204) * s);
  }} else {{
    const s = (t - 0.5) * 2;
    r = 255;
    g = Math.round(255 + (68  - 255) * s);
    b = Math.round(255 + (68  - 255) * s);
  }}
  return '#' + [r, g, b].map(v => Math.round(v).toString(16).padStart(2, '0')).join('');
}}

// ── State ──────────────────────────────────────────────────────────────────
let darkMode       = true;
let showResLabels  = false;
let showDistLabels = false;
let showHydrophob  = false;
let ligandModel    = null;
let activeToggles  = Object.fromEntries(Object.keys(COLORS).map(k => [k, true]));
let currentShapes  = [];
let distLabels     = [];
let resLabels      = [];
let hydrophobSurf  = null;
let currentData    = null;

// ── Viewer init ────────────────────────────────────────────────────────────
const viewer = $3Dmol.createViewer("viewer", {{
  backgroundColor: "#0d1117",
  antialias: true,
}});

viewer.addModel(RECEPTOR, "pdb");
viewer.setStyle({{ model: 0 }}, {{ cartoon: {{ color: "spectrum", opacity: 0.9 }} }});
viewer.zoomTo({{ model: 0 }});
viewer.render();

// Hover tooltip — receptor atoms show residue info; ligand atoms show atom type
viewer.setHoverable({{}}, true,
  function(atom) {{
    if (atom._label) return;
    const isLig = atom.model !== 0;
    const text  = isLig
      ? `${{atom.resn}} · ${{atom.atom}}`
      : `${{atom.resn}} ${{atom.resi}} · Chain ${{atom.chain}} · ${{atom.atom}}`;
    atom._label = viewer.addLabel(text, {{
      position:        atom,
      backgroundColor: "rgba(13,17,23,0.88)",
      fontColor:       "#ffffff",
      fontSize:        11,
      borderRadius:    4,
      padding:         3,
      inFront:         true,
    }});
    viewer.render();
  }},
  function(atom) {{
    if (atom._label) {{
      viewer.removeLabel(atom._label);
      delete atom._label;
      viewer.render();
    }}
  }}
);

// ── Compound dropdown ──────────────────────────────────────────────────────
const sel = document.getElementById("compound-select");
COMPOUNDS.forEach(c => {{
  const opt       = document.createElement("option");
  opt.value       = c.lig_id;
  opt.textContent = `#${{c.rank}}  ${{c.name || c.lig_id}}  (${{c.score}} kcal/mol)`;
  sel.appendChild(opt);
}});

// ── Interaction type toggles ───────────────────────────────────────────────
const togglesDiv = document.getElementById("toggles");
Object.entries(COLORS).forEach(([type, color]) => {{
  const row = document.createElement("label");
  row.className = "toggle-row";
  row.innerHTML = `
    <input type="checkbox" checked onchange="toggleType('${{type}}', this.checked)">
    <span class="color-dot" style="background:${{color}}"></span>
    <span>${{type.replace(/_/g, " ")}}</span>`;
  togglesDiv.appendChild(row);
}});

// ── Toolbar handlers ───────────────────────────────────────────────────────
function toggleBackground() {{
  darkMode = !darkMode;
  viewer.setBackgroundColor(darkMode ? "#0d1117" : "#f6f8fa");
  viewer.render();
  document.body.classList.toggle("light", !darkMode);
  const btn = document.getElementById("btn-bg");
  btn.textContent = darkMode ? "Dark BG" : "Light BG";
  btn.classList.toggle("active", !darkMode);
  if (currentData) draw2DDiagram(currentData);
}}

function toggleResLabels() {{
  showResLabels = !showResLabels;
  document.getElementById("btn-labels").classList.toggle("active", showResLabels);
  refreshResLabels();
}}

function toggleDistLabels() {{
  showDistLabels = !showDistLabels;
  document.getElementById("btn-dist").classList.toggle("active", showDistLabels);
  refreshInteractions();
}}

function toggleHydrophob() {{
  showHydrophob = !showHydrophob;
  document.getElementById("btn-hydro").classList.toggle("active", showHydrophob);
  refreshHydrophobSurface();
}}

// ── Core clear ────────────────────────────────────────────────────────────
function clearLigand() {{
  if (ligandModel !== null) {{ viewer.removeModel(ligandModel); ligandModel = null; }}
  currentShapes.forEach(s => viewer.removeShape(s)); currentShapes = [];
  distLabels.forEach(l => viewer.removeLabel(l));    distLabels    = [];
  resLabels.forEach(l  => viewer.removeLabel(l));    resLabels     = [];
  if (hydrophobSurf !== null) {{ viewer.removeSurface(hydrophobSurf); hydrophobSurf = null; }}
  viewer.setStyle({{ model: 0 }}, {{ cartoon: {{ color: "spectrum", opacity: 0.9 }} }});
}}

// ── Interaction lines + distance labels ───────────────────────────────────
function drawInteractions(interactions) {{
  interactions.forEach(ix => {{
    if (!activeToggles[ix.type]) return;
    const color = COLORS[ix.type] || "#ffffff";

    currentShapes.push(viewer.addCylinder({{
      start:  {{ x: ix.protein_coords[0], y: ix.protein_coords[1], z: ix.protein_coords[2] }},
      end:    {{ x: ix.ligand_coords[0],  y: ix.ligand_coords[1],  z: ix.ligand_coords[2]  }},
      radius: 0.1, color, opacity: 0.9, dashed: true,
    }}));

    if (showDistLabels) {{
      const lbl = viewer.addLabel(`${{ix.distance.toFixed(1)}} Å`, {{
        position: {{
          x: (ix.protein_coords[0] + ix.ligand_coords[0]) / 2,
          y: (ix.protein_coords[1] + ix.ligand_coords[1]) / 2,
          z: (ix.protein_coords[2] + ix.ligand_coords[2]) / 2,
        }},
        fontSize:        10,
        fontColor:       color,
        backgroundColor: "rgba(13,17,23,0.75)",
        borderRadius:    3,
        padding:         2,
        inFront:         true,
      }});
      distLabels.push(lbl);
    }}
  }});
  viewer.render();
}}

function refreshInteractions() {{
  currentShapes.forEach(s => viewer.removeShape(s)); currentShapes = [];
  distLabels.forEach(l => viewer.removeLabel(l));    distLabels    = [];
  if (currentData) drawInteractions(currentData.interactions || []);
}}

// ── Residue labels ────────────────────────────────────────────────────────
function drawResidueLabels(compound) {{
  const seen = new Set();
  (compound.interactions || []).forEach(ix => {{
    const key = `${{ix.residue_name}}${{ix.residue_number}}`;
    if (seen.has(key)) return;
    seen.add(key);
    resLabels.push(viewer.addLabel(`${{ix.residue_name}} ${{ix.residue_number}}`, {{
      position:        {{ x: ix.protein_coords[0], y: ix.protein_coords[1], z: ix.protein_coords[2] }},
      fontSize:        10,
      fontColor:       "#f0f6fc",
      backgroundColor: "rgba(0,0,0,0.65)",
      borderRadius:    3,
      padding:         2,
      inFront:         false,
    }}));
  }});
  viewer.render();
}}

function refreshResLabels() {{
  resLabels.forEach(l => viewer.removeLabel(l)); resLabels = [];
  if (currentData && showResLabels) drawResidueLabels(currentData);
  viewer.render();
}}

// ── Hydrophobicity surface (async — addSurface returns a Promise) ──────────
async function refreshHydrophobSurface() {{
  if (hydrophobSurf !== null) {{ viewer.removeSurface(hydrophobSurf); hydrophobSurf = null; }}
  if (!showHydrophob || !currentData) {{ viewer.render(); return; }}

  const bsRes = [...new Set((currentData.interactions || []).map(ix => ix.residue_number))];
  if (!bsRes.length) return;

  hydrophobSurf = await viewer.addSurface(
    $3Dmol.SurfaceType.MS,
    {{
      opacity:   0.65,
      colorfunc: (atom) => kdHex(atom.resn),
    }},
    {{ model: 0, resi: bsRes }}
  );
  viewer.render();
}}

// ── Sidebar interaction list ───────────────────────────────────────────────
function renderSidebar(compound) {{
  const list = document.getElementById("ilist");
  list.innerHTML = "";

  if (!compound || compound.plip_failed) {{
    list.innerHTML = `<div id="no-compound" style="color:#f85149">${{
      compound ? (compound.plip_error || "PLIP analysis failed") : "No data"
    }}</div>`;
    return;
  }}

  const active = (compound.interactions || []).filter(ix => activeToggles[ix.type]);
  if (!active.length) {{
    list.innerHTML = `<div id="no-compound">No visible interactions (check toggles)</div>`;
    return;
  }}

  active.forEach(ix => {{
    const div = document.createElement("div");
    div.className = "i-row";
    const color = COLORS[ix.type] || "#ffffff";
    div.innerHTML = `
      <div class="i-type" style="color:${{color}}">${{ix.type.replace(/_/g, " ")}}</div>
      <div class="i-detail">
        ${{ix.residue_name}}&nbsp;${{ix.residue_number}} (${{ix.chain}})
        &nbsp;&middot;&nbsp; ${{ix.distance.toFixed(2)}}&nbsp;&Aring;
      </div>`;
    list.appendChild(div);
  }});
}}

// ── 2D interaction diagram (LIGPLOT-style SVG) ─────────────────────────────
function draw2DDiagram(compound) {{
  const container = document.getElementById("diagram-container");

  if (!compound || !compound.interactions || !compound.interactions.length) {{
    container.innerHTML = '<p class="diagram-placeholder">No interaction data</p>';
    return;
  }}

  const visible = compound.interactions.filter(ix => activeToggles[ix.type]);
  if (!visible.length) {{
    container.innerHTML = '<p class="diagram-placeholder">No visible interactions</p>';
    return;
  }}

  const resMap = new Map();
  visible.forEach(ix => {{
    const key = `${{ix.residue_name}}${{ix.residue_number}}`;
    if (!resMap.has(key)) resMap.set(key, ix);
  }});
  const residues = [...resMap.values()];
  const n = residues.length;

  const W = 310, H = 230;
  const cx = W / 2, cy = H / 2;
  const ell_rx = W * 0.41, ell_ry = H * 0.40;

  const positions = residues.map((_, i) => {{
    const angle = (i / n) * 2 * Math.PI - Math.PI / 2;
    return {{ x: cx + ell_rx * Math.cos(angle), y: cy + ell_ry * Math.sin(angle) }};
  }});

  const textFill  = darkMode ? "#e6edf3" : "#24292f";
  const subFill   = "#8b949e";
  const ligFill   = darkMode ? "#21262d" : "#fff";
  const ligStroke = darkMode ? "#c9d1d9" : "#57606a";

  let svg = `<svg id="svg2d" width="${{W}}" height="${{H}}" xmlns="http://www.w3.org/2000/svg"
    style="overflow:visible">`;

  svg += `<defs>`;
  ["hbond", "salt_bridge"].forEach(type => {{
    const c = COLORS[type];
    svg += `<marker id="arr-${{type}}" markerWidth="5" markerHeight="5"
        refX="4.5" refY="2.5" orient="auto">
      <path d="M0,0 L0,5 L5,2.5 Z" fill="${{c}}" opacity="0.8"/>
    </marker>`;
  }});
  svg += `</defs>`;

  residues.forEach((res, i) => {{
    const pos   = positions[i];
    const color = COLORS[res.type] || "#888";
    const isDashed = ["hydrophobic", "pi_stack", "pi_cation", "halogen"].includes(res.type);
    const hasArrow = ["hbond", "salt_bridge"].includes(res.type);
    const dashAttr  = isDashed ? 'stroke-dasharray="5 3"' : '';
    const markerEnd = hasArrow ? `marker-end="url(#arr-${{res.type}})"` : '';

    const dx = pos.x - cx, dy = pos.y - cy;
    const dist = Math.sqrt(dx*dx + dy*dy);
    const ux = dx / dist, uy = dy / dist;
    const x2 = pos.x - ux * 19;
    const y2 = pos.y - uy * 19;

    svg += `<line x1="${{cx}}" y1="${{cy}}" x2="${{x2}}" y2="${{y2}}"
      stroke="${{color}}" stroke-width="1.4" opacity="0.8"
      ${{dashAttr}} ${{markerEnd}}/>`;
  }});

  const ligName = (compound.name || compound.lig_id || "LIG").slice(0, 9);
  svg += `
    <rect x="${{cx - 26}}" y="${{cy - 13}}" width="52" height="26" rx="6"
      fill="${{ligFill}}" stroke="${{ligStroke}}" stroke-width="1.5"/>
    <text x="${{cx}}" y="${{cy + 4}}" text-anchor="middle"
      fill="${{textFill}}" font-size="9" font-weight="bold"
      font-family="system-ui">${{ligName}}</text>`;

  residues.forEach((res, i) => {{
    const pos   = positions[i];
    const color = COLORS[res.type] || "#556677";
    svg += `
      <circle cx="${{pos.x}}" cy="${{pos.y}}" r="18"
        fill="${{color}}25" stroke="${{color}}" stroke-width="1.5"/>
      <text x="${{pos.x}}" y="${{pos.y - 2}}" text-anchor="middle"
        fill="${{textFill}}" font-size="8" font-weight="bold"
        font-family="system-ui">${{res.residue_name}}</text>
      <text x="${{pos.x}}" y="${{pos.y + 9}}" text-anchor="middle"
        fill="${{subFill}}" font-size="7" font-family="system-ui">${{res.residue_number}}</text>`;
  }});

  svg += `</svg>`;
  container.innerHTML = svg;
}}

// ── Main compound select ───────────────────────────────────────────────────
async function selectCompound(lig_id) {{
  clearLigand();
  if (!lig_id) return;
  const compound = COMPOUNDS.find(c => c.lig_id === lig_id);
  if (!compound) return;
  currentData = compound;

  if (compound.sdf_b64) {{
    const sdf   = atob(compound.sdf_b64);
    ligandModel = viewer.addModel(sdf, "sdf");
    viewer.setStyle({{ model: ligandModel }}, {{
      stick: {{ colorscheme: "default", radius: 0.18 }},
    }});
  }}

  const bsRes = [...new Set((compound.interactions || []).map(ix => ix.residue_number))];
  if (bsRes.length) {{
    viewer.addStyle({{ model: 0, resi: bsRes }}, {{
      stick: {{ colorscheme: "whiteCarbon", radius: 0.12 }},
    }});
  }}

  drawInteractions(compound.interactions || []);
  if (showResLabels) drawResidueLabels(compound);
  await refreshHydrophobSurface();

  renderSidebar(compound);
  draw2DDiagram(compound);

  if (ligandModel !== null) {{
    viewer.center({{ model: ligandModel }});
    viewer.zoom(2.0);
  }}
  viewer.render();
}}

function toggleType(type, checked) {{
  activeToggles[type] = checked;
  if (!currentData) return;
  refreshInteractions();
  renderSidebar(currentData);
  draw2DDiagram(currentData);
}}

// ── Export: PNG ────────────────────────────────────────────────────────────
function exportPNG() {{
  const name = currentData ? (currentData.name || currentData.lig_id) : "pose";
  const uri  = viewer.pngURI();
  const a    = Object.assign(document.createElement("a"), {{ href: uri, download: `${{name}}_interaction.png` }});
  a.click();
}}

// ── Export: 2D SVG ─────────────────────────────────────────────────────────
function exportSVG() {{
  const svgEl = document.getElementById("svg2d");
  if (!svgEl) {{ alert("Select a compound first."); return; }}
  const name  = currentData ? (currentData.name || currentData.lig_id) : "pose";
  const blob  = new Blob([svgEl.outerHTML], {{ type: "image/svg+xml" }});
  const url   = URL.createObjectURL(blob);
  const a     = Object.assign(document.createElement("a"), {{ href: url, download: `${{name}}_2d_interactions.svg` }});
  a.click();
  URL.revokeObjectURL(url);
}}

// ── Export: 360° WebM video ────────────────────────────────────────────────
function exportVideo() {{
  const canvas = document.querySelector("#viewer canvas");
  if (!canvas) {{ alert("3D canvas not ready."); return; }}

  const mime = ["video/webm;codecs=vp9", "video/webm"].find(m => MediaRecorder.isTypeSupported(m));
  if (!mime) {{ alert("WebM recording not supported in this browser. Try Chrome."); return; }}

  const stream   = canvas.captureStream(30);
  const recorder = new MediaRecorder(stream, {{ mimeType: mime }});
  const chunks   = [];

  recorder.ondataavailable = e => {{ if (e.data.size) chunks.push(e.data); }};
  recorder.onstop = () => {{
    const name = currentData ? (currentData.name || currentData.lig_id) : "pose";
    const blob = new Blob(chunks, {{ type: "video/webm" }});
    const url  = URL.createObjectURL(blob);
    const a    = Object.assign(document.createElement("a"), {{ href: url, download: `${{name}}_360.webm` }});
    a.click();
    URL.revokeObjectURL(url);
    document.getElementById("record-overlay").style.display = "none";
    const btn = document.getElementById("btn-video");
    btn.disabled    = false;
    btn.textContent = "Export 360° Video";
  }};

  const overlay = document.getElementById("record-overlay");
  overlay.style.display = "block";
  const btn = document.getElementById("btn-video");
  btn.disabled    = true;
  btn.textContent = "Recording…";

  recorder.start();

  const totalSteps = 72;
  let step = 0;

  function tick() {{
    if (step >= totalSteps) {{ recorder.stop(); return; }}
    viewer.rotate(5, "y");
    viewer.render();
    document.getElementById("record-pct").textContent = Math.round((step / totalSteps) * 100) + "%";
    step++;
    setTimeout(tick, 50);
  }}
  tick();
}}

viewer.render();
</script>
</body>
</html>
"""


def generate_viewer(work_dir: Path) -> Path:
    output_dir = work_dir / "output"
    viewer_path = output_dir / "interaction_viewer.html"

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
