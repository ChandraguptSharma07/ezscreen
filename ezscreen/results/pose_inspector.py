from __future__ import annotations

import json
import re as _re
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

# RDKit float tuples for DrawMolecule highlight colours
_ITYPE_RDKIT_COLOR = {
    "hbond":       (0.231, 0.510, 0.965),
    "hydrophobic": (0.976, 0.451, 0.086),
    "pi_stack":    (0.133, 0.773, 0.333),
    "pi_cation":   (0.659, 0.333, 0.965),
    "salt_bridge": (0.937, 0.318, 0.286),
    "halogen":     (0.078, 0.718, 0.647),
}

_LIG_SVG_W  = 560   # width of RDKit ligand SVG
_LIG_SVG_H  = 380   # height of RDKit ligand SVG
_GLYPH_MAR  = 170   # px margin around ligand for residue glyphs


def _strip_svg_inner(svg_text: str) -> tuple[str, str]:
    """Return (inner_content, viewBox) with outer <svg> wrapper and bg rect removed."""
    s = _re.sub(r"<\?xml[^?]*\?>\s*", "", svg_text)
    s = _re.sub(r"<!DOCTYPE[^>]*>\s*", "", s)
    vb_m = _re.search(r'viewBox="([^"]*)"', s)
    vb   = vb_m.group(1) if vb_m else f"0 0 {_LIG_SVG_W} {_LIG_SVG_H}"
    s = _re.sub(r"^\s*<svg[^>]*>", "", s.strip())
    s = _re.sub(r"</svg>\s*$",      "", s.strip())
    # Remove RDKit's white background rect (first <rect>, paired or self-closing)
    s = _re.sub(r"<rect\b[^>]*(?:/>|>.*?</rect>)", "", s, count=1, flags=_re.DOTALL)
    return s.strip(), vb


# ── Python-side 2D enrichment ─────────────────────────────────────────────

def _enrich_2d(compounds: list[dict]) -> list[dict]:
    """Add RDKit 2D SVG and atom draw-coordinates to each compound dict."""
    try:
        import base64

        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return compounds

    for c in compounds:
        if not c.get("sdf_b64") or c.get("plip_failed"):
            continue
        try:
            sdf_text = base64.b64decode(c["sdf_b64"]).decode(errors="replace")

            # --- 3-D mol: kept for coordinate matching only ---
            mol_3d = Chem.MolFromMolBlock(sdf_text, removeHs=True)
            if mol_3d is None or mol_3d.GetNumAtoms() == 0:
                continue
            conf_3d = mol_3d.GetConformer()
            coords_3d = [
                (conf_3d.GetAtomPosition(i).x,
                 conf_3d.GetAtomPosition(i).y,
                 conf_3d.GetAtomPosition(i).z)
                for i in range(mol_3d.GetNumAtoms())
            ]

            def nearest_atom(lc: list) -> int:
                best_i, best_d = 0, float("inf")
                for i, c3 in enumerate(coords_3d):
                    d = (lc[0]-c3[0])**2 + (lc[1]-c3[1])**2 + (lc[2]-c3[2])**2
                    if d < best_d:
                        best_d, best_i = d, i
                return best_i

            # --- 2-D mol: fresh copy with computed layout ---
            mol_2d = Chem.MolFromMolBlock(sdf_text, removeHs=True)
            AllChem.Compute2DCoords(mol_2d)
            rdMolDraw2D.PrepareMolForDrawing(mol_2d)

            # Map each interaction to the nearest atom in 3-D space
            interactions = c.get("interactions", [])
            per_ix_atom: list[int | None] = []
            interacting: dict[int, str] = {}   # atom_idx -> first interaction type

            for ix in interactions:
                lc = ix.get("ligand_coords")
                if lc:
                    idx = nearest_atom(lc)
                    per_ix_atom.append(idx)
                    if idx not in interacting:
                        interacting[idx] = ix["type"]
                else:
                    per_ix_atom.append(None)

            h_atoms = list(interacting.keys())

            # --- Draw the full-compound SVG (no highlight blobs; connection lines show interactions) ---
            drawer = rdMolDraw2D.MolDraw2DSVG(_LIG_SVG_W, _LIG_SVG_H)
            drawer.drawOptions().addStereoAnnotation = True
            drawer.drawOptions().padding = 0.12
            drawer.DrawMolecule(mol_2d)
            drawer.FinishDrawing()
            svg_full = drawer.GetDrawingText()

            # Transparent background (RDKit emits various white-background forms)
            svg_full = _re.sub(
                r"(style=['\"])background[^;'\"]*;?",
                r"\1background:transparent;",
                svg_full,
            )

            # --- Collect atom 2-D draw coordinates ---
            atom_2d: dict[int, list[float]] = {}
            for idx in h_atoms:
                try:
                    pt = drawer.GetDrawCoords(idx)
                    atom_2d[idx] = [round(pt.x, 2), round(pt.y, 2)]
                except Exception:
                    pass

            # Per-interaction 2D point (parallel to `interactions`)
            ix_atom_pts: list[list[float] | None] = []
            for idx in per_ix_atom:
                if idx is not None and idx in atom_2d:
                    ix_atom_pts.append(atom_2d[idx])
                else:
                    ix_atom_pts.append(None)

            # --- Site SVG: crop viewBox to interacting region ---
            svg_site     = svg_full
            site_viewbox: list[float] | None = None
            if atom_2d:
                expanded: list[list[float]] = list(atom_2d.values())
                for base_idx in list(interacting.keys()):
                    for nb in mol_2d.GetAtomWithIdx(base_idx).GetNeighbors():
                        try:
                            pt = drawer.GetDrawCoords(nb.GetIdx())
                            expanded.append([pt.x, pt.y])
                        except Exception:
                            pass
                for ring in mol_2d.GetRingInfo().AtomRings():
                    if any(ri in interacting for ri in ring):
                        for ri in ring:
                            try:
                                pt = drawer.GetDrawCoords(ri)
                                expanded.append([pt.x, pt.y])
                            except Exception:
                                pass

                xs  = [p[0] for p in expanded]
                ys  = [p[1] for p in expanded]
                pad = 55.0
                vx  = max(0.0, min(xs) - pad)
                vy  = max(0.0, min(ys) - pad)
                vw  = min(float(_LIG_SVG_W), max(xs) - min(xs) + 2 * pad)
                vh  = min(float(_LIG_SVG_H), max(ys) - min(ys) + 2 * pad)

                svg_site = _re.sub(
                    r'viewBox="[^"]*"',
                    f'viewBox="{vx:.1f} {vy:.1f} {vw:.1f} {vh:.1f}"',
                    svg_full,
                )
                site_viewbox = [round(vx, 1), round(vy, 1),
                                round(vw, 1), round(vh, 1)]

            inner_full, vb_full = _strip_svg_inner(svg_full)
            inner_site, vb_site = _strip_svg_inner(svg_site)

            c["lig_svg_full"]       = svg_full
            c["lig_svg_site"]       = svg_site
            c["lig_svg_inner_full"] = inner_full
            c["lig_svg_inner_site"] = inner_site
            c["lig_svg_vb_full"]    = vb_full
            c["lig_svg_vb_site"]    = vb_site
            c["ix_atom_pts"]        = ix_atom_pts
            c["site_viewbox"]       = site_viewbox
            c["svg_w"]              = _LIG_SVG_W
            c["svg_h"]              = _LIG_SVG_H

        except Exception:
            pass

    return compounds


# ── HTML builder ──────────────────────────────────────────────────────────

def _build_html(compounds: list[dict], receptor_pdb_text: str) -> str:
    receptor_escaped = receptor_pdb_text.replace("\\", "\\\\").replace("`", "\\`")
    compounds_json   = json.dumps(compounds)
    colors_json      = json.dumps(_INTERACTION_COLORS)
    lig_svg_w        = _LIG_SVG_W
    lig_svg_h        = _LIG_SVG_H
    glyph_mar        = _GLYPH_MAR

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
  background: #0d1117; color: #c9d1d9;
  display: flex; flex-direction: column;
  height: 100vh; overflow: hidden;
  transition: background .2s, color .2s;
}}
body.light {{ background: #f6f8fa; color: #24292f; }}

#banner {{
  background: #161b22; border-bottom: 1px solid #30363d;
  padding: 4px 16px; font-size: 11px; color: #8b949e;
  text-align: center; flex-shrink: 0;
}}
body.light #banner {{ background: #eaeef2; border-color: #d0d7de; }}

#toolbar {{
  background: #161b22; border-bottom: 1px solid #30363d;
  padding: 6px 12px; display: flex; align-items: center;
  gap: 6px; flex-wrap: wrap; flex-shrink: 0;
}}
body.light #toolbar {{ background: #eaeef2; border-color: #d0d7de; }}

.tb-sep {{ width: 1px; height: 20px; background: #30363d; margin: 0 3px; flex-shrink: 0; }}
body.light .tb-sep {{ background: #d0d7de; }}
.tb-label {{ font-size: 11px; color: #8b949e; white-space: nowrap; }}

.tb-btn {{
  background: #21262d; color: #c9d1d9;
  border: 1px solid #30363d; border-radius: 6px;
  padding: 4px 10px; font-size: 12px;
  cursor: pointer; white-space: nowrap; transition: background .12s;
}}
.tb-btn:hover {{ background: #30363d; }}
.tb-btn:disabled {{ opacity: .4; cursor: not-allowed; }}
.tb-btn.active {{ background: #1f6feb; border-color: #388bfd; color: #fff; }}

body.light .tb-btn {{ background: #fff; color: #24292f; border-color: #d0d7de; }}
body.light .tb-btn:hover {{ background: #f3f4f6; }}
body.light .tb-btn.active {{ background: #0969da; border-color: #0969da; color: #fff; }}

.tb-select {{
  background: #21262d; color: #c9d1d9;
  border: 1px solid #30363d; border-radius: 6px;
  padding: 4px 8px; font-size: 12px; cursor: pointer;
}}
body.light .tb-select {{ background: #fff; color: #24292f; border-color: #d0d7de; }}

/* view-toggle pair */
.view-pair {{ display: flex; gap: 0; }}
.view-pair .tb-btn:first-child {{ border-radius: 6px 0 0 6px; border-right: none; }}
.view-pair .tb-btn:last-child  {{ border-radius: 0 6px 6px 0; }}

#main {{ display: flex; flex: 1; overflow: hidden; }}

/* 3D viewer */
#viewer-wrap {{ flex: 1; display: flex; position: relative; }}
#viewer {{ flex: 1; position: relative; }}

/* Fixed 90° inset — sits in the corner, no mouse, gives the user a second angle on the pocket */
#viewer-inset {{
  position: absolute; top: 12px; right: 12px;
  width: 220px; height: 220px;
  background: rgba(13,17,23,.55);
  border: 1px solid #30363d; border-radius: 8px;
  overflow: hidden; pointer-events: none;
  z-index: 10;
}}
body.light #viewer-inset {{ background: rgba(246,248,250,.7); border-color: #d0d7de; }}
#viewer-inset .inset-label {{
  position: absolute; top: 6px; left: 8px;
  font-size: 10px; color: #8b949e;
  font-family: system-ui, sans-serif; letter-spacing: .04em;
  pointer-events: none;
}}

/* 2D diagram area */
#diagram2d-wrap {{
  flex: 1; display: none; align-items: center;
  justify-content: center; overflow: hidden;
  padding: 12px; position: relative;
}}

#diagram2d-wrap svg {{ max-width: 100%; max-height: 100%; }}

.d2-placeholder {{
  color: #8b949e; font-size: 14px; text-align: center; padding: 40px;
}}

/* Sidebar */
#sidebar {{
  width: 300px; background: #161b22;
  border-left: 1px solid #30363d;
  display: flex; flex-direction: column; overflow: hidden;
}}
body.light #sidebar {{ background: #f6f8fa; border-color: #d0d7de; }}

.sb-section {{
  padding: 10px 12px;
  border-bottom: 1px solid #30363d;
  flex-shrink: 0;
}}
body.light .sb-section {{ border-color: #d0d7de; }}

.sb-section select {{
  width: 100%; background: #21262d; color: #c9d1d9;
  border: 1px solid #30363d; border-radius: 6px;
  padding: 6px; font-size: 12px;
}}
body.light .sb-section select {{
  background: #fff; color: #24292f; border-color: #d0d7de;
}}

/* Site / full sub-toggle */
#site-toggle {{
  display: none; gap: 0; margin-top: 8px;
}}
#site-toggle .tb-btn:first-child {{ border-radius: 6px 0 0 6px; border-right: none; flex: 1; }}
#site-toggle .tb-btn:last-child  {{ border-radius: 0 6px 6px 0; flex: 1; }}

.toggle-row {{
  display: flex; align-items: center; gap: 7px;
  margin: 3px 0; font-size: 12px; cursor: pointer;
}}
.toggle-row input {{ width: 13px; height: 13px; cursor: pointer; }}
.color-dot {{ width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }}

#ilist {{ flex: 1; overflow-y: auto; padding: 8px; min-height: 0; }}
.i-row {{
  background: #21262d; border-radius: 6px;
  margin: 3px 0; padding: 7px 8px; font-size: 12px;
}}
body.light .i-row {{ background: #fff; border: 1px solid #d0d7de; }}
.i-type {{ font-weight: 600; text-transform: capitalize; }}
.i-detail {{ color: #8b949e; margin-top: 2px; font-size: 11px; }}
#no-compound {{ padding: 16px; color: #8b949e; font-size: 13px; text-align: center; }}

h3 {{
  font-size: 11px; font-weight: 600; color: #8b949e;
  letter-spacing: .05em; text-transform: uppercase;
}}

/* Record overlay */
#record-overlay {{
  display: none; position: absolute;
  top: 50%; left: 50%; transform: translate(-50%,-50%);
  background: rgba(0,0,0,.78); color: #fff;
  padding: 12px 24px; border-radius: 8px;
  font-size: 13px; z-index: 200;
  pointer-events: none; text-align: center;
}}
</style>
</head>
<body>

<div id="banner">
  Predicted pose — not experimentally validated &nbsp;&middot;&nbsp;
  Pipeline: UniDock &rarr; PLIP &rarr; SwiftScreen &nbsp;&middot;&nbsp;
  Cite: UniDock (Yu&nbsp;et&nbsp;al.&nbsp;2023) &nbsp;&middot;&nbsp; PLIP (Salentin&nbsp;et&nbsp;al.&nbsp;2015,&nbsp;NAR)
</div>

<div id="toolbar">
  <span class="tb-label">Preset</span>
  <select id="preset-select" class="tb-select" onchange="setPreset(this.value)" title="Overall rendering style">
    <option value="publication" selected>Publication</option>
    <option value="sticks">Sticks only</option>
    <option value="surface">Surface + ligand</option>
  </select>

  <div class="tb-sep"></div>

  <span class="tb-label">Display</span>
  <button class="tb-btn" id="btn-bg"     onclick="toggleBackground()" title="Switch light / dark background">Dark BG</button>
  <button class="tb-btn" id="btn-labels" onclick="toggleResLabels()"  title="Residue name labels in 3D">Res Labels</button>
  <button class="tb-btn" id="btn-dist"   onclick="toggleDistLabels()"  title="Distance labels on interaction lines">Distances</button>
  <button class="tb-btn" id="btn-surf"   onclick="togglePocketSurf()"  title="Translucent surface of the pocket">Pocket Surface</button>
  <button class="tb-btn" id="btn-hydro"  onclick="toggleHydrophob()"   title="Kyte-Doolittle pocket surface">Hydrophobicity</button>
  <button class="tb-btn active" id="btn-depth"  onclick="toggleDepth()" title="Depth fog — gives the pocket a sense of foreground / background">Depth</button>

  <div class="tb-sep"></div>

  <span class="tb-label">View</span>
  <div class="view-pair">
    <button class="tb-btn active" id="btn-3d" onclick="switchMode('3d')">3D</button>
    <button class="tb-btn"        id="btn-2d" onclick="switchMode('2d')">2D</button>
  </div>

  <div class="tb-sep"></div>

  <span class="tb-label">Export</span>
  <button class="tb-btn" id="btn-exp-png" onclick="exportPNG()"   title="Save current view as PNG">Export PNG</button>
  <button class="tb-btn" id="btn-exp-vid" onclick="exportVideo()" title="360° rotation WebM (3D only)">Export 360&deg; Video</button>
  <button class="tb-btn" id="btn-exp-svg" onclick="exportSVG()"   title="Download 2D diagram as SVG (2D mode only)" disabled>Export SVG</button>
</div>

<div id="main">

  <!-- ── 3D viewer ─────────────────────────────── -->
  <div id="viewer-wrap">
    <div id="viewer">
      <div id="record-overlay">
        Recording&hellip; <span id="record-pct">0%</span><br><small>Please wait</small>
      </div>
    </div>
    <div id="viewer-inset"><span class="inset-label">90&deg; view</span></div>
  </div>

  <!-- ── 2D diagram area ──────────────────────── -->
  <div id="diagram2d-wrap">
    <p class="d2-placeholder">Select a compound to view the 2D interaction map</p>
  </div>

  <!-- ── Sidebar ───────────────────────────────── -->
  <div id="sidebar">

    <div class="sb-section">
      <h3 style="margin-bottom:6px">Compound</h3>
      <select id="compound-select" onchange="selectCompound(this.value)">
        <option value="">— select compound —</option>
      </select>
      <!-- site / full sub-toggle (visible only in 2D mode) -->
      <div id="site-toggle">
        <button class="tb-btn active" id="btn-site" onclick="setSiteMode(true)">Site View</button>
        <button class="tb-btn"        id="btn-full" onclick="setSiteMode(false)">Full Compound</button>
      </div>
    </div>

    <div class="sb-section" id="toggles-section">
      <h3>Interaction Types</h3>
    </div>

    <div id="ilist">
      <div id="no-compound">Select a compound above</div>
    </div>

  </div><!-- /sidebar -->

</div><!-- /main -->

<script>
const COMPOUNDS = {compounds_json};
const COLORS    = {colors_json};
const RECEPTOR  = `{receptor_escaped}`;
const LIG_W     = {lig_svg_w};
const LIG_H     = {lig_svg_h};
const GM        = {glyph_mar};   // glyph margin around ligand SVG

// ── Kyte-Doolittle ───────────────────────────────────────────────────────
const KD = {{
  ILE:4.5,VAL:4.2,LEU:3.8,PHE:2.8,CYS:2.5,MET:1.9,ALA:1.8,
  GLY:-0.4,THR:-0.7,SER:-0.8,TRP:-0.9,TYR:-1.3,PRO:-1.6,
  HIS:-3.2,GLU:-3.5,GLN:-3.5,ASP:-3.5,ASN:-3.5,LYS:-3.9,ARG:-4.5,
}};
function kdHex(resn) {{
  const t = Math.max(0, Math.min(1, ((KD[resn] ?? 0) + 4.5) / 9));
  let r,g,b;
  if (t < .5) {{ const s=t*2; r=Math.round(68+(255-68)*s); g=Math.round(136+(255-136)*s); b=Math.round(204+(255-204)*s); }}
  else {{ const s=(t-.5)*2; r=255; g=Math.round(255+(68-255)*s); b=Math.round(255+(68-255)*s); }}
  return '#'+[r,g,b].map(v=>Math.round(v).toString(16).padStart(2,'0')).join('');
}}

// ── Amino-acid character colours ────────────────────────────────────────
const AA_CLS = {{
  ALA:'hp',VAL:'hp',LEU:'hp',ILE:'hp',PRO:'hp',PHE:'hp',MET:'hp',TRP:'hp',
  SER:'pol',THR:'pol',CYS:'pol',TYR:'pol',ASN:'pol',GLN:'pol',GLY:'pol',
  LYS:'pos',ARG:'pos',HIS:'pos',
  ASP:'neg',GLU:'neg',
}};
const AA_COL = {{ hp:'#f97316', pol:'#22c55e', pos:'#3b82f6', neg:'#ef4444' }};
function aaColor(resn) {{ return AA_COL[AA_CLS[resn]] || '#8b949e'; }}

// ── State ────────────────────────────────────────────────────────────────
let darkMode       = true;
let currentMode    = '3d';
let siteMode       = true;
let showResLabels  = false;
let showDistLabels = false;
let showHydrophob  = false;
let showPocketSurf = false;
let depthFog       = true;
let currentPreset  = 'publication';
let ligandModel    = null;
let activeToggles  = Object.fromEntries(Object.keys(COLORS).map(k=>[k,true]));
let currentShapes  = [];
let distLabels     = [];
let resLabels      = [];
let hydrophobSurf  = null;
let pocketSurf     = null;
let currentData    = null;
let pocketResi     = [];   // residue numbers within 5 Å of current ligand
let pinnedLabels   = new Map();   // atom-key -> 3Dmol label, survives until clicked again

// ── 3D Viewer init ───────────────────────────────────────────────────────
const viewer = $3Dmol.createViewer("viewer", {{ backgroundColor:"#0d1117", antialias:true }});
viewer.addModel(RECEPTOR,"pdb");
viewer.setStyle({{model:0}},{{cartoon:{{color:"spectrum",opacity:.9}}}});
viewer.enableFog(true);
viewer.zoomTo({{model:0}});
viewer.render();

// Secondary viewer pinned to a 90° rotation — useful for figure prep where
// readers want to see both faces of the pocket without re-rendering.
const insetViewer = $3Dmol.createViewer("viewer-inset", {{
  backgroundColor: "#0d1117", backgroundAlpha: 0, antialias: true,
}});
insetViewer.addModel(RECEPTOR,"pdb");
insetViewer.setStyle({{model:0}},{{cartoon:{{color:"#8b949e", opacity:0.4}}}});
let insetLigandModel = null;

// Hover tooltips are wired up per-compound in setupHover() — at init time
// there is nothing useful to label, and the cartoon ribbon swallows hover
// events anyway which is why the old global handler felt unreliable.
function setupHover() {{
  const hoverSel = (ligandModel !== null && pocketResi.length)
    ? {{or:[{{model:ligandModel}}, {{model:0, resi:pocketResi}}]}}
    : (ligandModel !== null ? {{model:ligandModel}} : {{model:0, resi:pocketResi}});

  viewer.setHoverable(hoverSel, 120,
    function(atom) {{
      if (atom._label) return;
      const isLig = (ligandModel !== null && atom.model === ligandModel);
      const text = isLig
        ? `${{atom.resn}} · ${{atom.atom}}`
        : `${{atom.resn}} ${{atom.resi}} · Chain ${{atom.chain}} · ${{atom.atom}}`;
      atom._label = viewer.addLabel(text, {{
        position: atom,
        backgroundColor: darkMode ? "rgba(13,17,23,.92)" : "rgba(255,255,255,.96)",
        fontColor:       darkMode ? "#f0f6fc" : "#1a1a1a",
        borderColor:     darkMode ? "#30363d" : "#d0d7de",
        borderThickness: 1, fontSize: 12, borderRadius: 5, padding: 5,
        inFront: true,
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

  // Click to pin — same selection, toggles a persistent label.
  viewer.setClickable(hoverSel, true, function(atom) {{
    const key = `${{atom.model}}:${{atom.chain||''}}:${{atom.resi}}:${{atom.atom}}`;
    if (pinnedLabels.has(key)) {{
      viewer.removeLabel(pinnedLabels.get(key));
      pinnedLabels.delete(key);
      viewer.render();
      return;
    }}
    const isLig = (ligandModel !== null && atom.model === ligandModel);
    const text = isLig
      ? `${{atom.resn}} · ${{atom.atom}}`
      : `${{atom.resn}} ${{atom.resi}} · Chain ${{atom.chain}}`;
    const label = viewer.addLabel(text, {{
      position: atom,
      backgroundColor: darkMode ? "rgba(31,111,235,.95)" : "rgba(9,105,218,.95)",
      fontColor: "#ffffff",
      borderColor: darkMode ? "#388bfd" : "#0969da",
      borderThickness: 1.5, fontSize: 12, borderRadius: 5, padding: 6,
      inFront: true,
    }});
    pinnedLabels.set(key, label);
    viewer.render();
  }});
}}

function clearPinnedLabels() {{
  pinnedLabels.forEach(l => viewer.removeLabel(l));
  pinnedLabels.clear();
}}

// ── Compound dropdown ────────────────────────────────────────────────────
const sel = document.getElementById("compound-select");
COMPOUNDS.forEach(c => {{
  const o = document.createElement("option");
  o.value = c.lig_id;
  o.textContent = `#${{c.rank}}  ${{c.name||c.lig_id}}  (${{c.score}} kcal/mol)`;
  sel.appendChild(o);
}});

// ── Interaction toggles ──────────────────────────────────────────────────
const togglesDiv = document.getElementById("toggles-section");
Object.entries(COLORS).forEach(([type,color]) => {{
  const row = document.createElement("label");
  row.className = "toggle-row";
  row.innerHTML = `<input type="checkbox" checked onchange="toggleType('${{type}}',this.checked)">
    <span class="color-dot" style="background:${{color}}"></span>
    <span>${{type.replace(/_/g," ")}}</span>`;
  togglesDiv.appendChild(row);
}});

// ── Mode switching ───────────────────────────────────────────────────────
function switchMode(mode) {{
  currentMode = mode;
  const is3d  = mode === '3d';

  document.getElementById("viewer-wrap").style.display    = is3d ? 'flex' : 'none';
  document.getElementById("diagram2d-wrap").style.display = is3d ? 'none' : 'flex';
  document.getElementById("site-toggle").style.display    = is3d ? 'none' : 'flex';

  document.getElementById("btn-3d").classList.toggle("active", is3d);
  document.getElementById("btn-2d").classList.toggle("active", !is3d);

  // Certain buttons only active in 3D
  ["btn-labels","btn-hydro"].forEach(id => {{
    const b = document.getElementById(id);
    if (b) b.disabled = !is3d;
  }});
  // Dark BG and Distances work in both modes
  document.getElementById("btn-bg").disabled = false;
  document.getElementById("btn-dist").disabled = false;

  document.getElementById("btn-exp-vid").disabled = !is3d;
  document.getElementById("btn-exp-svg").disabled =  is3d;

  if (!is3d && currentData) draw2DView(currentData, siteMode);
  if ( is3d) viewer.render();
}}

function setSiteMode(site) {{
  siteMode = site;
  document.getElementById("btn-site").classList.toggle("active",  site);
  document.getElementById("btn-full").classList.toggle("active", !site);
  if (currentData && currentMode === '2d') draw2DView(currentData, siteMode);
}}

// ── 3D: clear ligand ────────────────────────────────────────────────────
function clearLigand() {{
  if (ligandModel !== null) {{ viewer.removeModel(ligandModel); ligandModel = null; }}
  currentShapes.forEach(s=>viewer.removeShape(s)); currentShapes=[];
  distLabels.forEach(l=>viewer.removeLabel(l));    distLabels=[];
  resLabels.forEach(l=>viewer.removeLabel(l));     resLabels=[];
  if (hydrophobSurf!==null) {{ viewer.removeSurface(hydrophobSurf); hydrophobSurf=null; }}
  if (pocketSurf!==null)    {{ viewer.removeSurface(pocketSurf);    pocketSurf=null;    }}
  clearPinnedLabels();
  pocketResi = [];
  viewer.setStyle({{model:0}},{{cartoon:{{color:"spectrum",opacity:.9}}}});
  if (typeof updateInset === 'function') updateInset(null);
}}

// ── 3D: draw interactions ────────────────────────────────────────────────
function drawInteractions(ixs) {{
  ixs.forEach(ix => {{
    if (!activeToggles[ix.type]) return;
    const color = COLORS[ix.type]||"#fff";

    // H-bonds: if PLIP gave us the H atom, render donor–H solid + H–acceptor
    // dashed, which is the convention every textbook uses. Otherwise fall
    // back to a single dashed cylinder so older runs still display.
    if (ix.type === 'hbond' && ix.h_coords && ix.donor_coords && ix.acceptor_coords) {{
      const d = ix.donor_coords, h = ix.h_coords, a = ix.acceptor_coords;
      currentShapes.push(viewer.addCylinder({{
        start:{{x:d[0],y:d[1],z:d[2]}}, end:{{x:h[0],y:h[1],z:h[2]}},
        radius:.06, color, opacity:.75, dashed:false,
      }}));
      currentShapes.push(viewer.addCylinder({{
        start:{{x:h[0],y:h[1],z:h[2]}}, end:{{x:a[0],y:a[1],z:a[2]}},
        radius:.1, color, opacity:.95, dashed:true,
      }}));
    }} else {{
      currentShapes.push(viewer.addCylinder({{
        start:{{x:ix.protein_coords[0],y:ix.protein_coords[1],z:ix.protein_coords[2]}},
        end:  {{x:ix.ligand_coords[0], y:ix.ligand_coords[1], z:ix.ligand_coords[2] }},
        radius:.1, color, opacity:.9, dashed:true,
      }}));
    }}
    if (showDistLabels) {{
      const lbl = viewer.addLabel(`${{ix.distance.toFixed(1)}} Å`,{{
        position:{{
          x:(ix.protein_coords[0]+ix.ligand_coords[0])/2,
          y:(ix.protein_coords[1]+ix.ligand_coords[1])/2,
          z:(ix.protein_coords[2]+ix.ligand_coords[2])/2,
        }},
        fontSize:10, fontColor:color,
        backgroundColor:"rgba(13,17,23,.75)",
        borderRadius:3, padding:2, inFront:true,
      }});
      distLabels.push(lbl);
    }}
  }});
  viewer.render();
}}

function refreshInteractions() {{
  currentShapes.forEach(s=>viewer.removeShape(s)); currentShapes=[];
  distLabels.forEach(l=>viewer.removeLabel(l));    distLabels=[];
  if (currentData) drawInteractions(currentData.interactions||[]);
}}

function drawResidueLabels(compound) {{
  const seen = new Set();
  (compound.interactions||[]).forEach(ix => {{
    const key = `${{ix.residue_name}}${{ix.residue_number}}`;
    if (seen.has(key)) return; seen.add(key);
    resLabels.push(viewer.addLabel(`${{ix.residue_name}} ${{ix.residue_number}}`,{{
      position:{{x:ix.protein_coords[0],y:ix.protein_coords[1],z:ix.protein_coords[2]}},
      fontSize:10, fontColor:"#f0f6fc",
      backgroundColor:"rgba(0,0,0,.65)", borderRadius:3, padding:2, inFront:false,
    }}));
  }});
  viewer.render();
}}

function refreshResLabels() {{
  resLabels.forEach(l=>viewer.removeLabel(l)); resLabels=[];
  if (currentData && showResLabels) drawResidueLabels(currentData);
  viewer.render();
}}

async function refreshPocketSurface() {{
  if (pocketSurf !== null) {{ viewer.removeSurface(pocketSurf); pocketSurf = null; }}
  if (!showPocketSurf || !pocketResi.length) {{ viewer.render(); return; }}
  pocketSurf = await viewer.addSurface(
    $3Dmol.SurfaceType.SES,
    {{ opacity: 0.55, color: darkMode ? "#c9d1d9" : "#6e7681" }},
    {{ model: 0, resi: pocketResi }},
  );
  viewer.render();
}}

async function refreshHydrophobSurface() {{
  if (hydrophobSurf!==null) {{ viewer.removeSurface(hydrophobSurf); hydrophobSurf=null; }}
  if (!showHydrophob||!currentData) {{ viewer.render(); return; }}
  const bsRes=[...new Set((currentData.interactions||[]).map(ix=>ix.residue_number))];
  if (!bsRes.length) return;
  hydrophobSurf = await viewer.addSurface(
    $3Dmol.SurfaceType.MS,
    {{ opacity:.65, colorfunc:(atom)=>kdHex(atom.resn) }},
    {{ model:0, resi:bsRes }}
  );
  viewer.render();
}}

// ── Toolbar toggles (3D) ─────────────────────────────────────────────────
function toggleBackground() {{
  darkMode=!darkMode;
  viewer.setBackgroundColor(darkMode?"#0d1117":"#f6f8fa"); viewer.render();
  document.body.classList.toggle("light",!darkMode);
  const btn=document.getElementById("btn-bg");
  btn.textContent=darkMode?"Dark BG":"Light BG";
  btn.classList.toggle("active",!darkMode);
  if (currentData&&currentMode==='2d') draw2DView(currentData,siteMode);
  if (showPocketSurf) refreshPocketSurface();
}}
function toggleResLabels() {{
  showResLabels=!showResLabels;
  document.getElementById("btn-labels").classList.toggle("active",showResLabels);
  refreshResLabels();
}}
function toggleDistLabels() {{
  showDistLabels=!showDistLabels;
  document.getElementById("btn-dist").classList.toggle("active",showDistLabels);
  refreshInteractions();
  if (currentData&&currentMode==='2d') draw2DView(currentData,siteMode);
}}
function toggleHydrophob() {{
  showHydrophob=!showHydrophob;
  document.getElementById("btn-hydro").classList.toggle("active",showHydrophob);
  refreshHydrophobSurface();
}}
function togglePocketSurf() {{
  showPocketSurf = !showPocketSurf;
  document.getElementById("btn-surf").classList.toggle("active", showPocketSurf);
  refreshPocketSurface();
}}
function toggleDepth() {{
  depthFog = !depthFog;
  document.getElementById("btn-depth").classList.toggle("active", depthFog);
  viewer.enableFog(depthFog);
  viewer.render();
}}

// Three presets that swap how the receptor is drawn around the ligand.
async function applyPreset(name) {{
  currentPreset = name;
  // Wipe receptor styles and rebuild from scratch — additive styling gets
  // messy fast when toggling between presets.
  viewer.setStyle({{model:0}}, {{}});

  if (name === 'publication') {{
    viewer.setStyle({{model:0}}, {{cartoon:{{color:"#8b949e", opacity:0.45}}}});
    if (pocketResi.length) {{
      viewer.addStyle({{model:0, resi:pocketResi}},
        {{stick:{{colorscheme:"cyanCarbon", radius:0.20}}}});
    }}
    showPocketSurf = false;
  }} else if (name === 'sticks') {{
    if (pocketResi.length) {{
      viewer.addStyle({{model:0, resi:pocketResi}},
        {{stick:{{colorscheme:"cyanCarbon", radius:0.22}}}});
    }}
    showPocketSurf = false;
  }} else if (name === 'surface') {{
    if (pocketResi.length) {{
      viewer.addStyle({{model:0, resi:pocketResi}},
        {{stick:{{colorscheme:"cyanCarbon", radius:0.18}}}});
    }}
    showPocketSurf = true;
  }}

  document.getElementById("btn-surf").classList.toggle("active", showPocketSurf);
  await refreshPocketSurface();
  viewer.render();
}}

function setPreset(name) {{ applyPreset(name); }}

function updateInset(compound) {{
  // Always rebuild from scratch — the inset is small and the cost is trivial.
  if (insetLigandModel !== null) {{
    insetViewer.removeModel(insetLigandModel);
    insetLigandModel = null;
  }}
  insetViewer.setStyle({{model:0}},{{cartoon:{{color:"#8b949e", opacity:0.4}}}});

  if (!compound || !compound.sdf_b64) {{
    insetViewer.zoomTo({{model:0}});
    insetViewer.render();
    return;
  }}

  insetLigandModel = insetViewer.addModel(atob(compound.sdf_b64), "sdf");
  insetViewer.setStyle({{model:insetLigandModel}},{{
    stick:  {{colorscheme:"Jmol", radius:0.16}},
    sphere: {{colorscheme:"Jmol", scale:0.25}},
  }});
  if (pocketResi.length) {{
    insetViewer.addStyle({{model:0, resi:pocketResi}},
      {{stick:{{colorscheme:"cyanCarbon", radius:0.18}}}});
  }}
  const sel = pocketResi.length
    ? {{or:[{{model:insetLigandModel}}, {{model:0, resi:pocketResi}}]}}
    : {{model:insetLigandModel}};
  insetViewer.zoomTo(sel);
  insetViewer.rotate(90, "y");
  insetViewer.render();
}}

// ── Sidebar ──────────────────────────────────────────────────────────────
function renderSidebar(compound) {{
  const list=document.getElementById("ilist");
  list.innerHTML="";
  if (!compound||compound.plip_failed) {{
    list.innerHTML=`<div id="no-compound" style="color:#f85149">${{
      compound?(compound.plip_error||"PLIP analysis failed"):"No data"}}</div>`;
    return;
  }}
  const active=(compound.interactions||[]).filter(ix=>activeToggles[ix.type]);
  if (!active.length) {{ list.innerHTML=`<div id="no-compound">No visible interactions</div>`; return; }}
  active.forEach(ix => {{
    const div=document.createElement("div"); div.className="i-row";
    const color=COLORS[ix.type]||"#fff";
    div.innerHTML=`<div class="i-type" style="color:${{color}}">${{ix.type.replace(/_/g," ")}}</div>
      <div class="i-detail">${{ix.residue_name}}&nbsp;${{ix.residue_number}} (${{ix.chain}})
        &nbsp;&middot;&nbsp; ${{ix.distance.toFixed(2)}}&nbsp;&Aring;</div>`;
    list.appendChild(div);
  }});
}}

// ── 2D diagram ───────────────────────────────────────────────────────────
function transformPt(ax, ay, compound, useSite) {{
  if (!useSite || !compound.site_viewbox) return [ax + GM, ay + GM];
  const [vx, vy, vw, vh] = compound.site_viewbox;
  const siteScale = Math.min(LIG_W / vw, LIG_H / vh);
  const siteDx = (LIG_W - vw * siteScale) / 2;
  const siteDy = (LIG_H - vh * siteScale) / 2;
  return [
    (ax - vx) * siteScale + siteDx + GM,
    (ay - vy) * siteScale + siteDy + GM,
  ];
}}

function eyelashGlyph(cx, cy, atomX, atomY, r, n, color) {{
  const ba = Math.atan2(cy-atomY, cx-atomX);
  const span = Math.PI * 0.65;
  const fa = ba - span/2, ta = ba + span/2;
  let s = '';
  for (let i=0; i<=n; i++) {{
    const a = fa + (ta-fa)*i/n;
    const x1=(cx+(r-4)*Math.cos(a)).toFixed(1), y1=(cy+(r-4)*Math.sin(a)).toFixed(1);
    const x2=(cx+(r+11)*Math.cos(a)).toFixed(1), y2=(cy+(r+11)*Math.sin(a)).toFixed(1);
    s += `<line x1="${{x1}}" y1="${{y1}}" x2="${{x2}}" y2="${{y2}}" stroke="${{color}}" stroke-width="2"/>`;
  }}
  const x1=(cx+r*Math.cos(fa)).toFixed(1), y1=(cy+r*Math.sin(fa)).toFixed(1);
  const x2=(cx+r*Math.cos(ta)).toFixed(1), y2=(cy+r*Math.sin(ta)).toFixed(1);
  s += `<path d="M${{x1}},${{y1}} A${{r}},${{r}} 0 0,1 ${{x2}},${{y2}}" fill="none" stroke="${{color}}" stroke-width="2"/>`;
  return s;
}}

function draw2DView(compound, useSite) {{
  const wrap = document.getElementById("diagram2d-wrap");

  if (!compound || compound.plip_failed) {{
    wrap.innerHTML=`<p class="d2-placeholder">${{
      compound?(compound.plip_error||"PLIP failed"):"Select a compound above"}}</p>`;
    return;
  }}

  const visible = (compound.interactions||[]).filter(ix=>activeToggles[ix.type]);
  if (!visible.length) {{
    wrap.innerHTML=`<p class="d2-placeholder">No visible interactions — check toggles</p>`;
    return;
  }}

  // Inline SVG content (avoids white-box <image> artefact)
  const ligInner = useSite
    ? (compound.lig_svg_inner_site || compound.lig_svg_inner_full)
    : compound.lig_svg_inner_full;
  const ligVb = useSite
    ? (compound.lig_svg_vb_site || compound.lig_svg_vb_full || `0 0 ${{LIG_W}} ${{LIG_H}}`)
    : (compound.lig_svg_vb_full || `0 0 ${{LIG_W}} ${{LIG_H}}`);
  const hasSvg = !!ligInner;

  // Recolor bonds for dark mode (black bonds invisible on dark bg)
  let svgContent = ligInner || '';
  if (darkMode && svgContent) {{
    svgContent = svgContent
      .replace(/stroke:#000000/g, 'stroke:#c9d1d9')
      .replace(/stroke:black/g,   'stroke:#c9d1d9');
    svgContent = svgContent.replace(/fill:#([0-9a-fA-F]{{6}})/g, (m, h) => {{
      const r=parseInt(h.slice(0,2),16), g=parseInt(h.slice(2,4),16), b=parseInt(h.slice(4,6),16);
      return (r<40 && g<40 && b<40) ? 'fill:#c9d1d9' : m;
    }});
  }}

  const OW    = LIG_W + 2*GM;
  const OH    = LIG_H + 2*GM + 52;
  const LIGCX = GM + LIG_W/2;
  const LIGCY = GM + LIG_H/2;
  const PUSH  = 140;
  const NODER = 25;
  const textFill = darkMode ? "#e6edf3" : "#1a1a1a";
  const badgeBg  = darkMode ? "#161b22"  : "#ffffff";

  // --- Group interactions by residue (iterate ALL interactions to keep ix_atom_pts indices correct) ---
  const resMap = new Map();
  (compound.interactions||[]).forEach((ix, origI) => {{
    if (!activeToggles[ix.type]) return;
    const key = `${{ix.residue_name}}${{ix.residue_number}}${{ix.chain}}`;
    if (!resMap.has(key)) resMap.set(key, {{ix, type:ix.type, atomPts:[], allIx:[]}});
    const e = resMap.get(key);
    e.allIx.push(ix);
    const raw = compound.ix_atom_pts?.[origI];
    if (raw) {{
      const [tx,ty] = transformPt(raw[0], raw[1], compound, useSite);
      e.atomPts.push([tx,ty]);
    }}
  }});

  // --- Compute glyph positions (radial push from mean atom position) ---
  const residues = [...resMap.values()].map((e, i, arr) => {{
    let ax, ay;
    if (e.atomPts.length) {{
      ax = e.atomPts.reduce((s,p)=>s+p[0],0)/e.atomPts.length;
      ay = e.atomPts.reduce((s,p)=>s+p[1],0)/e.atomPts.length;
    }} else {{
      ax = LIGCX;
      ay = LIGCY;
    }}
    let dx = ax - LIGCX;
    let dy = ay - LIGCY;
    if (dx === 0 && dy === 0) {{
      const angle = (i/arr.length)*2*Math.PI - Math.PI/2;
      dx = Math.cos(angle);
      dy = Math.sin(angle);
    }}
    const r_atom = Math.sqrt(dx*dx + dy*dy);
    const initialAngle = Math.atan2(dy, dx);
    return {{...e, ax, ay, r_atom, angle: initialAngle}};
  }});

  // --- Enforce angular separation ---
  const MIN_SEP = 0.40; // radians (~23 degrees)
  residues.sort((a,b) => a.angle - b.angle);
  for (let pass=0; pass<10; pass++) {{
    for (let i=0; i<residues.length; i++) {{
      let j = (i + 1) % residues.length;
      let diff = residues[j].angle - residues[i].angle;
      if (diff < 0) diff += 2*Math.PI;
      if (diff < MIN_SEP) {{
        let push = (MIN_SEP - diff) / 2;
        residues[i].angle -= push;
        residues[j].angle += push;
        if (residues[i].angle < -Math.PI) residues[i].angle += 2*Math.PI;
        if (residues[j].angle > Math.PI) residues[j].angle -= 2*Math.PI;
      }}
    }}
    residues.sort((a,b) => a.angle - b.angle);
  }}

  // --- Compute base gx, gy ---
  residues.forEach(r => {{
    let r_glyph = r.r_atom + PUSH;
    r_glyph = Math.max(r_glyph, 180); // Ensure minimum distance from center
    r.gx = LIGCX + Math.cos(r.angle) * r_glyph;
    r.gy = LIGCY + Math.sin(r.angle) * r_glyph;
  }});

  // --- Collision spread (to fix labels/glyphs overlapping radially or closely) ---
  // Using an elliptical distance since labels drop down below the glyph
  for (let pass=0; pass<10; pass++) {{
    for (let a=0; a<residues.length; a++) {{
      for (let b=a+1; b<residues.length; b++) {{
        const dx=residues[b].gx-residues[a].gx, dy=residues[b].gy-residues[a].gy;
        // scale dy slightly to require more vertical distance than horizontal
        const d=Math.sqrt(dx*dx + (dy*1.3)*(dy*1.3));
        const minD = NODER*2 + 55; // increased padding for label clearance
        if (d < minD && d > 0.01) {{
          const push=(minD-d)/2, nx=dx/d, ny=dy/d; // use unscaled d for displacement direction
          residues[a].gx -= nx*push; residues[a].gy -= ny*push;
          residues[b].gx += nx*push; residues[b].gy += ny*push;
        }}
      }}
    }}
  }}

  // --- Final clamp to SVG boundaries ---
  residues.forEach(r => {{
    r.gx=Math.max(NODER+30, Math.min(OW-NODER-30, r.gx));
    r.gy=Math.max(NODER+30, Math.min(OH-52-NODER-30, r.gy));
  }});

  // --- Build SVG ---
  let s = `<svg id="diagram2d-svg" xmlns="http://www.w3.org/2000/svg"
    width="100%" height="100%" viewBox="0 0 ${{OW}} ${{OH}}"
    preserveAspectRatio="xMidYMid meet">`;

  // Arrowhead markers (larger, crisper)
  s += `<defs>`;
  ["hbond","salt_bridge"].forEach(type => {{
    s += `<marker id="d2a-${{type}}" markerWidth="9" markerHeight="9"
            refX="8" refY="4.5" orient="auto">
          <path d="M0,0 L0,9 L9,4.5 Z" fill="${{COLORS[type]}}" opacity=".95"/></marker>`;
  }});
  s += `</defs>`;


  // --- Connection lines: one per residue, from nearest ligand atom ---
  residues.forEach(res => {{
    const col     = COLORS[res.type] || "#888";
    const isHbond = res.type === "hbond" || res.type === "salt_bridge";

    // Find nearest atom point to the finalized glyph position
    let px = LIGCX, py = LIGCY;
    if (res.atomPts.length > 0) {{
      let minDist = Infinity;
      res.atomPts.forEach(p => {{
        const dSq = (p[0]-res.gx)**2 + (p[1]-res.gy)**2;
        if (dSq < minDist) {{
          minDist = dSq;
          px = p[0];
          py = p[1];
        }}
      }});
    }} else {{
      const angle = res.angle;
      px = LIGCX + Math.cos(angle) * (LIG_W * 0.42);
      py = LIGCY + Math.sin(angle) * (LIG_H * 0.42);
    }}

    const ddx=res.gx-px, ddy=res.gy-py;
    const dl=Math.sqrt(ddx*ddx+ddy*ddy)||1;

    if (isHbond) {{
      const ex=res.gx-(ddx/dl)*NODER, ey=res.gy-(ddy/dl)*NODER;
      s += `<line x1="${{px.toFixed(1)}}" y1="${{py.toFixed(1)}}"
              x2="${{ex.toFixed(1)}}" y2="${{ey.toFixed(1)}}"
              stroke="${{col}}" stroke-width="2.2" stroke-dasharray="6 3.5"
              marker-end="url(#d2a-${{res.type}})"/>`;
    }} else {{
      const ex=res.gx-(ddx/dl)*(NODER+2), ey=res.gy-(ddy/dl)*(NODER+2);
      s += `<line x1="${{px.toFixed(1)}}" y1="${{py.toFixed(1)}}"
              x2="${{ex.toFixed(1)}}" y2="${{ey.toFixed(1)}}"
              stroke="${{col}}" stroke-width="1.4" opacity=".5" stroke-dasharray="5 3"/>`;
    }}

    if (showDistLabels && res.ix.distance > 0.1) {{
      const mx=(px+res.gx)/2, my=(py+res.gy)/2;
      s += `<rect x="${{(mx-18).toFixed(1)}}" y="${{(my-9).toFixed(1)}}"
              width="36" height="18" rx="5"
              fill="${{badgeBg}}" stroke="${{col}}" stroke-width="1.3"/>
            <text x="${{mx.toFixed(1)}}" y="${{my.toFixed(1)}}" text-anchor="middle"
              dominant-baseline="middle" fill="${{col}}"
              font-size="9.5" font-weight="700" font-family="Arial,sans-serif">
              ${{res.ix.distance.toFixed(1)}}&thinsp;&Aring;
            </text>`;
    }}
  }});

  // Ligand structure inlined via <g transform> — drawn on top of connection lines
  if (hasSvg) {{
    if (useSite && compound.site_viewbox) {{
      const [vx, vy, vw, vh] = compound.site_viewbox;
      const siteScale = Math.min(LIG_W / vw, LIG_H / vh);
      const siteDx = (LIG_W - vw * siteScale) / 2;
      const siteDy = (LIG_H - vh * siteScale) / 2;
      s += `<g transform="translate(${{GM + siteDx}},${{GM + siteDy}}) scale(${{siteScale}}) translate(${{-vx}},${{-vy}})">${{svgContent}}</g>`;
    }} else {{
      s += `<g transform="translate(${{GM}},${{GM}})">${{svgContent}}</g>`;
    }}
  }} else {{
    s += `<rect x="${{GM}}" y="${{GM}}" width="${{LIG_W}}" height="${{LIG_H}}"
            rx="8" fill="none" stroke="#30363d" stroke-width="1" stroke-dasharray="4 3"/>
          <text x="${{LIGCX}}" y="${{LIGCY}}" text-anchor="middle"
            fill="#8b949e" font-size="13" font-family="Arial,sans-serif">
            Install rdkit for 2D structure</text>`;
  }}

  // --- Residue glyphs + labels ---
  residues.forEach(res => {{
    const {{gx, gy, ix, type}} = res;
    const col = COLORS[type] || "#888";
    const aac = aaColor(ix.residue_name);

    // Eyelash faces the nearest atom (same logic as connection line)
    let srcX = LIGCX, srcY = LIGCY;
    if (res.atomPts.length > 0) {{
      let minDist = Infinity;
      res.atomPts.forEach(p => {{
        const dSq = (p[0]-gx)**2 + (p[1]-gy)**2;
        if (dSq < minDist) {{
          minDist = dSq;
          srcX = p[0];
          srcY = p[1];
        }}
      }});
    }} else {{
      const angle = res.angle;
      srcX = LIGCX + Math.cos(angle) * (LIG_W * 0.42);
      srcY = LIGCY + Math.sin(angle) * (LIG_H * 0.42);
    }}

    // --- Glyph ---
    if (type === "hydrophobic") {{
      s += eyelashGlyph(gx, gy, srcX, srcY, NODER, 9, col);
    }} else {{
      let fillCol = aac + "22";
      let dash = "";
      if (type === "pi_stack" || type === "pi_cation") {{
         fillCol = col + "15";
         dash = ' stroke-dasharray="5 2.5"';
      }} else if (type === "halogen") {{
         fillCol = aac + "15";
      }}
      s += `<circle cx="${{gx.toFixed(1)}}" cy="${{gy.toFixed(1)}}" r="${{NODER}}"
              fill="${{fillCol}}" stroke="${{col}}" stroke-width="2.2"${{dash}}/>`;
    }}

    // --- Inner Text (Name & Number inside the glyph area) ---
    s += `<text x="${{gx.toFixed(1)}}" y="${{(gy-4).toFixed(1)}}" text-anchor="middle"
            dominant-baseline="middle" fill="${{textFill}}"
            font-size="9.5" font-weight="700" font-family="Arial,sans-serif">
            ${{ix.residue_name}}
          </text>
          <text x="${{gx.toFixed(1)}}" y="${{(gy+7).toFixed(1)}}" text-anchor="middle"
            dominant-baseline="middle" fill="${{col}}"
            font-size="8.5" font-weight="600" font-family="Arial,sans-serif">
            ${{ix.residue_number}} (${{ix.chain}})
          </text>`;

    // --- Symbols (Superscripts at top-right of the glyph) ---
    let symbol = "";
    if (type === "salt_bridge") {{
      symbol = ["LYS","ARG","HIS"].includes(ix.residue_name) ? "+" : "−";
    }} else if (type === "pi_stack" || type === "pi_cation") {{
      symbol = "&pi;";
    }} else if (type === "halogen") {{
      symbol = "X";
    }}
    if (symbol) {{
      s += `<text x="${{(gx+NODER-4).toFixed(1)}}" y="${{(gy-NODER+8).toFixed(1)}}"
              fill="${{col}}" font-size="14" font-weight="bold"
              font-family="'Times New Roman',Georgia,serif">${{symbol}}</text>`;
    }}
  }});

  // --- Legend ---
  const LY  = OH - 38;
  const LX0 = 16;
  const COL_W = Math.floor((OW-32) / Object.keys(COLORS).length);
  Object.entries(COLORS).forEach(([type,color], i) => {{
    const lx = LX0 + i*COL_W;
    s += `<rect x="${{lx}}" y="${{LY}}" width="10" height="10" rx="2" fill="${{color}}"/>
          <text x="${{lx+14}}" y="${{LY+9}}" fill="#8b949e" font-size="8.5"
            font-family="Arial,Helvetica,sans-serif">${{type.replace(/_/g,' ')}}</text>`;
  }});

  s += `</svg>`;
  wrap.innerHTML = s;
}}

// ── Main compound select ─────────────────────────────────────────────────
async function selectCompound(lig_id) {{
  clearLigand();
  if (!lig_id) return;
  const compound = COMPOUNDS.find(c=>c.lig_id===lig_id);
  if (!compound) return;
  currentData = compound;

  // Disable site toggle if no viewbox crop available
  document.getElementById("btn-site").disabled = !compound.site_viewbox;

  // 3D
  if (compound.sdf_b64) {{
    const sdf = atob(compound.sdf_b64);
    ligandModel = viewer.addModel(sdf,"sdf");
    viewer.setStyle({{model:ligandModel}},{{
      stick:  {{colorscheme:"Jmol", radius:0.16}},
      sphere: {{colorscheme:"Jmol", scale:0.25}},
    }});
  }}
  // Pull every residue with at least one atom inside 5 Å of the ligand — gives
  // a fuller picture of the pocket than the PLIP-interaction residues alone.
  if (ligandModel !== null) {{
    const near = viewer.selectedAtoms({{
      model:0, within:{{distance:5.0, sel:{{model:ligandModel}}}},
    }});
    pocketResi = [...new Set(near.map(a=>a.resi))];
  }}
  await applyPreset(currentPreset);
  setupHover();
  drawInteractions(compound.interactions||[]);
  updateInset(compound);
  if (showResLabels) drawResidueLabels(compound);
  await refreshPocketSurface();
  await refreshHydrophobSurface();

  renderSidebar(compound);

  if (currentMode==='3d') {{
    if (ligandModel!==null) {{
      // Frame ligand + pocket together so the user sees the binding site in context,
      // then nudge off-axis so it doesn't look like a flat schematic.
      const sel = pocketResi.length
        ? {{or:[{{model:ligandModel}}, {{model:0, resi:pocketResi}}]}}
        : {{model:ligandModel}};
      viewer.zoomTo(sel, 500);
      viewer.rotate(15, "x");
    }}
    viewer.render();
  }} else {{
    draw2DView(compound, siteMode);
  }}
}}

function toggleType(type, checked) {{
  activeToggles[type]=checked;
  if (!currentData) return;
  refreshInteractions();
  renderSidebar(currentData);
  if (currentMode==='2d') draw2DView(currentData, siteMode);
}}

// ── Exports ──────────────────────────────────────────────────────────────
function exportPNG() {{
  if (currentMode==='2d') {{
    const svg = document.getElementById("diagram2d-svg");
    if (!svg) {{ alert("Select a compound first."); return; }}
    const name = currentData?(currentData.name||currentData.lig_id):"pose";
    const str  = new XMLSerializer().serializeToString(svg);
    const img  = new Image();
    const blob = new Blob([str],{{type:'image/svg+xml'}});
    const url  = URL.createObjectURL(blob);
    img.onload = () => {{
      const sc=2, cv=document.createElement('canvas');
      cv.width=svg.viewBox.baseVal.width*sc||1680;
      cv.height=svg.viewBox.baseVal.height*sc||1320;
      const ctx=cv.getContext('2d');
      ctx.scale(sc,sc);
      ctx.fillStyle=darkMode?"#0d1117":"#f6f8fa";
      ctx.fillRect(0,0,cv.width,cv.height);
      ctx.drawImage(img,0,0);
      cv.toBlob(b => {{
        const a=Object.assign(document.createElement('a'),
          {{href:URL.createObjectURL(b),download:`${{name}}_2d.png`}});
        a.click();
      }},'image/png');
      URL.revokeObjectURL(url);
    }};
    img.src=url;
  }} else {{
    const name=currentData?(currentData.name||currentData.lig_id):"pose";
    const a=Object.assign(document.createElement("a"),
      {{href:viewer.pngURI(),download:`${{name}}_3d.png`}});
    a.click();
  }}
}}

function exportSVG() {{
  const svg=document.getElementById("diagram2d-svg");
  if (!svg) {{ alert("Select a compound first."); return; }}
  const name=currentData?(currentData.name||currentData.lig_id):"pose";
  const blob=new Blob([new XMLSerializer().serializeToString(svg)],
    {{type:'image/svg+xml'}});
  const a=Object.assign(document.createElement('a'),
    {{href:URL.createObjectURL(blob),download:`${{name}}_2d_interactions.svg`}});
  a.click();
}}

function exportVideo() {{
  const canvas=document.querySelector("#viewer canvas");
  if (!canvas) {{ alert("3D canvas not ready."); return; }}
  const mime=["video/webm;codecs=vp9","video/webm"].find(m=>MediaRecorder.isTypeSupported(m));
  if (!mime) {{ alert("WebM not supported — try Chrome."); return; }}
  const rec=new MediaRecorder(canvas.captureStream(30),{{mimeType:mime}});
  const chunks=[];
  rec.ondataavailable=e=>{{if(e.data.size)chunks.push(e.data);}};
  rec.onstop=()=>{{
    const name=currentData?(currentData.name||currentData.lig_id):"pose";
    const a=Object.assign(document.createElement('a'),{{
      href:URL.createObjectURL(new Blob(chunks,{{type:'video/webm'}})),
      download:`${{name}}_360.webm`}});
    a.click();
    document.getElementById("record-overlay").style.display="none";
    const btn=document.getElementById("btn-exp-vid");
    btn.disabled=false; btn.textContent="Export 360° Video";
  }};
  document.getElementById("record-overlay").style.display="block";
  const btn=document.getElementById("btn-exp-vid");
  btn.disabled=true; btn.textContent="Recording…";
  rec.start();
  let step=0;
  (function tick(){{
    if(step>=72){{rec.stop();return;}}
    viewer.rotate(5,"y"); viewer.render();
    document.getElementById("record-pct").textContent=Math.round(step/72*100)+"%";
    step++; setTimeout(tick,50);
  }})();
}}

viewer.render();
</script>
</body>
</html>
"""


def generate_viewer(work_dir: Path) -> Path:
    output_dir    = work_dir / "output"
    viewer_path   = output_dir / "interaction_viewer.html"

    interactions_json = output_dir / "interactions_top_n.json"
    if not interactions_json.exists():
        raise FileNotFoundError(
            "interactions_top_n.json not found — run PLIP analysis first"
        )

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
    compounds     = json.loads(interactions_json.read_text())
    compounds     = _enrich_2d(compounds)

    html = _build_html(compounds, receptor_text)
    viewer_path.write_text(html, encoding="utf-8")
    return viewer_path
