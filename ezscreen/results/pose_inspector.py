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

_MOLSTAR_VERSION = "4.4.0"
_MOLSTAR_CSS = f"https://unpkg.com/molstar@{_MOLSTAR_VERSION}/build/viewer/molstar.css"
_MOLSTAR_JS  = f"https://unpkg.com/molstar@{_MOLSTAR_VERSION}/build/viewer/molstar.js"

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

# Floating residue layer keeps full atoms (including backbone) so the
# Cα coincides with the receptor's Cα — ball-and-stick then appears to
# branch out of the cartoon ribbon instead of floating in space.
_RES3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _extract_sequence(receptor_text: str) -> list[dict]:
    """Walk ATOM lines, return [{chain, items:[{resi, resn, one, ca:[x,y,z]}]}].

    Order within a chain follows the order of first appearance in the PDB.
    Cα coordinates come from the CA atom of each residue; if no CA is present
    (rare), the first atom seen for that residue is used as a fallback.
    """
    if not receptor_text:
        return []
    chains: dict[str, list] = {}
    seen: dict[tuple[str, int], dict] = {}
    chain_order: list[str] = []
    for line in receptor_text.splitlines():
        if not line.startswith("ATOM  ") or len(line) < 54:
            continue
        chain = line[21]
        try:
            resi = int(line[22:26])
        except ValueError:
            continue
        atom = line[12:16].strip()
        resn = line[17:20].strip()
        key = (chain, resi)
        if key not in seen:
            entry = {
                "resi": resi,
                "resn": resn,
                "one":  _RES3TO1.get(resn, "X"),
                "ca":   None,
            }
            seen[key] = entry
            if chain not in chains:
                chains[chain] = []
                chain_order.append(chain)
            chains[chain].append(entry)
        if seen[key]["ca"] is None or atom == "CA":
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                seen[key]["ca"] = [round(x, 3), round(y, 3), round(z, 3)]
            except ValueError:
                pass
    return [{"chain": ch, "items": chains[ch]} for ch in chain_order]


def _extract_residue_pdb(receptor_text: str,
                         targets: set[tuple[str, int]]) -> str:
    if not targets or not receptor_text:
        return ""
    out: list[str] = []
    for line in receptor_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 26:
            continue
        chain = line[21]
        try:
            resi = int(line[22:26])
        except ValueError:
            continue
        if (chain, resi) not in targets:
            continue
        out.append(line)
    if not out:
        return ""
    out.append("END")
    return "\n".join(out) + "\n"


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

            mol_2d = Chem.MolFromMolBlock(sdf_text, removeHs=True)
            AllChem.Compute2DCoords(mol_2d)
            rdMolDraw2D.PrepareMolForDrawing(mol_2d)

            interactions = c.get("interactions", [])
            per_ix_atom: list[int | None] = []
            interacting: dict[int, str] = {}

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

            drawer = rdMolDraw2D.MolDraw2DSVG(_LIG_SVG_W, _LIG_SVG_H)
            drawer.drawOptions().addStereoAnnotation = True
            drawer.drawOptions().padding = 0.12
            drawer.DrawMolecule(mol_2d)
            drawer.FinishDrawing()
            svg_full = drawer.GetDrawingText()

            svg_full = _re.sub(
                r"(style=['\"])background[^;'\"]*;?",
                r"\1background:transparent;",
                svg_full,
            )

            atom_2d: dict[int, list[float]] = {}
            for idx in h_atoms:
                try:
                    pt = drawer.GetDrawCoords(idx)
                    atom_2d[idx] = [round(pt.x, 2), round(pt.y, 2)]
                except Exception:
                    pass

            ix_atom_pts: list[list[float] | None] = []
            for idx in per_ix_atom:
                if idx is not None and idx in atom_2d:
                    ix_atom_pts.append(atom_2d[idx])
                else:
                    ix_atom_pts.append(None)

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


# ── HTML builder (Mol* — stage 1) ─────────────────────────────────────────
#
# This is the Mol*-based viewer scaffold. Stage 1 ships:
#   • Mol* viewer with cartoon receptor + ligand auto-styled
#   • Native hover residue highlight (the whole reason for the migration)
#   • Compound dropdown that loads / removes the ligand structure
#   • Sidebar: interaction-type toggles + per-compound interaction list
#   • 3D / 2D mode switch (the 2D LIGPLOT diagram is unchanged from the
#     prior viewer — same _enrich_2d data, same SVG generation code)
#   • Light / dark BG toggle
#
# Later stages will rebuild on top of this: interaction cylinders (Mol* shape
# API), click-to-pin selection, protein / ligand style controls, pocket
# detection, surfaces, export PNG / WebM, 90° inset.

def _build_html(compounds: list[dict], receptor_pdb_text: str) -> str:
    receptor_escaped = receptor_pdb_text.replace("\\", "\\\\").replace("`", "\\`")
    compounds_json   = json.dumps(compounds)
    colors_json      = json.dumps(_INTERACTION_COLORS)
    sequence_json    = json.dumps(_extract_sequence(receptor_pdb_text))
    lig_svg_w        = _LIG_SVG_W
    lig_svg_h        = _LIG_SVG_H
    glyph_mar        = _GLYPH_MAR

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SwiftScreen — Interaction Viewer</title>
<link rel="stylesheet" type="text/css" href="{_MOLSTAR_CSS}">
<script type="text/javascript" src="{_MOLSTAR_JS}"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}

:root{{
  --bg:#0e1114; --panel:#161a20; --panel2:#1b2027; --elev:#222932;
  --border:#2a313b; --border2:#3a434f;
  --text:#dde4ec; --muted:#8a95a3; --faint:#5a6471; --tile-fg:#aeb9c6;
  --accent:#2dd4bf; --accent-fg:#04130f; --accent-soft:rgba(45,212,191,.15);
  --radius:6px; --radius-sm:4px;
  --font:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;
  --mono:ui-monospace,'SF Mono','JetBrains Mono',Menlo,Consolas,monospace;
  --tile-w:16px; --tile-h:20px; --tile-font:11px;
  --shadow:0 10px 28px rgba(0,0,0,.55);
  --btn:var(--panel2); --btn-hover:var(--elev);
}}

body{{
  font-family:var(--font);
  background:var(--bg); color:var(--text);
  display:flex; flex-direction:column; height:100vh; overflow:hidden;
  transition:background .2s,color .2s;
}}

#banner{{
  background:var(--panel); border-bottom:1px solid var(--border);
  padding:4px 16px; font-size:11px; color:var(--muted);
  text-align:center; flex-shrink:0;
}}

#toolbar{{
  background:var(--panel); border-bottom:1px solid var(--border);
  padding:6px 12px; display:flex; align-items:center;
  gap:6px; flex-wrap:wrap; flex-shrink:0;
}}
.tb-sep{{ width:1px; height:20px; background:var(--border); margin:0 3px; flex-shrink:0; }}
.tb-label{{ font-size:11px; color:var(--muted); white-space:nowrap; }}

.tb-btn{{
  background:var(--btn); color:var(--text);
  border:1px solid var(--border); border-radius:var(--radius);
  padding:4px 10px; font-size:12px; font-family:var(--font);
  cursor:pointer; white-space:nowrap;
  transition:background .15s,border-color .15s,color .15s;
}}
.tb-btn:hover{{ background:var(--btn-hover); border-color:var(--border2); }}
.tb-btn:disabled{{ opacity:.4; cursor:not-allowed; }}
.tb-btn.active{{ background:var(--accent); border-color:var(--accent); color:var(--accent-fg); }}

.view-pair{{ display:flex; gap:0; }}
.view-pair .tb-btn:first-child{{ border-radius:var(--radius) 0 0 var(--radius); border-right:none; }}
.view-pair .tb-btn:last-child{{ border-radius:0 var(--radius) var(--radius) 0; }}

.view-dropdown,.rep-dropdown{{ position:relative; display:inline-block; }}
.view-menu{{
  position:absolute; top:calc(100% + 4px); left:0;
  background:var(--elev); border:1px solid var(--border); border-radius:var(--radius);
  min-width:180px; max-height:300px; overflow-y:auto;
  z-index:100; padding:4px; box-shadow:var(--shadow);
}}
.view-menu.up{{ top:auto; bottom:calc(100% + 4px); }}
.view-menu-item{{
  display:flex; align-items:center; gap:4px;
  padding:5px 6px; border-radius:var(--radius-sm); font-size:12px;
}}
.view-menu-item:hover{{ background:var(--panel2); }}
.rep-menu-item.active{{ background:var(--accent-soft); }}
.vm-name{{ flex:1; cursor:pointer; padding:1px 2px; }}
.vm-del{{
  width:18px; height:18px; border-radius:var(--radius-sm);
  display:flex; align-items:center; justify-content:center;
  color:var(--muted); font-size:15px; line-height:1; cursor:pointer;
}}
.vm-del:hover{{ background:#d23b3b; color:#fff; }}
.view-menu-empty{{ padding:8px; font-size:12px; color:var(--muted); text-align:center; }}

#style-menu{{ overflow:visible; max-height:none; min-width:0; padding:4px; }}
.style-cat{{ position:relative; }}
.style-cat-head{{ cursor:default; white-space:nowrap; }}
.style-cat-cur{{ color:var(--muted); font-size:11px; margin-left:16px; }}
.style-arrow{{ color:var(--muted); font-size:9px; margin-left:8px; }}
.style-cat:hover > .style-cat-head{{ background:var(--panel2); }}
.view-submenu{{
  display:none; position:absolute; top:-5px; left:calc(100% + 3px);
  background:var(--elev); border:1px solid var(--border); border-radius:var(--radius);
  min-width:150px; z-index:110; padding:4px; box-shadow:var(--shadow);
}}
.style-cat:hover > .view-submenu{{ display:block; }}

.seg{{ display:inline-flex; border:1px solid var(--border); border-radius:var(--radius); overflow:hidden; }}
.seg-btn{{
  background:var(--btn); color:var(--text); border:none;
  border-right:1px solid var(--border); padding:4px 11px; font-size:12px;
  cursor:pointer; font-family:var(--font); transition:background .15s,color .15s;
}}
.seg-btn:last-child{{ border-right:none; }}
.seg-btn:hover{{ background:var(--btn-hover); }}
.seg-btn.active{{ background:var(--accent); color:var(--accent-fg); }}

#color-menu{{ min-width:236px; }}
.color-sep{{ height:1px; background:var(--border); margin:6px 2px; }}
.color-sec-head{{
  font-size:10px; letter-spacing:.06em; text-transform:uppercase;
  color:var(--muted); padding:3px 6px 4px; font-weight:600;
}}
.color-swatch-grid{{ display:grid; grid-template-columns:repeat(2,1fr); gap:2px 6px; padding:2px 4px 4px; }}
.color-swatch{{ display:flex; align-items:center; gap:6px; font-size:12px; padding:2px; border-radius:var(--radius-sm); cursor:pointer; }}
.color-swatch:hover{{ background:var(--panel2); }}
.color-swatch input[type=color],.pr-color{{
  width:22px; height:16px; padding:0; border:1px solid var(--border);
  border-radius:3px; background:none; cursor:pointer; flex-shrink:0;
}}
.color-reset{{ margin-top:3px; justify-content:center; }}
.color-reset .vm-name{{ flex:0; color:var(--muted); font-size:11px; white-space:nowrap; }}

.pr-row{{ display:flex; align-items:center; gap:5px; padding:4px 6px; }}
.pr-row select{{
  flex:1; min-width:0; background:var(--btn); color:var(--text);
  border:1px solid var(--border); border-radius:var(--radius-sm);
  padding:3px 4px; font-size:11px; font-family:var(--font);
}}
.pr-apply{{
  background:var(--accent); color:var(--accent-fg); border:none;
  border-radius:var(--radius-sm); padding:4px 9px; font-size:11px;
  cursor:pointer; white-space:nowrap; font-family:var(--font);
}}
.pr-apply:hover{{ filter:brightness(1.08); }}
.pr-hint{{ font-size:11px; color:var(--muted); padding:3px 6px; }}
.ov-list{{ max-height:124px; overflow-y:auto; padding:2px; }}
.ov-item{{ display:flex; align-items:center; gap:6px; padding:3px 6px; border-radius:var(--radius-sm); font-size:11px; }}
.ov-item:hover{{ background:var(--panel2); }}
.ov-swatch{{ width:11px; height:11px; border-radius:2px; flex-shrink:0; border:1px solid rgba(0,0,0,.28); }}
.ov-label{{ flex:1; font-family:var(--mono); font-variant-numeric:tabular-nums; }}
.ov-del{{ color:var(--muted); cursor:pointer; font-size:14px; line-height:1; padding:0 3px; }}
.ov-del:hover{{ color:#d23b3b; }}

#main{{ display:flex; flex:1; overflow:hidden; min-height:0; }}
#viewer{{ flex:1; position:relative; background:var(--bg); }}
body.light #viewer,body.light #diagram2d-wrap{{ background:#f6f8fa; }}
#diagram2d-wrap{{ flex:1; display:none; align-items:center; justify-content:center; overflow:hidden; padding:12px; position:relative; }}
#diagram2d-wrap svg{{ max-width:100%; max-height:100%; }}
.d2-placeholder{{ color:var(--muted); font-size:14px; text-align:center; padding:40px; }}

#sidebar{{ width:300px; background:var(--panel); border-left:1px solid var(--border); display:flex; flex-direction:column; overflow:hidden; }}
.sb-section{{ padding:10px 12px; border-bottom:1px solid var(--border); flex-shrink:0; }}
.sb-section select{{ width:100%; background:var(--btn); color:var(--text); border:1px solid var(--border); border-radius:var(--radius); padding:6px; font-size:12px; font-family:var(--font); }}
#site-toggle{{ display:none; gap:0; margin-top:8px; }}
#site-toggle .tb-btn:first-child{{ border-radius:var(--radius) 0 0 var(--radius); border-right:none; flex:1; }}
#site-toggle .tb-btn:last-child{{ border-radius:0 var(--radius) var(--radius) 0; flex:1; }}
.toggle-row{{ display:flex; align-items:center; gap:7px; margin:3px 0; font-size:12px; cursor:pointer; }}
.toggle-row input{{ width:13px; height:13px; cursor:pointer; }}
.color-dot{{ width:9px; height:9px; border-radius:50%; flex-shrink:0; }}
#ilist{{ flex:1; overflow-y:auto; padding:8px; min-height:0; }}
.i-row{{ background:var(--panel2); border:1px solid var(--border); border-radius:var(--radius); margin:3px 0; padding:7px 8px; font-size:12px; }}
.i-type{{ font-weight:600; text-transform:capitalize; }}
.i-detail{{ color:var(--muted); margin-top:2px; font-size:11px; font-family:var(--mono); font-variant-numeric:tabular-nums; }}
#no-compound{{ padding:16px; color:var(--muted); font-size:13px; text-align:center; }}
h3{{ font-size:11px; font-weight:600; color:var(--muted); letter-spacing:.05em; text-transform:uppercase; }}

#seqpanel{{ flex-shrink:0; background:var(--panel); border-top:1px solid var(--border); display:flex; flex-direction:column; max-height:176px; }}
#seqpanel.collapsed #seq-body{{ display:none; }}
#seq-header{{ display:flex; align-items:center; gap:10px; padding:6px 12px; border-bottom:1px solid var(--border); font-size:11px; }}
.seq-tool{{ padding:3px 9px; font-size:11px; }}
.seq-actions{{ display:flex; align-items:center; gap:8px; }}
.seq-count{{ font-size:11px; color:var(--muted); font-family:var(--mono); font-variant-numeric:tabular-nums; }}
.seq-paint{{ width:22px; height:18px; padding:0; border:1px solid var(--border); border-radius:3px; background:none; cursor:pointer; }}
.seq-link{{ background:none; border:none; color:var(--accent); font-size:11px; cursor:pointer; padding:0; font-family:var(--font); }}
.seq-link:hover{{ text-decoration:underline; }}
#seq-chainbar{{ display:flex; gap:4px; padding:4px 12px; border-bottom:1px solid var(--border); flex-wrap:wrap; align-items:center; flex-shrink:0; }}
#seq-chainbar .cb-label{{ font-size:10px; letter-spacing:.05em; text-transform:uppercase; color:var(--muted); margin-right:2px; }}
.chain-chip{{ display:inline-flex; align-items:center; gap:5px; background:var(--btn); color:var(--text); border:1px solid var(--border); border-radius:var(--radius-sm); padding:2px 9px; font-size:11px; cursor:pointer; font-family:var(--font); transition:background .15s,color .15s; }}
.chain-chip:hover{{ background:var(--btn-hover); }}
.chain-chip.active{{ background:var(--accent); color:var(--accent-fg); border-color:var(--accent); }}
.chain-dot{{ width:5px; height:5px; border-radius:50%; background:var(--accent); flex-shrink:0; }}
.chain-chip.active .chain-dot{{ background:var(--accent-fg); }}
#seq-body{{ overflow-x:auto; overflow-y:auto; padding:6px 12px 10px; }}
.seq-chain{{ display:flex; align-items:flex-end; gap:8px; margin:4px 0; font-family:var(--mono); white-space:nowrap; }}
.seq-chain-label{{ font-size:11px; color:var(--muted); min-width:48px; flex-shrink:0; font-family:var(--font); padding-bottom:2px; }}
.seq-tiles{{ display:inline-flex; gap:1px; }}
.seq-col{{ display:inline-flex; flex-direction:column; align-items:center; }}
.seq-tick{{ height:11px; font-size:8.5px; line-height:11px; color:var(--faint); font-family:var(--mono); white-space:nowrap; overflow:visible; }}
.seq-tick.on{{ color:var(--muted); }}
.seq-tile{{
  display:inline-flex; align-items:center; justify-content:center;
  width:var(--tile-w); height:var(--tile-h); font-size:var(--tile-font);
  background:transparent; color:var(--tile-fg); border-radius:2px; cursor:pointer;
  position:relative; user-select:none; transition:background .1s,color .1s,opacity .1s;
}}
.seq-tile:hover{{ background:var(--panel2); color:var(--text); outline:1px solid var(--accent); }}
.seq-tile.ix{{ color:#fff; font-weight:600; }}
.seq-tile.sel{{ outline:2px solid var(--accent); outline-offset:-1px; color:var(--text); }}
.seq-tile.dim{{ opacity:.2; }}
.seq-tile.ov::after{{ content:''; position:absolute; bottom:1px; left:50%; transform:translateX(-50%); width:5px; height:5px; border-radius:50%; background:var(--ov-dot,#fff); border:1px solid rgba(0,0,0,.3); }}

#viewer .msp-plugin{{ background:transparent; }}
#viewer .msp-viewport{{ background:transparent !important; }}
#viewer .msp-viewport-controls,
#viewer .msp-viewport-top-left-controls,
#viewer .msp-canvas-renderer-target-control{{ display:none !important; }}
</style>
</head>
<body>

<div id="banner">
  Predicted pose — not experimentally validated &nbsp;&middot;&nbsp;
  Pipeline: UniDock &rarr; PLIP &rarr; SwiftScreen &nbsp;&middot;&nbsp;
  Viewer: Mol*&nbsp;{_MOLSTAR_VERSION}
</div>

<div id="toolbar">
  <span class="tb-label">View</span>
  <div class="view-pair">
    <button class="tb-btn active" id="btn-3d" onclick="switchMode('3d')">3D</button>
    <button class="tb-btn"        id="btn-2d" onclick="switchMode('2d')">2D</button>
  </div>

  <div class="tb-sep"></div>

  <span class="tb-label">Display</span>
  <button class="tb-btn" id="btn-bg"   onclick="toggleBackground()" title="Switch light / dark background">Light BG</button>
  <button class="tb-btn active" id="btn-fx" onclick="togglePostFx()" title="Edge outlines and ambient occlusion shading">FX</button>
  <button class="tb-btn" id="btn-bindings" onclick="toggleBindings()" title="Show interaction cylinders in 3D">Bindings</button>
  <button class="tb-btn" id="btn-measure" onclick="toggleMeasure()" title="Click two atoms to drop a distance label; toggle off to clear">Measure</button>
  <button class="tb-btn" id="btn-dist" onclick="toggleDistLabels()" title="Distance labels in 2D mode">Distances</button>
  <button class="tb-btn" id="btn-reset" onclick="resetCamera()" title="Recenter on the current selection">Reset View</button>

  <div class="tb-sep"></div>

  <div class="rep-dropdown" id="style-dropdown">
    <button class="tb-btn" id="btn-style" onclick="toggleStyleMenu()" title="Protein, ligand and interacting-residue representations">Style &#9662;</button>
    <div id="style-menu" class="view-menu" style="display:none">
      <div class="style-cat">
        <div class="view-menu-item style-cat-head">
          <span class="vm-name">Protein</span>
          <span class="style-cat-cur" id="cur-protein"></span>
          <span class="style-arrow">&#9656;</span>
        </div>
        <div id="protein-menu" class="view-submenu"></div>
      </div>
      <div class="style-cat">
        <div class="view-menu-item style-cat-head">
          <span class="vm-name">Ligand</span>
          <span class="style-cat-cur" id="cur-ligand"></span>
          <span class="style-arrow">&#9656;</span>
        </div>
        <div id="ligand-menu" class="view-submenu"></div>
      </div>
      <div class="style-cat">
        <div class="view-menu-item style-cat-head">
          <span class="vm-name">Interacting Residue</span>
          <span class="style-cat-cur" id="cur-residue"></span>
          <span class="style-arrow">&#9656;</span>
        </div>
        <div id="residue-menu" class="view-submenu"></div>
      </div>
    </div>
  </div>

  <div class="rep-dropdown" id="color-dropdown">
    <button class="tb-btn" id="btn-color" onclick="toggleColorMenu()" title="Protein colour scheme">Colour &#9662;</button>
    <div id="color-menu" class="view-menu" style="display:none"></div>
  </div>

  <div class="tb-sep"></div>

  <span class="tb-label">Views</span>
  <button class="tb-btn" id="btn-view-save" onclick="saveCurrentView()" title="Save the current camera as a new view">Save view</button>
  <div class="view-dropdown">
    <button class="tb-btn" id="btn-view-current" onclick="toggleViewMenu()" title="Pick a saved view to fly to it">No saved views &#9662;</button>
    <div id="view-menu" class="view-menu" style="display:none"></div>
  </div>
</div>

<div id="main">
  <div id="viewer"></div>
  <div id="diagram2d-wrap">
    <p class="d2-placeholder">Select a compound to view the 2D interaction map</p>
  </div>

  <div id="sidebar">
    <div class="sb-section">
      <h3 style="margin-bottom:6px">Compound</h3>
      <select id="compound-select" onchange="selectCompound(this.value)">
        <option value="">— select compound —</option>
      </select>
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
  </div>
</div>

<div id="seqpanel">
  <div id="seq-header">
    <span class="tb-label">Sequence</span>
    <div class="seg" id="seq-mode">
      <button class="seg-btn active" id="seqm-fly" onclick="setStripMode('fly')">Fly</button>
      <button class="seg-btn" id="seqm-sel" onclick="setStripMode('select')">Select</button>
    </div>
    <div class="rep-dropdown" id="seqfilter-dropdown">
      <button class="tb-btn seq-tool" id="btn-seqfilter" onclick="toggleSeqFilterMenu()">Filter &#9662;</button>
      <div id="seqfilter-menu" class="view-menu up" style="display:none"></div>
    </div>
    <div class="rep-dropdown" id="seqselect-dropdown">
      <button class="tb-btn seq-tool" id="btn-seqselect" onclick="toggleSeqSelectMenu()">Quick select &#9662;</button>
      <div id="seqselect-menu" class="view-menu up" style="display:none"></div>
    </div>
    <span id="seq-hint" class="tb-label">click a residue to fly to it</span>
    <span class="seq-actions" id="seq-actions" style="display:none">
      <span class="seq-count" id="seq-count">0 selected</span>
      <button class="seq-link" onclick="deselectStrip()" title="Clear the selection">Deselect</button>
    </span>
    <button class="tb-btn" id="btn-seq-toggle" onclick="toggleSeqPanel()" style="margin-left:auto">Hide &#9662;</button>
  </div>
  <div id="seq-chainbar"></div>
  <div id="seq-body"></div>
</div>

<script>
const COMPOUNDS = {compounds_json};
const COLORS    = {colors_json};
const SEQUENCE  = {sequence_json};
const RECEPTOR  = `{receptor_escaped}`;
const LIG_W     = {lig_svg_w};
const LIG_H     = {lig_svg_h};
const GM        = {glyph_mar};

// 3-letter HETATM residue names + numeric colours for synthetic interaction
// cylinders. The residue name is what Mol* sees inside the synthesised PDB;
// the colour is fed into the uniform colour theme when loading.
const IX_RESN = {{
  hbond:       'HBD',
  hydrophobic: 'HYP',
  pi_stack:    'PST',
  pi_cation:   'PCA',
  salt_bridge: 'SLT',
  halogen:     'HLG',
}};
const IX_COLOR_INT = {{
  hbond:       0x3b82f6,
  hydrophobic: 0xf97316,
  pi_stack:    0x22c55e,
  pi_cation:   0xa855f7,
  salt_bridge: 0xef4444,
  halogen:     0x14b8a6,
}};

// Amino-acid colour classes (used by the 2D diagram glyphs).
const AA_CLS = {{
  ALA:'hp',VAL:'hp',LEU:'hp',ILE:'hp',PRO:'hp',PHE:'hp',MET:'hp',TRP:'hp',
  SER:'pol',THR:'pol',CYS:'pol',TYR:'pol',ASN:'pol',GLN:'pol',GLY:'pol',
  LYS:'pos',ARG:'pos',HIS:'pos',
  ASP:'neg',GLU:'neg',
}};
const AA_COL = {{ hp:'#f97316', pol:'#22c55e', pos:'#3b82f6', neg:'#ef4444' }};
function aaColor(resn) {{ return AA_COL[AA_CLS[resn]] || '#8b949e'; }}

// State
let molViewer      = null;
let ligandStruct   = null;   // Hierarchy structure entry for the current ligand
let residueStruct  = null;   // Hierarchy structure entry for the floating residue layer
let ixStructs      = {{}};     // {{type: hierarchy structure}} for cylinder groups
let ixRebuildSeq   = 0;      // Cancels stale rebuilds when toggles fire quickly
let currentData    = null;
let darkMode       = true;
let currentMode    = '3d';
let siteMode       = true;
let showDistLabels = false;
let showBindings   = false;
let postFxOn       = true;
let measureMode    = false;
let measurePicks   = [];     // Loci accumulator: 2 picks → distance
let measureSub     = null;   // RxJS subscription to click events
let measureBaseline = null;  // Set<ref> snapshot of state cells before measure mode
let cameraBookmarks = [];     // [{{name, snap}}, ...] — names auto-fill as max+1
let proteinRepType  = 'cartoon';
let ligandRepType   = 'ball-and-stick';
let sticksRepType   = 'ball-and-stick';
let proteinColorType = 'chain-id';
let styleMenuOpen   = false;
let colorMenuOpen   = false;

const PROTEIN_REPS = [
  {{ id: 'cartoon',   label: 'Cartoon' }},
  {{ id: 'lines',     label: 'Lines' }},
  {{ id: 'backbone',  label: 'Backbone' }},
  {{ id: 'surface',   label: 'Surface' }},
  {{ id: 'spacefill', label: 'Spacefill' }},
  {{ id: 'off',       label: 'Off' }},
];
const LIGAND_REPS = [
  {{ id: 'ball-and-stick', label: 'Ball-and-stick' }},
  {{ id: 'sticks',         label: 'Sticks only' }},
  {{ id: 'spacefill',      label: 'Spacefill' }},
  {{ id: 'lines',          label: 'Lines' }},
  {{ id: 'off',            label: 'Off' }},
];
const STICKS_REPS = [
  {{ id: 'ball-and-stick', label: 'Ball-and-stick' }},
  {{ id: 'sticks',         label: 'Sticks only' }},
  {{ id: 'lines',          label: 'Lines' }},
  {{ id: 'spacefill',      label: 'Spacefill' }},
  {{ id: 'off',             label: 'Off' }},
];
// Swiss-PdbViewer-style Colour panel. The scheme list maps to Mol* built-in
// colour themes, except 'cpk' which drives a custom theme we register so the
// per-element palette below is editable. 'uniform' carries its own value.
const COLOR_SCHEMES = [
  {{ id: 'cpk',                 label: 'CPK' }},
  {{ id: 'secondary-structure', label: 'Secondary structure' }},
  {{ id: 'chain-id',            label: 'Chain' }},
  {{ id: 'residue-name',        label: 'Type (residue)' }},
  {{ id: 'uncertainty',         label: 'B-factor' }},
  {{ id: 'sequence-id',         label: 'Rainbow N→C' }},
  {{ id: 'uniform',            label: 'Solid' }},
];
// Editable CPK palette, SPDBV defaults. Keyed by element symbol; HAL covers the
// halogens as one swatch (SPDBV greens them together).
const ELEMENT_PALETTE = {{
  C: 0x909090, N: 0x3050f8, O: 0xff0d0d, S: 0xffff30,
  H: 0xffffff, P: 0xff8000, HAL: 0x1ff01f,
}};
const ELEMENT_DEFAULTS = Object.assign({{}}, ELEMENT_PALETTE);
const ELEMENT_SWATCHES = [
  {{ key: 'C', label: 'C' }}, {{ key: 'N', label: 'N' }}, {{ key: 'O', label: 'O' }},
  {{ key: 'S', label: 'S' }}, {{ key: 'H', label: 'H' }}, {{ key: 'P', label: 'P' }},
  {{ key: 'HAL', label: 'Halogens' }},
];
const HALOGENS = new Set(['F', 'CL', 'BR', 'I', 'AT']);

function _intToHex(n) {{ return '#' + (n >>> 0).toString(16).padStart(6, '0').slice(-6); }}
function _hexToInt(h) {{ return parseInt(h.replace('#', ''), 16); }}

let activeToggles  = Object.fromEntries(Object.keys(COLORS).map(k => [k, true]));
let viewerReady    = false;
const pendingTasks = [];

function whenReady(fn) {{
  if (viewerReady) fn();
  else pendingTasks.push(fn);
}}

// ── Mol* init ────────────────────────────────────────────────────────────
(async () => {{
  try {{
    molViewer = await molstar.Viewer.create('viewer', {{
      layoutIsExpanded: false,
      layoutShowControls: false,
      layoutShowRemoteState: false,
      layoutShowSequence: false,
      layoutShowLog: false,
      layoutShowLeftPanel: false,
      viewportShowExpand: false,
      viewportShowSelectionMode: false,
      viewportShowAnimation: false,
      viewportShowControls: false,
      viewportShowSettings: false,
      pdbProvider: 'rcsb',
      emdbProvider: 'rcsb',
    }});
    registerCpkTheme();
    await molViewer.loadStructureFromData(RECEPTOR, 'pdb', false);
    applyBackground();
    applyPostFx();
    // Per-atom hover: 'element' = atom in Mol*'s vocab. Without this Mol*
    // falls back to residue-level labels which hide useful detail like
    // which atom of the side chain the cursor is on.
    try {{
      molViewer.plugin.managers.interactivity.setProps({{ granularity: 'element' }});
    }} catch (err) {{
      console.warn('granularity setProps failed:', err);
    }}
    viewerReady = true;
    renderSequencePanel();
    pendingTasks.splice(0).forEach(fn => fn());
  }} catch (err) {{
    console.error('Mol* init failed:', err);
    document.getElementById('viewer').innerHTML =
      `<div style="padding:24px;color:#f85149">Mol* failed to load: ${{err.message}}</div>`;
  }}
}})();

// ── Compound dropdown ────────────────────────────────────────────────────
const sel = document.getElementById('compound-select');
COMPOUNDS.forEach(c => {{
  const o = document.createElement('option');
  o.value = c.lig_id;
  o.textContent = `#${{c.rank}}  ${{c.name || c.lig_id}}  (${{c.score}} kcal/mol)`;
  sel.appendChild(o);
}});

// ── Interaction-type toggles (filter both 3D list and 2D diagram) ────────
const togDiv = document.getElementById('toggles-section');
Object.entries(COLORS).forEach(([type, color]) => {{
  const row = document.createElement('label');
  row.className = 'toggle-row';
  row.innerHTML = `<input type="checkbox" checked onchange="toggleType('${{type}}', this.checked)">
    <span class="color-dot" style="background:${{color}}"></span>
    <span>${{type.replace(/_/g, ' ')}}</span>`;
  togDiv.appendChild(row);
}});

function toggleType(type, checked) {{
  activeToggles[type] = checked;
  if (!currentData) return;
  renderSidebar(currentData);
  applySequenceHighlight(currentData);
  if (currentMode === '2d') draw2DView(currentData, siteMode);
  rebuildInteractions();
}}

// ── Compound load / unload via Mol* hierarchy manager ────────────────────
async function removeCurrentLigand() {{
  if (ligandStruct === null || !molViewer) return;
  try {{
    await molViewer.plugin.managers.structure.hierarchy.remove([ligandStruct]);
  }} catch (err) {{
    console.warn('ligand removal failed:', err);
  }}
  ligandStruct = null;
}}

// Side-chain ball-and-stick rides as a small companion structure containing
// the *full* atoms (backbone included) of each interacting residue. Cα
// coordinates coincide with the receptor's Cα, so the ball-and-stick layer
// visually branches out of the cartoon ribbon instead of floating in space.
async function removeResidueHighlight() {{
  if (residueStruct === null || !molViewer) return;
  try {{
    await molViewer.plugin.managers.structure.hierarchy.remove([residueStruct]);
  }} catch (err) {{
    console.warn('residue highlight removal failed:', err);
  }}
  residueStruct = null;
}}

function _sticksRepParams(type) {{
  switch (type) {{
    case 'ball-and-stick':
      return {{ type: 'ball-and-stick', color: 'element-symbol',
               typeParams: {{ sizeFactor: 0.22, sizeAspectRatio: 0.55, aromaticBonds: false }} }};
    case 'sticks':
      return {{ type: 'ball-and-stick', color: 'element-symbol',
               typeParams: {{ sizeFactor: 0.22, sizeAspectRatio: 0.01, aromaticBonds: false }} }};
    case 'lines':
      return {{ type: 'line', color: 'element-symbol',
               typeParams: {{ sizeFactor: 2.0 }} }};
    case 'spacefill':
      return {{ type: 'spacefill', color: 'element-symbol',
               typeParams: {{ sizeFactor: 0.5 }} }};
    default:
      return null;
  }}
}}

async function applyResidueHighlight(compound) {{
  if (!compound || !compound.residue_pdb || !molViewer) return;
  if (proteinRepType === 'off' || sticksRepType === 'off') return;
  const plugin = molViewer.plugin;
  try {{
    const data  = await plugin.builders.data.rawData({{ data: compound.residue_pdb, label: 'Interacting residues' }});
    const traj  = await plugin.builders.structure.parseTrajectory(data, 'pdb');
    const model = await plugin.builders.structure.createModel(traj);
    const struc = await plugin.builders.structure.createStructure(model);
    const comp  = await plugin.builders.structure.tryCreateComponentStatic(struc, 'all');
    const params = _sticksRepParams(sticksRepType);
    if (comp && params) {{
      await plugin.builders.structure.representation.addRepresentation(comp, params);
    }}
    const all = plugin.managers.structure.hierarchy.current.structures;
    if (all.length) residueStruct = all[all.length - 1];
  }} catch (err) {{
    console.warn('residue highlight load failed:', err);
  }}
}}

// ── Interaction cylinders via synthetic PDB structures ───────────────────
function _pad(v, n)        {{ return String(v).padStart(n); }}
function _fmtF(v, w, p)    {{ return v.toFixed(p).padStart(w); }}
function _isZeroVec(v)     {{
  return !v || (Math.abs(v[0]) + Math.abs(v[1]) + Math.abs(v[2]) < 1e-6);
}}

// V2000 atom record (69 columns). All atoms are carbon — only the bond
// cylinder is meaningful here.
function _molAtom(x, y, z) {{
  return _fmtF(x, 10, 4) + _fmtF(y, 10, 4) + _fmtF(z, 10, 4) +
         ' C   0  0  0  0  0  0  0  0  0  0  0  0';
}}

// V2000 mol block. Mol*'s PDB parser ignores CONECT for atoms further apart
// than the covalent-radius cutoff (3-4 A interactions get dropped), but
// SDF/mol has an explicit bond block that the parser always honours.
function buildIxMol(type, ixs) {{
  const pairs = [];
  for (const ix of ixs) {{
    const lc = ix.ligand_coords, pc = ix.protein_coords;
    if (_isZeroVec(lc) || _isZeroVec(pc)) continue;
    pairs.push([lc, pc]);
  }}
  if (!pairs.length) return null;

  const nAtoms = pairs.length * 2;
  const nBonds = pairs.length;
  const lines = [
    'ix-' + type,
    '  ezscreen          3D',
    '',
    _pad(nAtoms, 3) + _pad(nBonds, 3) + '  0  0  0  0  0  0  0  0999 V2000',
  ];
  for (const [lc, pc] of pairs) {{
    lines.push(_molAtom(lc[0], lc[1], lc[2]));
    lines.push(_molAtom(pc[0], pc[1], pc[2]));
  }}
  let serial = 1;
  for (let i = 0; i < pairs.length; i++) {{
    lines.push(_pad(serial, 3) + _pad(serial + 1, 3) + '  1  0  0  0  0');
    serial += 2;
  }}
  lines.push('M  END');
  return lines.join('\\n');
}}

async function clearInteractions() {{
  const toRemove = Object.values(ixStructs).filter(s => s);
  ixStructs = {{}};
  if (!toRemove.length || !molViewer) return;
  try {{
    await molViewer.plugin.managers.structure.hierarchy.remove(toRemove);
  }} catch (err) {{
    console.warn('interaction cleanup failed:', err);
  }}
}}

async function rebuildInteractions() {{
  const seq = ++ixRebuildSeq;
  await clearInteractions();
  if (seq !== ixRebuildSeq || !molViewer || !currentData) return;
  if (!showBindings) return;

  const groups = {{}};
  for (const ix of currentData.interactions || []) {{
    if (!activeToggles[ix.type]) continue;
    (groups[ix.type] = groups[ix.type] || []).push(ix);
  }}

  const plugin = molViewer.plugin;
  for (const [type, ixs] of Object.entries(groups)) {{
    if (seq !== ixRebuildSeq) return;
    const mol = buildIxMol(type, ixs);
    if (!mol) continue;

    // Build the structure ourselves so we can pin a ball-and-stick representation
    // with a uniform colour — Mol*'s default preset would render the synthetic
    // HETATM block as a space-filling blob with chain-auto coloring instead.
    try {{
      const data  = await plugin.builders.data.rawData({{ data: mol, label: 'ix-' + type }});
      const traj  = await plugin.builders.structure.parseTrajectory(data, 'mol');
      const model = await plugin.builders.structure.createModel(traj);
      const struc = await plugin.builders.structure.createStructure(model);
      const comp  = await plugin.builders.structure.tryCreateComponentStatic(struc, 'all');
      if (comp) {{
        await plugin.builders.structure.representation.addRepresentation(comp, {{
          type: 'ball-and-stick',
          typeParams: {{ sizeFactor: 0.05, sizeAspectRatio: 1.0, aromaticBonds: false }},
          color: 'uniform',
          colorParams: {{ value: IX_COLOR_INT[type] }},
        }});
      }}
    }} catch (err) {{
      console.warn('interaction load failed for', type, err);
      continue;
    }}

    const all = plugin.managers.structure.hierarchy.current.structures;
    if (all.length) ixStructs[type] = all[all.length - 1];
  }}
}}

async function selectCompound(ligId) {{
  whenReady(async () => {{
    await clearInteractions();
    await removeResidueHighlight();
    await removeCurrentLigand();

    if (!ligId) {{
      currentData = null;
      renderSidebar(null);
      applySequenceHighlight(null);
      document.getElementById('btn-site').disabled = true;
      return;
    }}

    const compound = COMPOUNDS.find(c => c.lig_id === ligId);
    currentData = compound;
    document.getElementById('btn-site').disabled = !compound.site_viewbox;

    if (compound && compound.sdf_b64) {{
      const sdf  = atob(compound.sdf_b64);
      const before = molViewer.plugin.managers.structure.hierarchy.current.structures.length;
      await molViewer.loadStructureFromData(sdf, 'sdf', false);
      const all = molViewer.plugin.managers.structure.hierarchy.current.structures;
      if (all.length > before) ligandStruct = all[all.length - 1];
      // Mol* auto-styles a fresh ligand as ball-and-stick; only re-apply when
      // the user has picked something else, so the default view is untouched.
      if (ligandRepType !== 'ball-and-stick') await applyLigandRep();
    }}

    if (showBindings && compound) {{
      await applyResidueHighlight(compound);
    }}

    renderSidebar(compound);
    applySequenceHighlight(compound);
    await rebuildInteractions();
    // Frame everything that's now in the scene
    molViewer.plugin.managers.camera.reset();
    if (currentMode === '2d') draw2DView(compound, siteMode);
  }});
}}

// ── Sequence panel ──────────────────────────────────────────────────────
let seqPanelOpen = true;

function toggleSeqPanel() {{
  seqPanelOpen = !seqPanelOpen;
  const panel = document.getElementById('seqpanel');
  panel.classList.toggle('collapsed', !seqPanelOpen);
  document.getElementById('btn-seq-toggle').innerHTML = seqPanelOpen ? 'Hide &#9662;' : 'Show &#9652;';
}}

function _tileId(chain, resi) {{ return 'seqt-' + chain + '-' + resi; }}

function flyToCoords(x, y, z) {{
  if (!molViewer || !molViewer.plugin.canvas3d) return;
  if (currentMode !== '3d') switchMode('3d');
  const cam = molViewer.plugin.canvas3d.camera;
  const snap = cam.getSnapshot();
  const dx = x - snap.target[0];
  const dy = y - snap.target[1];
  const dz = z - snap.target[2];
  snap.target[0] = x; snap.target[1] = y; snap.target[2] = z;
  snap.position[0] += dx; snap.position[1] += dy; snap.position[2] += dz;
  cam.setState(snap, 350);
}}

function resetCamera() {{
  whenReady(() => {{
    const plg = molViewer.plugin;
    if (plg.canvas3d && plg.canvas3d.requestCameraReset) {{
      plg.canvas3d.requestCameraReset({{ durationMs: 350 }});
    }} else {{
      plg.managers.camera.reset();
    }}
  }});
}}

let viewMenuOpen = false;

function _viewLabel() {{
  const n = cameraBookmarks.length;
  if (n === 0) return 'No saved views ▾';
  return n + ' saved view' + (n === 1 ? '' : 's') + ' ▾';
}}

function updateViewLabel() {{
  document.getElementById('btn-view-current').innerHTML = _viewLabel();
}}

function renderViewMenu() {{
  const menu = document.getElementById('view-menu');
  menu.innerHTML = '';
  if (cameraBookmarks.length === 0) {{
    const empty = document.createElement('div');
    empty.className = 'view-menu-empty';
    empty.textContent = 'No saved views yet';
    menu.appendChild(empty);
    return;
  }}
  cameraBookmarks.forEach((b, i) => {{
    const row = document.createElement('div');
    row.className = 'view-menu-item';

    const name = document.createElement('span');
    name.className = 'vm-name';
    name.textContent = b.name;
    name.onclick = () => restoreView(i);

    const del = document.createElement('span');
    del.className = 'vm-del';
    del.textContent = '×';
    del.title = 'Delete ' + b.name;
    del.onclick = (e) => {{ e.stopPropagation(); deleteView(i); }};

    row.appendChild(name);
    row.appendChild(del);
    menu.appendChild(row);
  }});
}}

function setViewMenuOpen(on) {{
  viewMenuOpen = on;
  const menu = document.getElementById('view-menu');
  menu.style.display = on ? '' : 'none';
  if (on) renderViewMenu();
}}

function toggleViewMenu() {{ setViewMenuOpen(!viewMenuOpen); }}

document.addEventListener('click', (e) => {{
  if (viewMenuOpen) {{
    const dd = document.querySelector('.view-dropdown');
    if (dd && !dd.contains(e.target)) setViewMenuOpen(false);
  }}
  if (styleMenuOpen) {{
    const dd = document.getElementById('style-dropdown');
    if (dd && !dd.contains(e.target)) setStyleMenuOpen(false);
  }}
  if (colorMenuOpen) {{
    const dd = document.getElementById('color-dropdown');
    if (dd && !dd.contains(e.target)) setColorMenuOpen(false);
  }}
}});

// ── Representation dropdowns (Protein / Sticks) ─────────────────────────
function _proteinRepParams(type) {{
  const col = _proteinColorSpec();
  switch (type) {{
    case 'cartoon':   return {{ type: 'cartoon', ...col }};
    case 'lines':     return {{ type: 'line', ...col,
                                typeParams: {{ sizeFactor: 1.5 }} }};
    case 'backbone':  return {{ type: 'backbone', ...col }};
    case 'surface':   return {{ type: 'molecular-surface', ...col,
                                typeParams: {{ alpha: 0.75, smoothness: 1.5 }} }};
    case 'spacefill': return {{ type: 'spacefill', ...col }};
    default:          return null;
  }}
}}

function _findPolymerComp() {{
  if (!molViewer) return null;
  const structures = molViewer.plugin.managers.structure.hierarchy.current.structures;
  if (!structures.length) return null;
  const comps = structures[0].components || [];
  if (!comps.length) return null;
  const polymer = comps.find(c => {{
    if (c.key && /polymer/i.test(c.key)) return true;
    const lbl = c.cell && c.cell.obj && c.cell.obj.label;
    return lbl && /polymer/i.test(lbl);
  }});
  return polymer || comps[0];
}}

async function applyProteinRep(refit = true) {{
  if (!molViewer) return;
  const polymer = _findPolymerComp();
  if (!polymer) return;
  const plugin = molViewer.plugin;
  if (polymer.representations && polymer.representations.length) {{
    const update = plugin.build();
    polymer.representations.forEach(rep => update.delete(rep.cell.transform.ref));
    await update.commit();
  }}
  const params = _proteinRepParams(proteinRepType);
  if (params) {{
    await plugin.builders.structure.representation.addRepresentation(polymer.cell, params);
  }}
  // Surface / spacefill / lines occupy a larger volume than the cartoon trace
  // the camera was first fitted to. Mol* only recomputes the zoom-out envelope
  // (radiusMax) on a camera reset, so without this the view stays clamped to
  // the old cartoon bounds — looking zoomed-in and refusing to pull back. A
  // colour-only change keeps the same geometry, so it passes refit=false.
  if (refit && plugin.canvas3d && plugin.canvas3d.requestCameraReset) {{
    plugin.canvas3d.requestCameraReset({{ durationMs: 350 }});
  }}
}}

// hierarchy.current is rebuilt on every state change, so the StructureRef we
// captured at load goes stale — its .components still report the original
// representations. Re-resolve the live entry by ref before reading them, or we
// delete nothing and just stack a new rep on top of the old one.
function _liveLigandStruct() {{
  if (!molViewer || !ligandStruct) return null;
  const ref = ligandStruct.cell.transform.ref;
  const structures = molViewer.plugin.managers.structure.hierarchy.current.structures;
  return structures.find(s => s.cell.transform.ref === ref) || ligandStruct;
}}

async function applyLigandRep() {{
  const live = _liveLigandStruct();
  if (!live) return;
  const plugin = molViewer.plugin;
  const comps = live.components || [];
  if (!comps.length) return;
  const params = _sticksRepParams(ligandRepType);
  for (const comp of comps) {{
    if (comp.representations && comp.representations.length) {{
      const update = plugin.build();
      comp.representations.forEach(rep => update.delete(rep.cell.transform.ref));
      await update.commit();
    }}
    if (params) {{
      await plugin.builders.structure.representation.addRepresentation(comp.cell, params);
    }}
  }}
}}

function _labelOf(list, id) {{
  const m = list.find(r => r.id === id);
  return m ? m.label : '?';
}}

// Each style category resolves to its option list, current selection and the
// submenu container — keyed by the category name used throughout the menu.
const REP_CATS = {{
  protein: {{ list: PROTEIN_REPS, menu: 'protein-menu', cur: 'cur-protein',
             get: () => proteinRepType, set: id => setProteinRep(id) }},
  ligand:  {{ list: LIGAND_REPS,  menu: 'ligand-menu',  cur: 'cur-ligand',
             get: () => ligandRepType,  set: id => setLigandRep(id) }},
  residue: {{ list: STICKS_REPS,  menu: 'residue-menu', cur: 'cur-residue',
             get: () => sticksRepType,  set: id => setSticksRep(id) }},
}};

function refreshRepButtons() {{
  Object.values(REP_CATS).forEach(cat => {{
    const el = document.getElementById(cat.cur);
    if (el) el.textContent = _labelOf(cat.list, cat.get());
  }});
}}

function renderRepMenu(which) {{
  const cat  = REP_CATS[which];
  if (!cat) return;
  const cur  = cat.get();
  const menu = document.getElementById(cat.menu);
  menu.innerHTML = '';
  cat.list.forEach(rep => {{
    const row = document.createElement('div');
    row.className = 'view-menu-item rep-menu-item' + (rep.id === cur ? ' active' : '');
    const name = document.createElement('span');
    name.className = 'vm-name';
    name.textContent = rep.label + (rep.id === cur ? ' ✓' : '');
    name.onclick = () => cat.set(rep.id);
    row.appendChild(name);
    menu.appendChild(row);
  }});
}}

function setStyleMenuOpen(on) {{
  styleMenuOpen = on;
  document.getElementById('style-menu').style.display = on ? '' : 'none';
  if (on) {{ Object.keys(REP_CATS).forEach(renderRepMenu); refreshRepButtons(); }}
}}
function toggleStyleMenu() {{ setStyleMenuOpen(!styleMenuOpen); }}

async function setProteinRep(id) {{
  proteinRepType = id;
  refreshRepButtons();
  if (styleMenuOpen) renderRepMenu('protein');
  await applyProteinRep();
  applyPostFx();
  if (showBindings && currentData) {{
    await removeResidueHighlight();
    if (proteinRepType !== 'off') await applyResidueHighlight(currentData);
  }}
}}

async function setLigandRep(id) {{
  ligandRepType = id;
  refreshRepButtons();
  if (styleMenuOpen) renderRepMenu('ligand');
  await applyLigandRep();
}}

async function setSticksRep(id) {{
  sticksRepType = id;
  refreshRepButtons();
  if (styleMenuOpen) renderRepMenu('residue');
  if (showBindings && currentData) {{
    await removeResidueHighlight();
    await applyResidueHighlight(currentData);
  }}
}}

function setColorMenuOpen(on) {{
  colorMenuOpen = on;
  document.getElementById('color-menu').style.display = on ? '' : 'none';
  if (on) renderColorMenu();
}}
function toggleColorMenu() {{ setColorMenuOpen(!colorMenuOpen); }}

function saveCurrentView() {{
  if (!molViewer || !molViewer.plugin.canvas3d) return;
  const maxN = cameraBookmarks.reduce((m, b) => {{
    const n = parseInt(String(b.name).replace(/^View\\s+/, ''), 10);
    return isNaN(n) ? m : Math.max(m, n);
  }}, 0);
  const name = 'View ' + (maxN + 1);
  const snap = molViewer.plugin.canvas3d.camera.getSnapshot();
  cameraBookmarks.push({{ name, snap }});
  updateViewLabel();
  if (viewMenuOpen) renderViewMenu();
}}

function restoreView(idx) {{
  if (!molViewer || !molViewer.plugin.canvas3d) return;
  molViewer.plugin.canvas3d.camera.setState(cameraBookmarks[idx].snap, 350);
  setViewMenuOpen(false);
}}

function deleteView(idx) {{
  cameraBookmarks.splice(idx, 1);
  updateViewLabel();
  renderViewMenu();
}}

// ── Sidebar ──────────────────────────────────────────────────────────────
function renderSidebar(compound) {{
  const list = document.getElementById('ilist');
  list.innerHTML = '';
  if (!compound || compound.plip_failed) {{
    list.innerHTML = `<div id="no-compound" style="color:#f85149">${{
      compound ? (compound.plip_error || 'PLIP analysis failed') : 'No data'}}</div>`;
    return;
  }}
  const active = (compound.interactions || []).filter(ix => activeToggles[ix.type]);
  if (!active.length) {{
    list.innerHTML = `<div id="no-compound">No visible interactions</div>`;
    return;
  }}
  active.forEach(ix => {{
    const div = document.createElement('div');
    div.className = 'i-row';
    const color = COLORS[ix.type] || '#fff';
    div.innerHTML = `<div class="i-type" style="color:${{color}}">${{ix.type.replace(/_/g, ' ')}}</div>
      <div class="i-detail">${{ix.residue_name}}&nbsp;${{ix.residue_number}} (${{ix.chain}})
        &nbsp;&middot;&nbsp; ${{ix.distance.toFixed(2)}}&nbsp;&Aring;</div>`;
    list.appendChild(div);
  }});
}}

// ── 3D / 2D mode switching ───────────────────────────────────────────────
function switchMode(mode) {{
  currentMode = mode;
  const is3d  = mode === '3d';
  document.getElementById('viewer').style.display          = is3d ? '' : 'none';
  document.getElementById('diagram2d-wrap').style.display  = is3d ? 'none' : 'flex';
  document.getElementById('site-toggle').style.display     = is3d ? 'none' : 'flex';
  document.getElementById('btn-3d').classList.toggle('active',  is3d);
  document.getElementById('btn-2d').classList.toggle('active', !is3d);
  document.getElementById('btn-reset').disabled = !is3d;
  document.getElementById('btn-view-save').disabled = !is3d;
  document.getElementById('btn-view-current').disabled = !is3d;
  if (!is3d) setViewMenuOpen(false);
  if (!is3d && currentData) draw2DView(currentData, siteMode);
}}

function setSiteMode(site) {{
  siteMode = site;
  document.getElementById('btn-site').classList.toggle('active',  site);
  document.getElementById('btn-full').classList.toggle('active', !site);
  if (currentData && currentMode === '2d') draw2DView(currentData, siteMode);
}}

function applyBackground() {{
  if (!molViewer || !molViewer.plugin.canvas3d) return;
  // Mol*'s Color is just a tagged number — passing the hex int directly works.
  const c = darkMode ? 0x0d1117 : 0xf6f8fa;
  molViewer.plugin.canvas3d.setProps({{
    renderer: {{ backgroundColor: c }},
  }});
}}

function applyPostFx() {{
  if (!molViewer || !molViewer.plugin.canvas3d) return;
  const outlineColor = darkMode ? 0x000000 : 0x202020;
  // FX (outlines + AO) only reads well on solid reps. Lines/backbone/off
  // get no benefit (often look noisier), so we no-op them regardless of
  // the user's FX toggle — toggle state is preserved for when they switch
  // back to a solid rep.
  const isSolidRep = proteinRepType !== 'lines'
                  && proteinRepType !== 'backbone'
                  && proteinRepType !== 'off';
  const fxActive = postFxOn && isSolidRep;
  molViewer.plugin.canvas3d.setProps({{
    postprocessing: {{
      occlusion: fxActive
        ? {{ name: 'on', params: {{
            samples: 32,
            multiScale: {{ name: 'off', params: {{}} }},
            radius: 5,
            bias: 0.8,
            blurKernelSize: 15,
            resolutionScale: 1,
            color: 0x000000,
            transparentThreshold: 0.4,
          }} }}
        : {{ name: 'off', params: {{}} }},
      outline: fxActive
        ? {{ name: 'on', params: {{
            scale: 1,
            threshold: 0.33,
            color: outlineColor,
            includeTransparent: true,
          }} }}
        : {{ name: 'off', params: {{}} }},
    }},
  }});
}}

function togglePostFx() {{
  postFxOn = !postFxOn;
  document.getElementById('btn-fx').classList.toggle('active', postFxOn);
  applyPostFx();
}}

function toggleBackground() {{
  darkMode = !darkMode;
  document.body.classList.toggle('light', !darkMode);
  const btn = document.getElementById('btn-bg');
  btn.textContent = darkMode ? 'Light BG' : 'Dark BG';
  btn.classList.toggle('active', !darkMode);
  applyBackground();
  applyPostFx();
  if (currentData && currentMode === '2d') draw2DView(currentData, siteMode);
}}

function toggleDistLabels() {{
  showDistLabels = !showDistLabels;
  document.getElementById('btn-dist').classList.toggle('active', showDistLabels);
  if (currentData && currentMode === '2d') draw2DView(currentData, siteMode);
}}

async function toggleBindings() {{
  showBindings = !showBindings;
  document.getElementById('btn-bindings').classList.toggle('active', showBindings);
  if (showBindings) {{
    if (currentData) {{
      await applyResidueHighlight(currentData);
    }}
  }} else {{
    await removeResidueHighlight();
  }}
  await rebuildInteractions();
}}

function _stateCellRefs() {{
  const refs = new Set();
  if (!molViewer) return refs;
  const cells = molViewer.plugin.state.data.cells;
  if (!cells) return refs;
  cells.forEach((_, ref) => refs.add(ref));
  return refs;
}}

async function clearAllMeasurements() {{
  if (!molViewer || !measureBaseline) return;
  // Diff approach: anything in the state tree that didn't exist when measure
  // mode was switched on must have been added by a measurement. Mol*'s
  // measurement manager state shape moves around between versions, so we
  // avoid coupling to it and just delete the new refs.
  const current = _stateCellRefs();
  const toDelete = [];
  current.forEach(ref => {{ if (!measureBaseline.has(ref)) toDelete.push(ref); }});
  console.debug('measure clear:', toDelete.length, 'new cells to remove');
  if (!toDelete.length) return;
  try {{
    const builder = molViewer.plugin.build();
    for (const r of toDelete) builder.delete(r);
    await builder.commit();
  }} catch (err) {{
    console.warn('measure clear failed:', err);
  }}
}}

async function toggleMeasure() {{
  measureMode = !measureMode;
  document.getElementById('btn-measure').classList.toggle('active', measureMode);

  if (measureMode) {{
    if (!molViewer) return;
    if (measureSub) return;
    // Snapshot the state tree so we can diff on toggle-off and remove
    // anything the measurement system added.
    measureBaseline = _stateCellRefs();
    // Subscribe to clicks while measure mode is active. Each click with a
    // real loci becomes a pick; two picks add a distance measurement.
    measureSub = molViewer.plugin.behaviors.interaction.click.subscribe(({{ current }}) => {{
      if (!measureMode) return;
      if (!current || !current.loci) return;
      const kind = current.loci.kind;
      if (kind !== 'element-loci') {{
        console.debug('measure: ignored click with loci kind', kind);
        return;
      }}
      measurePicks.push(current.loci);
      console.debug('measure: pick', measurePicks.length, '/ 2');
      if (measurePicks.length >= 2) {{
        const a = measurePicks[0], b = measurePicks[1];
        measurePicks = [];
        // Override Mol*'s default label styling so the number is actually
        // readable against the dark canvas: larger white text on a
        // semi-opaque dark plate.
        molViewer.plugin.managers.structure.measurement
          .addDistance(a, b, {{
            labelParams: {{
              textColor: 0xffffff,
              textSize: 0.55,
              background: true,
              backgroundColor: 0x0d1117,
              backgroundOpacity: 0.9,
              borderColor: 0xffd93d,
              borderWidth: 0.12,
            }},
            lineParams: {{
              linesColor: 0xffd93d,
              linesSize: 0.08,
            }},
          }})
          .then(() => console.debug('measure: added distance'))
          .catch(err => console.warn('measure failed:', err));
      }}
    }});
    return;
  }}

  // Leaving measure mode: stop listening + drop the user's distance labels.
  if (measureSub) {{ measureSub.unsubscribe(); measureSub = null; }}
  measurePicks = [];
  await clearAllMeasurements();
  measureBaseline = null;
}}

// ── 2D diagram (preserved from prior viewer) ─────────────────────────────
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
  const wrap = document.getElementById('diagram2d-wrap');

  if (!compound || compound.plip_failed) {{
    wrap.innerHTML = `<p class="d2-placeholder">${{
      compound ? (compound.plip_error || 'PLIP failed') : 'Select a compound above'}}</p>`;
    return;
  }}

  const visible = (compound.interactions || []).filter(ix => activeToggles[ix.type]);
  if (!visible.length) {{
    wrap.innerHTML = `<p class="d2-placeholder">No visible interactions — check toggles</p>`;
    return;
  }}

  const ligInner = useSite
    ? (compound.lig_svg_inner_site || compound.lig_svg_inner_full)
    : compound.lig_svg_inner_full;
  const hasSvg = !!ligInner;

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

  const resMap = new Map();
  (compound.interactions || []).forEach((ix, origI) => {{
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

  const residues = [...resMap.values()].map((e, i, arr) => {{
    let ax, ay;
    if (e.atomPts.length) {{
      ax = e.atomPts.reduce((s,p)=>s+p[0],0)/e.atomPts.length;
      ay = e.atomPts.reduce((s,p)=>s+p[1],0)/e.atomPts.length;
    }} else {{
      ax = LIGCX; ay = LIGCY;
    }}
    let dx = ax - LIGCX, dy = ay - LIGCY;
    if (dx === 0 && dy === 0) {{
      const angle = (i/arr.length)*2*Math.PI - Math.PI/2;
      dx = Math.cos(angle); dy = Math.sin(angle);
    }}
    const r_atom = Math.sqrt(dx*dx + dy*dy);
    return {{...e, ax, ay, r_atom, angle: Math.atan2(dy, dx)}};
  }});

  const MIN_SEP = 0.40;
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

  residues.forEach(r => {{
    let r_glyph = Math.max(r.r_atom + PUSH, 180);
    r.gx = LIGCX + Math.cos(r.angle) * r_glyph;
    r.gy = LIGCY + Math.sin(r.angle) * r_glyph;
  }});

  for (let pass=0; pass<10; pass++) {{
    for (let a=0; a<residues.length; a++) {{
      for (let b=a+1; b<residues.length; b++) {{
        const dx=residues[b].gx-residues[a].gx, dy=residues[b].gy-residues[a].gy;
        const d=Math.sqrt(dx*dx + (dy*1.3)*(dy*1.3));
        const minD = NODER*2 + 55;
        if (d < minD && d > 0.01) {{
          const push=(minD-d)/2, nx=dx/d, ny=dy/d;
          residues[a].gx -= nx*push; residues[a].gy -= ny*push;
          residues[b].gx += nx*push; residues[b].gy += ny*push;
        }}
      }}
    }}
  }}

  residues.forEach(r => {{
    r.gx = Math.max(NODER+30, Math.min(OW-NODER-30, r.gx));
    r.gy = Math.max(NODER+30, Math.min(OH-52-NODER-30, r.gy));
  }});

  let s = `<svg id="diagram2d-svg" xmlns="http://www.w3.org/2000/svg"
    width="100%" height="100%" viewBox="0 0 ${{OW}} ${{OH}}"
    preserveAspectRatio="xMidYMid meet">`;

  s += `<defs>`;
  ["hbond","salt_bridge"].forEach(type => {{
    s += `<marker id="d2a-${{type}}" markerWidth="9" markerHeight="9"
            refX="8" refY="4.5" orient="auto">
          <path d="M0,0 L0,9 L9,4.5 Z" fill="${{COLORS[type]}}" opacity=".95"/></marker>`;
  }});
  s += `</defs>`;

  residues.forEach(res => {{
    const col     = COLORS[res.type] || "#888";
    const isHbond = res.type === "hbond" || res.type === "salt_bridge";

    let px = LIGCX, py = LIGCY;
    if (res.atomPts.length > 0) {{
      let minDist = Infinity;
      res.atomPts.forEach(p => {{
        const dSq = (p[0]-res.gx)**2 + (p[1]-res.gy)**2;
        if (dSq < minDist) {{ minDist = dSq; px = p[0]; py = p[1]; }}
      }});
    }} else {{
      px = LIGCX + Math.cos(res.angle) * (LIG_W * 0.42);
      py = LIGCY + Math.sin(res.angle) * (LIG_H * 0.42);
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

  residues.forEach(res => {{
    const {{gx, gy, ix, type}} = res;
    const col = COLORS[type] || "#888";
    const aac = aaColor(ix.residue_name);

    let srcX = LIGCX, srcY = LIGCY;
    if (res.atomPts.length > 0) {{
      let minDist = Infinity;
      res.atomPts.forEach(p => {{
        const dSq = (p[0]-gx)**2 + (p[1]-gy)**2;
        if (dSq < minDist) {{ minDist = dSq; srcX = p[0]; srcY = p[1]; }}
      }});
    }} else {{
      srcX = LIGCX + Math.cos(res.angle) * (LIG_W * 0.42);
      srcY = LIGCY + Math.sin(res.angle) * (LIG_H * 0.42);
    }}

    if (type === "hydrophobic") {{
      s += eyelashGlyph(gx, gy, srcX, srcY, NODER, 9, col);
    }} else {{
      let fillCol = aac + "22";
      let dash = "";
      if (type === "pi_stack" || type === "pi_cation") {{
        fillCol = col + "15"; dash = ' stroke-dasharray="5 2.5"';
      }} else if (type === "halogen") {{
        fillCol = aac + "15";
      }}
      s += `<circle cx="${{gx.toFixed(1)}}" cy="${{gy.toFixed(1)}}" r="${{NODER}}"
              fill="${{fillCol}}" stroke="${{col}}" stroke-width="2.2"${{dash}}/>`;
    }}

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
// ── Colour layers + interactive residue strip ───────────────────────────
// Two colour layers compose in one custom Mol* theme (ezs-protein): a global
// default atom/scheme colour, with optional per-residue overrides painted on
// top. The strip drives fly-to / multi-select, with filter + quick-select.
var RESIDUE_OVERRIDES = {{}};
var _hasOverrides = false;
var proteinThemeRegistered = false;
var stripMode = 'fly';
var stripSel = new Set();
var stripLastKey = null;
var seqFilter = 'all';
var seqFilterMenuOpen = false;
var seqSelectMenuOpen = false;
// With several chains the strip stacks one full row each, so default to a
// single-chain view past a couple of chains; the chip bar switches / shows all.
var seqChain = (SEQUENCE && SEQUENCE.length > 2) ? SEQUENCE[0].chain : 'all';
var SOLID_VALUE = 0x9aa4af;

// chain -> {{ items, idx:{{resi->index}} }} for range select and residue pickers
var CHAIN_MAP = (function () {{
  var m = {{}};
  (SEQUENCE || []).forEach(function (ch) {{
    var idx = {{}};
    ch.items.forEach(function (it, i) {{ idx[it.resi] = i; }});
    m[ch.chain] = {{ items: ch.items, idx: idx }};
  }});
  return m;
}})();

function _seqKey(chain, resi) {{ return chain + '|' + resi; }}

// The viewer build of Mol* does not expose molstar.StructureProperties, so we
// read atom/residue/chain identity straight off the model's atomic hierarchy.
// location.element is a model-level ElementIndex; the segment index arrays map
// it to the owning residue / chain.
function _atomSymbol(location) {{
  try {{
    var m = location && location.unit && location.unit.model;
    if (!m || !m.atomicHierarchy) return '';
    return String(m.atomicHierarchy.atoms.type_symbol.value(location.element) || '');
  }} catch (e) {{ return ''; }}
}}

function _atomChainResi(location) {{
  try {{
    var m = location && location.unit && location.unit.model;
    if (!m || !m.atomicHierarchy) return null;
    var AH = m.atomicHierarchy;
    var el = location.element;
    var rI = AH.residueAtomSegments.index[el];
    var cI = AH.chainAtomSegments.index[el];
    return {{ chain: String(AH.chains.auth_asym_id.value(cI)), resi: AH.residues.auth_seq_id.value(rI) }};
  }} catch (e) {{ return null; }}
}}

// CPK element colour, hierarchy-based.
function elementColor(location) {{
  var sym = _atomSymbol(location).toUpperCase();
  if (HALOGENS.has(sym)) return ELEMENT_PALETTE.HAL;
  return ELEMENT_PALETTE[sym] != null ? ELEMENT_PALETTE[sym] : ELEMENT_PALETTE.C;
}}

// Per-residue override lookup, composed on top of the base scheme below.
function residueOverrideAt(location) {{
  if (!_hasOverrides) return null;
  var cr = _atomChainResi(location);
  if (!cr) return null;
  var v = RESIDUE_OVERRIDES[cr.chain + '|' + cr.resi];
  return (v == null) ? null : v;
}}

// Resolve the base (default) colour function for the active scheme. CPK and
// Solid are computed directly; the rest delegate to Mol*'s built-in theme so
// chain / secondary-structure / b-factor / rainbow keep rendering as before.
function _baseColorFn(ctx) {{
  var scheme = proteinColorType;
  if (scheme === 'cpk') return function (loc) {{ return elementColor(loc); }};
  if (scheme === 'uniform') return function () {{ return SOLID_VALUE; }};
  try {{
    var reg = molViewer.plugin.representation.structure.themes.colorThemeRegistry;
    var provider = reg.get(scheme);
    if (provider && provider.factory) {{
      var params = provider.getParams ? provider.getParams(ctx) : {{}};
      var dv = {{}};
      try {{
        dv = molstar.ParamDefinition
          ? molstar.ParamDefinition.getDefaultValues(params)
          : (provider.defaultValues || {{}});
      }} catch (e2) {{ dv = provider.defaultValues || {{}}; }}
      var inst = provider.factory(ctx, dv);
      if (inst && typeof inst.color === 'function') return inst.color;
    }}
  }} catch (e) {{ console.warn('base theme delegation failed for', scheme, e); }}
  return function (loc) {{ return elementColor(loc); }};
}}

function _proteinThemeFactory(ctx, props) {{
  var base = _baseColorFn(ctx);
  return {{
    factory: _proteinThemeFactory,
    granularity: 'group',
    color: function (loc) {{ var o = residueOverrideAt(loc); return (o != null) ? o : base(loc); }},
    props: props || {{}},
  }};
}}

function registerCpkTheme() {{
  if (proteinThemeRegistered || !molViewer) return proteinThemeRegistered;
  try {{
    var reg = molViewer.plugin.representation.structure.themes.colorThemeRegistry;
    if (!reg || !reg.add) return false;
    reg.add({{
      name: 'ezs-protein', label: 'SwiftScreen protein', category: 'Atom Property',
      factory: _proteinThemeFactory, getParams: function () {{ return {{}}; }},
      defaultValues: {{}}, isApplicable: function () {{ return true; }},
    }});
    proteinThemeRegistered = true;
  }} catch (e) {{
    console.warn('protein theme registration failed:', e);
    proteinThemeRegistered = false;
  }}
  return proteinThemeRegistered;
}}

// Route through the unified theme only when CPK is active or overrides exist;
// otherwise hand back the built-in theme untouched.
function _proteinColorSpec() {{
  if (proteinThemeRegistered && (_hasOverrides || proteinColorType === 'cpk')) {{
    return {{ color: 'ezs-protein' }};
  }}
  if (proteinColorType === 'uniform') return {{ color: 'uniform', colorParams: {{ value: SOLID_VALUE }} }};
  if (proteinColorType === 'cpk') return {{ color: 'element-symbol' }};
  return {{ color: proteinColorType }};
}}

function refreshOverrideState() {{
  _hasOverrides = Object.keys(RESIDUE_OVERRIDES).length > 0;
  if (colorMenuOpen) renderColorMenu();
  decorateOverrideTiles();
}}

function paintResidues(keys, hex) {{
  if (!keys || !keys.length) return;
  var c = _hexToInt(hex);
  keys.forEach(function (k) {{ RESIDUE_OVERRIDES[k] = c; }});
  refreshOverrideState();
  if (proteinRepType !== 'off') applyProteinRep(false);
}}

function clearResidueOverride(k) {{
  delete RESIDUE_OVERRIDES[k];
  refreshOverrideState();
  if (proteinRepType !== 'off') applyProteinRep(false);
}}

function clearAllOverrides() {{
  Object.keys(RESIDUE_OVERRIDES).forEach(function (k) {{ delete RESIDUE_OVERRIDES[k]; }});
  refreshOverrideState();
  if (proteinRepType !== 'off') applyProteinRep(false);
}}

// ── Residue strip ────────────────────────────────────────────────────────
function setStripMode(m) {{
  stripMode = m;
  document.getElementById('seqm-fly').classList.toggle('active', m === 'fly');
  document.getElementById('seqm-sel').classList.toggle('active', m === 'select');
  document.getElementById('seq-hint').textContent =
    m === 'fly' ? 'click a residue to fly to it' : 'click to select · shift-click for a range';
  updateSeqActions();
}}

function updateSeqActions() {{
  var bar = document.getElementById('seq-actions');
  bar.style.display = (stripMode === 'select') ? '' : 'none';
  document.getElementById('seq-count').textContent = stripSel.size + ' selected';
}}

function onTileClick(chain, resi, ev) {{
  if (stripMode === 'fly') {{
    var info = CHAIN_MAP[chain];
    var it = info && info.items[info.idx[resi]];
    if (it && it.ca) flyToCoords(it.ca[0], it.ca[1], it.ca[2]);
    return;
  }}
  var key = _seqKey(chain, resi);
  if (ev && ev.shiftKey && stripLastKey && stripLastKey.indexOf(chain + '|') === 0) {{
    var info2 = CHAIN_MAP[chain];
    var lastResi = parseInt(stripLastKey.split('|')[1], 10);
    var a = info2.idx[lastResi], b = info2.idx[resi];
    if (a != null && b != null) {{
      var lo = Math.min(a, b), hi = Math.max(a, b);
      for (var i = lo; i <= hi; i++) stripSel.add(_seqKey(chain, info2.items[i].resi));
    }}
  }} else {{
    if (stripSel.has(key)) stripSel.delete(key); else stripSel.add(key);
    stripLastKey = key;
  }}
  refreshStripSelection();
}}

function refreshStripSelection() {{
  document.querySelectorAll('.seq-tile.sel').forEach(function (t) {{ t.classList.remove('sel'); }});
  stripSel.forEach(function (k) {{
    var p = k.split('|');
    var t = document.getElementById(_tileId(p[0], parseInt(p[1], 10)));
    if (t) t.classList.add('sel');
  }});
  updateSeqActions();
}}

function deselectStrip() {{ stripSel.clear(); stripLastKey = null; refreshStripSelection(); }}

function paintSelection(hex) {{
  if (!stripSel.size) return;
  paintResidues(Array.from(stripSel), hex);
}}

// Drops any per-residue overrides on the current selection, returning those
// residues to the default atom-colour layer.
function clearSelectionOverrides() {{
  if (!stripSel.size) return;
  var changed = false;
  stripSel.forEach(function (k) {{
    if (k in RESIDUE_OVERRIDES) {{ delete RESIDUE_OVERRIDES[k]; changed = true; }}
  }});
  if (!changed) return;
  refreshOverrideState();
  if (proteinRepType !== 'off') applyProteinRep(false);
}}

function decorateOverrideTiles() {{
  document.querySelectorAll('.seq-tile.ov').forEach(function (t) {{
    t.classList.remove('ov'); t.style.removeProperty('--ov-dot');
  }});
  Object.keys(RESIDUE_OVERRIDES).forEach(function (k) {{
    var p = k.split('|');
    var t = document.getElementById(_tileId(p[0], parseInt(p[1], 10)));
    if (t) {{ t.classList.add('ov'); t.style.setProperty('--ov-dot', _intToHex(RESIDUE_OVERRIDES[k])); }}
  }});
}}

// ── Strip filter + quick-select ──────────────────────────────────────────
// Residues that the current compound interacts with, honouring the type toggles.
function _interactingKeys() {{
  var keys = new Set();
  if (currentData && currentData.interactions) {{
    currentData.interactions.forEach(function (ix) {{
      if (!activeToggles[ix.type]) return;
      keys.add(ix.chain + '|' + ix.residue_number);
    }});
  }}
  return keys;
}}

function setSeqFilter(mode) {{
  seqFilter = mode;
  applySeqFilter();
  if (seqFilterMenuOpen) renderSeqFilterMenu();
  var btn = document.getElementById('btn-seqfilter');
  if (btn) btn.classList.toggle('active', mode !== 'all');
}}

// Dims tiles outside the interacting set when the filter is on; they stay
// clickable so you can still inspect them.
function applySeqFilter() {{
  var inter = seqFilter === 'interacting' ? _interactingKeys() : null;
  (SEQUENCE || []).forEach(function (ch) {{
    ch.items.forEach(function (it) {{
      var t = document.getElementById(_tileId(ch.chain, it.resi));
      if (!t) return;
      if (inter && !inter.has(ch.chain + '|' + it.resi)) t.classList.add('dim');
      else t.classList.remove('dim');
    }});
  }});
}}

function selectInteracting() {{
  setStripMode('select');
  _interactingKeys().forEach(function (k) {{ stripSel.add(k); }});
  refreshStripSelection();
}}

function selectChain(chain) {{
  setStripMode('select');
  var info = CHAIN_MAP[chain];
  if (!info) return;
  info.items.forEach(function (it) {{ stripSel.add(_seqKey(chain, it.resi)); }});
  refreshStripSelection();
}}

function renderSeqFilterMenu() {{
  var menu = document.getElementById('seqfilter-menu');
  menu.innerHTML = '';
  menu.appendChild(_menuRow('Show all residues', seqFilter === 'all', function () {{ setSeqFilter('all'); }}));
  menu.appendChild(_menuRow('Only interacting', seqFilter === 'interacting', function () {{ setSeqFilter('interacting'); }}));
}}

function renderSeqSelectMenu() {{
  var menu = document.getElementById('seqselect-menu');
  menu.innerHTML = '';
  menu.appendChild(_menuRow('All interacting residues', false, function () {{ selectInteracting(); setSeqSelectMenuOpen(false); }}));
  var sep = document.createElement('div'); sep.className = 'color-sep'; menu.appendChild(sep);
  (SEQUENCE || []).forEach(function (ch) {{
    menu.appendChild(_menuRow('All of chain ' + ch.chain, false, function () {{ selectChain(ch.chain); setSeqSelectMenuOpen(false); }}));
  }});
  var sep2 = document.createElement('div'); sep2.className = 'color-sep'; menu.appendChild(sep2);
  menu.appendChild(_menuRow('Clear selection', false, function () {{ deselectStrip(); setSeqSelectMenuOpen(false); }}));
}}

function setSeqFilterMenuOpen(on) {{
  seqFilterMenuOpen = on;
  document.getElementById('seqfilter-menu').style.display = on ? '' : 'none';
  if (on) renderSeqFilterMenu();
}}
function toggleSeqFilterMenu() {{ setSeqFilterMenuOpen(!seqFilterMenuOpen); }}

function setSeqSelectMenuOpen(on) {{
  seqSelectMenuOpen = on;
  document.getElementById('seqselect-menu').style.display = on ? '' : 'none';
  if (on) renderSeqSelectMenu();
}}
function toggleSeqSelectMenu() {{ setSeqSelectMenuOpen(!seqSelectMenuOpen); }}

document.addEventListener('click', function (e) {{
  if (seqFilterMenuOpen) {{
    var d = document.getElementById('seqfilter-dropdown');
    if (d && !d.contains(e.target)) setSeqFilterMenuOpen(false);
  }}
  if (seqSelectMenuOpen) {{
    var d2 = document.getElementById('seqselect-dropdown');
    if (d2 && !d2.contains(e.target)) setSeqSelectMenuOpen(false);
  }}
}});

// Chains the current compound contacts (honouring the type toggles).
function _interactingChains() {{
  var s = new Set();
  if (currentData && currentData.interactions) {{
    currentData.interactions.forEach(function (ix) {{ if (activeToggles[ix.type]) s.add(ix.chain); }});
  }}
  return s;
}}

// Chain chips: focus one chain (default for many-chain receptors) or show all.
// A dot marks chains the ligand contacts. Hidden for single-chain receptors.
function renderChainBar() {{
  var bar = document.getElementById('seq-chainbar');
  if (!bar) return;
  if (!SEQUENCE || SEQUENCE.length <= 1) {{ bar.style.display = 'none'; bar.innerHTML = ''; return; }}
  bar.style.display = '';
  bar.innerHTML = '';
  var inter = _interactingChains();
  var lab = document.createElement('span'); lab.className = 'cb-label'; lab.textContent = 'Chain';
  bar.appendChild(lab);
  function chip(label, value, dot) {{
    var b = document.createElement('button');
    b.className = 'chain-chip' + (seqChain === value ? ' active' : '');
    b.appendChild(document.createTextNode(label));
    if (dot) {{ var d = document.createElement('span'); d.className = 'chain-dot'; b.appendChild(d); }}
    b.onclick = function () {{ setSeqChain(value); }};
    return b;
  }}
  bar.appendChild(chip('All', 'all', false));
  SEQUENCE.forEach(function (ch) {{ bar.appendChild(chip(ch.chain, ch.chain, inter.has(ch.chain))); }});
}}

function setSeqChain(c) {{ seqChain = c; renderSequencePanel(); }}

// Rebuilds the strip with a per-column tick ruler and interactive tiles.
function renderSequencePanel() {{
  renderChainBar();
  var body = document.getElementById('seq-body');
  body.innerHTML = '';
  if (!SEQUENCE || SEQUENCE.length === 0) {{
    body.innerHTML = '<div style="color:var(--muted);font-size:11px;padding:4px">No receptor sequence available</div>';
    return;
  }}
  var chains = (seqChain === 'all') ? SEQUENCE : SEQUENCE.filter(function (c) {{ return c.chain === seqChain; }});
  if (!chains.length) chains = SEQUENCE;
  chains.forEach(function (chainObj) {{
    var row = document.createElement('div'); row.className = 'seq-chain';
    var label = document.createElement('span'); label.className = 'seq-chain-label';
    label.textContent = 'Chain ' + chainObj.chain; row.appendChild(label);
    var tiles = document.createElement('span'); tiles.className = 'seq-tiles';
    chainObj.items.forEach(function (it) {{
      var col = document.createElement('span'); col.className = 'seq-col';
      var tick = document.createElement('span'); tick.className = 'seq-tick';
      if (it.resi % 10 === 0) {{ tick.textContent = it.resi; tick.classList.add('on'); }}
      col.appendChild(tick);
      var tile = document.createElement('span'); tile.className = 'seq-tile';
      tile.id = _tileId(chainObj.chain, it.resi);
      tile.textContent = it.one;
      tile.title = it.resn + ' ' + it.resi + ' (chain ' + chainObj.chain + ')';
      tile.onclick = function (ev) {{ onTileClick(chainObj.chain, it.resi, ev); }};
      col.appendChild(tile);
      tiles.appendChild(col);
    }});
    row.appendChild(tiles);
    body.appendChild(row);
  }});
  if (currentData) applySequenceHighlight(currentData);
  refreshStripSelection();
  decorateOverrideTiles();
  applySeqFilter();
}}

// Contact highlight only; selection / override / filter decorations live on
// separate classes so they survive a re-highlight.
function applySequenceHighlight(compound) {{
  document.querySelectorAll('.seq-tile.ix').forEach(function (t) {{
    t.classList.remove('ix'); t.style.background = '';
  }});
  if (compound && compound.interactions) {{
    var seen = {{}};
    compound.interactions.forEach(function (ix) {{
      if (!activeToggles[ix.type]) return;
      var key = ix.chain + '|' + ix.residue_number;
      if (seen[key]) return;
      seen[key] = true;
      var tile = document.getElementById(_tileId(ix.chain, ix.residue_number));
      if (!tile) return;
      tile.classList.add('ix');
      tile.style.background = COLORS[ix.type] || 'var(--accent)';
    }});
  }}
  renderChainBar();
  applySeqFilter();
}}

// ── Colour menu: Layer 1 atom colours + Layer 2 per-residue ──────────────
function _menuRow(label, active, onclick) {{
  var row = document.createElement('div');
  row.className = 'view-menu-item rep-menu-item' + (active ? ' active' : '');
  var name = document.createElement('span'); name.className = 'vm-name';
  name.textContent = label + (active ? ' ✓' : ''); name.onclick = onclick;
  row.appendChild(name);
  return row;
}}

function renderColorMenu() {{
  var menu = document.getElementById('color-menu');
  menu.innerHTML = '';

  var h1 = document.createElement('div');
  h1.className = 'color-sec-head'; h1.textContent = 'Atom colours (default)';
  menu.appendChild(h1);
  COLOR_SCHEMES.forEach(function (c) {{
    menu.appendChild(_menuRow(c.label, c.id === proteinColorType, function () {{ setProteinColor(c.id); }}));
  }});

  var sub = document.createElement('div');
  sub.className = 'color-sec-head'; sub.textContent = 'Customize elements';
  sub.style.marginTop = '4px'; menu.appendChild(sub);
  var grid = document.createElement('div'); grid.className = 'color-swatch-grid';
  ELEMENT_SWATCHES.forEach(function (sw) {{
    var cell = document.createElement('label'); cell.className = 'color-swatch';
    var input = document.createElement('input'); input.type = 'color';
    input.value = _intToHex(ELEMENT_PALETTE[sw.key]);
    input.oninput = function () {{ setElementColor(sw.key, input.value); }};
    var lbl = document.createElement('span'); lbl.textContent = sw.label;
    cell.appendChild(input); cell.appendChild(lbl); grid.appendChild(cell);
  }});
  menu.appendChild(grid);
  var reset = document.createElement('div');
  reset.className = 'view-menu-item rep-menu-item color-reset';
  reset.innerHTML = '<span class="vm-name">Reset elements to CPK</span>';
  reset.onclick = resetElementPalette; menu.appendChild(reset);

  var sep = document.createElement('div'); sep.className = 'color-sep'; menu.appendChild(sep);

  var h2 = document.createElement('div');
  h2.className = 'color-sec-head'; h2.textContent = 'Per-residue colours';
  menu.appendChild(h2);

  var pick = document.createElement('div'); pick.className = 'pr-row';
  var chSel = document.createElement('select'); chSel.id = 'pr-chain';
  (SEQUENCE || []).forEach(function (ch) {{
    var o = document.createElement('option'); o.value = ch.chain; o.textContent = ch.chain; chSel.appendChild(o);
  }});
  chSel.onchange = function () {{ populateResidueOptions(); }};
  var resSel = document.createElement('select'); resSel.id = 'pr-res';
  var col = document.createElement('input'); col.type = 'color'; col.className = 'pr-color'; col.id = 'pr-color'; col.value = '#ffd23f';
  var apply = document.createElement('button'); apply.className = 'pr-apply'; apply.textContent = 'Apply';
  apply.onclick = applyPerResidue;
  pick.appendChild(chSel); pick.appendChild(resSel); pick.appendChild(col); pick.appendChild(apply);
  menu.appendChild(pick);
  populateResidueOptions();

  if (stripSel.size) {{
    var fromSel = document.createElement('div'); fromSel.className = 'pr-row';
    var btn = document.createElement('button'); btn.className = 'pr-apply'; btn.style.flex = '1';
    btn.textContent = 'Colour ' + stripSel.size + ' selected residue' + (stripSel.size === 1 ? '' : 's');
    btn.onclick = function () {{ paintSelection(document.getElementById('pr-color').value); }};
    fromSel.appendChild(btn); menu.appendChild(fromSel);
  }}

  var keys = Object.keys(RESIDUE_OVERRIDES);
  if (keys.length) {{
    keys.sort(function (a, b) {{
      var pa = a.split('|'), pb = b.split('|');
      return pa[0] === pb[0] ? (parseInt(pa[1], 10) - parseInt(pb[1], 10)) : (pa[0] < pb[0] ? -1 : 1);
    }});
    var list = document.createElement('div'); list.className = 'ov-list';
    keys.forEach(function (k) {{
      var p = k.split('|'); var chain = p[0]; var resi = parseInt(p[1], 10);
      var info = CHAIN_MAP[chain]; var it = info && info.items[info.idx[resi]];
      var item = document.createElement('div'); item.className = 'ov-item';
      var sw = document.createElement('span'); sw.className = 'ov-swatch';
      sw.style.background = _intToHex(RESIDUE_OVERRIDES[k]);
      var lab = document.createElement('span'); lab.className = 'ov-label';
      lab.textContent = (it ? it.resn : '') + ' ' + resi + ' · ' + chain;
      var del = document.createElement('span'); del.className = 'ov-del'; del.textContent = '×';
      del.title = 'Remove'; del.onclick = function () {{ clearResidueOverride(k); }};
      item.appendChild(sw); item.appendChild(lab); item.appendChild(del); list.appendChild(item);
    }});
    menu.appendChild(list);
  }} else {{
    var hint = document.createElement('div'); hint.className = 'pr-hint';
    hint.textContent = 'None painted. Pick a residue, or use Select mode in the strip.';
    menu.appendChild(hint);
  }}

  // Decolour actions stay visible regardless of selection / overrides.
  function _decolourRow(label, fn) {{
    var r = document.createElement('div');
    r.className = 'view-menu-item rep-menu-item color-reset';
    r.innerHTML = '<span class="vm-name">' + label + '</span>';
    r.onclick = fn;
    return r;
  }}
  menu.appendChild(_decolourRow('Decolour selected', clearSelectionOverrides));
  menu.appendChild(_decolourRow('Decolour all', clearAllOverrides));
}}

function populateResidueOptions() {{
  var chSel = document.getElementById('pr-chain');
  var resSel = document.getElementById('pr-res');
  if (!chSel || !resSel) return;
  var info = CHAIN_MAP[chSel.value];
  resSel.innerHTML = '';
  if (!info) return;
  info.items.forEach(function (it) {{
    var o = document.createElement('option'); o.value = it.resi; o.textContent = it.resn + ' ' + it.resi;
    resSel.appendChild(o);
  }});
}}

function applyPerResidue() {{
  var chSel = document.getElementById('pr-chain');
  var resSel = document.getElementById('pr-res');
  var col = document.getElementById('pr-color');
  if (!chSel || !resSel || !resSel.value) return;
  paintResidues([_seqKey(chSel.value, parseInt(resSel.value, 10))], col.value);
}}

function setProteinColor(id) {{
  proteinColorType = id;
  if (colorMenuOpen) renderColorMenu();
  if (proteinRepType !== 'off') applyProteinRep(false);
}}

function setElementColor(key, hex) {{
  ELEMENT_PALETTE[key] = _hexToInt(hex);
  if (proteinColorType !== 'cpk') proteinColorType = 'cpk';
  if (colorMenuOpen) renderColorMenu();
  if (proteinRepType !== 'off') applyProteinRep(false);
}}

function resetElementPalette() {{
  Object.assign(ELEMENT_PALETTE, ELEMENT_DEFAULTS);
  if (colorMenuOpen) renderColorMenu();
  if (proteinColorType === 'cpk' && proteinRepType !== 'off') applyProteinRep(false);
}}
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

    for c in compounds:
        targets: set[tuple[str, int]] = set()
        for ix in c.get("interactions") or []:
            ch = ix.get("chain")
            rn = ix.get("residue_number")
            if ch and rn is not None:
                try:
                    targets.add((str(ch), int(rn)))
                except (TypeError, ValueError):
                    pass
        if targets:
            c["residue_pdb"] = _extract_residue_pdb(receptor_text, targets)

    html = _build_html(compounds, receptor_text)
    viewer_path.write_text(html, encoding="utf-8")
    return viewer_path
