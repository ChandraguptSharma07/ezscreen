from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from ezscreen.errors import ReceptorPrepError


def _error(detail: str) -> ReceptorPrepError:
    return ReceptorPrepError(
        f"Modified-residue preservation failed: {detail}\n"
        "The selected chemistry could not be verified.\n"
        "Review the explicit policy/template and receptor before retrying."
    )


def _identity(item: dict[str, Any]) -> str:
    chain, number = item["chain"], item["resseq"]
    if not isinstance(chain, str) or len(chain) > 1 or chain in {":", ",", "="}:
        raise ValueError("invalid chain ID")
    if type(number) is not int or not -999 <= number <= 9999:
        raise ValueError("invalid residue number")
    if item.get("icode", ""):
        raise ValueError("insertion codes are not supported in policy version 1")
    return f"{chain}:{number}"


def load_modified_residue_policy(path: Path) -> dict[str, Any]:
    """Validate an explicit linked-residue policy without preparing a receptor.

    Args:
        path: JSON policy containing reviewed templates and residue assignments.

    Returns:
        Validated policy including the source file SHA256.

    Raises:
        ReceptorPrepError: If any required identity or chemical definition is invalid.
    """
    from meeko import ResidueChemTemplates
    from rdkit import Chem

    try:
        data = path.read_bytes()
        policy = json.loads(data)
        if policy["version"] != 1:
            raise ValueError("unsupported policy version")
        residues, definitions = policy["residues"], policy["templates"]
        if (
            not isinstance(residues, list)
            or not residues
            or not isinstance(definitions, dict)
        ):
            raise ValueError("nonempty residue selections and templates required")
        seen, used = set(), set()
        for item in residues:
            identity = _identity(item)
            if identity in seen:
                raise ValueError("duplicate selected residue")
            seen.add(identity)
            if not re.fullmatch(r"[A-Z0-9]{1,3}", item["resname"]):
                raise ValueError("invalid residue name")
            template_name = item["template"]
            if not re.fullmatch(r"EZ_[A-Za-z0-9_]+", template_name):
                raise ValueError("custom template names must start with EZ_")
            used.add(template_name)
            if (
                not isinstance(item["state_rationale"], str)
                or not item["state_rationale"].strip()
            ):
                raise ValueError("chemical-state rationale required")
            if type(item["formal_charge"]) is not int:
                raise ValueError("explicit integer formal charge required")
            definition = definitions[template_name]
            if set(definition) != {"smiles", "atom_name", "link_labels"}:
                raise ValueError(
                    "template must define smiles, atom_name and link_labels"
                )
            parser = Chem.SmilesParserParams()
            parser.removeHs = False
            mol = Chem.MolFromSmiles(definition["smiles"], parser)
            names = definition["atom_name"]
            if (
                mol is None
                or len(names) != mol.GetNumAtoms()
                or len(set(names)) != len(names)
            ):
                raise ValueError("invalid template atoms")
            if len(Chem.GetMolFrags(mol)) != 1:
                raise ValueError("modified residue template must be connected")
            if any(not isinstance(n, str) or not n or len(n) > 4 for n in names):
                raise ValueError("template atom names must fit PDB columns")
            if (
                sum(a.GetFormalCharge() for a in mol.GetAtoms())
                != item["formal_charge"]
            ):
                raise ValueError("template formal charge disagrees with policy")
            labels = {int(k): v for k, v in definition["link_labels"].items()}
            if sorted(labels.values()) != ["C-term", "N-term"]:
                raise ValueError(
                    "version 1 requires an internal peptide residue with two links"
                )
            if {names[i]: label for i, label in labels.items()} != {
                "N": "N-term",
                "C": "C-term",
            }:
                raise ValueError(
                    "version 1 requires unmodified N/C backbone link atoms"
                )
            links = item["links"]
            if len(links) != 2 or {link["atom"] for link in links} != {"N", "C"}:
                raise ValueError("declare both peptide neighbors")
            for link in links:
                if _identity(link) == identity:
                    raise ValueError("self-link is invalid")
                if link["partner_atom"] != {"N": "C", "C": "N"}[link["atom"]]:
                    raise ValueError("expected N-C peptide links")
        if set(definitions) != used:
            raise ValueError("only explicitly assigned templates may be supplied")
        templates = ResidueChemTemplates.create_from_defaults()
        if used.intersection(templates.residue_templates):
            raise ValueError("custom templates cannot override built-in templates")
        templates.add_dict({"residue_templates": definitions})
        for name in used:
            template = templates.residue_templates[name]
            template.check(template.mol, template.link_labels, template.atom_names)
    except Exception as exc:
        raise _error(str(exc)) from exc
    return {**policy, "source_sha256": hashlib.sha256(data).hexdigest()}


def _atoms(path: Path, identity: str) -> dict[str, tuple[str, tuple[float, ...]]]:
    atoms = {}
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if f"{line[21:22].strip()}:{int(line[22:26])}" != identity:
            continue
        if line[26:27].strip():
            raise _error(f"ambiguous insertion code at {identity}")
        name = line[12:16].strip()
        if name in atoms:
            raise _error(f"duplicate atom {identity}/{name}")
        xyz = tuple(float(line[start : start + 8]) for start in (30, 38, 46))
        if not all(math.isfinite(v) for v in xyz):
            raise _error(f"nonfinite coordinates at {identity}/{name}")
        atoms[name] = (line[17:20].strip(), xyz)
    return atoms


def validate_modified_atoms(
    policy: dict[str, Any],
    source: Path,
    prepared: Path | None = None,
) -> None:
    """Check exact selected heavy-atom names, residue identities and coordinates."""
    from rdkit import Chem

    for item in policy["residues"]:
        identity = _identity(item)
        definition = policy["templates"][item["template"]]
        parser = Chem.SmilesParserParams()
        parser.removeHs = False
        mol = Chem.MolFromSmiles(definition["smiles"], parser)
        expected = {
            definition["atom_name"][a.GetIdx()]
            for a in mol.GetAtoms()
            if a.GetAtomicNum() != 1
        }
        original = _atoms(source, identity)
        hydrogen_names = {
            definition["atom_name"][a.GetIdx()]
            for a in mol.GetAtoms()
            if a.GetAtomicNum() == 1
        }
        for path in [source] + ([prepared] if prepared is not None else []):
            actual = _atoms(path, identity)
            heavy = set(actual) - hydrogen_names
            if heavy != expected or any(
                r != item["resname"] for r, _ in actual.values()
            ):
                raise _error(f"atom/residue mismatch at {identity} in {path.name}")
            if any(math.dist(original[n][1], actual[n][1]) > 0.002 for n in expected):
                raise _error(f"heavy-atom coordinates changed at {identity}")
            if path.suffix == ".pdbqt":
                for line in path.read_text().splitlines():
                    if not line.startswith(("ATOM  ", "HETATM")):
                        continue
                    if f"{line[21:22].strip()}:{int(line[22:26])}" != identity:
                        continue
                    try:
                        charge = float(line[70:76])
                        if not math.isfinite(charge) or not line[77:].strip():
                            raise ValueError("missing atom type or nonfinite charge")
                    except ValueError as exc:
                        raise _error(f"invalid PDBQT parameters at {identity}") from exc


def validate_modified_polymer(policy: dict[str, Any], path: Path) -> dict[str, Any]:
    """Verify Meeko's materialized states and inferred inter-residue connections."""
    from meeko import Polymer
    from meeko.polymer import find_inter_mols_bonds
    from rdkit import Chem

    try:
        polymer = Polymer.from_json(path.read_text())
        valid = polymer.get_valid_monomers()
        raw = {key: (m.raw_rdkit_mol, m.input_resname) for key, m in valid.items()}
        bonds = find_inter_mols_bonds(raw)
        verified = []
        for item in policy["residues"]:
            identity = _identity(item)
            monomer = valid[identity]
            if (
                monomer.residue_template_key != item["template"]
                or monomer.input_resname != item["resname"]
            ):
                raise ValueError(f"wrong template/residue at {identity}")
            definition = policy["templates"][item["template"]]
            parser = Chem.SmilesParserParams()
            parser.removeHs = False
            expected_mol = Chem.MolFromSmiles(definition["smiles"], parser)
            if Chem.MolToSmiles(expected_mol) != Chem.MolToSmiles(monomer.rdkit_mol):
                raise ValueError(f"materialized chemistry differs at {identity}")
            if list(monomer.atom_names) != definition["atom_name"]:
                raise ValueError(f"template atom mapping differs at {identity}")
            actual_links = set()
            for pair, pairs in bonds.items():
                if identity not in pair:
                    continue
                side = pair.index(identity)
                partner_id = pair[1 - side]
                partner = valid[partner_id]
                for indices in pairs:
                    own_name = monomer.atom_names[
                        monomer.mapidx_from_raw[indices[side]]
                    ]
                    partner_name = partner.atom_names[
                        partner.mapidx_from_raw[indices[1 - side]]
                    ]
                    actual_links.add((own_name, partner_id, partner_name))
            expected_links = {
                (link["atom"], _identity(link), link["partner_atom"])
                for link in item["links"]
            }
            if actual_links != expected_links:
                raise ValueError(f"unexpected/missing covalent links at {identity}")
            verified.append({**item, "verified": True})
    except Exception as exc:
        raise _error(str(exc)) from exc
    return {
        "complete": True,
        "source_sha256": policy["source_sha256"],
        "residues": verified,
    }
