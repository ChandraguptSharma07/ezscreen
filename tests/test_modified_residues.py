from __future__ import annotations

import json
import subprocess

import pytest

from ezscreen.errors import ReceptorPrepError
from ezscreen.prep import receptor


@pytest.fixture
def policy_file(tmp_path):
    policy = {
        "version": 1,
        "templates": {
            "EZ_TEST": {
                "smiles": "C(=O)C([H])(N[H])[H]",
                "atom_name": ["C", "O", "CA", "HA2", "N", "H", "HA3"],
                "link_labels": {"0": "C-term", "4": "N-term"},
            }
        },
        "residues": [
            {
                "resname": "TST",
                "chain": "A",
                "resseq": 2,
                "template": "EZ_TEST",
                "formal_charge": 0,
                "state_rationale": "Synthetic glycine-like test fixture, not an LLP template.",
                "links": [
                    {"atom": "N", "chain": "A", "resseq": 1, "partner_atom": "C"},
                    {"atom": "C", "chain": "A", "resseq": 3, "partner_atom": "N"},
                ],
            }
        ],
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    return path


def _atom(name, number=2, resname="TST", record="HETATM", x=0.0):
    return (
        f"{record:<6}{1:5d} {name:>4} {resname:>3} A{number:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}          {name[0]:>2}"
    )


def test_policy_is_hash_frozen(policy_file):
    import hashlib

    from ezscreen.prep.modified_residues import load_modified_residue_policy

    result = load_modified_residue_policy(policy_file)
    assert (
        result["source_sha256"] == hashlib.sha256(policy_file.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("formal_charge", 1),
        ("state_rationale", ""),
        ("icode", "A"),
        ("template", "GLY"),
        ("links", []),
        ("chain", "A,B"),
    ],
)
def test_policy_rejects_ambiguous_chemistry(policy_file, field, value):
    from ezscreen.prep.modified_residues import load_modified_residue_policy

    policy = json.loads(policy_file.read_text())
    policy["residues"][0][field] = value
    policy_file.write_text(json.dumps(policy))
    with pytest.raises(ReceptorPrepError):
        load_modified_residue_policy(policy_file)


@pytest.mark.parametrize("record", ["ATOM", "HETATM"])
def test_protected_atom_identity_and_coordinates(policy_file, tmp_path, record):
    from ezscreen.prep.modified_residues import (
        load_modified_residue_policy,
        validate_modified_atoms,
    )

    policy = load_modified_residue_policy(policy_file)
    src, dst = tmp_path / "raw.pdb", tmp_path / "fixed.pdb"
    text = "\n".join(_atom(n, record=record) for n in ["N", "CA", "C", "O"]) + "\n"
    src.write_text(text)
    dst.write_text(text)
    validate_modified_atoms(policy, src, dst)
    dst.write_text(text + _atom("CA", resname="LYS") + "\n")
    with pytest.raises(ReceptorPrepError, match="duplicate"):
        validate_modified_atoms(policy, src, dst)
    dst.write_text(
        text.replace(_atom("CA", record=record), _atom("CA", record=record, x=0.1))
    )
    with pytest.raises(ReceptorPrepError, match="coordinates"):
        validate_modified_atoms(policy, src, dst)
    dst.write_text(text.replace(_atom("O", record=record), ""))
    with pytest.raises(ReceptorPrepError, match="mismatch"):
        validate_modified_atoms(policy, src, dst)


def test_pdbfixer_protects_only_selected_residue(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    residues = [
        SimpleNamespace(name=name, id=str(number), chain=SimpleNamespace(id="A"))
        for name, number in [("TST", 2), ("TST", 4), ("HOH", 8), ("LIG", 9)]
    ]
    captured = {}

    class Fixer:
        def __init__(self, **kwargs):
            self.topology = SimpleNamespace(
                residues=lambda: iter(residues),
                getNumResidues=lambda: len(residues),
            )
            self.positions = []

        def findMissingResidues(self):
            self.missingResidues = {}

        def findNonstandardResidues(self):
            self.nonstandardResidues = [(r, "GLY") for r in residues if r.name == "TST"]

        def replaceNonstandardResidues(self):
            captured["replaced"] = [r.id for r, _ in self.nonstandardResidues]
            for r, name in self.nonstandardResidues:
                r.name = name

    class Modeller:
        def __init__(self, topology, positions):
            self.topology, self.positions = topology, positions

        def delete(self, selected):
            captured["removed"] = [r.id for r in selected]

    app = ModuleType("openmm.app")
    app.Modeller = Modeller
    app.PDBFile = SimpleNamespace(writeFile=lambda *args, **kwargs: None)
    fixer_module = ModuleType("pdbfixer")
    fixer_module.PDBFixer = Fixer
    definitions = ModuleType("pdbfixer.pdbfixer")
    definitions.proteinResidues = ["GLY"]
    definitions.dnaResidues = []
    definitions.rnaResidues = []
    monkeypatch.setitem(sys.modules, "openmm.app", app)
    monkeypatch.setitem(sys.modules, "pdbfixer", fixer_module)
    monkeypatch.setitem(sys.modules, "pdbfixer.pdbfixer", definitions)
    receptor._run_pdbfixer(
        tmp_path / "unused.pdb",
        tmp_path / "out.pdb",
        keep_waters=True,
        protected_keys={("TST", "A", 2)},
    )
    assert captured == {"replaced": ["4"], "removed": ["9"]}


def test_polymer_gate_checks_links_and_materialized_state(
    policy_file, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import meeko
    import meeko.polymer
    from rdkit import Chem

    from ezscreen.prep.modified_residues import (
        load_modified_residue_policy,
        validate_modified_polymer,
    )

    policy = load_modified_residue_policy(policy_file)
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    mol = Chem.MolFromSmiles(policy["templates"]["EZ_TEST"]["smiles"], parser)
    monomer = SimpleNamespace(
        raw_rdkit_mol=mol,
        rdkit_mol=mol,
        input_resname="TST",
        residue_template_key="EZ_TEST",
        atom_names=["C", "O", "CA", "HA2", "N", "H", "HA3"],
        mapidx_from_raw={i: i for i in range(7)},
    )
    neighbor = SimpleNamespace(
        raw_rdkit_mol=mol,
        input_resname="GLY",
        atom_names=monomer.atom_names,
        mapidx_from_raw=monomer.mapidx_from_raw,
    )
    polymer = SimpleNamespace(
        get_valid_monomers=lambda: {"A:1": neighbor, "A:2": monomer, "A:3": neighbor}
    )
    monkeypatch.setattr(meeko.Polymer, "from_json", lambda *args: polymer)
    inferred = {("A:1", "A:2"): [(0, 4)], ("A:2", "A:3"): [(0, 4)]}
    monkeypatch.setattr(meeko.polymer, "find_inter_mols_bonds", lambda *args: inferred)
    path = tmp_path / "polymer.json"
    path.write_text("{}")
    assert validate_modified_polymer(policy, path)["complete"]
    inferred[("A:1", "A:2")].append((2, 2))
    with pytest.raises(ReceptorPrepError, match="links"):
        validate_modified_polymer(policy, path)
    inferred[("A:1", "A:2")].pop()
    monomer.rdkit_mol = Chem.MolFromSmiles("C")
    with pytest.raises(ReceptorPrepError, match="chemistry"):
        validate_modified_polymer(policy, path)


def test_modified_policy_requires_explicit_chemistry(tmp_path):
    from ezscreen.prep.modified_residues import load_modified_residue_policy

    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"version": 1, "residues": [{}], "templates": {}}))
    with pytest.raises(ReceptorPrepError):
        load_modified_residue_policy(path)


def test_modified_residues_cannot_use_automatic_protonation(tmp_path):
    with pytest.raises(ReceptorPrepError, match="automatic"):
        receptor.prep_receptor(
            tmp_path / "unused.pdb",
            ["A"],
            tmp_path / "out",
            receptor_isomers="variants",
            modified_residue_policy=tmp_path / "policy.json",
        )


def test_protected_meeko_has_no_permissive_fallback(tmp_path, monkeypatch):
    src = tmp_path / "receptor.pdb"
    src.write_text("END\n")
    monkeypatch.setattr(receptor, "_meeko_receptor_command", lambda: "meeko")
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"bad template")

    monkeypatch.setattr(receptor.subprocess, "run", run)
    monkeypatch.setattr(
        receptor,
        "_run_obabel_receptor",
        lambda *args: pytest.fail("protected residue must never use fallback"),
    )
    with pytest.raises(ReceptorPrepError, match="bad template"):
        receptor._run_meeko_receptor(
            src,
            tmp_path,
            modified_templates=tmp_path / "templates.json",
            template_assignments={"A:680": "EZ_LLP"},
        )
    assert "-a" not in calls[0]
    assert "--add_templates" in calls[0]
    assert "--set_template" in calls[0]
    assert "-j" in calls[0]


def test_append_does_not_duplicate_protected_residue(tmp_path):
    fixed = tmp_path / "fixed.pdb"
    selected = tmp_path / "selected.pdb"
    output = tmp_path / "output.pdb"
    line = (
        "HETATM    1  CA  LLP A 680       0.000   0.000   0.000  1.00 20.00           C"
    )
    fixed.write_text(line + "\nEND\n")
    selected.write_text(line + "\nEND\n")
    receptor._append_selected_heterogens(
        fixed,
        selected,
        output,
        protected_keys={("LLP", "A", 680)},
    )
    assert output.read_text().count("LLP") == 1


@pytest.mark.parametrize("keep_hetero", [None, []])
def test_prep_threads_policy_and_writes_verified_report(
    policy_file, tmp_path, monkeypatch, keep_hetero
):
    from ezscreen.prep import modified_residues

    source = tmp_path / "source.pdb"
    source.write_text("\n".join(_atom(n) for n in ["N", "CA", "C", "O"]) + "\nEND\n")
    captured = {}

    def fixer(src, dst, **kwargs):
        captured["fixer"] = kwargs
        dst.write_text(src.read_text())
        return {"missing_residues": 0, "residue_count": 1, "pdbfixer_version": "mock"}

    def meeko(src, output_dir, **kwargs):
        captured["meeko"] = kwargs
        path = output_dir / "receptor.pdbqt"
        lines = [
            line[:66] + f"    {0.0:6.3f} {line[76:78].strip()}"
            for line in src.read_text().splitlines()
            if line.startswith(("ATOM", "HETATM"))
        ]
        path.write_text("\n".join(lines) + "\n")
        return path, "mock"

    monkeypatch.setattr(receptor, "_run_pdbfixer", fixer)
    monkeypatch.setattr(receptor, "_run_meeko_receptor", meeko)
    monkeypatch.setattr(
        modified_residues, "validate_modified_polymer", lambda *args: {"complete": True}
    )
    result = receptor.prep_receptor(
        source,
        ["A"],
        tmp_path / "out",
        keep_hetero=keep_hetero,
        modified_residue_policy=policy_file,
    )
    assert captured["fixer"]["protected_keys"] == {("TST", "A", 2)}
    assert captured["meeko"]["template_assignments"] == {"A:2": "EZ_TEST"}
    assert result["report"]["modified_residue_preservation"]["complete"]
    assert result["fixed_pdb_path"].read_text().count("TST") == 4
    assert (tmp_path / "out" / "modified_residue_policy.json").is_file()
    assert (tmp_path / "out" / "receptor_modified_retention.json").is_file()


async def test_wizard_policy_is_opt_in_and_blocks_automatic_states(policy_file):
    from textual.widgets import Input, Switch

    from ezscreen.tui.app import EzscreenApp
    from ezscreen.tui.screens.run_wizard import RunWizardScreen

    app, screen = EzscreenApp(), RunWizardScreen()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(screen)
        await pilot.pause()
        screen._step = 1
        assert screen._validate_step() is None
        assert screen._ctx["modified_residue_policy"] is None
        screen.query_one("#modified-residue-policy", Input).value = str(policy_file)
        assert screen._validate_step() is None
        assert screen._ctx["modified_residue_policy"] == str(policy_file.resolve())
        assert len(screen._ctx["modified_residue_policy_sha256"]) == 64
        screen.query_one("#opt-receptor-isomers", Switch).value = True
        await pilot.pause()
        assert "automatic protonation" in screen._validate_step()
