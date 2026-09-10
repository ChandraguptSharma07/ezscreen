from pathlib import Path

import pytest

from ezscreen.prep.receptor import _filter_chains


def atom(chain: str, serial: int = 1) -> str:
    return (
        f"ATOM  {serial:5d}  CA  ALA {chain}   1    "
        "   1.000   2.000   3.000  1.00 20.00           C  "
    )


@pytest.mark.parametrize("terminator", ["TER", "TER       2      ALA A   1"])
def test_discarded_leading_chain_has_no_terminator(
    tmp_path: Path, terminator: str
) -> None:
    source = tmp_path / "input.pdb"
    output = tmp_path / "output.pdb"
    source.write_text(f"{atom('A')}\n{terminator}\n{atom('B', 3)}\nTER\nEND\n")
    _filter_chains(source, ["B"], output)
    assert output.read_text() == f"{atom('B', 3)}\nTER\nEND\n"


@pytest.mark.parametrize("chain", ["B", " "])
def test_selected_segments_keep_atoms_and_terminators(
    tmp_path: Path, chain: str
) -> None:
    source = tmp_path / "input.pdb"
    output = tmp_path / "output.pdb"
    text = f"{atom(chain)}\nTER\n{atom(chain, 3)}\nTER\nEND\n"
    source.write_text(text)
    _filter_chains(source, [chain], output)
    assert output.read_text() == text


def test_orphan_duplicate_and_cross_model_terminators_are_removed(tmp_path: Path) -> None:
    source = tmp_path / "input.pdb"
    output = tmp_path / "output.pdb"
    source.write_text(
        f"TER\nMODEL        1\n{atom('B')}\nTER\nTER\nENDMDL\n"
        f"MODEL        2\nTER\n{atom('A')}\nTER\nENDMDL\nEND\n"
    )
    _filter_chains(source, ["B"], output)
    assert output.read_text() == (
        f"MODEL        1\n{atom('B')}\nTER\nENDMDL\nMODEL        2\nENDMDL\nEND\n"
    )


def test_removed_trailing_chain_cannot_borrow_selected_segment(tmp_path: Path) -> None:
    source = tmp_path / "input.pdb"
    output = tmp_path / "output.pdb"
    source.write_text(f"{atom('B')}\n{atom('A', 3)}\nTER\nEND\n")
    _filter_chains(source, ["B"], output)
    assert output.read_text() == f"{atom('B')}\nEND\n"
