"""Smoke tests for the TUI application."""

from __future__ import annotations

from textual.widgets import (
    Checkbox,
    DataTable,
    Input,
    ListItem,
    ProgressBar,
    RadioButton,
    Select,
    Switch,
)

from ezscreen.tui.app import EzscreenApp
from ezscreen.tui.screens._placeholder import PlaceholderScreen
from ezscreen.tui.screens.home import HomeScreen
from ezscreen.tui.screens.md_simulation import MDSimulationScreen
from ezscreen.tui.screens.run_wizard import RunWizardScreen


async def test_app_title() -> None:
    app = EzscreenApp()
    async with app.run_test():
        assert app.title == "ezscreen"


async def test_home_screen_mounts() -> None:
    app = EzscreenApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)
        assert app.screen.query_one("#action-md", ListItem).id == "action-md"


async def test_quit_with_q() -> None:
    app = EzscreenApp()
    async with app.run_test() as pilot:
        await pilot.press("q")
    # reaches here only if app exited cleanly


async def test_placeholder_navigates_back() -> None:
    app = EzscreenApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(PlaceholderScreen("Test"))
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)
        assert app.screen.query_one("#action-md", ListItem).id == "action-md"


async def test_run_wizard_hetero_presets_and_category_actions() -> None:
    app = EzscreenApp()
    screen = RunWizardScreen()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(screen)
        await pilot.pause()

        screen._hetero = [
            {
                "resname": "HOH",
                "chain": "A",
                "resseq": 10,
                "n_atoms": 1,
                "category": "water",
                "keep_default": False,
            },
            {
                "resname": "LIG",
                "chain": "A",
                "resseq": 20,
                "n_atoms": 8,
                "category": "ligand",
                "keep_default": False,
            },
            {
                "resname": "HEM",
                "chain": "A",
                "resseq": 30,
                "n_atoms": 43,
                "category": "cofactor",
                "keep_default": True,
            },
            {
                "resname": "ZN",
                "chain": "A",
                "resseq": 40,
                "n_atoms": 1,
                "category": "metal",
                "keep_default": False,
            },
        ]
        screen._populate_hetero()
        await pilot.pause()

        assert screen._hetero_keep == {2, 3}

        screen._apply_hetero_preset("crystal")
        await pilot.pause()
        assert screen._hetero_keep == {1, 2, 3}

        screen._set_hetero_category("water", True)
        await pilot.pause()
        assert screen._hetero_keep == {0, 1, 2, 3}

        screen._set_hetero_category("water", False)
        screen._step = 1
        assert screen._validate_step() is None
        assert [h["resname"] for h in screen._ctx["keep_hetero"]] == [
            "LIG",
            "HEM",
            "ZN",
        ]


async def test_run_wizard_records_receptor_isomer_choice() -> None:
    app = EzscreenApp()
    screen = RunWizardScreen()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(screen)
        await pilot.pause()

        screen._step = 1
        assert screen._validate_step() is None
        assert screen._ctx["receptor_isomers"] == "off"

        screen.query_one("#opt-receptor-isomers", Switch).value = True
        screen.query_one("#rb-rec-isomer-variants", RadioButton).value = True
        await pilot.pause()

        assert screen._validate_step() is None
        assert screen._ctx["receptor_isomers"] == "variants"


async def test_run_wizard_records_ensemble_receptors(tmp_path) -> None:
    conf = tmp_path / "conf_a.pdbqt"
    conf.write_text("RECEPTOR\n")
    app = EzscreenApp()
    screen = RunWizardScreen()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(screen)
        await pilot.pause()

        screen._step = 1
        screen.query_one("#rb-ensemble-manual", RadioButton).value = True
        await pilot.pause()
        screen.query_one("#ensemble-receptors-input", Input).value = str(conf)

        assert screen._validate_step() is None
        assert screen._ctx["receptor_isomers"] == "off"
        assert screen._ctx["ensemble_mode"] == "manual"
        assert screen._ctx["ensemble_manual_paths"] == [conf]
        assert screen._ctx["ensemble_receptor_paths"] == [conf]

        screen.query_one("#opt-receptor-isomers", Switch).value = True
        assert screen._validate_step() == (
            "Use ensemble conformations or receptor protonation variants, not both."
        )

        screen.query_one("#opt-receptor-isomers", Switch).value = False
        screen._ctx["ensemble_receptor_paths"] = [conf]
        screen._step = 5
        screen.query_one("#opt-consensus", Switch).value = True
        assert (
            screen._validate_step()
            == "Ensemble plus consensus ranking is not supported yet."
        )


async def test_run_wizard_records_auto_ensemble_settings() -> None:
    app = EzscreenApp()
    screen = RunWizardScreen()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(screen)
        await pilot.pause()

        screen._step = 1
        screen.query_one("#rb-ensemble-auto", RadioButton).value = True
        screen.query_one("#ensemble-max-conformers", Input).value = "4"
        await pilot.pause()

        assert screen._validate_step() is None
        assert screen._ctx["ensemble_mode"] == "auto"
        assert screen._ctx["ensemble_max_conformers"] == 4
        assert screen._ctx["ensemble_method"] == "pdb_then_nma"
        assert screen._ctx["ensemble_manual_paths"] == []

        screen.query_one("#ensemble-max-conformers", Input).value = "9"
        assert screen._validate_step() == "Max conformers must be between 1 and 8."


async def test_run_wizard_records_selected_cocrystal_ligand() -> None:
    app = EzscreenApp()
    screen = RunWizardScreen()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(screen)
        await pilot.pause()

        ligands = [
            {
                "resname": "PTR",
                "chain": "A",
                "resseq": 393,
                "coords": [(0.0, 0.0, 0.0)] * 12,
                "centroid": [0.0, 0.0, 0.0],
            },
            {
                "resname": "1N1",
                "chain": "A",
                "resseq": 501,
                "coords": [(1.0, 1.0, 1.0)] * 33,
                "centroid": [1.0, 1.0, 1.0],
            },
        ]
        screen._show_rec_result(["A"], False, None, ligands)
        await pilot.pause()

        screen.query_one("#rb-cocrystal", RadioButton).value = True
        await pilot.pause()

        select = screen.query_one("#cocrystal-select", Select)
        select.value = screen._cocrystal_option_value(ligands[1])

        screen._step = 2
        assert screen._validate_step() is None
        assert screen._ctx["site_method"] == "Co-crystal ligand"
        assert screen._ctx["site_details"]["type"] == "cocrystal"
        assert screen._ctx["site_details"]["ligands"] == [ligands[1]]
        assert screen._ctx["site_details"]["available_ligands"] == ligands


async def test_run_wizard_filters_cocrystal_choices_by_selected_chain() -> None:
    app = EzscreenApp()
    screen = RunWizardScreen()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(screen)
        await pilot.pause()

        ligands = [
            {
                "resname": "1N1",
                "chain": "A",
                "resseq": 501,
                "coords": [(1.0, 1.0, 1.0)] * 33,
                "centroid": [1.0, 1.0, 1.0],
            },
            {
                "resname": "1N1",
                "chain": "B",
                "resseq": 502,
                "coords": [(2.0, 2.0, 2.0)] * 33,
                "centroid": [2.0, 2.0, 2.0],
            },
        ]
        screen._show_rec_result(["A", "B"], False, None, ligands)
        await pilot.pause()

        screen.query_one("#chain-B", Checkbox).value = False
        await pilot.pause()

        assert screen._visible_cocrystal_ligands() == [ligands[0]]
        assert screen._selected_cocrystal_ligand() == ligands[0]


async def test_run_wizard_prepares_ensemble_receptor_files(tmp_path) -> None:
    copied = tmp_path / "copy_me.pdbqt"
    copied.write_text("PDBQT\n")
    source_pdb = tmp_path / "prep_me.pdb"
    source_pdb.write_text("ATOM\nEND\n")
    prepared = tmp_path / "prepared.pdbqt"
    prepared.write_text("PREPARED\n")

    class FakeRecPrep:
        @staticmethod
        def prep_receptor(**kwargs):
            return {"pdbqt_path": prepared}

    screen = RunWizardScreen()
    paths = screen._prepare_ensemble_receptors(
        [copied, source_pdb],
        rec_prep=FakeRecPrep,
        work_dir=tmp_path / "run",
        chains=["A"],
        ph=7.4,
        keep_hetero=None,
        primary_pdbqt=tmp_path / "primary.pdbqt",
    )

    assert len(paths) == 3
    assert paths[1].name.startswith("receptor_conformer_001_copy_me")
    assert paths[1].read_text() == "PDBQT\n"
    assert paths[2].name.startswith("receptor_conformer_002_prep_me")
    assert paths[2].read_text() == "PREPARED\n"


async def test_run_wizard_diffdock_uses_native_confidence() -> None:
    app = EzscreenApp()
    screen = RunWizardScreen()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(screen)
        await pilot.pause()

        screen.query_one("#engine-diffdock", RadioButton).value = True
        screen._update_scoring_options()
        await pilot.pause()

        assert screen._selected_engine() == "diffdock"
        assert str(screen.query_one("#opt-scoring").value) == "diffdock_confidence"


async def test_run_wizard_records_consensus_toggle() -> None:
    app = EzscreenApp()
    screen = RunWizardScreen()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(screen)
        await pilot.pause()

        screen._step = 5
        screen.query_one("#opt-consensus", Switch).value = True
        assert screen._validate_step() is None
        assert screen._ctx["engine"] == "unidock"
        assert screen._ctx["consensus_enabled"] is True

        screen.query_one("#engine-diffdock", RadioButton).value = True
        screen._update_scoring_options()
        screen._update_consensus_visibility()
        await pilot.pause()

        assert screen.query_one("#opt-consensus", Switch).value is False


async def test_home_md_simulation_action_opens_screen() -> None:
    app = EzscreenApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)
        app.screen.action_open_md()
        await pilot.pause()
        assert isinstance(app.screen, MDSimulationScreen)


async def test_md_simulation_screen_has_browser_controls() -> None:
    app = EzscreenApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(MDSimulationScreen())
        await pilot.pause()

        assert app.screen.query_one("#md-runs", DataTable)
        assert app.screen.query_one("#md-progress", ProgressBar)
        assert app.screen.query_one("#md-account", Select)
        assert app.screen.query_one("#md-run-filter", Input)
