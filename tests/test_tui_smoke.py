"""Smoke tests for the TUI application."""

from __future__ import annotations

from textual.widgets import RadioButton, Switch

from ezscreen.tui.app import EzscreenApp
from ezscreen.tui.screens._placeholder import PlaceholderScreen
from ezscreen.tui.screens.home import HomeScreen
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
