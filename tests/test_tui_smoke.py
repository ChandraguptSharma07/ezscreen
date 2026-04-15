"""Smoke tests for the TUI application."""
from __future__ import annotations

import pytest

from ezscreen.tui.app import EzscreenApp
from ezscreen.tui.screens.home import HomeScreen
from ezscreen.tui.screens._placeholder import PlaceholderScreen


async def test_app_title() -> None:
    app = EzscreenApp()
    async with app.run_test() as pilot:
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
