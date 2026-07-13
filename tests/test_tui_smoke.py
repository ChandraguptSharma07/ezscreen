"""Smoke tests for the TUI application."""

from __future__ import annotations

from pathlib import Path

from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Input,
    ListItem,
    ProgressBar,
    RadioButton,
    Select,
    Static,
    Switch,
)

from ezscreen.tui.app import EzscreenApp
from ezscreen.tui.screens._placeholder import PlaceholderScreen
from ezscreen.tui.screens.auth_setup import AuthScreen
from ezscreen.tui.screens.home import HomeScreen
from ezscreen.tui.screens.md_results import MDResultsScreen
from ezscreen.tui.screens.md_simulation import MDSimulationScreen
from ezscreen.tui.screens.run_wizard import RunWizardScreen
from ezscreen.tui.screens.team_runs import TeamRunsScreen


async def test_app_title() -> None:
    app = EzscreenApp()
    async with app.run_test():
        assert app.title == "ezscreen"


async def test_app_starts_in_required_account_setup_when_onboarding_incomplete(monkeypatch) -> None:
    from ezscreen import auth

    monkeypatch.setattr(auth, "onboarding_status", lambda: {"github_ok": False, "kaggle_ok": False, "complete": False})
    app = EzscreenApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, AuthScreen)
        assert app.screen.query_one("#btn-team-signin", Button)


async def test_home_screen_mounts_after_required_onboarding(monkeypatch) -> None:
    from ezscreen import auth

    monkeypatch.setattr(auth, "onboarding_status", lambda: {"github_ok": True, "kaggle_ok": True, "complete": True})
    app = EzscreenApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)
        assert app.screen.query_one("#action-md", ListItem).id == "action-md"

async def test_home_screen_has_single_team_action(monkeypatch) -> None:
    from ezscreen import auth

    monkeypatch.setattr(auth, "onboarding_status", lambda: {"github_ok": True, "kaggle_ok": True, "complete": True})
    app = EzscreenApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.screen.query_one("#action-team", ListItem).id == "action-team"
        assert not app.screen.query("#action-team-create")
        assert not app.screen.query("#action-team-join")
        assert not app.screen.query("#action-team-pause")

async def test_required_account_screen_can_log_out_without_leaving_screen(monkeypatch, tmp_path) -> None:
    from ezscreen import auth
    from ezscreen.team import client as team_client

    state_path = tmp_path / "team-client.json"
    monkeypatch.setattr(team_client, "LOCAL_STATE_PATH", state_path)
    team_client.save_local_state({
        "server_url": "http://team.invalid",
        "user_name": "Alice",
        "session_token": "session",
        "user_id": "gh_1",
        "github_login": "octocat",
        "team_id": "team_1",
    })
    monkeypatch.setattr(auth, "onboarding_status", lambda: {"github_ok": False, "kaggle_ok": False, "complete": False})
    monkeypatch.setattr(team_client.TeamClient, "logout", lambda self: {"ok": True})

    app = EzscreenApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert isinstance(app.screen, AuthScreen)
        await pilot.click("#btn-team-signout")
        await pilot.pause()
        assert isinstance(app.screen, AuthScreen)
        state = team_client.load_local_state()
        assert state == {"server_url": "http://team.invalid", "user_name": "Alice"}
        assert "Signed out" in app.screen.query_one("#github-status", Static).renderable


async def test_quit_with_q() -> None:
    app = EzscreenApp()
    async with app.run_test() as pilot:
        await pilot.press("q")
    # reaches here only if app exited cleanly


async def test_placeholder_navigates_back(monkeypatch) -> None:
    from ezscreen import auth

    monkeypatch.setattr(auth, "onboarding_status", lambda: {"github_ok": True, "kaggle_ok": True, "complete": True})
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


async def test_team_runs_screen_hides_server_and_repo_fields() -> None:
    app = EzscreenApp()
    screen = TeamRunsScreen()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(screen)
        await pilot.pause()

        assert screen.query_one("#team-service-text", Static)
        assert screen.query_one("#team-auth-text", Static)
        assert screen.query_one("#team-account-note", Static)
        assert screen.query_one("#btn-team-signin", Button)
        assert screen.query_one("#btn-team-auth-poll", Button)
        assert screen.query_one("#team-github-token", Input)
        assert screen.query_one("#btn-team-connect", Button)
        assert screen.query_one("#btn-team-disconnect", Button)
        assert not screen.query("#team-server-url")
        assert not screen.query("#team-repo")


def test_team_runs_sign_in_errors_are_actionable() -> None:
    from ezscreen.tui.screens.team_runs import format_sign_in_error

    assert "older code" in format_sign_in_error(RuntimeError('HTTP 404: {"error": "not_found"}'))
    assert "EZSCREEN_GITHUB_CLIENT_ID" in format_sign_in_error(
        RuntimeError("HTTP 400: EZSCREEN_GITHUB_CLIENT_ID is not configured")
    )


async def test_run_wizard_team_assignment_section_stays_hidden_without_team_toggle(monkeypatch) -> None:
    from ezscreen.team import client as team_client

    monkeypatch.setattr(team_client, "LOCAL_STATE_PATH", Path("C:/tmp/ezscreen-missing-team-state.json"))
    app = EzscreenApp()
    screen = RunWizardScreen()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(screen)
        await pilot.pause()

        section = screen.query_one("#team-assignment-section")
        assert section.display is False


async def test_run_wizard_team_toggle_hides_local_account_assignments(monkeypatch, tmp_path) -> None:
    import json

    from ezscreen.team import client as team_client

    state_path = tmp_path / "team-state.json"
    state_path.write_text(
        json.dumps({"server_url": "http://team.invalid", "user_id": "u1"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(team_client, "LOCAL_STATE_PATH", state_path)

    def fake_status(self, user_id):
        return {
            "team": {"team_id": "team_1"},
            "members": [
                {"member_id": "mem_a", "display_name": "Alice", "enabled": 1, "left_at": None},
                {"member_id": "mem_b", "display_name": "Bob", "enabled": 1, "left_at": None},
            ],
        }

    monkeypatch.setattr(team_client.TeamClient, "status", fake_status)
    app = EzscreenApp()
    screen = RunWizardScreen()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(screen)
        await pilot.pause()

        screen.query_one("#opt-team-runners", Switch).value = True
        await pilot.pause()

        assert screen.query_one("#acct-assignment-section").display is False
        assert screen.query_one("#team-assignment-section").display is True


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


async def test_home_md_simulation_action_opens_screen(monkeypatch) -> None:
    from ezscreen import auth

    monkeypatch.setattr(auth, "onboarding_status", lambda: {"github_ok": True, "kaggle_ok": True, "complete": True})
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
        assert app.screen.query_one("#md-leaderboard", DataTable)
        assert app.screen.query_one("#md-run-filter", Input)
        assert app.screen.query_one("#md-hit-filter", Input)


async def test_team_runs_screen_prioritizes_worker_actions_when_already_in_team(monkeypatch, tmp_path) -> None:
    import json

    from ezscreen.team import client as team_client

    state_path = tmp_path / "team-state.json"
    state_path.write_text(
        json.dumps({
            "server_url": "http://team.invalid",
            "user_id": "u1",
            "session_token": "session-token",
            "github_login": "alice",
            "user_name": "Alice",
            "team_id": "team_1",
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(team_client, "LOCAL_STATE_PATH", state_path)

    def fake_status(self, user_id):
        return {
            "team": {"team_id": "team_1"},
            "members": [
                {"member_id": "mem_a", "display_name": "Alice", "github_repo": "", "enabled": 1, "paused": 0, "left_at": None},
            ],
            "jobs": [],
        }

    monkeypatch.setattr(team_client.TeamClient, "status", fake_status)

    app = EzscreenApp()
    screen = TeamRunsScreen()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(screen)
        await pilot.pause()

        assert screen.query_one("#team-worker-actions").display is True
        assert screen.query_one("#team-setup-section").display is False
        assert screen.query_one("#btn-team-connect", Button)



async def test_md_results_screen_lists_jobs(monkeypatch, tmp_path) -> None:
    from ezscreen import checkpoint

    db_file = tmp_path / "checkpoints.db"
    monkeypatch.setattr(checkpoint, "DB_DIR", tmp_path)
    monkeypatch.setattr(checkpoint, "DB_PATH", db_file)
    checkpoint.init_db()
    checkpoint.create_run("ezs-md", {}, 10)
    out_dir = tmp_path / "md-out"
    out_dir.mkdir()
    (out_dir / "md_summary.json").write_text('{"platform": "CUDA", "simulated_ps": 10.0}')
    (out_dir / "md_metrics.json").write_text('{"frames": 2, "temperature": {"first": 298.0, "last": 300.0, "min": 298.0, "max": 300.0}}')
    checkpoint.create_md_run(
        md_run_id="ezs-md-md-lig1",
        run_id="ezs-md",
        ligand_id="lig1",
        preset="smoke",
        pose_source="unidock",
        accelerator="nvidiaTeslaT4",
        output_dir=out_dir,
    )
    checkpoint.mark_md_run_complete("ezs-md-md-lig1", out_dir, summary={"platform": "CUDA"})

    app = EzscreenApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.push_screen(MDResultsScreen("ezs-md"))
        await pilot.pause()

        assert app.screen.query_one("#md-results-table", DataTable)
        assert app.screen.query_one("#md-results-card", Static)
        assert app.screen.query_one("#btn-md-report", Button)




def test_checkpoint_partial_status_survives_for_results_runs(monkeypatch, tmp_path) -> None:
    from ezscreen import checkpoint

    db_file = tmp_path / "checkpoints.db"
    monkeypatch.setattr(checkpoint, "DB_DIR", tmp_path)
    monkeypatch.setattr(checkpoint, "DB_PATH", db_file)
    checkpoint.init_db()
    checkpoint.create_run("ezs-partial", {}, 10)
    checkpoint.mark_run_partial("ezs-partial")

    row = checkpoint.get_run("ezs-partial")
    assert row is not None
    assert row["status"] == "partial"


def test_md_results_action_targets_generate_local_viewers(tmp_path) -> None:
    out_dir = tmp_path / "md-out"
    out_dir.mkdir()
    (out_dir / "minimized.pdb").write_text("ATOM      1  N   GLY A   1      11.104  13.207   9.947  1.00 20.00           N\nEND\n", encoding="utf-8")
    (out_dir / "final.pdb").write_text("ATOM      1  N   GLY A   1      12.104  14.207   8.947  1.00 20.00           N\nEND\n", encoding="utf-8")
    (out_dir / "md_summary.json").write_text('{"run_id": "ezs-md", "ligand_id": "lig1"}', encoding="utf-8")
    (out_dir / "md_metrics.json").write_text('{"frames": 2}', encoding="utf-8")
    (out_dir / "md_report.html").write_text('<html></html>', encoding="utf-8")

    assert MDResultsScreen._resolve_action_target("btn-md-min", out_dir).name == "minimized_viewer.html"
    assert MDResultsScreen._resolve_action_target("btn-md-final", out_dir).name == "final_viewer.html"
    assert MDResultsScreen._resolve_action_target("btn-md-traj", out_dir).name == "trajectory_viewer.html"
