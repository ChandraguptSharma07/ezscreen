from __future__ import annotations

import re
import secrets
from collections import Counter, defaultdict
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    ContentSwitcher,
    Footer,
    Header,
    Input,
    Label,
    RadioButton,
    RadioSet,
    RichLog,
    Select,
    Static,
    Switch,
)

from ezscreen.backends import engines
from ezscreen.tui.widgets.breadcrumb import Breadcrumb

_FORCE_FIELD_OPTIONS: list[tuple[str, str]] = [
    ("MMFF94 â€” general purpose", "MMFF94"),
    ("MMFF94s â€” planar amides/aromatics", "MMFF94s"),
    ("UFF â€” broad element coverage", "UFF"),
]
_RECEPTOR_ISOMER_MODES = {"off", "single", "variants"}
_ENSEMBLE_MODES = {"off", "manual", "auto"}
_AUTO_ENSEMBLE_METHOD = "pdb_then_nma"
_AUTO_ENSEMBLE_DEFAULT_MAX = 3
_AUTO_ENSEMBLE_HARD_MAX = 8

_HETERO_CATEGORY_ORDER = ("cofactor", "metal", "ligand", "water", "additive", "other")
_HETERO_CATEGORY_LABELS = {
    "water": "Waters",
    "metal": "Metals",
    "cofactor": "Cofactors",
    "ligand": "Ligands",
    "additive": "Additives",
    "other": "Other heteroatoms",
}
_HETERO_NOISY_CATEGORIES = {"water", "additive"}


_STEPS = [
    "step-receptor",
    "step-recprep",
    "step-site",
    "step-ligands",
    "step-prep",
    "step-engine",
    "step-compute",
    "step-confirm",
]

_STEP_LABELS = [
    "Step 1 of 8 â€” Receptor & Chains",
    "Step 2 of 8 â€” Receptor Prep",
    "Step 3 of 8 â€” Binding Site",
    "Step 4 of 8 â€” Ligand Library",
    "Step 5 of 8 â€” Ligand Prep",
    "Step 6 of 8 â€” Engine & Scoring",
    "Step 7 of 8 â€” Compute",
    "Step 8 of 8 â€” Confirm & Submit",
]


class RunWizardScreen(Screen):
    """Multi-step wizard for configuring and submitting a virtual screening run."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def __init__(self) -> None:
        super().__init__()
        self._step = 0
        self._ctx: dict = {}
        self._chains: list[str] = []
        self._hetero: list[dict] = []
        self._hetero_keep: set[int] = set()
        self._hetero_expanded: set[str] = set()
        self._refreshing_hetero_controls = False
        self._cocrystal_ligands: list[dict] = []
        self._pockets: list[dict] = []
        self._submitted = False

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        yield Breadcrumb(["Home", "New Run"])
        with Vertical(id="wizard-body"):
            yield Static("", id="step-indicator")
            with ContentSwitcher(initial="step-receptor", id="wizard-switcher"):
                # â”€â”€ Step 1: Receptor & Chains â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                with VerticalScroll(id="step-receptor", classes="wizard-step"):
                    yield Label(
                        "PDB ID (e.g. 7L11), AF:UniProt (e.g. AF:P00533), or path to a .pdb file",
                        classes="form-label",
                    )
                    with Horizontal(classes="form-row"):
                        yield Input(
                            placeholder="7L11  or  AF:P00533  or  /path/to/receptor.pdb",
                            id="rec-input",
                        )
                        yield Button("Validate", id="btn-rec-validate")
                    yield Static("", id="rec-status", classes="form-status")
                    with Vertical(id="af-warning-box"):
                        yield Static("", id="af-warning")
                    with Vertical(id="chain-section"):
                        yield Label("Select chains to include", classes="form-label")
                        yield Vertical(id="chain-list")

                # â”€â”€ Step 2: Receptor Prep â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                with Vertical(id="step-recprep", classes="wizard-step"):
                    yield Label(
                        "Heteroatoms (waters, ions, cofactors, ligands)",
                        classes="form-section",
                    )
                    yield Static(
                        "[#6e7681]Choose the receptor cleanup policy, then adjust any group "
                        "that needs special handling. Validate the receptor first to populate "
                        "this list.[/#6e7681]"
                    )
                    with Horizontal(classes="hetero-presets"):
                        yield Button("Recommended", id="hetero-preset-recommended")
                        yield Button("Apo receptor", id="hetero-preset-apo")
                        yield Button("Crystal complex", id="hetero-preset-crystal")
                    yield VerticalScroll(id="hetero-list")
                    yield Static("", id="hetero-status", classes="form-status")
                    yield Label("Receptor protonation variants", classes="form-section")
                    yield Static(
                        "[#6e7681]Use one corrected receptor state by default, or dock against "
                        "multiple binding-site protonation variants when needed.[/#6e7681]"
                    )
                    yield Switch(id="opt-receptor-isomers", value=False)
                    with Vertical(id="receptor-isomer-mode-box"):
                        with RadioSet(id="opt-receptor-isomer-mode"):
                            yield RadioButton(
                                "A - corrected single protonation state",
                                id="rb-rec-isomer-single",
                                value=True,
                            )
                            yield RadioButton(
                                "B - multiple binding-site variants",
                                id="rb-rec-isomer-variants",
                            )
                    yield Label(
                        "Ensemble receptor conformations", classes="form-section"
                    )
                    yield Static(
                        "[#6e7681]Dock one receptor, provide conformers, or let ezscreen "
                        "generate a small ensemble from PDB structures with normal-mode fallback.[/#6e7681]"
                    )
                    with RadioSet(id="ensemble-mode"):
                        yield RadioButton(
                            "Single receptor",
                            id="rb-ensemble-off",
                            value=True,
                        )
                        yield RadioButton(
                            "Use my conformers",
                            id="rb-ensemble-manual",
                        )
                        yield RadioButton(
                            "Generate automatically",
                            id="rb-ensemble-auto",
                        )
                    with Vertical(id="ensemble-manual-box"):
                        yield Input(
                            placeholder="/path/conf1.pdbqt, /path/conf2.pdb",
                            id="ensemble-receptors-input",
                        )
                    with Vertical(id="ensemble-auto-box"):
                        yield Static(
                            "[#6e7681]Method: PDB + normal-mode fallback[/#6e7681]"
                        )
                        yield Input(
                            value=str(_AUTO_ENSEMBLE_DEFAULT_MAX),
                            placeholder="Max conformers (1-8)",
                            id="ensemble-max-conformers",
                        )
                    yield Static("", id="ensemble-status", classes="form-status")
                # â”€â”€ Step 3: Binding Site â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                with VerticalScroll(id="step-site", classes="wizard-step"):
                    yield Label("Binding site method", classes="form-label")
                    with RadioSet(id="site-method"):
                        yield RadioButton("Co-crystal ligand", id="rb-cocrystal")
                        yield RadioButton("Active site residues", id="rb-residues")
                        yield RadioButton(
                            "P2Rank pocket prediction", id="rb-p2rank", value=True
                        )
                        yield RadioButton("Blind docking  \u26a0", id="rb-blind")
                        yield RadioButton("Enter coordinates manually", id="rb-manual")
                    with Vertical(id="sub-cocrystal", classes="site-sub"):
                        yield Static("", id="cocrystal-info")
                        yield Select(
                            [],
                            id="cocrystal-select",
                            prompt="Choose reference ligand",
                        )
                    with Vertical(id="sub-residues", classes="site-sub"):
                        yield Label(
                            "Residue numbers (comma-separated, e.g. 42,45,78)",
                            classes="form-label",
                        )
                        yield Input(placeholder="42,45,78", id="residues-input")
                    with Vertical(id="sub-p2rank", classes="site-sub"):
                        yield Static("", id="p2rank-status")
                        yield Vertical(id="p2rank-picks")
                    with Vertical(id="sub-blind", classes="site-sub"):
                        yield Static(
                            "[\u26a0] Blind docking scans the entire protein â€” "
                            "slow and noisy. Only use if no binding pocket is known.",
                            id="blind-warning",
                        )
                    with Vertical(id="sub-manual", classes="site-sub"):
                        yield Label("Box center (Ã…)", classes="form-label")
                        with Horizontal(classes="form-row"):
                            yield Input(placeholder="X", id="manual-cx")
                            yield Input(placeholder="Y", id="manual-cy")
                            yield Input(placeholder="Z", id="manual-cz")
                        yield Label("Box size (Ã…)", classes="form-label")
                        with Horizontal(classes="form-row"):
                            yield Input(placeholder="X", id="manual-sx")
                            yield Input(placeholder="Y", id="manual-sy")
                            yield Input(placeholder="Z", id="manual-sz")

                # â”€â”€ Step 3: Ligand Library â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                with Vertical(id="step-ligands", classes="wizard-step"):
                    yield Label(
                        "Ligand library  (.sdf, .smi, .smiles, or .ism file)",
                        classes="form-label",
                    )
                    with Horizontal(classes="form-row"):
                        yield Input(placeholder="/path/to/library.sdf", id="lig-input")
                        yield Button("Download library", id="btn-lib-browser")
                    yield Static("", id="lig-status", classes="form-status")

                # -- Step 4: Ligand Prep --
                with VerticalScroll(id="step-prep", classes="wizard-step"):
                    yield Label("ADMET pre-filter", classes="form-section")
                    yield Static(
                        "[#6e7681]Remove obvious drug-like failures before docking"
                        " (recommended)[/#6e7681]"
                    )
                    yield Switch(id="opt-admet", value=True)
                    yield Label("Conformer force field", classes="form-section")
                    yield Static(
                        "[#6e7681]Defaults to your settings choice; change it for this run only.[/#6e7681]"
                    )
                    yield Select(
                        _FORCE_FIELD_OPTIONS, id="opt-force-field", allow_blank=False
                    )
                    yield Label("Enumerate ligand variants", classes="form-section")
                    yield Static(
                        "[#6e7681]Protonation/tautomer/stereo/ring variants via Gypsum-DL on Kaggle."
                        " Which variant types & the cap come from Settings.[/#6e7681]"
                    )
                    yield Switch(id="opt-enumerate")

                # -- Step 5: Engine & Scoring --
                with VerticalScroll(id="step-engine", classes="wizard-step"):
                    # Scoring-function selector + CNN-rescore toggle land here next
                    # (Phase 30 Features D/E). The engine selector is below.
                    yield Label(
                        "Docking engine", classes="form-section", id="lbl-engine"
                    )
                    with RadioSet(id="opt-engine"):
                        for _i, _eng in enumerate(engines.dockable_engines()):
                            yield RadioButton(
                                _eng.label, id=f"engine-{_eng.key}", value=(_i == 0)
                            )
                    yield Static("", id="engine-note", classes="form-status")
                    yield Label(
                        "Scoring function", classes="form-section", id="lbl-scoring"
                    )
                    yield Select(
                        [("Vina", "vina")], id="opt-scoring", allow_blank=False
                    )
                    yield Label("Consensus", classes="form-section", id="lbl-consensus")
                    yield Static(
                        "[#6e7681]Run UniDock + GNINA and rank by mean normalized rank.[/#6e7681]",
                        id="consensus-note",
                        classes="form-status",
                    )
                    yield Switch(id="opt-consensus", value=False)
                    yield Label("Search depth", classes="form-section", id="lbl-depth")
                    with RadioSet(id="opt-depth"):
                        yield RadioButton(
                            "Fast       \u2014 triage only, misses ~25% best poses",
                            id="rb-fast",
                        )
                        yield RadioButton(
                            "Balanced \u2605 \u2014 standard VS, good for rigid pockets",
                            id="rb-balanced",
                            value=True,
                        )
                        yield RadioButton(
                            "Thorough   \u2014 flexible ligands, induced-fit targets",
                            id="rb-thorough",
                        )

                # -- Step 6: Compute --
                with VerticalScroll(id="step-compute", classes="wizard-step"):
                    yield Label("Compute backend", classes="form-section")
                    yield Static(
                        "[#6e7681]Run locally on CPU (AutoDock Vina) â€” no Kaggle account needed."
                        " Slower than GPU for large libraries.[/#6e7681]"
                    )
                    yield Switch(id="opt-local", value=False)
                    yield Label("Use team runners", id="lbl-team-runners", classes="form-section")
                    yield Static(
                        "[#6e7681]Split this Kaggle run across enabled team workers.[/#6e7681]",
                        id="team-runners-note",
                        classes="form-status",
                    )
                    yield Switch(id="opt-team-runners", value=False)
                    yield Label(
                        "Exhaustiveness (Vina)",
                        id="lbl-exhaustiveness",
                        classes="form-label",
                    )
                    yield Input(
                        id="opt-exhaustiveness", placeholder="4 (from settings)"
                    )
                    yield Label("Kaggle GPU", id="lbl-gpu", classes="form-section")
                    with RadioSet(id="opt-gpu-type"):
                        yield RadioButton(
                            "P100 (16 GB) â˜… â€” standard, widest availability",
                            id="rb-p100",
                            value=True,
                        )
                        yield RadioButton(
                            "T4 Ã— 2 (32 GB) â€” more memory, uses both GPUs",
                            id="rb-t4x2",
                        )
                    with Vertical(id="compute-assignment-sections"):
                        yield Vertical(id="acct-assignment-section")
                        yield Vertical(id="team-assignment-section")

                # -- Step 7: Confirm & Submit --
                with Vertical(id="step-confirm", classes="wizard-step"):
                    yield Label("Ready to submit", classes="form-section")
                    yield Static("", id="confirm-summary", classes="confirm-summary")
                    yield Label("Output", classes="form-section")
                    yield RichLog(id="run-log", highlight=True, markup=True)

            with Horizontal(id="wizard-nav"):
                yield Button("\u2190 Back", id="btn-back")
                yield Button("Next \u2192", id="btn-next", variant="primary")

        yield Footer()

    # ------------------------------------------------------------------
    # Mount
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        self._update_ui()
        self.query_one("#af-warning-box").display = False
        self.query_one("#chain-section").display = False
        for sid in (
            "sub-cocrystal",
            "sub-residues",
            "sub-p2rank",
            "sub-blind",
            "sub-manual",
        ):
            self.query_one(f"#{sid}").display = False
        self.query_one("#lbl-exhaustiveness").display = False
        self.query_one("#receptor-isomer-mode-box").display = False
        self.query_one("#ensemble-manual-box").display = False
        self.query_one("#ensemble-auto-box").display = False
        self.query_one("#opt-exhaustiveness").display = False
        self._update_team_runner_visibility()
        self.query_one("#lbl-gpu").display = True
        self.query_one("#opt-gpu-type").display = True
        self._load_defaults()
        self._update_engine_note()
        self._update_scoring_options()
        self._update_consensus_visibility()
        self._update_compute_assignment_visibility()

    def _load_defaults(self) -> None:
        try:
            from ezscreen import config

            cfg = config.load()
            admet = cfg.get("run", {}).get("admet_pre_filter", True)
            prep_cfg = cfg.get("prep", {})
            self.query_one("#opt-admet", Switch).value = bool(admet)
            ff = str(prep_cfg.get("force_field", "MMFF94"))
            if ff in {v for _, v in _FORCE_FIELD_OPTIONS}:
                self.query_one("#opt-force-field", Select).value = ff
            self.query_one("#opt-enumerate", Switch).value = bool(
                prep_cfg.get("enumerate_enabled", False)
            )
            receptor_isomers = str(prep_cfg.get("receptor_isomers", "off"))
            if receptor_isomers not in _RECEPTOR_ISOMER_MODES:
                receptor_isomers = "off"
            self.query_one("#opt-receptor-isomers", Switch).value = (
                receptor_isomers != "off"
            )
            self.query_one("#receptor-isomer-mode-box").display = (
                receptor_isomers != "off"
            )
            if receptor_isomers == "variants":
                self.query_one("#rb-rec-isomer-variants", RadioButton).value = True
            else:
                self.query_one("#rb-rec-isomer-single", RadioButton).value = True
        except Exception:
            pass

    def _populate_account_assignment(self) -> None:
        from ezscreen import auth

        accounts = auth.get_all_kaggle_accounts()
        section = self.query_one("#acct-assignment-section", Vertical)
        section.remove_children()
        if len(accounts) <= 1:
            section.display = False
            return

        section.display = True
        section.mount(
            Label("Account notebook assignment (Kaggle)", classes="form-section")
        )
        section.mount(
            Static(
                "[#6e7681]Notebooks per account (leave blank for 1 per account).[/#6e7681]"
            )
        )
        for acct in accounts:
            name = acct["name"]
            safe = name.replace(" ", "-").lower()
            section.mount(
                Horizontal(
                    Label(f"  {name}", classes="acct-row-label"),
                    Input(
                        id=f"acct-shards-{safe}",
                        placeholder="1",
                        classes="acct-row-input",
                    ),
                    classes="acct-row",
                )
            )

    def _team_member_input_id(self, member_id: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", member_id)
        return f"team-shards-{safe}"

    def _populate_team_member_assignment(self) -> None:
        section = self.query_one("#team-assignment-section", Vertical)
        section.remove_children()
        if self.query_one("#opt-local", Switch).value:
            section.display = False
            return
        section.display = True
        section.mount(Label("Team runner shard assignment", classes="form-section"))
        try:
            from ezscreen.team.client import TeamClient, load_local_state

            state = load_local_state()
            if not state.get("server_url") or not state.get("user_id"):
                section.mount(
                    Static(
                        "[#6e7681]No team runner is connected. Open Team to connect or resume a worker before using team runners.[/#6e7681]"
                    )
                )
                return
            status = TeamClient(state["server_url"]).status(state["user_id"])
            members = [
                member
                for member in status.get("members", [])
                if member.get("enabled") and not member.get("left_at") and (member.get("github_repo") or "").strip()
            ]
        except Exception as exc:
            section.mount(Static(f"[#f85149]Team status unavailable: {exc}[/#f85149]"))
            return
        if not members:
            section.mount(Static("[#6e7681]No connected team workers are available. Open Team to connect or resume a worker.[/#6e7681]"))
            return
        section.mount(
            Static(
                "[#6e7681]Ligand shards per team worker (blank = auto, 0 = exclude). Receptor and notebook are sent to each assigned worker.[/#6e7681]"
            )
        )
        for member in members:
            name = member.get("display_name") or member.get("github_repo") or member.get("member_id", "")
            state = "paused" if member.get("paused") else "enabled"
            if member.get("active_worker_job_id"):
                state = "busy"
            section.mount(
                Horizontal(
                    Label(f"  {name} ({state})", classes="acct-row-label"),
                    Input(
                        id=self._team_member_input_id(member["member_id"]),
                        placeholder="auto",
                        classes="acct-row-input",
                    ),
                    classes="acct-row",
                )
            )

    def _update_compute_assignment_visibility(self) -> None:
        local_on = self.query_one("#opt-local", Switch).value
        team_on = (not local_on) and self.query_one("#opt-team-runners", Switch).value
        acct_section = self.query_one("#acct-assignment-section", Vertical)
        team_section = self.query_one("#team-assignment-section", Vertical)
        if local_on:
            acct_section.display = False
            team_section.display = False
            return
        if team_on:
            acct_section.display = False
            self._populate_team_member_assignment()
            team_section.display = True
            return
        self._populate_account_assignment()
        acct_section.display = bool(acct_section.children)
        team_section.display = False

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "opt-receptor-isomers":
            self.query_one("#receptor-isomer-mode-box").display = event.value
        elif event.switch.id == "opt-local":
            local_on = event.value
            self.query_one("#lbl-exhaustiveness").display = local_on
            self.query_one("#opt-exhaustiveness").display = local_on
            self.query_one("#lbl-gpu").display = not local_on
            self.query_one("#opt-gpu-type").display = not local_on
            for selector in ("#lbl-team-runners", "#team-runners-note", "#opt-team-runners"):
                self.query_one(selector).display = (not local_on) and self._has_local_team_state()
            if local_on:
                self.query_one("#opt-team-runners", Switch).value = False
            self.query_one("#lbl-depth").display = not local_on
            self.query_one("#opt-depth").display = not local_on
            self.query_one("#lbl-engine").display = not local_on
            self.query_one("#opt-engine").display = not local_on
            self.query_one("#engine-note").display = not local_on
            self.query_one("#lbl-scoring").display = not local_on
            self.query_one("#opt-scoring").display = not local_on
            self._update_consensus_visibility(local_on=local_on)
            self._update_compute_assignment_visibility()
        elif event.switch.id == "opt-team-runners":
            self._update_compute_assignment_visibility()

    def _has_local_team_state(self) -> bool:
        try:
            from ezscreen.team.client import load_local_state

            state = load_local_state()
            return bool(state.get("server_url") and state.get("user_id"))
        except Exception:
            return False

    def _update_team_runner_visibility(self) -> None:
        visible = self._has_local_team_state()
        for selector in ("#lbl-team-runners", "#team-runners-note", "#opt-team-runners"):
            self.query_one(selector).display = visible
        if not visible:
            self.query_one("#opt-team-runners", Switch).value = False
        self._update_compute_assignment_visibility()
    def _update_ui(self) -> None:
        self.query_one("#step-indicator", Static).update(
            f"[#6e7681]{_STEP_LABELS[self._step]}[/#6e7681]"
        )
        self.query_one("#wizard-switcher", ContentSwitcher).current = _STEPS[self._step]
        is_last = self._step == len(_STEPS) - 1
        self.query_one("#btn-next", Button).label = (
            "Submit" if is_last else "Next \u2192"
        )
        self.query_one("#btn-back", Button).disabled = (
            self._step == 0 or self._submitted
        )
        self.query_one("#btn-next", Button).disabled = self._submitted

    # ------------------------------------------------------------------
    # Button / RadioSet events
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        dispatch = {
            "btn-rec-validate": self._start_rec_validate,
            "btn-lib-browser": self._open_lib_browser,
            "btn-back": self._go_back,
            "btn-next": self._go_next,
        }
        if fn := dispatch.get(event.button.id):
            fn()
            return

        button_id = event.button.id or ""
        if button_id.startswith("hetero-preset-"):
            self._apply_hetero_preset(button_id.removeprefix("hetero-preset-"))
        elif button_id.startswith("hetcat-keep-"):
            self._set_hetero_category(button_id.removeprefix("hetcat-keep-"), True)
        elif button_id.startswith("hetcat-remove-"):
            self._set_hetero_category(button_id.removeprefix("hetcat-remove-"), False)
        elif button_id.startswith("hetcat-toggle-"):
            self._toggle_hetero_category(button_id.removeprefix("hetcat-toggle-"))

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if self._refreshing_hetero_controls:
            return
        checkbox_id = event.checkbox.id or ""
        if checkbox_id.startswith("chain-"):
            self._refresh_cocrystal_selector()
            return
        if not checkbox_id.startswith("het-"):
            return
        try:
            idx = int(checkbox_id.removeprefix("het-"))
        except ValueError:
            return
        if event.value:
            self._hetero_keep.add(idx)
        else:
            self._hetero_keep.discard(idx)
        self._update_hetero_summary()

    def _open_lib_browser(self) -> None:
        from ezscreen.tui.screens.library_browser import LibraryBrowserScreen

        def _on_dismiss(path: str | None) -> None:
            if path:
                self.query_one("#lig-input", Input).value = path

        self.app.push_screen(LibraryBrowserScreen(), _on_dismiss)

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id == "site-method":
            label = str(event.pressed.label) if event.pressed else ""
            self._show_site_sub(label)
        elif event.radio_set.id == "opt-engine":
            self._update_engine_note()
            self._update_scoring_options()
            self._update_consensus_visibility()
        elif event.radio_set.id == "ensemble-mode":
            self._update_ensemble_mode_visibility()

    def _selected_receptor_isomer_mode(self) -> str:
        if not self.query_one("#opt-receptor-isomers", Switch).value:
            return "off"
        btn = self.query_one("#opt-receptor-isomer-mode", RadioSet).pressed_button
        if btn and btn.id == "rb-rec-isomer-variants":
            return "variants"
        return "single"

    def _selected_ensemble_mode(self) -> str:
        btn = self.query_one("#ensemble-mode", RadioSet).pressed_button
        if btn and btn.id == "rb-ensemble-manual":
            return "manual"
        if btn and btn.id == "rb-ensemble-auto":
            return "auto"
        return "off"

    def _update_ensemble_mode_visibility(self) -> None:
        mode = self._selected_ensemble_mode()
        self.query_one("#ensemble-manual-box").display = mode == "manual"
        self.query_one("#ensemble-auto-box").display = mode == "auto"

    def _auto_ensemble_max_conformers(self) -> int:
        raw = self.query_one("#ensemble-max-conformers", Input).value.strip()
        if not raw:
            return _AUTO_ENSEMBLE_DEFAULT_MAX
        value = int(raw)
        if value < 1 or value > _AUTO_ENSEMBLE_HARD_MAX:
            raise ValueError(
                f"Max conformers must be between 1 and {_AUTO_ENSEMBLE_HARD_MAX}."
            )
        return value

    def _ensemble_receptor_paths(self) -> list[Path]:
        raw = self.query_one("#ensemble-receptors-input", Input).value.strip()
        if not raw:
            return []
        parts = [p.strip() for p in re.split(r"[\n,;]+", raw) if p.strip()]
        paths: list[Path] = []
        seen: set[Path] = set()
        for part in parts:
            path = Path(part).expanduser()
            if path in seen:
                continue
            seen.add(path)
            paths.append(path)
        return paths

    def _update_consensus_visibility(self, local_on: bool | None = None) -> None:
        if local_on is None:
            try:
                local_on = self.query_one("#opt-local", Switch).value
            except Exception:
                local_on = False
        visible = (not local_on) and self._selected_engine() in {"unidock", "gnina"}
        for selector in ("#lbl-consensus", "#consensus-note", "#opt-consensus"):
            self.query_one(selector).display = visible
        if not visible:
            self.query_one("#opt-consensus", Switch).value = False

    def _selected_engine(self) -> str:
        btn = self.query_one("#opt-engine", RadioSet).pressed_button
        return btn.id.removeprefix("engine-") if btn and btn.id else "unidock"

    def _update_engine_note(self) -> None:
        prof = engines.get(self._selected_engine())
        self.query_one("#engine-note", Static).update(f"[#6e7681]{prof.note}[/#6e7681]")

    def _update_scoring_options(self) -> None:
        key = self._selected_engine()
        fns = engines.scoring_functions(key)
        sel = self.query_one("#opt-scoring", Select)
        current = sel.value if sel.value != Select.BLANK else None
        if not fns:
            sel.set_options([("Native confidence", "diffdock_confidence")])
            sel.value = "diffdock_confidence"
            return
        sel.set_options([(f.capitalize(), f) for f in fns])
        sel.value = (
            current if current in fns else (engines.default_scoring(key) or fns[0])
        )

    def _show_site_sub(self, label: str) -> None:
        for sid in (
            "sub-cocrystal",
            "sub-residues",
            "sub-p2rank",
            "sub-blind",
            "sub-manual",
        ):
            self.query_one(f"#{sid}").display = False
        if "Co-crystal" in label:
            self.query_one("#sub-cocrystal").display = True
        elif "residues" in label:
            self.query_one("#sub-residues").display = True
        elif "P2Rank" in label:
            self.query_one("#sub-p2rank").display = True
            if not self._pockets and self._ctx.get("pdb_path"):
                self._start_p2rank()
        elif "Blind" in label or "\u26a0" in label:
            self.query_one("#sub-blind").display = True
        elif "manually" in label:
            self.query_one("#sub-manual").display = True

    def _cocrystal_option_value(self, ligand: dict[str, object]) -> str:
        return (
            f"{ligand.get('resname', '')}|"
            f"{ligand.get('chain', '')}|"
            f"{int(ligand.get('resseq', 0))}"
        )

    def _selected_chain_ids(self) -> list[str]:
        selected: list[str] = []
        for ch in self._chains:
            safe_id = ch if ch.strip() else "_blank"
            try:
                checked = self.query_one(f"#chain-{safe_id}", Checkbox).value
            except Exception:
                checked = True
            if checked:
                selected.append(ch)
        return selected

    def _visible_cocrystal_ligands(self) -> list[dict]:
        selected_chains = set(self._selected_chain_ids())
        if not selected_chains:
            return list(self._cocrystal_ligands)
        visible = [
            ligand
            for ligand in self._cocrystal_ligands
            if str(ligand.get("chain", "")) in selected_chains
        ]
        return visible or list(self._cocrystal_ligands)

    def _cocrystal_options(self) -> list[tuple[str, str]]:
        options: list[tuple[str, str]] = []
        for ligand in self._visible_cocrystal_ligands():
            chain = str(ligand.get("chain", "")).strip() or "?"
            resseq = ligand.get("resseq", "?")
            n_atoms = len(ligand.get("coords") or [])
            label = f"{ligand.get('resname', '?')}  chain {chain}  residue {resseq}"
            if n_atoms:
                label += f"  ({n_atoms} atoms)"
            options.append((label, self._cocrystal_option_value(ligand)))
        return options

    def _selected_cocrystal_ligand(self) -> dict | None:
        visible_ligands = self._visible_cocrystal_ligands()
        if not visible_ligands:
            return None
        select = self.query_one("#cocrystal-select", Select)
        value = select.value
        if value == Select.BLANK:
            return visible_ligands[0]
        selected = str(value)
        for ligand in visible_ligands:
            if self._cocrystal_option_value(ligand) == selected:
                return ligand
        return None

    def _refresh_cocrystal_selector(self) -> None:
        select = self.query_one("#cocrystal-select", Select)
        current = select.value
        options = self._cocrystal_options()
        select.set_options(options)
        values = {value for _, value in options}
        if current != Select.BLANK and current in values:
            select.value = current
        elif options:
            select.value = options[0][1]

    # ------------------------------------------------------------------
    # Step 1 â€” Receptor validation
    # ------------------------------------------------------------------

    def _start_rec_validate(self) -> None:
        raw = self.query_one("#rec-input", Input).value.strip()
        if not raw:
            self.query_one("#rec-status", Static).update(
                "[#f85149]Enter a PDB ID or file path.[/#f85149]"
            )
            return
        self.query_one("#rec-status", Static).update("[#e3b341]Validating...[/#e3b341]")
        self.query_one("#btn-rec-validate").disabled = True
        self.query_one("#chain-section").display = False
        self.query_one("#af-warning-box").display = False
        self._chains = []
        self._cocrystal_ligands = []
        self.query_one("#cocrystal-select", Select).set_options([])
        self._pockets = []
        self.run_worker(lambda: self._do_rec_validate(raw), thread=True)

    def _do_rec_validate(self, raw: str) -> None:
        from ezscreen.prep import receptor as rec_prep

        try:
            work_dir = Path.home() / ".ezscreen" / "tmp" / "wizard"
            work_dir.mkdir(parents=True, exist_ok=True)

            if raw.upper().startswith("AF:"):
                from ezscreen.prep.alphafold import download_alphafold_structure

                uid = raw[3:].strip().upper()
                out = work_dir / f"AF-{uid}-model.pdb"
                pdb_path, _ = download_alphafold_structure(uid, out)
                self._ctx["pdb_source"] = "alphafold"
                self._ctx["pdb_id"] = uid
            elif len(raw) == 4 and raw.isalnum():
                pdb_path = rec_prep.fetch_pdb(raw.upper(), work_dir / "receptor_raw")
                self._ctx["pdb_source"] = "rcsb"
                self._ctx["pdb_id"] = raw.upper()
            else:
                pdb_path = Path(raw).expanduser()
                if not pdb_path.exists():
                    raise FileNotFoundError(f"File not found: {pdb_path}")
                self._ctx["pdb_source"] = "local"
                self._ctx["pdb_id"] = None

            self._ctx["pdb_path"] = pdb_path
            self._ctx["receptor_raw"] = raw

            chains = rec_prep.get_chains(pdb_path)
            is_af, af_ver = rec_prep.detect_alphafold(pdb_path)
            self._ctx["chains"] = chains
            self._ctx["is_alphafold"] = is_af
            self._ctx["af_version"] = af_ver

            from ezscreen.pocket import detect as pocket

            cocrystal = [] if is_af else pocket.find_cocrystal_ligands(pdb_path)
            self._cocrystal_ligands = cocrystal
            self._ctx["cocrystal_ligands"] = cocrystal

            try:
                self._hetero = rec_prep.heteroatom_inventory(pdb_path)
            except Exception:
                self._hetero = []

            self.app.call_from_thread(
                self._show_rec_result, chains, is_af, af_ver, cocrystal
            )
        except Exception as exc:
            self.app.call_from_thread(self._show_rec_error, str(exc))

    def _show_rec_result(
        self,
        chains: list[str],
        is_af: bool,
        af_ver: str | None,
        cocrystal: list[dict],
    ) -> None:
        self._cocrystal_ligands = list(cocrystal)
        co_txt = f"  {len(cocrystal)} co-crystal ligand(s) found." if cocrystal else ""
        self.query_one("#rec-status", Static).update(
            f"[#3fb950]Valid â€” {len(chains)} chain(s) found.{co_txt}[/#3fb950]"
        )
        self.query_one("#btn-rec-validate").disabled = False

        chain_list = self.query_one("#chain-list", Vertical)
        chain_list.remove_children()
        for ch in chains:
            label = f"Chain {ch}" if ch.strip() else "All (no chain ID)"
            safe_id = ch if ch.strip() else "_blank"
            chain_list.mount(Checkbox(label, id=f"chain-{safe_id}", value=True))
        self.query_one("#chain-section").display = True
        self._chains = chains
        self._populate_hetero()

        if is_af:
            ver_label = {"af2": "AlphaFold 2", "af3": "AlphaFold 3"}.get(
                af_ver or "", "AlphaFold"
            )
            self.query_one("#af-warning", Static).update(
                f"[#e3b341]\u26a0  {ver_label} structure detected â€” "
                "P2Rank will use AF profile. Pocket reliability is lower.[/#e3b341]"
            )
            self.query_one("#af-warning-box").display = True

        if cocrystal:
            names = ", ".join(lig["resname"] for lig in cocrystal)
            self.query_one("#cocrystal-info", Static).update(
                f"[#3fb950]Found: {names}[/#3fb950]"
            )
            self._refresh_cocrystal_selector()
        else:
            self.query_one("#cocrystal-info", Static).update(
                "[#6e7681]No co-crystal ligand found in this structure.[/#6e7681]"
            )
            self.query_one("#cocrystal-select", Select).set_options([])

    def _show_rec_error(self, msg: str) -> None:
        self.query_one("#rec-status", Static).update(f"[#f85149]{msg}[/#f85149]")
        self.query_one("#btn-rec-validate").disabled = False

    def _populate_hetero(self) -> None:
        hlist = self.query_one("#hetero-list", VerticalScroll)
        hlist.remove_children()
        status = self.query_one("#hetero-status", Static)
        if not self._hetero:
            self._hetero_keep = set()
            self._hetero_expanded = set()
            status.update("[#6e7681]No heteroatoms found in this structure.[/#6e7681]")
            return
        self._hetero_keep = {
            i for i, h in enumerate(self._hetero) if self._recommended_hetero_keep(h)
        }
        self._hetero_expanded = {
            category
            for category, items in self._hetero_by_category().items()
            if category not in _HETERO_NOISY_CATEGORIES or len(items) <= 8
        }
        self._refresh_hetero_controls()

    def _recommended_hetero_keep(self, hetero: dict) -> bool:
        category = str(hetero.get("category", "other"))
        return bool(hetero.get("keep_default")) or category in {"cofactor", "metal"}

    def _hetero_by_category(self) -> dict[str, list[tuple[int, dict]]]:
        grouped: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        for i, h in enumerate(self._hetero):
            category = str(h.get("category") or "other")
            grouped[category].append((i, h))
        return grouped

    def _refresh_hetero_controls(self) -> None:
        hlist = self.query_one("#hetero-list", VerticalScroll)
        self._refreshing_hetero_controls = True
        try:
            hlist.remove_children()
            grouped = self._hetero_by_category()
            for category in _HETERO_CATEGORY_ORDER:
                items = grouped.get(category, [])
                if not items:
                    continue

                count = len(items)
                atoms = sum(int(h.get("n_atoms", 0)) for _, h in items)
                kept = sum(1 for i, _ in items if i in self._hetero_keep)
                label = _HETERO_CATEGORY_LABELS.get(category, category.title())
                expanded = category in self._hetero_expanded
                toggle_label = "Hide individual" if expanded else "Show individual"
                children = [
                    Static(
                        f"{label}: {kept}/{count} kept, {atoms} atom(s)",
                        classes="hetero-category-header",
                    ),
                    Horizontal(
                        Button("Keep all", id=f"hetcat-keep-{category}"),
                        Button("Remove all", id=f"hetcat-remove-{category}"),
                        Button(toggle_label, id=f"hetcat-toggle-{category}"),
                        classes="hetero-category-actions",
                    ),
                ]
                if expanded:
                    for i, h in items:
                        children.append(
                            Checkbox(
                                self._hetero_label(h),
                                id=f"het-{i}",
                                value=i in self._hetero_keep,
                            )
                        )
                else:
                    children.append(
                        Static(
                            f"{count} {label.lower()} hidden. Use the category "
                            "buttons above or show individual rows.",
                            classes="hetero-collapsed-note",
                        )
                    )
                hlist.mount(Vertical(*children, classes="hetero-category"))
        finally:
            self._refreshing_hetero_controls = False
        self._update_hetero_summary()

    def _hetero_label(self, hetero: dict) -> str:
        chain = str(hetero.get("chain") or "").strip() or "-"
        resseq = hetero.get("resseq", "?")
        resname = hetero.get("resname", "?")
        category = hetero.get("category", "other")
        n_atoms = hetero.get("n_atoms", 0)
        return f"{resname} {chain}{resseq} - {category}, {n_atoms} atom(s)"

    def _apply_hetero_preset(self, preset: str) -> None:
        keep_categories = {
            "recommended": {"cofactor", "metal"},
            "apo": set(),
            "crystal": {"cofactor", "metal", "ligand"},
        }.get(preset)
        if keep_categories is None:
            return
        self._hetero_keep = {
            i
            for i, h in enumerate(self._hetero)
            if str(h.get("category", "other")) in keep_categories
        }
        self._refresh_hetero_controls()

    def _set_hetero_category(self, category: str, keep: bool) -> None:
        for i, h in enumerate(self._hetero):
            if str(h.get("category", "other")) != category:
                continue
            if keep:
                self._hetero_keep.add(i)
            else:
                self._hetero_keep.discard(i)
        self._refresh_hetero_controls()

    def _toggle_hetero_category(self, category: str) -> None:
        if category in self._hetero_expanded:
            self._hetero_expanded.remove(category)
        else:
            self._hetero_expanded.add(category)
        self._refresh_hetero_controls()

    def _update_hetero_summary(self) -> None:
        status = self.query_one("#hetero-status", Static)
        if not self._hetero:
            return
        kept = [h for i, h in enumerate(self._hetero) if i in self._hetero_keep]
        removed = [h for i, h in enumerate(self._hetero) if i not in self._hetero_keep]
        kept_counts = Counter(str(h.get("category", "other")) for h in kept)
        removed_counts = Counter(str(h.get("category", "other")) for h in removed)
        kept_txt = self._format_hetero_counts(kept_counts) or "none"
        removed_txt = self._format_hetero_counts(removed_counts) or "none"
        warnings = []
        if kept_counts.get("ligand", 0):
            warnings.append("Keeping ligands can block the docking pocket.")
        if removed_counts.get("metal", 0) or removed_counts.get("cofactor", 0):
            warnings.append(
                "Removing metals/cofactors can change active-site chemistry."
            )
        warning_txt = ""
        if warnings:
            warning_txt = f"\n[#e3b341]{' '.join(warnings)}[/#e3b341]"
        status.update(
            f"Keeping {len(kept)} group(s): {kept_txt}\n"
            f"Removing {len(removed)} group(s): {removed_txt}{warning_txt}"
        )

    def _format_hetero_counts(self, counts: Counter[str]) -> str:
        parts = []
        for category in _HETERO_CATEGORY_ORDER:
            count = counts.get(category, 0)
            if count:
                label = _HETERO_CATEGORY_LABELS.get(category, category).lower()
                parts.append(f"{count} {label}")
        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Step 2 - P2Rank
    # ------------------------------------------------------------------

    def _start_p2rank(self) -> None:
        self.query_one("#p2rank-status", Static).update(
            "[#e3b341]Running P2Rank...[/#e3b341]"
        )
        self.run_worker(lambda: self._do_p2rank(), thread=True)

    def _do_p2rank(self) -> None:
        from ezscreen.pocket import detect as pocket

        try:
            pdb_path = self._ctx["pdb_path"]
            is_af = self._ctx.get("is_alphafold", False)
            work_dir = Path.home() / ".ezscreen" / "tmp" / "wizard" / "p2rank"
            if pocket._p2rank_exe() is None:
                self.app.call_from_thread(
                    self.query_one("#p2rank-status", Static).update,
                    "[#e3b341]Downloading P2Rank (~50 MB, first run only)...[/#e3b341]",
                )
            pockets = pocket.run_p2rank(pdb_path, work_dir, alphafold=is_af)
            self._pockets = pockets
            self.app.call_from_thread(self._show_p2rank_results, pockets)
        except Exception as exc:
            self.app.call_from_thread(
                self.query_one("#p2rank-status", Static).update,
                f"[#f85149]P2Rank failed: {exc}[/#f85149]",
            )

    def _show_p2rank_results(self, pockets: list[dict]) -> None:
        if not pockets:
            self.query_one("#p2rank-status", Static).update(
                "[#e3b341]P2Rank found no pockets for this structure â€” "
                "try Active site residues or Blind docking.[/#e3b341]"
            )
            return

        self.query_one("#p2rank-status", Static).update(
            f"[#3fb950]{len(pockets)} pocket(s) found â€” select one below.[/#3fb950]"
        )
        picks = self.query_one("#p2rank-picks", Vertical)
        picks.remove_children()
        buttons = [
            RadioButton(
                f"#{p['rank']}  score {p['score']:.2f}  prob {p['probability']:.2f}",
                id=f"pocket-{i}",
                value=(i == 0),
            )
            for i, p in enumerate(pockets[:5])
        ]
        picks.mount(RadioSet(*buttons, id="p2rank-radioset"))

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _go_back(self) -> None:
        if self._step > 0:
            self._step -= 1
            self._update_ui()

    def _go_next(self) -> None:
        if self._step == len(_STEPS) - 1:
            self._submit()
            return
        err = self._validate_step()
        if err:
            self.app.notify(err, severity="error", timeout=4)
            return
        self._step += 1
        if self._step == len(_STEPS) - 1:
            self._populate_confirm()
        self._update_ui()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_step(self) -> str | None:
        step = _STEPS[self._step]

        if step == "step-receptor":
            if not self._ctx.get("pdb_path"):
                return "Validate the receptor first."
            selected = [
                ch
                for ch in self._chains
                if self.query_one(
                    f"#chain-{ch if ch.strip() else '_blank'}", Checkbox
                ).value
            ]
            if not selected:
                return "Select at least one chain."
            self._ctx["selected_chains"] = selected

        elif step == "step-recprep":
            self._ctx["keep_hetero"] = [
                h for i, h in enumerate(self._hetero) if i in self._hetero_keep
            ]
            self._ctx["receptor_isomers"] = self._selected_receptor_isomer_mode()
            ensemble_mode = self._selected_ensemble_mode()
            self._ctx["ensemble_mode"] = ensemble_mode
            self._ctx["ensemble_method"] = None
            self._ctx["ensemble_manual_paths"] = []
            self._ctx["ensemble_receptor_paths"] = []
            self._ctx["ensemble_max_conformers"] = _AUTO_ENSEMBLE_DEFAULT_MAX
            if ensemble_mode != "off" and self._ctx["receptor_isomers"] != "off":
                return "Use ensemble conformations or receptor protonation variants, not both."
            if ensemble_mode == "manual":
                ensemble_paths = self._ensemble_receptor_paths()
                if not ensemble_paths:
                    return "Enter at least one ensemble receptor conformer path."
                for path in ensemble_paths:
                    if not path.exists():
                        return f"Ensemble receptor not found: {path}"
                    if path.suffix.lower() not in {".pdb", ".pdbqt"}:
                        return f"Unsupported ensemble receptor format: {path.suffix}"
                self._ctx["ensemble_manual_paths"] = ensemble_paths
                self._ctx["ensemble_receptor_paths"] = ensemble_paths
            elif ensemble_mode == "auto":
                try:
                    self._ctx["ensemble_max_conformers"] = (
                        self._auto_ensemble_max_conformers()
                    )
                except ValueError as exc:
                    return str(exc)
                self._ctx["ensemble_method"] = _AUTO_ENSEMBLE_METHOD

        elif step == "step-site":
            btn = self.query_one("#site-method", RadioSet).pressed_button
            if btn is None:
                return "Select a binding site method."
            label = str(btn.label)
            if "Co-crystal" in label and not self._cocrystal_ligands:
                return "No co-crystal ligand found. Choose a different method."
            if "Co-crystal" in label and self._selected_cocrystal_ligand() is None:
                return "Choose a co-crystal reference ligand."
            if "P2Rank" in label and not self._pockets:
                return "P2Rank has not finished yet. Wait or choose a different method."
            if "manually" in label:
                err = self._validate_manual_coords()
                if err:
                    return err
            self._ctx["site_method"] = label
            self._ctx["site_details"] = self._collect_site_details(label)

        elif step == "step-ligands":
            raw = self.query_one("#lig-input", Input).value.strip()
            if not raw:
                return "Enter a ligand file path."
            p = Path(raw).expanduser()
            if not p.exists():
                return f"File not found: {p}"
            self._ctx["ligand_path"] = p

        elif step == "step-prep":
            self._ctx["admet_pre_filter"] = self.query_one("#opt-admet", Switch).value
            self._ctx["force_field"] = str(
                self.query_one("#opt-force-field", Select).value
            )
            self._ctx["enumerate_enabled"] = self.query_one(
                "#opt-enumerate", Switch
            ).value

        elif step == "step-engine":
            self._ctx["engine"] = self._selected_engine()
            has_ensemble = self._ctx.get(
                "ensemble_mode", "off"
            ) != "off" or self._ctx.get("ensemble_receptor_paths")
            if has_ensemble and self._ctx["engine"] == "diffdock":
                return (
                    "Ensemble docking currently supports UniDock/GNINA, not DiffDock-L."
                )
            _scoring = self.query_one("#opt-scoring", Select).value
            self._ctx["scoring"] = str(_scoring) if _scoring != Select.BLANK else "vina"
            self._ctx["consensus_enabled"] = (
                self._ctx["engine"] in {"unidock", "gnina"}
                and self.query_one("#opt-consensus", Switch).value
            )
            if has_ensemble and self._ctx["consensus_enabled"]:
                return "Ensemble plus consensus ranking is not supported yet."
            depth_btn = self.query_one("#opt-depth", RadioSet).pressed_button
            label = str(depth_btn.label) if depth_btn else ""
            if "Fast" in label:
                self._ctx["search_params"] = {"search_mode": "fast"}
                self._ctx["search_label"] = "Fast"
            elif "Thorough" in label:
                self._ctx["search_params"] = {"search_mode": "detail"}
                self._ctx["search_label"] = "Thorough"
            else:
                self._ctx["search_params"] = {"search_mode": "balance"}
                self._ctx["search_label"] = "Balanced"

        elif step == "step-compute":
            self._ctx["run_locally"] = self.query_one("#opt-local", Switch).value
            self._ctx["use_team_runners"] = (not self._ctx["run_locally"]) and self.query_one("#opt-team-runners", Switch).value
            has_ensemble = self._ctx.get(
                "ensemble_mode", "off"
            ) != "off" or self._ctx.get("ensemble_receptor_paths")
            if self._ctx["run_locally"] and has_ensemble:
                return "Ensemble docking requires Kaggle GPU in this version."
            gpu_btn = self.query_one("#opt-gpu-type", RadioSet).pressed_button
            self._ctx["gpu_type"] = (
                "nvidiaTeslaT4"
                if gpu_btn and "T4" in str(gpu_btn.label)
                else "nvidiaTeslaP100"
            )
            raw_exh = self.query_one("#opt-exhaustiveness", Input).value.strip()
            try:
                self._ctx["exhaustiveness"] = int(raw_exh) if raw_exh else None
            except ValueError:
                return "Exhaustiveness must be a whole number."
            # Collect account shard assignments when Kaggle backend is selected
            if not self._ctx["run_locally"]:
                if self._ctx.get("use_team_runners"):
                    self._ctx["account_assignments"] = None
                    self._ctx["team_member_assignments"] = []
                    try:
                        from ezscreen.team.client import TeamClient, load_local_state

                        state = load_local_state()
                        status = TeamClient(state["server_url"]).status(state["user_id"])
                        team_assignments = []
                        for member in status.get("members", []):
                            if not member.get("enabled") or member.get("left_at") or not (member.get("github_repo") or "").strip():
                                continue
                            input_id = self._team_member_input_id(member["member_id"])
                            raw = self.query_one(f"#{input_id}", Input).value.strip()
                            count = int(raw) if raw else None
                            if count is not None and count < 0:
                                return "Team runner shard counts must be zero or greater."
                            team_assignments.append(
                                {
                                    "member_id": member["member_id"],
                                    "display_name": member.get("display_name", ""),
                                    "github_repo": member.get("github_repo") or "",
                                    "shard_count": count,
                                }
                            )
                        self._ctx["team_member_assignments"] = team_assignments
                    except ValueError:
                        return "Team runner shard counts must be whole numbers."
                    except Exception as exc:
                        return f"Team runner assignments are unavailable: {exc}"
                else:
                    from ezscreen import auth

                    accounts = auth.get_all_kaggle_accounts()
                    if len(accounts) > 1:
                        assignments = []
                        for acct in accounts:
                            safe = acct["name"].replace(" ", "-").lower()
                            try:
                                raw = self.query_one(
                                    f"#acct-shards-{safe}", Input
                                ).value.strip()
                                count = (
                                    int(raw) if raw else None
                                )  # None = auto, 0 = exclude, N = explicit
                            except Exception:
                                count = None
                            assignments.append({"account": acct, "shard_count": count})
                        self._ctx["account_assignments"] = assignments
                    else:
                        self._ctx["account_assignments"] = None
                    self._ctx["team_member_assignments"] = []
            else:
                self._ctx["account_assignments"] = None
                self._ctx["team_member_assignments"] = []

        return None

    def _validate_manual_coords(self) -> str | None:
        fields = [
            "manual-cx",
            "manual-cy",
            "manual-cz",
            "manual-sx",
            "manual-sy",
            "manual-sz",
        ]
        for fid in fields:
            val = self.query_one(f"#{fid}", Input).value.strip()
            if not val:
                return "Fill in all six box coordinate fields."
            try:
                float(val)
            except ValueError:
                return f"Invalid number in box coordinates: '{val}'"
        return None

    def _collect_site_details(self, label: str) -> dict:
        if "Co-crystal" in label:
            ligand = self._selected_cocrystal_ligand()
            return {
                "type": "cocrystal",
                "ligands": [ligand] if ligand else [],
                "available_ligands": list(self._cocrystal_ligands),
            }
        if "residues" in label:
            raw = self.query_one("#residues-input", Input).value.strip()
            res_ids = [int(r.strip()) for r in raw.split(",") if r.strip().isdigit()]
            return {"type": "residues", "residues": res_ids}
        if "P2Rank" in label:
            try:
                rs = self.query_one("#p2rank-radioset", RadioSet)
                idx = rs.pressed_index if rs.pressed_index is not None else 0
                return {"type": "p2rank", "pocket": self._pockets[idx]}
            except Exception:
                return {
                    "type": "p2rank",
                    "pocket": self._pockets[0] if self._pockets else {},
                }
        if "manually" in label:
            cx = float(self.query_one("#manual-cx", Input).value)
            cy = float(self.query_one("#manual-cy", Input).value)
            cz = float(self.query_one("#manual-cz", Input).value)
            sx = float(self.query_one("#manual-sx", Input).value)
            sy = float(self.query_one("#manual-sy", Input).value)
            sz = float(self.query_one("#manual-sz", Input).value)
            return {
                "type": "manual",
                "center": [cx, cy, cz],
                "size": [sx, sy, sz],
            }
        return {"type": "blind"}

    # ------------------------------------------------------------------
    # Step 5 â€” Confirm summary
    # ------------------------------------------------------------------

    def _populate_confirm(self) -> None:
        ctx = self._ctx
        pdb = ctx.get("pdb_id") or (
            ctx["pdb_path"].name if ctx.get("pdb_path") else "â€”"
        )
        chains = ", ".join(ctx.get("selected_chains", []))
        site = ctx.get("site_method", "â€”")
        lig = ctx["ligand_path"].name if ctx.get("ligand_path") else "â€”"
        admet = "yes" if ctx.get("admet_pre_filter") else "no"
        depth = ctx.get("search_label", "Balanced")
        force_field = ctx.get("force_field", "MMFF94")
        receptor_isomers = {
            "off": "off",
            "single": "single corrected state",
            "variants": "multiple variants",
        }.get(ctx.get("receptor_isomers", "off"), "off")
        ensemble_mode = ctx.get("ensemble_mode", "off")
        if ensemble_mode == "manual":
            ensemble_count = (
                len(
                    ctx.get("ensemble_manual_paths")
                    or ctx.get("ensemble_receptor_paths")
                    or []
                )
                + 1
            )
            ensemble_label = f"manual ({ensemble_count} conformations)"
        elif ensemble_mode == "auto":
            max_conformers = ctx.get(
                "ensemble_max_conformers", _AUTO_ENSEMBLE_DEFAULT_MAX
            )
            ensemble_label = f"auto PDB + NMA fallback (max {max_conformers})"
        else:
            ensemble_label = "off"
        engine_label = (
            "AutoDock Vina (local)"
            if ctx.get("run_locally")
            else engines.get(ctx.get("engine", "unidock")).label
        )
        scoring_label = ctx.get("scoring", "vina").capitalize()
        consensus_label = "on" if ctx.get("consensus_enabled") else "off"
        if ctx.get("run_locally"):
            exh = ctx.get("exhaustiveness") or "default (from settings)"
            backend = f"Local CPU (AutoDock Vina, exhaustiveness={exh})"
        elif ctx.get("engine") == "diffdock":
            backend = "NVIDIA NIM API"
        else:
            gpu_label = "T4 x 2" if ctx.get("gpu_type") == "nvidiaTeslaT4" else "P100"
            backend = "Team runners" if ctx.get("use_team_runners") else f"Kaggle GPU ({gpu_label})"
        af_note = (
            "\n[#e3b341]\u26a0  AlphaFold structure â€” P2Rank AF profile active[/#e3b341]"
            if ctx.get("is_alphafold")
            else ""
        )
        self.query_one("#confirm-summary", Static).update(
            "\n".join(
                [
                    f"[bold #6e7681]Receptor     [/bold #6e7681][#f0f6fc]{pdb}[/#f0f6fc]  chains {chains}",
                    f"[bold #6e7681]Binding site [/bold #6e7681][#f0f6fc]{site}[/#f0f6fc]",
                    f"[bold #6e7681]Rec isomers  [/bold #6e7681][#f0f6fc]{receptor_isomers}[/#f0f6fc]",
                    f"[bold #6e7681]Ensemble     [/bold #6e7681][#f0f6fc]{ensemble_label}[/#f0f6fc]",
                    f"[bold #6e7681]Ligands      [/bold #6e7681][#f0f6fc]{lig}[/#f0f6fc]",
                    f"[bold #6e7681]ADMET filter [/bold #6e7681][#f0f6fc]{admet}[/#f0f6fc]",
                    f"[bold #6e7681]Engine       [/bold #6e7681][#f0f6fc]{engine_label}[/#f0f6fc]",
                    f"[bold #6e7681]Scoring      [/bold #6e7681][#f0f6fc]{scoring_label}[/#f0f6fc]",
                    f"[bold #6e7681]Consensus    [/bold #6e7681][#f0f6fc]{consensus_label}[/#f0f6fc]",
                    f"[bold #6e7681]Search depth [/bold #6e7681][#f0f6fc]{depth}[/#f0f6fc]",
                    f"[bold #6e7681]Force field  [/bold #6e7681][#f0f6fc]{force_field}[/#f0f6fc]",
                    f"[bold #6e7681]Backend      [/bold #6e7681][#f0f6fc]{backend}[/#f0f6fc]",
                    af_note,
                ]
            )
        )

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def _prepare_ensemble_receptors(
        self,
        paths: list[Path],
        *,
        rec_prep,
        work_dir: Path,
        chains: list[str],
        ph: float,
        keep_hetero: list[dict] | None,
        primary_pdbqt: Path,
    ) -> list[Path]:
        if not paths:
            return [primary_pdbqt]
        from ezscreen.prep.ensemble import prepare_manual_ensemble

        conformers = prepare_manual_ensemble(
            paths,
            output_dir=work_dir / "receptor_ensemble",
            chains=chains,
            ph=ph,
            keep_hetero=keep_hetero,
            rec_prep_module=rec_prep,
        )
        return [primary_pdbqt] + [c.pdbqt_path for c in conformers]

    def _submit(self) -> None:
        if self._submitted:
            return
        self._submitted = True
        self._update_ui()
        self.run_worker(lambda: self._do_submit(), thread=True)

    def _do_submit(self) -> None:
        from ezscreen import auth, checkpoint, config
        from ezscreen.admet.filter import filter_library
        from ezscreen.backends import engines
        from ezscreen.backends.kaggle import runner as kaggle_runner
        from ezscreen.pocket import detect as pocket
        from ezscreen.prep import ligands as ligand_prep
        from ezscreen.prep import receptor as rec_prep

        _log_file = (Path.home() / ".ezscreen" / "runs" / "last_run.log").open(
            "w", buffering=1, encoding="utf-8"
        )

        def log(msg: str) -> None:
            self.app.call_from_thread(self.query_one("#run-log", RichLog).write, msg)
            import re

            _log_file.write(re.sub(r"\[/?[^\[\]]*\]", "", msg) + "\n")

        try:
            run_id = "ezs-" + secrets.token_hex(3)
            work_dir = Path.home() / ".ezscreen" / "runs" / run_id
            work_dir.mkdir(parents=True, exist_ok=True)
            ctx = self._ctx
            cfg = config.load()

            log(f"[#6e7681]Run ID: {run_id}[/#6e7681]")

            # Kaggle username
            try:
                import json

                creds = auth.load_credentials()
                with open(creds.get("kaggle_json_path", "")) as f:
                    kaggle_username = json.load(f).get("username", "user")
            except Exception:
                kaggle_username = "user"

            # Receptor prep
            log("[#79c0ff]Prepping receptor...[/#79c0ff]")
            rec_result = rec_prep.prep_receptor(
                pdb_path=ctx["pdb_path"],
                chains=ctx["selected_chains"],
                output_dir=work_dir / "receptor",
                ph=cfg["run"].get("default_ph", 7.4),
                keep_hetero=ctx.get("keep_hetero"),
                receptor_isomers=ctx.get("receptor_isomers", "off"),
            )
            receptor_pdbqt = rec_result["pdbqt_path"]
            receptor_pdb = rec_result.get("fixed_pdb_path") or rec_result.get(
                "receptor_pdb_path"
            )
            ensemble_mode = ctx.get("ensemble_mode", "off")
            if ensemble_mode == "off" and ctx.get("ensemble_receptor_paths"):
                ensemble_mode = "manual"
            if ensemble_mode == "manual":
                ensemble_paths = list(
                    ctx.get("ensemble_manual_paths")
                    or ctx.get("ensemble_receptor_paths")
                    or []
                )
                log(
                    f"[#79c0ff]Preparing {len(ensemble_paths)} ensemble receptor conformation(s)...[/#79c0ff]"
                )
                receptor_pdbqt_paths = self._prepare_ensemble_receptors(
                    ensemble_paths,
                    rec_prep=rec_prep,
                    work_dir=work_dir,
                    chains=ctx["selected_chains"],
                    ph=cfg["run"].get("default_ph", 7.4),
                    keep_hetero=ctx.get("keep_hetero"),
                    primary_pdbqt=receptor_pdbqt,
                )
                receptor_variant_modes = None
            elif ensemble_mode == "auto":
                from ezscreen.prep.ensemble import generate_auto_ensemble

                max_conformers = int(
                    ctx.get("ensemble_max_conformers", _AUTO_ENSEMBLE_DEFAULT_MAX)
                )
                log(
                    f"[#79c0ff]Generating up to {max_conformers} receptor ensemble "
                    "conformation(s) with PDB + normal-mode fallback...[/#79c0ff]"
                )
                conformers = generate_auto_ensemble(
                    Path(receptor_pdb or ctx["pdb_path"]),
                    ctx["selected_chains"],
                    ctx.get("site_details") or {},
                    max_conformers=max_conformers,
                    output_dir=work_dir / "receptor_ensemble_auto",
                    ph=cfg["run"].get("default_ph", 7.4),
                    keep_hetero=ctx.get("keep_hetero"),
                    rec_prep_module=rec_prep,
                    logger=lambda message: log(f"[#6e7681]{message}[/#6e7681]"),
                )
                ctx["ensemble_generated_conformers"] = [
                    {
                        "name": c.name,
                        "source": c.source,
                        "pdb_path": str(c.pdb_path) if c.pdb_path else None,
                        "pdbqt_path": str(c.pdbqt_path),
                        "metadata": c.metadata,
                    }
                    for c in conformers
                ]
                receptor_pdbqt_paths = [receptor_pdbqt] + [
                    c.pdbqt_path for c in conformers
                ]
                if not conformers:
                    log(
                        "[#e3b341]Auto ensemble did not yield any prep-valid conformers; "
                        "continuing with the primary receptor only.[/#e3b341]"
                    )
                receptor_variant_modes = None
            else:
                receptor_pdbqt_paths = rec_result.get("variant_pdbqt_paths") or [
                    receptor_pdbqt
                ]
                receptor_variant_modes = rec_result.get("receptor_variant_modes") or [
                    "default"
                ]
            _has_receptor_ensemble = (
                len(receptor_pdbqt_paths) > 1 and not receptor_variant_modes
            )
            _has_kaggle_receptor_prep = bool(receptor_variant_modes) and any(
                m != "default" for m in receptor_variant_modes
            )
            if _has_kaggle_receptor_prep:
                log(
                    f"[#6e7681]{len(receptor_variant_modes)} receptor variant mode(s) queued; "
                    "PROPKA/PDB2PQR prep will run on Kaggle[/#6e7681]"
                )
            if _has_receptor_ensemble:
                log(
                    f"[#6e7681]{len(receptor_pdbqt_paths)} receptor conformation(s) "
                    "queued for ensemble docking[/#6e7681]"
                )

            # Binding box
            engine_key = ctx.get("engine", "unidock")
            engine_profile = engines.get(engine_key)
            site = ctx["site_details"]
            if engine_profile.requires_box:
                log("[#79c0ff]Computing binding box...[/#79c0ff]")
                if site["type"] == "cocrystal":
                    box = pocket.box_from_cocrystal(site["ligands"][0])
                elif site["type"] == "residues":
                    box = pocket.box_from_residues(
                        ctx["pdb_path"], site["residues"], ctx["selected_chains"]
                    )
                elif site["type"] == "p2rank":
                    box = {**site["pocket"]}
                elif site["type"] == "manual":
                    box = {"center": site["center"], "size": site["size"]}
                else:
                    box = pocket.box_blind(ctx["pdb_path"])
                ctx["box"] = box
                log(
                    f"[#6e7681]Box: center {box['center']}  size {box['size']}[/#6e7681]"
                )
            else:
                box = None
                ctx["box"] = None
                ctx["site_method"] = "Blind docking (box-free)"
                log(
                    f"[#6e7681]{engine_profile.label}: box step skipped; blind docking.[/#6e7681]"
                )

            # Optional box confidence
            # Optional box confidence: redock the co-crystal ligand and measure how
            # well the docked pose reproduces the crystal pose. Fail-soft, opt-in.
            if (
                engine_profile.requires_box
                and site["type"] == "cocrystal"
                and bool(cfg.get("prep", {}).get("redock_validation", False))
            ):
                try:
                    from ezscreen.benchmark.redock import (
                        RMSD_WARN_THRESHOLD,
                        redock_cocrystal,
                    )

                    log(
                        "[#79c0ff]Validating box (redocking co-crystal ligand)...[/#79c0ff]"
                    )
                    validation_receptors = []
                    for idx, receptor_path in enumerate(receptor_pdbqt_paths):
                        variant_name = (
                            "default" if idx == 0 else Path(receptor_path).stem
                        )
                        validation_receptors.append((variant_name, receptor_path))

                    _rd_results = []
                    for variant_name, receptor_path in validation_receptors:
                        _rd = redock_cocrystal(
                            Path(receptor_path),
                            ctx["pdb_path"],
                            site["ligands"][0],
                            box,
                            work_dir,
                        )
                        _rd["receptor_variant"] = variant_name
                        _rd_results.append(_rd)

                    _rd_ok = [
                        result
                        for result in _rd_results
                        if result.get("rmsd") is not None
                    ]
                    if _rd_ok:
                        _rd_best = min(_rd_ok, key=lambda result: result["rmsd"])
                        ctx["redock_rmsd"] = _rd_best["rmsd"]
                        ctx["redock_best_receptor_variant"] = _rd_best[
                            "receptor_variant"
                        ]
                        ctx["redock_best_mode_rank"] = _rd_best.get("best_mode_rank")
                        _variant_note = ""
                        if len(validation_receptors) > 1:
                            _variant_note = f" on {_rd_best['receptor_variant']}"
                        _mode_note = ""
                        if _rd_best.get("best_mode_rank"):
                            _mode_note = f" (mode {_rd_best['best_mode_rank']})"
                        if _rd_best["reliable"]:
                            log(
                                f"[#3fb950]Redock best RMSD: {_rd_best['rmsd']:.2f} A{_variant_note}{_mode_note} - box reproduces the crystal pose[/#3fb950]"
                            )
                        else:
                            log(
                                f"[#e3b341]Warning: redock best RMSD: {_rd_best['rmsd']:.2f} A{_variant_note}{_mode_note} (> {RMSD_WARN_THRESHOLD:.0f} A) - box may be unreliable[/#e3b341]"
                            )
                    else:
                        _rd = _rd_results[0]
                        ref_desc = (
                            f"{_rd.get('reference_ligand', '?')}/"
                            f"{_rd.get('reference_chain', '?')}/"
                            f"{_rd.get('reference_resseq', '?')}"
                        )
                        reason = _rd.get("reason", "unknown")
                        detail = _rd.get("detail")
                        extra = f" ({detail})" if detail else ""
                        log(
                            f"[#6e7681]Redock validation skipped for {ref_desc}: {reason}{extra}.[/#6e7681]"
                        )
                except Exception:
                    pass

            # ADMET pre-filter
            ligand_input = ctx["ligand_path"]
            if ctx.get("admet_pre_filter"):
                log("[#79c0ff]Running ADMET pre-filter...[/#79c0ff]")
                admet_out = work_dir / "admet_filtered.sdf"
                admet_summary = filter_library(str(ligand_input), str(admet_out))
                total = admet_summary.get("total_input", 0)
                removed = admet_summary.get("admet_removed", 0)
                log(f"[#6e7681]ADMET: {total - removed:,}/{total:,} passed[/#6e7681]")
                ligand_input = admet_out

            if ctx.get("engine") == "diffdock":
                from ezscreen.backends.diffdock.runner import run_diffdock_screening

                log(
                    "[#79c0ff]Running DiffDock-L blind screening via NVIDIA NIM...[/#79c0ff]"
                )
                checkpoint.init_db()
                checkpoint.create_run(run_id, ctx, 0)
                result = run_diffdock_screening(
                    run_id=run_id,
                    receptor_pdb=Path(receptor_pdb),
                    ligand_path=Path(ligand_input),
                    work_dir=work_dir,
                    progress_cb=log,
                )
                if result["status"] == "complete":
                    checkpoint.mark_run_complete(run_id)
                    log(f"[#3fb950]Done! Results -> {result['output_dir']}[/#3fb950]")
                    self.app.call_from_thread(self._on_complete, run_id)
                    return
                checkpoint.mark_run_failed(run_id)
                log(
                    f"[#f85149]Run failed: {result.get('error_type') or result.get('status', 'unknown')}[/#f85149]"
                )
                self.app.call_from_thread(self._on_done)
                return

            # Ligand prep
            _prep_cfg = cfg.get("prep", {})
            _force_field = ctx.get("force_field") or str(
                _prep_cfg.get("force_field", "MMFF94")
            )
            _enum_enabled = bool(
                ctx.get("enumerate_enabled", _prep_cfg.get("enumerate_enabled", False))
            )
            _enum_opts = {
                "enabled": _enum_enabled,
                "protonation": bool(_prep_cfg.get("enumerate_protonation", True)),
                "tautomers": bool(_prep_cfg.get("enumerate_tautomers", True)),
                "stereo": bool(_prep_cfg.get("enumerate_stereo", False)),
                "ring": bool(_prep_cfg.get("enumerate_ring", False)),
                "max_variants": int(_prep_cfg.get("max_variants_per_ligand", 4)),
            }
            if ctx.get("use_team_runners"):
                team_assignments = ctx.get("team_member_assignments")
                _active = [a for a in (team_assignments or []) if a["shard_count"] != 0]
                if _active:
                    n_shards_target = sum(a["shard_count"] or 1 for a in _active)
                else:
                    n_shards_target = None
            else:
                account_assignments = ctx.get("account_assignments")
                _active = [a for a in (account_assignments or []) if a["shard_count"] != 0]
                if _active and len(_active) > 1:
                    n_shards_target = sum(a["shard_count"] or 1 for a in _active)
                else:
                    n_shards_target = None

            if ctx.get("run_locally"):
                log("[#79c0ff]Prepping ligands locally...[/#79c0ff]")
                lig_result = ligand_prep.prep_ligands(
                    input_path=ligand_input,
                    output_dir=work_dir / "shards",
                    ph=cfg["run"].get("default_ph", 7.4),
                    n_shards=n_shards_target,
                    force_field=_force_field,
                    enumerate_opts=_enum_opts,
                )
                log(
                    f"[#6e7681]{len(lig_result['shard_paths'])} shard(s) ready ({lig_result['report']['prep_passed']:,} ligands prepped)[/#6e7681]"
                )
            else:
                _prep_on_kaggle = bool(_prep_cfg.get("prep_on_kaggle", True))
                if _prep_on_kaggle:
                    log("[#79c0ff]Sharding ligands for Kaggle prep...[/#79c0ff]")
                else:
                    log("[#79c0ff]Prepping ligands locally before upload...[/#79c0ff]")
                _prep_kwargs = {
                    "input_path": ligand_input,
                    "output_dir": work_dir / "shards",
                    "n_shards": n_shards_target,
                }
                if not _prep_on_kaggle:
                    _prep_kwargs["force_field"] = _force_field
                    _prep_kwargs["enumerate_opts"] = _enum_opts
                lig_result = (
                    ligand_prep.shard_raw
                    if _prep_on_kaggle
                    else ligand_prep.prep_ligands
                )(**_prep_kwargs)
                log(
                    f"[#6e7681]{len(lig_result['shard_paths'])} shard(s) ready ({lig_result['report']['total_input']:,} ligands)[/#6e7681]"
                )

            shard_paths = lig_result["shard_paths"]
            if not shard_paths:
                raise RuntimeError("Ligand prep produced 0 dockable compounds.")

            if ctx.get("run_locally") or not _prep_on_kaggle:
                _protonation_status = lig_result["report"].get(
                    "protonation_status", "unknown"
                )
                _protonation_method = lig_result["report"].get("protonation_method")
                _preparation_location = "local"
            elif _enum_enabled and bool(_enum_opts.get("protonation")):
                _protonation_status = "enumeration_requested_fail_soft"
                _protonation_method = "Gypsum-DL"
                _preparation_location = "kaggle"
            else:
                _protonation_status = "not_applied_input_state_preserved"
                _protonation_method = None
                _preparation_location = "kaggle"
            ctx["ligand_prep_provenance"] = {
                "preparation_location": _preparation_location,
                "protonation_ph": cfg["run"].get("default_ph", 7.4),
                "protonation_status": _protonation_status,
                "protonation_method": _protonation_method,
            }

            # Checkpoint
            checkpoint.init_db()
            checkpoint.create_run(run_id, ctx, lig_result["report"]["total_input"])
            for i, _ in enumerate(shard_paths):
                n_per = lig_result["report"]["total_input"] // len(shard_paths)
                checkpoint.add_shard(run_id, i, n_per)

            if ctx.get("use_team_runners"):
                from ezscreen.team.client import TeamClient, load_local_state
                from ezscreen.team.coordinator import (
                    dispatch_team_job_runs,
                    dispatch_team_job_workers,
                    download_team_result_zips,
                    merge_team_result_zips,
                    stage_team_job_receptor_files,
                    start_team_job,
                    upload_job_files,
                    wait_for_team_job,
                )

                state = load_local_state()
                if not state.get("server_url") or not state.get("user_id"):
                    raise RuntimeError("Team runner state is missing. Open Team Runs first.")
                _gpu_type = ctx.get("gpu_type", "nvidiaTeslaP100")
                team_receptor_files, team_receptor_variants = stage_team_job_receptor_files(
                    work_dir=work_dir,
                    receptor_pdbqt=Path(receptor_pdbqt),
                    receptor_pdbqts=[Path(p) for p in receptor_pdbqt_paths],
                    receptor_variant_modes=receptor_variant_modes,
                    receptor_pdb=Path(receptor_pdb) if receptor_pdb else None,
                )
                notebook_src = kaggle_runner.render_vina_notebook(
                    run_id=run_id,
                    shard_filenames=[p.name for p in shard_paths],
                    receptor_filename=team_receptor_variants[0]["filename"],
                    receptor_variant=team_receptor_variants[0]["name"],
                    receptor_variants=team_receptor_variants,
                    receptor_pdb_filename="receptor_fixed.pdb",
                    box_center=box["center"],
                    box_size=box["size"],
                    search_mode=ctx["search_params"].get("search_mode", "balance"),
                    ph=cfg["run"].get("default_ph", 7.4),
                    prep_on_kaggle=_prep_on_kaggle,
                    accelerator=_gpu_type,
                    force_field=_force_field,
                    enumerate_opts=_enum_opts,
                    prep_cfg=_prep_cfg,
                    results_cfg=cfg.get("results", {}),
                    engine=ctx.get("engine", "unidock"),
                    scoring=ctx.get("scoring", "vina"),
                )
                notebook_path = work_dir / "team_notebook.ipynb"
                notebook_path.write_text(notebook_src, encoding="utf-8")
                client = TeamClient(state["server_url"], token=state.get("session_token"))
                team_job = start_team_job(
                    client=client,
                    user_id=state["user_id"],
                    run_id=run_id,
                    receptor_pdbqt=team_receptor_files[0],
                    receptor_files=team_receptor_files,
                    shard_paths=[Path(p) for p in shard_paths],
                    notebook_path=notebook_path,
                    engine=ctx.get("engine", "unidock"),
                    accelerator=_gpu_type,
                    team_member_assignments=ctx.get("team_member_assignments") or [],
                )
                log(f"[#79c0ff]Team job queued: {team_job['team_job_id']}[/#79c0ff]")
                prepare_dispatch = dispatch_team_job_workers(
                    client=client,
                    user_id=state["user_id"],
                    team_job=team_job,
                )
                if prepare_dispatch["dispatched"]:
                    log(
                        f"[#79c0ff]Triggered {len(prepare_dispatch['dispatched'])} worker upload-ticket workflow(s).[/#79c0ff]"
                    )
                if prepare_dispatch["skipped"]:
                    log(
                        "[#e3b341]Some workers could not be dispatched by the team server yet; "
                        "check GitHub App setup for those repos.[/#e3b341]"
                    )
                if prepare_dispatch["failed"]:
                    log(
                        f"[#e3b341]Immediate GitHub dispatch failed for {len(prepare_dispatch['failed'])} worker(s); "
                        "check the team server GitHub App configuration.[/#e3b341]"
                    )
                try:
                    upload_job_files(
                        client=client,
                        user_id=state["user_id"],
                        team_job_id=team_job["team_job_id"],
                        file_paths=[*team_receptor_files, notebook_path, *[Path(p) for p in shard_paths]],
                        expected_ticket_count=sum(len(worker_job.get("files", [])) for worker_job in team_job.get("worker_jobs", [])),
                        poll_timeout_s=int(cfg.get("run", {}).get("team_upload_ticket_timeout_s", 360)),
                    )
                    log("[#3fb950]Team upload tickets consumed; files uploaded directly to Kaggle.[/#3fb950]")
                except TimeoutError:
                    log("[#e3b341]Team job is waiting for workers to mint upload tickets.[/#e3b341]")
                    self.app.call_from_thread(self._on_done)
                    return
                run_dispatch = dispatch_team_job_runs(
                    client=client,
                    user_id=state["user_id"],
                    team_job=team_job,
                )
                if run_dispatch["dispatched"]:
                    log(
                        f"[#79c0ff]Triggered {len(run_dispatch['dispatched'])} worker run workflow(s).[/#79c0ff]"
                    )
                if run_dispatch["skipped"]:
                    log(
                        "[#e3b341]Some workers could not be dispatched for the run phase; "
                        "check GitHub App setup for those repos.[/#e3b341]"
                    )
                if run_dispatch["failed"]:
                    log(
                        f"[#e3b341]Immediate GitHub run dispatch failed for {len(run_dispatch['failed'])} worker(s); "
                        "check the team server GitHub App configuration.[/#e3b341]"
                    )
                team_result = wait_for_team_job(
                    client=client,
                    user_id=state["user_id"],
                    team_job_id=team_job["team_job_id"],
                    poll_interval_s=int(cfg.get("run", {}).get("team_status_poll_interval_s", 15)),
                    timeout_s=int(cfg.get("run", {}).get("team_job_timeout_s", 7200)),
                )
                if team_result.get("status") != "complete":
                    result = {
                        "status": "failed",
                        "error_type": f"team_job_{team_result.get('status', 'failed')}",
                        "output_dir": work_dir,
                    }
                else:
                    result_zips = download_team_result_zips(
                        client=client,
                        user_id=state["user_id"],
                        team_job=team_job,
                        output_dir=work_dir / "team_result_zips",
                    )
                    merged = merge_team_result_zips(
                        result_zips,
                        work_dir,
                        engine=ctx.get("engine", "unidock"),
                    )
                    result = {
                        "status": "complete",
                        "output_dir": work_dir,
                        **merged,
                    }
            elif ctx.get("run_locally"):
                # Local AutoDock Vina path
                from ezscreen.backends.local.runner import run_local_screening

                log("[#79c0ff]Running local AutoDock Vina docking...[/#79c0ff]")
                result = run_local_screening(
                    run_id=run_id,
                    receptor_pdbqt=receptor_pdbqt,
                    shard_paths=shard_paths,
                    box_center=box["center"],
                    box_size=box["size"],
                    work_dir=work_dir,
                    exhaustiveness=ctx.get("exhaustiveness"),
                )
            else:
                if ctx.get("consensus_enabled"):
                    from ezscreen.results.merger import build_consensus_results

                    first_engine = ctx.get("engine", "unidock")
                    second_engine = "gnina" if first_engine != "gnina" else "unidock"
                    consensus_engines = [first_engine, second_engine]
                    engine_outputs = []
                    result = {"status": "failed", "error_type": "consensus_not_started"}

                    for _engine in consensus_engines:
                        log(f"[#79c0ff]Consensus engine: {_engine}...[/#79c0ff]")
                        engine_work = work_dir / f"consensus_{_engine}"
                        engine_work.mkdir(parents=True, exist_ok=True)
                        account_assignments = ctx.get("account_assignments")
                        _active_assignments = [
                            a
                            for a in (account_assignments or [])
                            if a["shard_count"] != 0
                        ]
                        _has_kaggle_receptor_prep = bool(
                            receptor_variant_modes
                        ) and any(m != "default" for m in receptor_variant_modes)
                        if (
                            _has_kaggle_receptor_prep or _has_receptor_ensemble
                        ) and not _active_assignments:
                            _creds = auth.load_credentials()
                            _active_assignments = [
                                {
                                    "account": {
                                        "name": kaggle_username,
                                        "username": kaggle_username,
                                        "kaggle_json_path": _creds.get(
                                            "kaggle_json_path", ""
                                        ),
                                    },
                                    "shard_count": 1,
                                }
                            ]
                        if _active_assignments and (
                            len(_active_assignments) > 1
                            or _has_kaggle_receptor_prep
                            or _has_receptor_ensemble
                        ):
                            result = kaggle_runner.run_multi_account_screening(
                                run_id=f"{run_id}-{_engine}",
                                receptor_pdbqt=receptor_pdbqt,
                                receptor_pdbqts=receptor_pdbqt_paths,
                                receptor_variant_modes=receptor_variant_modes,
                                receptor_pdb=(
                                    Path(receptor_pdb) if receptor_pdb else None
                                ),
                                shard_paths=shard_paths,
                                account_assignments=_active_assignments,
                                work_dir=engine_work,
                                box_center=box["center"],
                                box_size=box["size"],
                                search_mode=ctx["search_params"].get(
                                    "search_mode", "balance"
                                ),
                                ph=cfg["run"].get("default_ph", 7.4),
                                retry_limit=cfg["run"].get("shard_retry_limit", 3),
                                accelerator=ctx.get("gpu_type", "nvidiaTeslaP100"),
                                prep_on_kaggle=_prep_on_kaggle,
                                force_field=_force_field,
                                enumerate_opts=_enum_opts,
                                engine=_engine,
                                scoring=ctx.get("scoring", "vina"),
                            )
                        else:
                            _gpu_type = ctx.get("gpu_type", "nvidiaTeslaP100")
                            notebook_src = kaggle_runner.render_vina_notebook(
                                run_id=f"{run_id}-{_engine}",
                                shard_filenames=[p.name for p in shard_paths],
                                receptor_filename=Path(receptor_pdbqt).name,
                                box_center=box["center"],
                                box_size=box["size"],
                                search_mode=ctx["search_params"].get(
                                    "search_mode", "balance"
                                ),
                                ph=cfg["run"].get("default_ph", 7.4),
                                prep_on_kaggle=_prep_on_kaggle,
                                accelerator=_gpu_type,
                                force_field=_force_field,
                                enumerate_opts=_enum_opts,
                                prep_cfg=_prep_cfg,
                                results_cfg=cfg.get("results", {}),
                                engine=_engine,
                                scoring=ctx.get("scoring", "vina"),
                            )
                            notebook_path = engine_work / "notebook.ipynb"
                            notebook_path.write_text(notebook_src, encoding="utf-8")
                            result = kaggle_runner.run_screening_job(
                                run_id=f"{run_id}-{_engine}",
                                receptor_pdbqt=receptor_pdbqt,
                                shard_paths=shard_paths,
                                notebook_path=notebook_path,
                                username=kaggle_username,
                                work_dir=engine_work,
                                retry_limit=cfg["run"].get("shard_retry_limit", 3),
                                accelerator=_gpu_type,
                                engine=_engine,
                            )
                        if result["status"] not in ("complete", "partial"):
                            break
                        engine_outputs.append(Path(result["output_dir"]))

                    if len(engine_outputs) == len(consensus_engines):
                        consensus = build_consensus_results(
                            engine_outputs, work_dir / "output"
                        )
                        result = {
                            "status": "complete",
                            "output_dir": consensus["scores_csv"].parent,
                            "error_type": None,
                        }
                else:
                    account_assignments = ctx.get("account_assignments")
                    _active_assignments = [
                        a for a in (account_assignments or []) if a["shard_count"] != 0
                    ]
                    _has_kaggle_receptor_prep = bool(receptor_variant_modes) and any(
                        m != "default" for m in receptor_variant_modes
                    )
                    if (
                        _has_kaggle_receptor_prep or _has_receptor_ensemble
                    ) and not _active_assignments:
                        _creds = auth.load_credentials()
                        _active_assignments = [
                            {
                                "account": {
                                    "name": kaggle_username,
                                    "username": kaggle_username,
                                    "kaggle_json_path": _creds.get(
                                        "kaggle_json_path", ""
                                    ),
                                },
                                "shard_count": 1,
                            }
                        ]
                    if _active_assignments and (
                        len(_active_assignments) > 1
                        or _has_kaggle_receptor_prep
                        or _has_receptor_ensemble
                    ):
                        # Multi-account/variant Kaggle path
                        if len(_active_assignments) > 1:
                            log(
                                "[#79c0ff]Submitting to multiple Kaggle accounts...[/#79c0ff]"
                            )
                        else:
                            log(
                                "[#79c0ff]Submitting packed receptor Kaggle run...[/#79c0ff]"
                            )
                        for a in _active_assignments:
                            n = a["shard_count"] or 1
                            log(
                                f"[#6e7681]  {a['account']['name']}: {n} notebook group(s)[/#6e7681]"
                            )
                        account_assignments = _active_assignments
                        result = kaggle_runner.run_multi_account_screening(
                            run_id=run_id,
                            receptor_pdbqt=receptor_pdbqt,
                            receptor_pdbqts=receptor_pdbqt_paths,
                            receptor_variant_modes=receptor_variant_modes,
                            receptor_pdb=Path(receptor_pdb) if receptor_pdb else None,
                            shard_paths=shard_paths,
                            account_assignments=account_assignments,
                            work_dir=work_dir,
                            box_center=box["center"],
                            box_size=box["size"],
                            search_mode=ctx["search_params"].get(
                                "search_mode", "balance"
                            ),
                            ph=cfg["run"].get("default_ph", 7.4),
                            retry_limit=cfg["run"].get("shard_retry_limit", 3),
                            accelerator=ctx.get("gpu_type", "nvidiaTeslaP100"),
                            prep_on_kaggle=_prep_on_kaggle,
                            force_field=_force_field,
                            enumerate_opts=_enum_opts,
                            engine=ctx.get("engine", "unidock"),
                            scoring=ctx.get("scoring", "vina"),
                        )
                    else:
                        # Single-account Kaggle path
                        _gpu_type = ctx.get("gpu_type", "nvidiaTeslaP100")
                        notebook_src = kaggle_runner.render_vina_notebook(
                            run_id=run_id,
                            shard_filenames=[p.name for p in shard_paths],
                            receptor_filename=Path(receptor_pdbqt).name,
                            box_center=box["center"],
                            box_size=box["size"],
                            search_mode=ctx["search_params"].get(
                                "search_mode", "balance"
                            ),
                            ph=cfg["run"].get("default_ph", 7.4),
                            prep_on_kaggle=_prep_on_kaggle,
                            accelerator=_gpu_type,
                            force_field=_force_field,
                            enumerate_opts=_enum_opts,
                            prep_cfg=_prep_cfg,
                            results_cfg=cfg.get("results", {}),
                            engine=ctx.get("engine", "unidock"),
                            scoring=ctx.get("scoring", "vina"),
                        )
                        notebook_path = work_dir / "notebook.ipynb"
                        notebook_path.write_text(notebook_src, encoding="utf-8")

                        log("[#79c0ff]Submitting to Kaggle...[/#79c0ff]")
                        result = kaggle_runner.run_screening_job(
                            run_id=run_id,
                            receptor_pdbqt=receptor_pdbqt,
                            shard_paths=shard_paths,
                            notebook_path=notebook_path,
                            username=kaggle_username,
                            work_dir=work_dir,
                            retry_limit=cfg["run"].get("shard_retry_limit", 3),
                            accelerator=_gpu_type,
                            engine=ctx.get("engine", "unidock"),
                        )

            if result["status"] in ("complete", "partial"):
                if result["status"] == "complete":
                    checkpoint.mark_run_complete(run_id)
                    log(
                        f"[#3fb950]\u2713 Done!  Results \u2192 {result['output_dir']}[/#3fb950]"
                    )
                else:
                    checkpoint.mark_run_partial(run_id)
                    log(
                        f"[#e3b341]\u26a0 Partial results â€” some notebooks failed. Results \u2192 {result['output_dir']}[/#e3b341]"
                    )
                try:
                    from ezscreen.results.methods import (
                        run_meta_from_checkpoint,
                        write_methods,
                    )

                    _rm = run_meta_from_checkpoint(run_id)
                    if _rm and result.get("output_dir"):
                        write_methods(_rm, Path(result["output_dir"]))
                except Exception:
                    pass
                self.app.call_from_thread(self._on_complete, run_id)
            else:
                checkpoint.mark_run_failed(run_id)
                err = result.get("error_type") or result.get("status", "unknown")
                log(f"[#f85149]\u2717 Run failed: {err}[/#f85149]")
                log(f"[#6e7681]Resume with: ezscreen resume {run_id}[/#6e7681]")
                self.app.call_from_thread(self._on_done)

        except Exception as exc:
            import traceback

            tb = traceback.format_exc()
            _log_file.write(f"Error: {exc}\n{tb}\n")
            _log_file.close()
            self.app.call_from_thread(
                self.query_one("#run-log", RichLog).write,
                f"[#f85149]Error: {exc}[/#f85149]\n[#6e7681]Full log: ~/.ezscreen/runs/last_run.log[/#6e7681]",
            )
            self.app.call_from_thread(self._on_done)
        else:
            _log_file.close()

    def _on_complete(self, run_id: str) -> None:
        from ezscreen.tui.screens.results_viewer import ResultsScreen

        self.app.switch_screen(ResultsScreen(run_id))

    def _on_done(self) -> None:
        self._submitted = False
        self._update_ui()
