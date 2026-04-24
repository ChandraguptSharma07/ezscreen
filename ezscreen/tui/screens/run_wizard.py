from __future__ import annotations

import secrets
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
    Static,
    Switch,
)

from ezscreen.tui.widgets.breadcrumb import Breadcrumb

_STEPS = [
    "step-receptor",
    "step-site",
    "step-ligands",
    "step-options",
    "step-confirm",
]

_STEP_LABELS = [
    "Step 1 of 5 — Receptor & Chains",
    "Step 2 of 5 — Binding Site",
    "Step 3 of 5 — Ligand Library",
    "Step 4 of 5 — Run Options",
    "Step 5 of 5 — Confirm & Submit",
]


class RunWizardScreen(Screen):
    """Multi-step wizard for configuring and submitting a virtual screening run."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Back")]

    def __init__(self) -> None:
        super().__init__()
        self._step = 0
        self._ctx: dict = {}
        self._chains: list[str] = []
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

                # ── Step 1: Receptor & Chains ─────────────────────────
                with Vertical(id="step-receptor", classes="wizard-step"):
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

                # ── Step 2: Binding Site ──────────────────────────────
                with Vertical(id="step-site", classes="wizard-step"):
                    yield Label("Binding site method", classes="form-label")
                    with RadioSet(id="site-method"):
                        yield RadioButton("Co-crystal ligand",         id="rb-cocrystal")
                        yield RadioButton("Active site residues",       id="rb-residues")
                        yield RadioButton("P2Rank pocket prediction",   id="rb-p2rank",  value=True)
                        yield RadioButton("Blind docking  \u26a0",      id="rb-blind")
                        yield RadioButton("Enter coordinates manually", id="rb-manual")
                    with Vertical(id="sub-cocrystal", classes="site-sub"):
                        yield Static("", id="cocrystal-info")
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
                            "[\u26a0] Blind docking scans the entire protein — "
                            "slow and noisy. Only use if no binding pocket is known.",
                            id="blind-warning",
                        )
                    with Vertical(id="sub-manual", classes="site-sub"):
                        yield Label("Box center (Å)", classes="form-label")
                        with Horizontal(classes="form-row"):
                            yield Input(placeholder="X", id="manual-cx")
                            yield Input(placeholder="Y", id="manual-cy")
                            yield Input(placeholder="Z", id="manual-cz")
                        yield Label("Box size (Å)", classes="form-label")
                        with Horizontal(classes="form-row"):
                            yield Input(placeholder="X", id="manual-sx")
                            yield Input(placeholder="Y", id="manual-sy")
                            yield Input(placeholder="Z", id="manual-sz")

                # ── Step 3: Ligand Library ────────────────────────────
                with Vertical(id="step-ligands", classes="wizard-step"):
                    yield Label(
                        "Ligand library  (.sdf, .smi, or .ism file)",
                        classes="form-label",
                    )
                    with Horizontal(classes="form-row"):
                        yield Input(placeholder="/path/to/library.sdf", id="lig-input")
                        yield Button("Download library", id="btn-lib-browser")
                    yield Static("", id="lig-status", classes="form-status")

                # ── Step 4: Run Options ───────────────────────────────
                with Vertical(id="step-options", classes="wizard-step"):
                    yield Label("ADMET pre-filter", classes="form-section")
                    yield Static(
                        "[#6e7681]Remove obvious drug-like failures before docking"
                        " (recommended)[/#6e7681]"
                    )
                    yield Switch(id="opt-admet", value=True)
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
                    yield Label("Compute backend", classes="form-section")
                    yield Static(
                        "[#6e7681]Run locally on CPU (AutoDock Vina) — no Kaggle account needed."
                        " Slower than GPU for large libraries.[/#6e7681]"
                    )
                    yield Switch(id="opt-local", value=False)
                    with Vertical(id="local-options-section"):
                        yield Label("Exhaustiveness (Vina)", classes="form-label")
                        yield Input(id="opt-exhaustiveness", placeholder="4 (from settings)")
                    with Vertical(id="kaggle-gpu-section"):
                        yield Label("Kaggle GPU", classes="form-section")
                        with RadioSet(id="opt-gpu-type"):
                            yield RadioButton(
                                "P100 (16 GB) ★ — standard, widest availability",
                                id="rb-p100",
                                value=True,
                            )
                            yield RadioButton(
                                "T4 × 2 (32 GB) — more memory, uses both GPUs",
                                id="rb-t4x2",
                            )
                    yield VerticalScroll(id="acct-assignment-section")

                # ── Step 5: Confirm & Submit ──────────────────────────
                with Vertical(id="step-confirm", classes="wizard-step"):
                    yield Label("Ready to submit", classes="form-section")
                    yield Static("", id="confirm-summary", classes="confirm-summary")
                    yield Label("Output", classes="form-section")
                    yield RichLog(id="run-log", highlight=True, markup=True)

            with Horizontal(id="wizard-nav"):
                yield Button("\u2190 Back", id="btn-back")
                yield Button("Next \u2192",  id="btn-next", variant="primary")

        yield Footer()

    # ------------------------------------------------------------------
    # Mount
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        self._update_ui()
        self.query_one("#af-warning-box").display = False
        self.query_one("#chain-section").display  = False
        for sid in ("sub-cocrystal", "sub-residues", "sub-p2rank", "sub-blind", "sub-manual"):
            self.query_one(f"#{sid}").display = False
        self.query_one("#local-options-section").display = False
        self.query_one("#kaggle-gpu-section").display = True
        self._load_defaults()
        self._populate_account_assignment()

    def _load_defaults(self) -> None:
        try:
            from ezscreen import config
            cfg   = config.load()
            admet = cfg.get("run", {}).get("admet_pre_filter", True)
            self.query_one("#opt-admet", Switch).value = bool(admet)
        except Exception:
            pass

    def _populate_account_assignment(self) -> None:
        from ezscreen import auth
        accounts = auth.get_all_kaggle_accounts()
        section  = self.query_one("#acct-assignment-section", VerticalScroll)
        section.remove_children()
        if len(accounts) <= 1:
            section.display = False
            return

        section.display = True
        section.mount(Label("Account notebook assignment (Kaggle)", classes="form-section"))
        section.mount(Static(
            "[#6e7681]Notebooks per account (leave blank for 1 per account).[/#6e7681]"
        ))
        for acct in accounts:
            name = acct["name"]
            safe = name.replace(" ", "-").lower()
            section.mount(
                Horizontal(
                    Label(f"  {name}", classes="acct-row-label"),
                    Input(id=f"acct-shards-{safe}", placeholder="1", classes="acct-row-input"),
                    classes="acct-row",
                )
            )

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "opt-local":
            local_on = event.value
            self.query_one("#local-options-section").display = local_on
            self.query_one("#kaggle-gpu-section").display = not local_on
            self.query_one("#lbl-depth").display = not local_on
            self.query_one("#opt-depth").display = not local_on
            acct_section = self.query_one("#acct-assignment-section", VerticalScroll)
            if local_on:
                acct_section.display = False
            else:
                self._populate_account_assignment()

    def _update_ui(self) -> None:
        self.query_one("#step-indicator", Static).update(
            f"[#6e7681]{_STEP_LABELS[self._step]}[/#6e7681]"
        )
        self.query_one("#wizard-switcher", ContentSwitcher).current = _STEPS[self._step]
        is_last = self._step == len(_STEPS) - 1
        self.query_one("#btn-next",  Button).label    = "Submit" if is_last else "Next \u2192"
        self.query_one("#btn-back",  Button).disabled = self._step == 0 or self._submitted
        self.query_one("#btn-next",  Button).disabled = self._submitted

    # ------------------------------------------------------------------
    # Button / RadioSet events
    # ------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        dispatch = {
            "btn-rec-validate": self._start_rec_validate,
            "btn-lib-browser":  self._open_lib_browser,
            "btn-back":         self._go_back,
            "btn-next":         self._go_next,
        }
        if fn := dispatch.get(event.button.id):
            fn()

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

    def _show_site_sub(self, label: str) -> None:
        for sid in ("sub-cocrystal", "sub-residues", "sub-p2rank", "sub-blind", "sub-manual"):
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

    # ------------------------------------------------------------------
    # Step 1 — Receptor validation
    # ------------------------------------------------------------------

    def _start_rec_validate(self) -> None:
        raw = self.query_one("#rec-input", Input).value.strip()
        if not raw:
            self.query_one("#rec-status", Static).update(
                "[#f85149]Enter a PDB ID or file path.[/#f85149]"
            )
            return
        self.query_one("#rec-status",    Static).update("[#e3b341]Validating...[/#e3b341]")
        self.query_one("#btn-rec-validate").disabled = True
        self.query_one("#chain-section").display     = False
        self.query_one("#af-warning-box").display    = False
        self._chains = []
        self._cocrystal_ligands = []
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
                self._ctx["pdb_id"]     = uid
            elif len(raw) == 4 and raw.isalnum():
                pdb_path = rec_prep.fetch_pdb(raw.upper(), work_dir / "receptor_raw")
                self._ctx["pdb_source"] = "rcsb"
                self._ctx["pdb_id"]     = raw.upper()
            else:
                pdb_path = Path(raw).expanduser()
                if not pdb_path.exists():
                    raise FileNotFoundError(f"File not found: {pdb_path}")
                self._ctx["pdb_source"] = "local"
                self._ctx["pdb_id"]     = None

            self._ctx["pdb_path"]     = pdb_path
            self._ctx["receptor_raw"] = raw

            chains        = rec_prep.get_chains(pdb_path)
            is_af, af_ver = rec_prep.detect_alphafold(pdb_path)
            self._ctx["chains"]       = chains
            self._ctx["is_alphafold"] = is_af
            self._ctx["af_version"]   = af_ver

            from ezscreen.pocket import detect as pocket
            cocrystal = [] if is_af else pocket.find_cocrystal_ligands(pdb_path)
            self._cocrystal_ligands        = cocrystal
            self._ctx["cocrystal_ligands"] = cocrystal

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
        co_txt = f"  {len(cocrystal)} co-crystal ligand(s) found." if cocrystal else ""
        self.query_one("#rec-status", Static).update(
            f"[#3fb950]Valid — {len(chains)} chain(s) found.{co_txt}[/#3fb950]"
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

        if is_af:
            ver_label = {"af2": "AlphaFold 2", "af3": "AlphaFold 3"}.get(
                af_ver or "", "AlphaFold"
            )
            self.query_one("#af-warning", Static).update(
                f"[#e3b341]\u26a0  {ver_label} structure detected — "
                "P2Rank will use AF profile. Pocket reliability is lower.[/#e3b341]"
            )
            self.query_one("#af-warning-box").display = True

        if cocrystal:
            names = ", ".join(lig["resname"] for lig in cocrystal)
            self.query_one("#cocrystal-info", Static).update(
                f"[#3fb950]Found: {names}[/#3fb950]"
            )
        else:
            self.query_one("#cocrystal-info", Static).update(
                "[#6e7681]No co-crystal ligand found in this structure.[/#6e7681]"
            )

    def _show_rec_error(self, msg: str) -> None:
        self.query_one("#rec-status", Static).update(f"[#f85149]{msg}[/#f85149]")
        self.query_one("#btn-rec-validate").disabled = False

    # ------------------------------------------------------------------
    # Step 2 — P2Rank
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
            is_af    = self._ctx.get("is_alphafold", False)
            work_dir = Path.home() / ".ezscreen" / "tmp" / "wizard" / "p2rank"
            pockets  = pocket.run_p2rank(pdb_path, work_dir, alphafold=is_af)
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
                "[#e3b341]No pockets found — will fall back to blind docking.[/#e3b341]"
            )
            return

        self.query_one("#p2rank-status", Static).update(
            f"[#3fb950]{len(pockets)} pocket(s) found — select one below.[/#3fb950]"
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
                ch for ch in self._chains
                if self.query_one(f"#chain-{ch if ch.strip() else '_blank'}", Checkbox).value
            ]
            if not selected:
                return "Select at least one chain."
            self._ctx["selected_chains"] = selected

        elif step == "step-site":
            btn = self.query_one("#site-method", RadioSet).pressed_button
            if btn is None:
                return "Select a binding site method."
            label = str(btn.label)
            if "Co-crystal" in label and not self._cocrystal_ligands:
                return "No co-crystal ligand found. Choose a different method."
            if "P2Rank" in label and not self._pockets:
                return "P2Rank has not finished yet. Wait or choose a different method."
            if "manually" in label:
                err = self._validate_manual_coords()
                if err:
                    return err
            self._ctx["site_method"]  = label
            self._ctx["site_details"] = self._collect_site_details(label)

        elif step == "step-ligands":
            raw = self.query_one("#lig-input", Input).value.strip()
            if not raw:
                return "Enter a ligand file path."
            p = Path(raw).expanduser()
            if not p.exists():
                return f"File not found: {p}"
            self._ctx["ligand_path"] = p

        elif step == "step-options":
            self._ctx["admet_pre_filter"] = self.query_one("#opt-admet", Switch).value
            self._ctx["run_locally"] = self.query_one("#opt-local", Switch).value
            gpu_btn = self.query_one("#opt-gpu-type", RadioSet).pressed_button
            self._ctx["gpu_type"] = (
                "nvidiaTeslaT4x2"
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
                from ezscreen import auth
                accounts = auth.get_all_kaggle_accounts()
                if len(accounts) > 1:
                    assignments = []
                    for acct in accounts:
                        safe = acct["name"].replace(" ", "-").lower()
                        try:
                            raw   = self.query_one(f"#acct-shards-{safe}", Input).value.strip()
                            count = int(raw) if raw else 0
                        except Exception:
                            count = 0
                        assignments.append({"account": acct, "shard_count": count})
                    self._ctx["account_assignments"] = assignments
                else:
                    self._ctx["account_assignments"] = None
            else:
                self._ctx["account_assignments"] = None
            depth_btn = self.query_one("#opt-depth", RadioSet).pressed_button
            label     = str(depth_btn.label) if depth_btn else ""
            if "Fast" in label:
                self._ctx["search_params"] = {"search_mode": "fast"}
                self._ctx["search_label"]  = "Fast"
            elif "Thorough" in label:
                self._ctx["search_params"] = {"search_mode": "detail"}
                self._ctx["search_label"]  = "Thorough"
            else:
                self._ctx["search_params"] = {"search_mode": "balance"}
                self._ctx["search_label"]  = "Balanced"

        return None

    def _validate_manual_coords(self) -> str | None:
        fields = ["manual-cx", "manual-cy", "manual-cz", "manual-sx", "manual-sy", "manual-sz"]
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
            return {"type": "cocrystal", "ligands": self._cocrystal_ligands}
        if "residues" in label:
            raw     = self.query_one("#residues-input", Input).value.strip()
            res_ids = [int(r.strip()) for r in raw.split(",") if r.strip().isdigit()]
            return {"type": "residues", "residues": res_ids}
        if "P2Rank" in label:
            try:
                rs  = self.query_one("#p2rank-radioset", RadioSet)
                idx = rs.pressed_index if rs.pressed_index is not None else 0
                return {"type": "p2rank", "pocket": self._pockets[idx]}
            except Exception:
                return {"type": "p2rank", "pocket": self._pockets[0] if self._pockets else {}}
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
                "size":   [sx, sy, sz],
            }
        return {"type": "blind"}

    # ------------------------------------------------------------------
    # Step 5 — Confirm summary
    # ------------------------------------------------------------------

    def _populate_confirm(self) -> None:
        ctx   = self._ctx
        pdb   = ctx.get("pdb_id") or (ctx["pdb_path"].name if ctx.get("pdb_path") else "—")
        chains = ", ".join(ctx.get("selected_chains", []))
        site  = ctx.get("site_method", "—")
        lig   = ctx["ligand_path"].name if ctx.get("ligand_path") else "—"
        admet   = "yes" if ctx.get("admet_pre_filter") else "no"
        depth   = ctx.get("search_label", "Balanced")
        if ctx.get("run_locally"):
            exh = ctx.get("exhaustiveness") or "default (from settings)"
            backend = f"Local CPU (AutoDock Vina, exhaustiveness={exh})"
        else:
            gpu_label = "T4 × 2" if ctx.get("gpu_type") == "nvidiaTeslaT4x2" else "P100"
            backend = f"Kaggle GPU ({gpu_label})"
        af_note = (
            "\n[#e3b341]\u26a0  AlphaFold structure — P2Rank AF profile active[/#e3b341]"
            if ctx.get("is_alphafold") else ""
        )
        self.query_one("#confirm-summary", Static).update("\n".join([
            f"[bold #6e7681]Receptor     [/bold #6e7681][#f0f6fc]{pdb}[/#f0f6fc]  chains {chains}",
            f"[bold #6e7681]Binding site [/bold #6e7681][#f0f6fc]{site}[/#f0f6fc]",
            f"[bold #6e7681]Ligands      [/bold #6e7681][#f0f6fc]{lig}[/#f0f6fc]",
            f"[bold #6e7681]ADMET filter [/bold #6e7681][#f0f6fc]{admet}[/#f0f6fc]",
            f"[bold #6e7681]Search depth [/bold #6e7681][#f0f6fc]{depth}[/#f0f6fc]",
            f"[bold #6e7681]Backend      [/bold #6e7681][#f0f6fc]{backend}[/#f0f6fc]",
            af_note,
        ]))

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def _submit(self) -> None:
        if self._submitted:
            return
        self._submitted = True
        self._update_ui()
        self.run_worker(lambda: self._do_submit(), thread=True)

    def _do_submit(self) -> None:
        from ezscreen import auth, checkpoint, config
        from ezscreen.admet.filter import filter_library
        from ezscreen.backends.kaggle import runner as kaggle_runner
        from ezscreen.pocket import detect as pocket
        from ezscreen.prep import ligands as ligand_prep
        from ezscreen.prep import receptor as rec_prep

        _log_file = (Path.home() / ".ezscreen" / "runs" / "last_run.log").open("w", buffering=1, encoding="utf-8")

        def log(msg: str) -> None:
            self.app.call_from_thread(self.query_one("#run-log", RichLog).write, msg)
            import re
            _log_file.write(re.sub(r"\[/?[^\[\]]*\]", "", msg) + "\n")

        try:
            run_id   = "ezs-" + secrets.token_hex(3)
            work_dir = Path.home() / ".ezscreen" / "runs" / run_id
            work_dir.mkdir(parents=True, exist_ok=True)
            ctx      = self._ctx
            cfg      = config.load()

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
            )
            receptor_pdbqt = rec_result["pdbqt_path"]

            # Binding box
            log("[#79c0ff]Computing binding box...[/#79c0ff]")
            site = ctx["site_details"]
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
            log(f"[#6e7681]Box: center {box['center']}  size {box['size']}[/#6e7681]")

            # ADMET pre-filter
            ligand_input = ctx["ligand_path"]
            if ctx.get("admet_pre_filter"):
                log("[#79c0ff]Running ADMET pre-filter...[/#79c0ff]")
                admet_out     = work_dir / "admet_filtered.sdf"
                admet_summary = filter_library(str(ligand_input), str(admet_out))
                total   = admet_summary.get("total_input", 0)
                removed = admet_summary.get("admet_removed", 0)
                log(f"[#6e7681]ADMET: {total - removed:,}/{total:,} passed[/#6e7681]")
                ligand_input  = admet_out

            # Ligand prep
            account_assignments = ctx.get("account_assignments")
            if account_assignments and len(account_assignments) > 1:
                n_shards_target = sum(a["shard_count"] or 1 for a in account_assignments)
            else:
                n_shards_target = None

            if ctx.get("run_locally"):
                log("[#79c0ff]Prepping ligands locally...[/#79c0ff]")
                lig_result = ligand_prep.prep_ligands(
                    input_path=ligand_input,
                    output_dir=work_dir / "shards",
                    ph=cfg["run"].get("default_ph", 7.4),
                    n_shards=n_shards_target,
                )
                log(f"[#6e7681]{len(lig_result['shard_paths'])} shard(s) ready ({lig_result['report']['prep_passed']:,} ligands prepped)[/#6e7681]")
            else:
                log("[#79c0ff]Sharding ligands for Kaggle prep...[/#79c0ff]")
                lig_result = ligand_prep.shard_raw(
                    input_path=ligand_input,
                    output_dir=work_dir / "shards",
                    n_shards=n_shards_target,
                )
                log(f"[#6e7681]{len(lig_result['shard_paths'])} shard(s) ready ({lig_result['report']['total_input']:,} ligands)[/#6e7681]")

            shard_paths = lig_result["shard_paths"]
            if not shard_paths:
                raise RuntimeError("Ligand prep produced 0 dockable compounds.")

            # Checkpoint
            checkpoint.init_db()
            checkpoint.create_run(run_id, ctx, lig_result["report"]["total_input"])
            for i, _ in enumerate(shard_paths):
                n_per = lig_result["report"]["total_input"] // len(shard_paths)
                checkpoint.add_shard(run_id, i, n_per)

            if ctx.get("run_locally"):
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
                account_assignments = ctx.get("account_assignments")
                if account_assignments and len(account_assignments) > 1:
                    # Multi-account Kaggle path
                    log("[#79c0ff]Submitting to multiple Kaggle accounts...[/#79c0ff]")
                    for a in account_assignments:
                        n = a["shard_count"] or 1
                        log(f"[#6e7681]  {a['account']['name']}: {n} notebook(s)[/#6e7681]")
                    result = kaggle_runner.run_multi_account_screening(
                        run_id=run_id,
                        receptor_pdbqt=receptor_pdbqt,
                        shard_paths=shard_paths,
                        account_assignments=account_assignments,
                        work_dir=work_dir,
                        box_center=box["center"],
                        box_size=box["size"],
                        search_mode=ctx["search_params"].get("search_mode", "balance"),
                        ph=cfg["run"].get("default_ph", 7.4),
                        retry_limit=cfg["run"].get("shard_retry_limit", 3),
                        accelerator=ctx.get("gpu_type", "nvidiaTeslaP100"),
                    )
                else:
                    # Single-account Kaggle path
                    import jinja2

                    from ezscreen import __version__

                    template_path = (
                        Path(__file__).parent.parent.parent
                        / "backends" / "kaggle" / "templates" / "vina_shard.ipynb.j2"
                    )
                    env = jinja2.Environment(
                        variable_start_string="<<",
                        variable_end_string=">>",
                        block_start_string="<%",
                        block_end_string="%>",
                        loader=jinja2.FileSystemLoader(str(template_path.parent)),
                    )
                    _prep_cfg = cfg.get("prep", {})
                    _gpu_type = ctx.get("gpu_type", "nvidiaTeslaP100")
                    notebook_src = env.get_template(template_path.name).render(
                        ezscreen_version=__version__,
                        run_id=run_id,
                        engine="unidock",
                        mode="hybrid",
                        box_center=box["center"],
                        box_size=box["size"],
                        notebook_index=0,
                        total_notebooks=1,
                        shard_filenames=[p.name for p in shard_paths],
                        ph=cfg["run"].get("default_ph", 7.4),
                        search_mode=ctx["search_params"].get("search_mode", "balance"),
                        enumerate_tautomers=False,
                        prep_on_kaggle=True,
                        max_heavy_atoms=int(_prep_cfg.get("max_heavy_atoms", 70)),
                        max_mw=float(_prep_cfg.get("max_mw", 700.0)),
                        max_rotatable_bonds=int(_prep_cfg.get("max_rotatable_bonds", 20)),
                        gpu_ids="0,1" if _gpu_type == "nvidiaTeslaT4x2" else "",
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
                    )

            if result["status"] in ("complete", "partial"):
                if result["status"] == "complete":
                    checkpoint.mark_run_complete(run_id)
                    log(f"[#3fb950]\u2713 Done!  Results \u2192 {result['output_dir']}[/#3fb950]")
                else:
                    checkpoint.mark_run_complete(run_id)
                    log(f"[#e3b341]\u26a0 Partial results — some notebooks failed. Results \u2192 {result['output_dir']}[/#e3b341]")
                self.app.call_from_thread(self._on_complete, run_id)
            else:
                checkpoint.mark_run_failed(run_id)
                err = result.get('error_type') or result.get('status', 'unknown')
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
