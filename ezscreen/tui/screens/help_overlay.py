from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Footer, Static

_HELP_TEXT = """\
[bold #79c0ff]ezscreen[/bold #79c0ff]  [#6e7681]GPU-accelerated virtual screening[/#6e7681]

[bold #f0f6fc]Global[/bold #f0f6fc]
  [#79c0ff]?[/#79c0ff]          Show this help
  [#79c0ff]q[/#79c0ff]          Quit
  [#79c0ff]Escape[/#79c0ff]     Go back / close overlay
  [#79c0ff]Tab[/#79c0ff]        Focus next widget
  [#79c0ff]Shift+Tab[/#79c0ff]  Focus previous widget

[bold #f0f6fc]Home Dashboard[/bold #f0f6fc]
  [#79c0ff]n[/#79c0ff]          New Run wizard
  [#79c0ff]s[/#79c0ff]          Status Monitor
  [#79c0ff]Enter[/#79c0ff]      Open selected run in Results Viewer

[bold #f0f6fc]Status Monitor[/bold #f0f6fc]
  [#79c0ff]r[/#79c0ff]          Refresh run table

[bold #f0f6fc]Results Viewer[/bold #f0f6fc]
  [#79c0ff]o[/#79c0ff]          Open 3D viewer in browser

[bold #f0f6fc]New Run Wizard[/bold #f0f6fc]
  [#79c0ff]\u2190 Back[/#79c0ff]      Previous step
  [#79c0ff]Next \u2192[/#79c0ff]      Advance / Submit on final step

[bold #f0f6fc]Screens[/bold #f0f6fc]
  New Run       Configure and submit a virtual screening job
  Status        Live view of all runs with elapsed time and progress
  ADMET Filter  Apply drug-likeness filters to a compound library
  Validate      Re-score top hits with DiffDock-L via NVIDIA NIM
  Auth Setup    Store Kaggle credentials and NIM API key
  Settings      Edit default pH, search depth, ADMET toggle, etc.

[#6e7681]Docs: https://github.com/ChandraguptSharma07/SwiftScreen[/#6e7681]
"""


class HelpScreen(ModalScreen):
    """Full-screen help overlay."""

    BINDINGS = [
        Binding("escape",          "app.pop_screen", "Close"),
        Binding("question_mark",   "app.pop_screen", "Close"),
        Binding("q",               "app.pop_screen", "Close"),
    ]

    def compose(self) -> ComposeResult:
        yield Static(_HELP_TEXT, id="help-content")
        yield Footer()
