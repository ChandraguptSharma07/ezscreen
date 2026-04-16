from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

console = Console()

_POLL_INTERVAL = 30   # seconds
_TIMEOUT = 21_600     # 6 hours


def _api():
    import kaggle
    kaggle.api.authenticate()
    return kaggle.api


def _fetch_and_show_logs(kernel_ref: str) -> None:
    """Download kernel output and print error.json if present."""
    api = _api()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            api.kernels_output(kernel_ref, path=tmp, quiet=True)
            error_file = Path(tmp) / "error.json"
            if error_file.exists():
                data = json.loads(error_file.read_text())
                cell  = data.get("cell", "?")
                msg   = data.get("message") or data.get("stderr") or data.get("cmd", "")
                console.print(Panel(
                    f"[bold]Cell:[/bold] {cell}\n\n{msg}",
                    title="[red]Notebook error[/red]",
                    border_style="red",
                ))
            else:
                console.print("  [dim](no error.json in kernel output — cell may have failed before writing it)[/dim]")
    except Exception as exc:
        console.print(f"  [dim]Could not fetch kernel logs: {exc}[/dim]")


def _notify(run_id: str, status: str) -> None:
    try:
        from ezscreen.notify import send_run_complete
        send_run_complete(run_id, status)
    except Exception:
        pass


def _classify_error(msg: str) -> str:
    m = msg.lower()
    if "preempt" in m or "interrupt" in m:
        return "preempted"
    if "out of memory" in m or "oom" in m or "cuda out" in m:
        return "gpu_oom"
    if "pdbfixer" in m or "meeko" in m or "prep" in m:
        return "prep_failure"
    if "timeout" in m:
        return "timeout"
    return "unknown"


def _status_grid(run_id: str, status: str, elapsed: int, retries: int) -> Table:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold dim")
    t.add_column()
    t.add_row("Run",     run_id)
    t.add_row("Status",  status)
    t.add_row("Elapsed", f"{elapsed // 60}m {elapsed % 60}s")
    if retries:
        t.add_row("Retries", str(retries))
    return t


def poll_until_done(
    kernel_ref: str,
    run_id: str,
    retry_limit: int = 3,
) -> dict[str, Any]:
    """Poll a Kaggle kernel until complete, failed, or timed out.

    Returns dict with keys: status, error_type, retry_count.
      status: 'complete' | 'failed' | 'timeout' | 'retry'
    """
    username, slug = kernel_ref.split("/", 1)
    api = _api()
    elapsed = retries = 0

    # Give Kaggle time to register the kernel before the first status check —
    # polling immediately after push returns "error" while the kernel is still queuing
    time.sleep(15)
    elapsed += 15

    with Live(console=console, refresh_per_second=0.1) as live:
        while elapsed < _TIMEOUT:
            try:
                resp = api.kernels_status(kernel_ref)
                # resp.status is an Enum like <KernelWorkerStatus.COMPLETE: 2>
                status_raw = str(getattr(resp, "status", "unknown"))
                status = status_raw.split('.')[-1].lower() # e.g. 'complete', 'error', 'queued'
                errmsg = getattr(resp, "failure_message", "") or ""
            except Exception as e:
                status = "error"
                errmsg = f"Poller crashed: {e}"

            live.update(_status_grid(run_id, status, elapsed, retries))

            if status == "complete":
                _notify(run_id, "complete")
                return {"status": "complete", "error_type": None, "retry_count": retries}

            if status in ("error", "cancelacknowledged", "unknown"):
                error_type = _classify_error(errmsg)
                if errmsg:
                    console.print(f"\n  [red]Kaggle failureMessage:[/red] {errmsg}")
                console.print("\n  [dim]Fetching notebook logs...[/dim]")
                _fetch_and_show_logs(kernel_ref)
                if error_type in ("preempted", "gpu_oom") and retries < retry_limit:
                    retries += 1
                    console.print(
                        f"\n  [yellow]⟳ Retry {retries}/{retry_limit} — {error_type}[/yellow]"
                    )
                    return {"status": "retry", "error_type": error_type, "retry_count": retries}
                _notify(run_id, f"failed ({error_type})")
                return {"status": "failed", "error_type": error_type, "retry_count": retries}

            time.sleep(_POLL_INTERVAL)
            elapsed += _POLL_INTERVAL

    _notify(run_id, "timeout")
    return {"status": "timeout", "error_type": "timeout", "retry_count": retries}
