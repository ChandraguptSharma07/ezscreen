from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.table import Table

console = Console()

_POLL_INTERVAL = 30   # seconds
_TIMEOUT = 21_600     # 6 hours


def _api():
    import kaggle
    kaggle.api.authenticate()
    return kaggle.api


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
    """
    Poll a Kaggle kernel until complete, failed, or timed out.
    Returns dict with keys: status, error_type, retry_count.
      status: 'complete' | 'failed' | 'timeout' | 'retry'
    """
    username, slug = kernel_ref.split("/", 1)
    api = _api()
    elapsed = retries = 0

    with Live(console=console, refresh_per_second=0.1) as live:
        while elapsed < _TIMEOUT:
            try:
                resp   = api.kernel_status(username, slug)
                status = resp.get("status", "unknown")
                errmsg = resp.get("failureMessage", "")
            except Exception:
                status = "error"
                errmsg = ""

            live.update(_status_grid(run_id, status, elapsed, retries))

            if status == "complete":
                return {"status": "complete", "error_type": None, "retry_count": retries}

            if status in ("error", "cancelAcknowledged"):
                error_type = _classify_error(errmsg)
                if error_type in ("preempted", "gpu_oom") and retries < retry_limit:
                    retries += 1
                    console.print(
                        f"\n  [yellow]⟳ Retry {retries}/{retry_limit} — {error_type}[/yellow]"
                    )
                    return {"status": "retry", "error_type": error_type, "retry_count": retries}
                return {"status": "failed", "error_type": error_type, "retry_count": retries}

            time.sleep(_POLL_INTERVAL)
            elapsed += _POLL_INTERVAL

    return {"status": "timeout", "error_type": "timeout", "retry_count": retries}
