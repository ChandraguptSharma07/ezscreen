from __future__ import annotations

import smtplib
from email.message import EmailMessage


def _notify_config() -> dict:
    try:
        from ezscreen import config
        return config.load().get("notify", {})
    except Exception:
        return {}


def _send_desktop(run_id: str, status: str, message: str | None) -> None:
    try:
        from plyer import notification  # type: ignore[import]
        notification.notify(
            title=f"ezscreen — {run_id}",
            message=message or f"Run {status}.",
            app_name="ezscreen",
            timeout=8,
        )
    except Exception:
        pass  # plyer not installed or no display


def _send_email(run_id: str, status: str, message: str | None) -> None:
    nc        = _notify_config()
    smtp_host = nc.get("smtp_host", "")
    smtp_port = int(nc.get("smtp_port", 587))
    from_addr = nc.get("from_address", "")
    to_addr   = nc.get("to_address", "")

    if not (smtp_host and from_addr and to_addr):
        return

    try:
        msg = EmailMessage()
        msg["Subject"] = f"[ezscreen] {run_id} — {status}"
        msg["From"]    = from_addr
        msg["To"]      = to_addr
        msg.set_content(message or f"Run {run_id} finished with status: {status}.")
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as smtp:
            smtp.starttls()
            smtp.send_message(msg)
    except Exception:
        pass


def send_run_complete(run_id: str, status: str, message: str | None = None) -> None:
    nc = _notify_config()
    if nc.get("desktop_enabled", False):
        _send_desktop(run_id, status, message)
    _send_email(run_id, status, message)
