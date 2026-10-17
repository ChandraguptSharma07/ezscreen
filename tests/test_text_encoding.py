from __future__ import annotations

from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "ezscreen"
GARBLED = ("â€", "â”", "â•", "â†", "âœ", "â˜", "âš", "Ã", "Â")


def test_package_text_has_no_garbled_characters():
    offenders = [
        f"{path.relative_to(PACKAGE.parent)}:{number}"
        for path in sorted(PACKAGE.rglob("*"))
        if path.suffix in {".py", ".j2", ".tcss", ".yml"}
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if any(token in line for token in GARBLED)
    ]

    assert offenders == []
