"""
ezscreen.vendor.scrubber — thin shim for the forlilab/scrubber package.

Scrubber is not on PyPI, so it cannot be listed as a normal dependency.
We re-export it here so the rest of the codebase imports from one place.

Install scrubber (one-time, optional but recommended):
    pip install git+https://github.com/forlilab/scrubber.git

If scrubber is not installed, ezscreen falls back to RDKit-only preparation
(no tautomer enumeration, no pH-driven protonation).
"""
from __future__ import annotations

try:
    # Real scrubber package — installed separately by the user
    from scrubber import (
        Scrubber,  # noqa: F401  (re-export)
        ScrubberError,  # noqa: F401
    )
    SCRUBBER_AVAILABLE = True
except ImportError:
    SCRUBBER_AVAILABLE = False

    class Scrubber:                        # type: ignore[no-redef]
        """Stub used when forlilab/scrubber is not installed."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "[ezscreen] scrubber is not installed.\n"
                "For tautomer enumeration and pH-driven protonation, run:\n"
                "  pip install git+https://github.com/forlilab/scrubber.git\n"
                "Without it, ezscreen falls back to RDKit-only preparation."
            )

    class ScrubberError(Exception):       # type: ignore[no-redef]
        pass


__all__ = ["Scrubber", "ScrubberError", "SCRUBBER_AVAILABLE"]
