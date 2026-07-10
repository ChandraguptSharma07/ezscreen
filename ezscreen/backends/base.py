from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ezscreen.backends import engines


@dataclass(frozen=True)
class DockingBox:
    center: tuple[float, float, float]
    size: tuple[float, float, float]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Sequence[float]]) -> "DockingBox":
        return cls(
            center=tuple(float(v) for v in data["center"]),  # type: ignore[arg-type]
            size=tuple(float(v) for v in data["size"]),      # type: ignore[arg-type]
        )


DockingConfig = Mapping[str, Any]
DockingResult = dict[str, Any]


class DockingBackend(ABC):
    """Runtime contract for engines declared in backends.engines."""

    engine_key = engines.DEFAULT_ENGINE

    @property
    def profile(self) -> engines.EngineProfile:
        return engines.get(self.engine_key)

    @property
    def score_type(self) -> str:
        return self.profile.native_score_type

    @property
    def requires_gpu(self) -> bool:
        return self.profile.requires_gpu

    @property
    def requires_box(self) -> bool:
        return self.profile.requires_box

    @abstractmethod
    def run(
        self,
        ligands: Sequence[Path],
        receptor: Path,
        box: DockingBox | None,
        config: DockingConfig,
    ) -> DockingResult:
        """Run docking and return the legacy result dict."""
