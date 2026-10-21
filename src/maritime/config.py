"""Region configuration.

A region is the only thing that changes when the pipeline moves to a new part of the
world: a bounding box, a data source, a spatial resolution, and optional reference
layers. Detection code should never hardcode geography; it should take a Region.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from maritime.paths import REGIONS, ROOT


@dataclass(frozen=True)
class BBox:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def __post_init__(self) -> None:
        if not (-180 <= self.min_lon < self.max_lon <= 180):
            raise ValueError(f"invalid longitude range: {self.min_lon}..{self.max_lon}")
        if not (-90 <= self.min_lat < self.max_lat <= 90):
            raise ValueError(f"invalid latitude range: {self.min_lat}..{self.max_lat}")

    def contains(self, lon: float, lat: float) -> bool:
        return self.min_lon <= lon <= self.max_lon and self.min_lat <= lat <= self.max_lat


@dataclass(frozen=True)
class SourceConfig:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Region:
    name: str
    display_name: str
    bbox: BBox
    h3_resolution: int
    source: SourceConfig
    default_start: date
    default_end: date
    zones_of_interest: Path | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Region:
        period = d.get("default_period", {})
        zones = d.get("zones_of_interest")
        res = int(d.get("h3_resolution", 6))
        if not 0 <= res <= 15:
            raise ValueError(f"h3_resolution must be 0..15, got {res}")
        return cls(
            name=d["name"],
            display_name=d.get("display_name", d["name"]),
            bbox=BBox(**d["bbox"]),
            h3_resolution=res,
            source=SourceConfig(**d["source"]),
            default_start=_as_date(period.get("start")),
            default_end=_as_date(period.get("end")),
            zones_of_interest=(ROOT / zones) if zones else None,
            notes=(d.get("notes") or "").strip(),
        )


def _as_date(v: Any) -> date:
    if isinstance(v, date):
        return v
    return date.fromisoformat(str(v))


def load_region(name_or_path: str | Path) -> Region:
    """Load by short name (e.g. 'baltic') or by path to a YAML file."""
    p = Path(name_or_path)
    if p.suffix not in {".yaml", ".yml"}:
        p = REGIONS / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"no region config at {p}; available: {available_regions()}")
    with p.open() as f:
        return Region.from_dict(yaml.safe_load(f))


def available_regions() -> list[str]:
    return sorted(p.stem for p in REGIONS.glob("*.yaml"))
