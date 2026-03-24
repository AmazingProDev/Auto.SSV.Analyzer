from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkbookSheet:
    name: str
    path: str
    drawing_path: str | None
    cells: dict[tuple[int, int], str]
    score: float


@dataclass
class ImageCandidate:
    sheet_name: str
    sheet_path: str
    drawing_path: str
    media_path: str
    target_key: str
    target_label: str
    anchor_row: int
    anchor_col: int
    score: float
    analysis_kind: str = "cross"
    metric_group: str | None = None
    metric_name: str | None = None
    nearby_text: list[str] = field(default_factory=list)
    caption_ref: str | None = None


@dataclass
class Bitmap:
    width: int
    height: int
    pixels: list[list[tuple[int, int, int]]] | None = None


@dataclass(slots=True)
class DotComponent:
    pixels: list[tuple[int, int]]
    area: int
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    width: int
    height: int
    fill_ratio: float
    mean_rgb: tuple[float, float, float]
    mean_lab: tuple[float, float, float]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass(slots=True)
class LegendSwatch:
    bbox: tuple[int, int, int, int]
    center_y: float
    rgb: tuple[float, float, float]
    lab: tuple[float, float, float]
    hue_degrees: float
    saturation: float
    value: float


@dataclass
class DetectedColor:
    name: str
    rgb: tuple[int, int, int]
    hex: str
    dominant_angle: float
    point_count: int
    site_angle: float | None = None


@dataclass
class AnalysisOutcome:
    cross: bool
    verdict: str
    detected_colors: list[DetectedColor]
    metrics: dict[str, Any]
    site_center: dict[str, float]
    annotated_preview: str
    analysis_kind: str = "cross"
    is_failure: bool | None = None
    warnings: list[str] = field(default_factory=list)
    warning_details: list[dict[str, Any]] = field(default_factory=list)
