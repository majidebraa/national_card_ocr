"""Card layouts: where each field, marker and probe lives on a rectified card.

A layout is plain JSON in the coordinate system of the rectified card (default 1011x638 px).
The same files drive the OCR crops, the side/version classifier and the synthetic generator,
so there is a single source of truth. To support a real card design, copy a layout, measure
the boxes on a rectified sample (``idcard-ocr overlay``) and point ``IDCARD_LAYOUT_DIR`` at it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from idcard_ocr.errors import LayoutError

VALID_KINDS = frozenset({"national_id", "name", "date", "serial"})
VALID_SIDES = frozenset({"front", "back"})
VALID_MARKERS = frozenset({"photo", "barcode", "chip"})


@dataclass(frozen=True, slots=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    def scaled(self, sx: float, sy: float) -> Box:
        return Box(round(self.x * sx), round(self.y * sy), round(self.w * sx), round(self.h * sy))

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


@dataclass(frozen=True, slots=True)
class FieldSpec:
    name: str
    roi: Box
    kind: str
    label: str = ""
    label_roi: Box | None = None
    required: bool = True
    regex: str | None = None


@dataclass(frozen=True, slots=True)
class CrnnSpec:
    """Geometry the line recogniser was trained with (see ``ocr/crnn.py``)."""

    window: tuple[float, float]  # card pixels covered by one network input at scale 1
    anchor: tuple[float, float]  # point the registration hypotheses scale around
    fields: dict[str, tuple[float, float, int, str]]  # name -> (cx, cy, plane id, kind)


@dataclass(frozen=True, slots=True)
class Layout:
    id: str
    version: str
    side: str
    size: tuple[int, int]
    palette_rgb: tuple[int, int, int]
    bg_probe: Box
    markers: dict[str, Box] = field(default_factory=dict)
    fields: dict[str, FieldSpec] = field(default_factory=dict)
    crnn: CrnnSpec | None = None


def _crnn_spec(raw: dict | None, lid: str, fields: dict[str, FieldSpec]) -> CrnnSpec | None:
    if not raw:
        return None
    window = raw["window"]
    anchor = raw["anchor"]
    spec: dict[str, tuple[float, float, int, str]] = {}
    for name, item in raw["fields"].items():
        if name not in fields:
            raise LayoutError(f"{lid}.crnn: unknown field {name!r}")
        cx, cy = item["centre"]
        spec[name] = (float(cx), float(cy), int(item["plane"]), str(item["kind"]))
    if len(window) != 2 or len(anchor) != 2 or not spec:
        raise LayoutError(f"{lid}.crnn: window, anchor and fields are required")
    return CrnnSpec(
        (float(window[0]), float(window[1])), (float(anchor[0]), float(anchor[1])), spec
    )


def _box(raw: Any, what: str, size: tuple[int, int]) -> Box:
    if not (isinstance(raw, list) and len(raw) == 4 and all(isinstance(v, int) for v in raw)):
        raise LayoutError(f"{what}: expected [x, y, w, h] integers")
    box = Box(*raw)
    if box.w <= 0 or box.h <= 0 or box.x < 0 or box.y < 0:
        raise LayoutError(f"{what}: box has non-positive size or negative origin")
    if box.x + box.w > size[0] or box.y + box.h > size[1]:
        raise LayoutError(f"{what}: box exceeds the card size {size}")
    return box


def parse_layout(raw: dict[str, Any]) -> Layout:
    try:
        lid = str(raw["id"])
        size_raw = raw["card_size"]
        size = (int(size_raw[0]), int(size_raw[1]))
        side = raw["side"]
        if side not in VALID_SIDES:
            raise LayoutError(f"{lid}: side must be one of {sorted(VALID_SIDES)}")
        palette = tuple(int(c) for c in raw["palette_rgb"])
        if len(palette) != 3 or not all(0 <= c <= 255 for c in palette):
            raise LayoutError(f"{lid}: palette_rgb must be three 0-255 integers")
        markers: dict[str, Box] = {}
        for name, box in raw.get("markers", {}).items():
            if name not in VALID_MARKERS:
                raise LayoutError(f"{lid}: unknown marker {name!r}")
            markers[name] = _box(box, f"{lid}.markers.{name}", size)
        fields: dict[str, FieldSpec] = {}
        for name, spec in raw.get("fields", {}).items():
            kind = spec["kind"]
            if kind not in VALID_KINDS:
                raise LayoutError(f"{lid}.{name}: unknown kind {kind!r}")
            label_raw = spec.get("label_roi")
            fields[name] = FieldSpec(
                name=name,
                roi=_box(spec["roi"], f"{lid}.{name}.roi", size),
                kind=kind,
                label=str(spec.get("label", "")),
                label_roi=_box(label_raw, f"{lid}.{name}.label_roi", size) if label_raw else None,
                required=bool(spec.get("required", True)),
                regex=spec.get("regex"),
            )
        return Layout(
            id=lid,
            version=str(raw["version"]),
            side=side,
            size=size,
            palette_rgb=(palette[0], palette[1], palette[2]),
            bg_probe=_box(raw["bg_probe"], f"{lid}.bg_probe", size),
            markers=markers,
            fields=fields,
            crnn=_crnn_spec(raw.get("crnn"), lid, fields),
        )
    except KeyError as exc:
        raise LayoutError(f"layout is missing required key {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise LayoutError(f"layout is malformed: {exc}") from exc


class LayoutRegistry:
    """All known layouts, keyed by ``id`` (e.g. ``new-front``)."""

    def __init__(self, layouts: list[Layout]) -> None:
        if not layouts:
            raise LayoutError("at least one layout is required")
        self._by_id = {layout.id: layout for layout in layouts}
        if len(self._by_id) != len(layouts):
            raise LayoutError("duplicate layout ids")

    @classmethod
    def load(cls, extra_dir: str | Path | None = None) -> LayoutRegistry:
        raw: dict[str, dict[str, Any]] = {}
        pkg = resources.files("idcard_ocr.layouts")
        for item in pkg.iterdir():
            if item.name.endswith(".json"):
                data = json.loads(item.read_text(encoding="utf-8"))
                raw[str(data["id"])] = data
        if extra_dir:
            base = Path(extra_dir)
            if not base.is_dir():
                raise LayoutError(f"layout directory does not exist: {base}")
            for path in sorted(base.glob("*.json")):
                data = json.loads(path.read_text(encoding="utf-8"))
                raw[str(data["id"])] = data  # overrides built-ins with the same id
        return cls([parse_layout(r) for r in raw.values()])

    def get(self, layout_id: str) -> Layout:
        try:
            return self._by_id[layout_id]
        except KeyError as exc:
            raise LayoutError(f"unknown layout {layout_id!r}") from exc

    def find(self, version: str, side: str) -> Layout | None:
        for layout in self._by_id.values():
            if layout.version == version and layout.side == side:
                return layout
        return None

    def all(self) -> list[Layout]:
        return list(self._by_id.values())

    @property
    def versions(self) -> set[str]:
        return {layout.version for layout in self._by_id.values()}
