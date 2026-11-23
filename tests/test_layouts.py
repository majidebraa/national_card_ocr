from __future__ import annotations

import json
from pathlib import Path

import pytest

from idcard_ocr.errors import LayoutError
from idcard_ocr.layouts import Box, LayoutRegistry, parse_layout


def test_builtin_layouts(registry: LayoutRegistry) -> None:
    assert {layout.id for layout in registry.all()} == {
        "old-front",
        "new-front",
        "old-back",
        "new-back",
    }
    assert registry.versions == {"old", "new"}
    assert registry.find("new", "front") is not None
    assert registry.find("alien", "front") is None


def test_every_field_is_inside_the_card(registry: LayoutRegistry) -> None:
    for layout in registry.all():
        w, h = layout.size
        for spec in layout.fields.values():
            assert spec.roi.x + spec.roi.w <= w and spec.roi.y + spec.roi.h <= h


def test_box_scaling() -> None:
    assert Box(10, 20, 30, 40).scaled(2, 0.5) == Box(20, 10, 60, 20)


def _raw(registry: LayoutRegistry) -> dict:
    return json.loads(
        json.dumps(
            {
                "id": "x-front",
                "version": "x",
                "side": "front",
                "card_size": [1011, 638],
                "palette_rgb": [1, 2, 3],
                "bg_probe": [0, 0, 10, 10],
                "markers": {"photo": [0, 0, 10, 10]},
                "fields": {"a": {"roi": [0, 0, 10, 10], "kind": "name"}},
            }
        )
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update(side="left"),
        lambda r: r.update(palette_rgb=[1, 2]),
        lambda r: r["markers"].update(hologram=[0, 0, 5, 5]),
        lambda r: r["fields"]["a"].update(kind="telepathy"),
        lambda r: r["fields"]["a"].update(roi=[0, 0, 5000, 10]),
        lambda r: r["fields"]["a"].update(roi=[0, 0, -1, 10]),
        lambda r: r["fields"]["a"].update(roi=["a", 0, 1, 1]),
        lambda r: r.pop("bg_probe"),
    ],
)
def test_invalid_layouts_are_rejected(registry: LayoutRegistry, mutate) -> None:  # type: ignore[no-untyped-def]
    raw = _raw(registry)
    mutate(raw)
    with pytest.raises(LayoutError):
        parse_layout(raw)


def test_valid_custom_layout_parses(registry: LayoutRegistry) -> None:
    assert parse_layout(_raw(registry)).id == "x-front"


def test_extra_dir_overrides_by_id(tmp_path: Path) -> None:
    raw = json.loads(
        (Path(__file__).parents[1] / "src/idcard_ocr/layouts/new-front.json").read_text("utf-8")
    )
    raw["fields"]["national_id"]["roi"] = [10, 10, 100, 50]
    (tmp_path / "custom.json").write_text(json.dumps(raw), encoding="utf-8")
    reg = LayoutRegistry.load(tmp_path)
    assert reg.get("new-front").fields["national_id"].roi == Box(10, 10, 100, 50)
    assert len(reg.all()) == 4


def test_missing_extra_dir() -> None:
    with pytest.raises(LayoutError):
        LayoutRegistry.load("/definitely/not/here")


def test_unknown_layout_and_empty_registry(registry: LayoutRegistry) -> None:
    with pytest.raises(LayoutError):
        registry.get("nope")
    with pytest.raises(LayoutError):
        LayoutRegistry([])
    one = registry.get("old-front")
    with pytest.raises(LayoutError):
        LayoutRegistry([one, one])
