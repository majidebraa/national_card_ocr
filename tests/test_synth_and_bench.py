from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from idcard_ocr.bench import canon, run_benchmark, to_markdown
from idcard_ocr.config import Settings
from idcard_ocr.layouts import LayoutRegistry
from idcard_ocr.pipeline import Pipeline
from idcard_ocr.synth.dataset import Sample, generate_dataset, load_labels
from idcard_ocr.synth.identity import random_identity
from idcard_ocr.synth.render import find_font, render_card, to_persian_digits
from idcard_ocr.synth.scene import DIFFICULTIES, compose_scene
from idcard_ocr.validation.dates import is_valid_jalali
from idcard_ocr.validation.national_id import is_valid_national_id
from idcard_ocr.validation.text import ZWNJ
from tests.conftest import FakeEngine


def test_identities_are_valid_and_varied() -> None:
    rng = random.Random(1)
    ids = [random_identity(rng) for _ in range(200)]
    assert all(is_valid_national_id(i.national_id) for i in ids)
    assert all(is_valid_jalali(*(int(p) for p in i.birth_date.split("/"))) for i in ids)
    assert all(is_valid_jalali(*(int(p) for p in i.expiry_date.split("/"))) for i in ids)
    assert len({i.national_id for i in ids}) > 190


def test_rendering_is_deterministic(registry: LayoutRegistry) -> None:
    layout = registry.get("new-front")
    a = render_card(layout, random_identity(random.Random(5)), random.Random(5))
    b = render_card(layout, random_identity(random.Random(5)), random.Random(5))
    assert np.array_equal(a, b) and a.shape == (638, 1011, 3)


def test_cards_are_visibly_marked_as_specimens(registry: LayoutRegistry) -> None:
    """Regression guard: the watermark must stay in the pixels."""
    layout = registry.get("old-front")
    with_mark = render_card(layout, random_identity(random.Random(2)), random.Random(2))
    centre = with_mark[260:420, 300:700].astype(int)
    red_tint = (centre[:, :, 2] - centre[:, :, 0]).mean()
    assert red_tint > 0


def test_persian_digit_conversion() -> None:
    assert to_persian_digits("1370/05/12") == "۱۳۷۰/۰۵/۱۲"


def test_font_override_and_missing_font(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    assert Path(find_font()).is_file()
    monkeypatch.setenv("IDCARD_SYNTH_FONT", str(Path(find_font(bold=True))))
    assert find_font().endswith("Bold.ttf")


def test_scene_geometry_ground_truth(registry: LayoutRegistry) -> None:
    layout = registry.get("old-back")
    flat = render_card(layout, random_identity(random.Random(1)), random.Random(1))
    for name in DIFFICULTIES:
        scene = compose_scene(flat, random.Random(7), name)
        assert scene.image.shape == (1200, 1600, 3)
        assert scene.quad.shape == (4, 2)
        assert (scene.quad > -50).all() and (scene.quad[:, 0] < 1650).all()


def test_dataset_generation_is_deterministic(tmp_path: Path) -> None:
    a = generate_dataset(tmp_path / "a", 4, seed=9, difficulty="easy")
    b = generate_dataset(tmp_path / "b", 4, seed=9, difficulty="easy")
    assert [s.fields for s in a] == [s.fields for s in b]
    assert (tmp_path / "a" / a[0].file).read_bytes() == (tmp_path / "b" / b[0].file).read_bytes()
    assert {s.layout_id for s in generate_dataset(tmp_path / "c", 8, seed=1)} == {
        "old-front",
        "new-front",
        "old-back",
        "new-back",
    }
    loaded = load_labels(tmp_path / "a")
    assert [s.sample_id for s in loaded] == [s.sample_id for s in a]
    line = (tmp_path / "a" / "labels.jsonl").read_text("utf-8").splitlines()[0]
    assert json.loads(line)["layout_id"] == a[0].layout_id


def test_dataset_argument_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        generate_dataset(tmp_path, 0)
    with pytest.raises(ValueError):
        generate_dataset(tmp_path, 1, difficulty="impossible")


def test_canonical_comparison() -> None:
    assert canon("national_id", "۰۴۹۹ ۳۷۰۸۹۹") == "0499370899"
    assert canon("last_name", f"ژیان{ZWNJ}فر") == canon("last_name", "ژیان فر")
    assert canon("serial_no", " ab12 ") == "AB12"
    assert canon("first_name", "عليرضا") == canon("first_name", "علیرضا")


def test_benchmark_report(easy_dataset: tuple[Path, list[Sample]], settings: Settings) -> None:
    pipe = Pipeline(settings, engine=FakeEngine())
    report = run_benchmark(pipe, easy_dataset[0], limit=6)
    assert report["images"] == 6 and report["processed"] + report["rejected"] == 6
    assert report["classification_accuracy_pct"] >= 80
    assert report["fields"]["national_id"]["exact_match_pct"] == 100.0
    assert report["fields"]["national_id"]["silent_error_pct"] == 0.0
    md = to_markdown(report, "t")
    assert "| national_id |" in md and "silent error" in md
