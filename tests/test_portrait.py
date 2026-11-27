from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from idcard_ocr.config import Settings
from idcard_ocr.layouts import Box
from idcard_ocr.pipeline import Pipeline
from idcard_ocr.portrait import PortraitExtractor, default_model_path
from idcard_ocr.store import EncryptedImageStore, generate_key
from idcard_ocr.synth.dataset import Sample
from tests.conftest import FakeEngine

PHOTO = Box(60, 170, 270, 305)


def test_bundled_model_is_the_audited_file() -> None:
    import hashlib

    from idcard_ocr.portrait import MODEL_SHA256

    assert hashlib.sha256(default_model_path().read_bytes()).hexdigest() == MODEL_SHA256


def test_blank_region_falls_back_to_the_photo_box() -> None:
    card = np.full((638, 1011, 3), 200, np.uint8)
    p = PortraitExtractor().extract(card, PHOTO)
    assert not p.face_found and p.face_box is None and p.confidence == 0.0
    assert p.box == (60, 170, 270, 305)
    assert p.image.shape[:2] == (305, 270)


def test_fallback_box_follows_the_offset_and_stays_inside_the_card() -> None:
    card = np.full((638, 1011, 3), 200, np.uint8)
    p = PortraitExtractor().extract(card, PHOTO, offset=(-100, 400))
    x, y, w, h = p.box
    assert x == 0 and y + h <= 638 and w >= 1 and h >= 1


def test_missing_model_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PortraitExtractor(tmp_path / "nope.onnx")


def test_pipeline_reports_and_retains_the_portrait(
    settings: Settings, tmp_path: Path, easy_dataset: tuple[Path, list[Sample]]
) -> None:
    store = EncryptedImageStore(tmp_path / "v", (generate_key(),), 30)
    pipe = Pipeline(settings, engine=FakeEngine(), store=store)
    directory, samples = easy_dataset
    front = next(s for s in samples if s.side == "front")
    result = pipe.process((directory / front.file).read_bytes(), store=True)
    assert result.portrait is not None and len(result.portrait.box) == 4
    assert "portrait" in result.timings_ms
    assert result.stored is not None and "portrait" in result.stored.variants
    assert store.load(result.stored.image_id, "portrait").startswith(b"\xff\xd8")


def test_portrait_can_be_disabled(
    settings: Settings, easy_dataset: tuple[Path, list[Sample]]
) -> None:
    from dataclasses import replace

    pipe = Pipeline(replace(settings, extract_portrait=False), engine=FakeEngine())
    directory, samples = easy_dataset
    result = pipe.process((directory / samples[0].file).read_bytes())
    assert result.portrait is None


def test_orient_flips_only_when_the_face_is_upside_down(monkeypatch: pytest.MonkeyPatch) -> None:
    card = np.zeros((638, 1011, 3), np.uint8)
    card[:10] = 255  # marks the top edge
    ex = PortraitExtractor()

    def run(upright: bool, flipped: bool) -> tuple[np.ndarray, bool]:
        calls = iter([upright, flipped])
        monkeypatch.setattr(ex, "has_face", lambda *_a, **_k: next(calls))
        return ex.orient(card, PHOTO)

    out, rotated = run(True, False)
    assert not rotated and out is card
    out, rotated = run(False, True)
    assert rotated and out[-1].min() == 255 and out[0].max() == 0
    out, rotated = run(False, False)
    assert not rotated and out is card


def test_level_makes_the_eye_line_horizontal() -> None:
    from idcard_ocr.portrait import level

    card = np.zeros((200, 200, 3), np.uint8)
    # eyes tilted 20 degrees (image-right eye lower), centred on (100, 100)
    import math

    dx, dy = 40 * math.cos(math.radians(20)), 40 * math.sin(math.radians(20))
    right, left = (100 - dx, 100 - dy), (100 + dx, 100 + dy)
    for cx, cy in (right, left):
        card[round(cy) - 2 : round(cy) + 3, round(cx) - 2 : round(cx) + 3] = 255
    out = level(card, (100.0, 100.0), right, left)
    ys, xs = np.nonzero(out[..., 0] > 128)
    left_y = ys[xs < 100].mean()
    right_y = ys[xs >= 100].mean()
    assert abs(left_y - right_y) < 2.5
    assert level(card, (100.0, 100.0), (60, 100), (140, 101)) is card  # < MIN_TILT_DEG


def test_single_layout_falls_back_to_the_face_when_classification_fails(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from idcard_ocr.errors import ClassificationError
    from idcard_ocr.pipeline import Pipeline

    pipe = Pipeline(settings, engine=FakeEngine())
    layout = pipe.registry.get("new-front")
    only = type(pipe.registry)([layout])
    pipe.registry = only
    card = np.full((638, 1011, 3), 128, np.uint8)

    def refuse(*_a: object, **_k: object) -> None:
        raise ClassificationError("nope")

    monkeypatch.setattr("idcard_ocr.pipeline.classify", refuse)
    assert pipe.portraits is not None
    answers = iter([False, True])
    monkeypatch.setattr(pipe.portraits, "has_face", lambda *_a, **_k: next(answers))
    warnings: list[str] = []
    cls = pipe._classify(card, "front", "new", warnings)
    assert cls.layout.id == "new-front" and cls.rotated_180
    assert warnings == ["classified_by_face"]

    monkeypatch.setattr(pipe.portraits, "has_face", lambda *_a, **_k: False)
    with pytest.raises(ClassificationError):
        pipe._classify(card, "front", "new", [])
