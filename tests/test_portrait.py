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
