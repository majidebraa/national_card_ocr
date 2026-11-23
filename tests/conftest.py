from __future__ import annotations

import os
import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from idcard_ocr.config import Settings
from idcard_ocr.errors import OcrUnavailableError
from idcard_ocr.layouts import LayoutRegistry
from idcard_ocr.ocr.base import OcrText
from idcard_ocr.ocr.tesseract import TesseractEngine
from idcard_ocr.pipeline import Pipeline
from idcard_ocr.synth.dataset import Sample, generate_dataset
from idcard_ocr.synth.identity import random_identity
from idcard_ocr.synth.render import render_card


def _tessdata() -> str | None:
    return os.environ.get("IDCARD_TESSDATA_DIR") or (
        "/home/claude/tessdata" if Path("/home/claude/tessdata").is_dir() else None
    )


@pytest.fixture(scope="session")
def tessdata_dir() -> str | None:
    return _tessdata()


@pytest.fixture(scope="session")
def registry() -> LayoutRegistry:
    return LayoutRegistry.load()


@pytest.fixture
def settings(tessdata_dir: str | None) -> Settings:
    return Settings(tessdata_dir=tessdata_dir, allow_anonymous=True)


class FakeEngine:
    """An OCR engine that needs no binaries: returns a fixed text."""

    name = "fake"

    def __init__(self, text: str = "علی", confidence: float = 0.9) -> None:
        self.text, self.confidence = text, confidence
        self.calls = 0

    def recognize(
        self,
        image: np.ndarray,
        *,
        psm: int = 7,
        whitelist: str | None = None,
        lang: str | None = None,
    ) -> OcrText:
        self.calls += 1
        return OcrText(self.text, self.confidence)

    def check_ready(self) -> None:
        return None


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def fake_pipeline(settings: Settings, fake_engine: FakeEngine) -> Pipeline:
    return Pipeline(settings, engine=fake_engine)


@pytest.fixture(scope="session")
def tesseract_ready(tessdata_dir: str | None) -> TesseractEngine:
    engine = TesseractEngine(tessdata_dir=tessdata_dir)
    try:
        engine.check_ready()
    except OcrUnavailableError:
        pytest.skip("tesseract with the fas language pack is not available")
    return engine


@pytest.fixture(scope="session")
def easy_dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[Sample]]:
    out = tmp_path_factory.mktemp("easy")
    samples = generate_dataset(out, 8, seed=42, difficulty="easy", upside_down_fraction=0.25)
    return out, samples


@pytest.fixture
def flat_cards(registry: LayoutRegistry) -> Iterator[dict[str, tuple[np.ndarray, dict[str, str]]]]:
    rng = random.Random(99)
    out = {}
    for layout in registry.all():
        ident = random_identity(rng)
        out[layout.id] = (render_card(layout, ident, rng), ident.values())
    yield out
