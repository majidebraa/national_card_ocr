from __future__ import annotations

import numpy as np
import pytesseract
import pytest

from idcard_ocr.errors import OcrError, OcrUnavailableError
from idcard_ocr.ocr.tesseract import TesseractEngine

IMG = np.full((60, 200), 255, np.uint8)


def _patch(monkeypatch: pytest.MonkeyPatch, result=None, exc: Exception | None = None) -> list:  # type: ignore[no-untyped-def]
    calls: list = []

    def fake(image, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        if exc:
            raise exc
        return result

    monkeypatch.setattr(pytesseract, "image_to_data", fake)
    return calls


def test_text_and_confidence_are_aggregated(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch(monkeypatch, {"text": ["", "علی", "رضا"], "conf": [-1, "90", "70"]})
    out = TesseractEngine(lang="fas", tessdata_dir="/t").recognize(IMG, whitelist="0123")
    assert out.text == "علی رضا" and out.confidence == pytest.approx(0.8)
    cfg = calls[0]["config"]
    assert "--tessdata-dir" in cfg and "tessedit_char_whitelist=0123" in cfg


def test_language_override(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch(monkeypatch, {"text": ["x"], "conf": ["50"]})
    TesseractEngine(lang="fas").recognize(IMG, lang="eng")
    assert calls[0]["lang"] == "eng"


def test_empty_result(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, {"text": ["", " "], "conf": ["-1", "-1"]})
    out = TesseractEngine().recognize(IMG)
    assert (out.text, out.confidence) == ("", 0.0)


def test_unparseable_confidence_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, {"text": ["a"], "conf": ["n/a"]})
    assert TesseractEngine().recognize(IMG).confidence == 0.0


def test_error_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    # TesseractError subclasses RuntimeError: it must not be mistaken for a timeout.
    _patch(monkeypatch, exc=pytesseract.TesseractError(1, "bad image"))
    with pytest.raises(OcrError, match="failed"):
        TesseractEngine().recognize(IMG)
    _patch(monkeypatch, exc=RuntimeError("Tesseract process timeout"))
    with pytest.raises(OcrError, match="timed out"):
        TesseractEngine().recognize(IMG)
    _patch(monkeypatch, exc=pytesseract.TesseractNotFoundError())
    with pytest.raises(OcrUnavailableError):
        TesseractEngine().recognize(IMG)


def test_check_ready_failures(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(OcrUnavailableError, match="binary"):
        TesseractEngine().check_ready()
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/tesseract")
    with pytest.raises(OcrUnavailableError, match="'fas'"):
        TesseractEngine(lang="fas", tessdata_dir=str(tmp_path)).check_ready()
    (tmp_path / "fas.traineddata").write_bytes(b"x")
    TesseractEngine(lang="fas", tessdata_dir=str(tmp_path)).check_ready()
    monkeypatch.setattr(pytesseract, "get_languages", lambda config="": ["eng"])
    with pytest.raises(OcrUnavailableError, match="'fas'"):
        TesseractEngine(lang="fas").check_ready()
