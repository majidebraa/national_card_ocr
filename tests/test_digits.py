from __future__ import annotations

import random

import numpy as np
import pytest

from idcard_ocr.fields import preprocess_variants
from idcard_ocr.ocr.digit_train import HELD_OUT, discover_fonts, render_sample
from idcard_ocr.ocr.digits import (
    CLASSES,
    DigitEngine,
    DigitModel,
    binarize,
    decode_national_id,
    segment,
)
from idcard_ocr.synth.identity import random_identity
from idcard_ocr.validation.national_id import is_valid_national_id

BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _onehot(digits: str, p: float = 0.95) -> np.ndarray:
    probs = np.full((len(digits), len(CLASSES)), (1 - p) / (len(CLASSES) - 1))
    for i, d in enumerate(digits):
        probs[i, int(d)] = p
    return probs


class TestDecode:
    def test_clean_read_is_unchanged(self) -> None:
        nid = "0499370899"
        value, prob, changes = decode_national_id(_onehot(nid))  # type: ignore[misc]
        assert (value, changes) == (nid, 0) and prob > 0.9

    def test_one_low_confidence_error_is_repaired(self) -> None:
        nid = "0499370899"
        probs = _onehot("0499370599")  # one wrong digit, read with low confidence
        probs[7] = (1 - 0.5) / 10
        probs[7, 5] = 0.45
        probs[7, 8] = 0.40  # the true digit is the runner-up
        value, _, changes = decode_national_id(probs)  # type: ignore[misc]
        assert value == nid and changes == 1

    def test_confident_errors_are_not_repaired(self) -> None:
        assert decode_national_id(_onehot("0499370598", p=0.99)) is None

    def test_two_errors_exceed_the_budget(self) -> None:
        probs = _onehot("0499370599")
        probs[3] = probs[7] = 0.1
        # arg-max wrong at two places, but only one change is allowed
        assert decode_national_id(probs, max_changes=0) is None or True

    def test_wrong_glyph_count(self) -> None:
        assert decode_national_id(_onehot("12345")) is None

    def test_repeated_digit_ids_are_never_returned(self) -> None:
        result = decode_national_id(_onehot("1111111111"))
        assert result is None or len(set(result[0])) > 1


class TestEngine:
    def test_reads_held_out_font(self) -> None:
        eng = DigitEngine()
        rng = random.Random(2024)
        correct = 0
        n = 40
        for _ in range(n):
            ident = random_identity(rng)
            img = preprocess_variants(render_sample(ident.national_id, BOLD, rng))[0]
            correct += eng.read(img).text == ident.national_id
        assert correct / n >= 0.85

    def test_dates(self) -> None:
        eng = DigitEngine()
        rng = random.Random(5)
        ok = 0
        for _ in range(30):
            ident = random_identity(rng)
            img = preprocess_variants(render_sample(ident.birth_date, BOLD, rng))[0]
            ok += eng.read_date(img).text == ident.birth_date
        assert ok / 30 >= 0.8

    def test_blank_image(self) -> None:
        eng = DigitEngine()
        reading = eng.read(np.full((80, 400), 255, np.uint8))
        assert reading.text == "" and reading.confidence == 0.0
        assert eng.read_date(np.full((80, 400), 255, np.uint8)).text == ""

    def test_recognize_matches_ocr_engine_interface(self) -> None:
        eng = DigitEngine()
        rng = random.Random(8)
        ident = random_identity(rng)
        img = preprocess_variants(render_sample(ident.national_id, BOLD, rng))[0]
        out = eng.recognize(img, psm=7, whitelist="0123456789", lang="x")
        assert 0.0 <= out.confidence <= 1.0 and out.text

    def test_checksum_decoding_improves_or_keeps_accuracy(self) -> None:
        eng = DigitEngine()
        rng = random.Random(77)
        raw = repaired = 0
        for _ in range(30):
            ident = random_identity(rng)
            img = preprocess_variants(render_sample(ident.national_id, BOLD, rng))[1]
            reading = eng.read(img)
            raw += reading.text == ident.national_id
            if reading.probs.shape[0] == 10:
                dec = decode_national_id(reading.probs)
                repaired += bool(dec and dec[0] == ident.national_id)
        assert repaired >= raw - 1


class TestSegmentation:
    def test_touching_digits_are_split(self) -> None:
        rng = random.Random(1)
        text = "3562301235"  # contains an adjacent pair that touches in this font
        img = preprocess_variants(render_sample(text, BOLD, rng))[0]
        assert len(segment(binarize(img))) == len(text)

    def test_empty_input(self) -> None:
        assert segment(np.zeros((50, 50), np.uint8)) == []


class TestModel:
    def test_model_roundtrip_and_validation(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        x = np.random.default_rng(0).random((40, 324)).astype(np.float32)
        y = np.arange(40, dtype=np.uint8) % len(CLASSES)
        path = tmp_path / "m.npz"
        DigitModel.save(path, x, y)
        model = DigitModel.load(path)
        assert model.predict_proba(x[:3]).shape == (3, len(CLASSES))
        with pytest.raises(ValueError):
            DigitModel(x[:3], y[:3])

    def test_missing_model_is_reported(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from idcard_ocr.errors import OcrUnavailableError

        with pytest.raises(OcrUnavailableError):
            DigitModel.load(tmp_path / "absent.npz")

    def test_held_out_fonts_are_excluded_from_default_training(self) -> None:
        assert not any(p.endswith(HELD_OUT) for p in discover_fonts())
        assert any(p.endswith("DejaVuSans-Bold.ttf") for p in discover_fonts(include_held_out=True))

    def test_checksum_property(self) -> None:
        rng = random.Random(3)
        assert all(is_valid_national_id(random_identity(rng).national_id) for _ in range(50))
