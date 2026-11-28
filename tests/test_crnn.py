from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from idcard_ocr.fields import read_field
from idcard_ocr.layouts import Box, CrnnSpec, FieldSpec, parse_layout
from idcard_ocr.ocr.base import OcrText
from idcard_ocr.ocr.crnn import (
    BLANK,
    CHARSET,
    GRID,
    INPUT_H,
    INPUT_W,
    NOMINAL,
    CrnnEngine,
    CrnnReading,
    centres,
    decode,
    normalise,
    well_formed,
    window,
)

SPEC = CrnnSpec(
    window=(360.0, 166.0),
    anchor=(690.0, 174.0),
    fields={
        "national_id": (690.0, 174.0, 0, "national_id"),
        "birth_date": (700.0, 357.0, 1, "date"),
    },
)
T = 60


def logits_for(text: str, confidence: float = 0.99) -> np.ndarray:
    """Logits whose greedy CTC decoding is ``text`` with the given top-1 probability."""
    path: list[int] = []
    for ch in text:
        path += [CHARSET.index(ch), BLANK]
    path += [BLANK] * (T - len(path))
    out = np.full((T, len(CHARSET) + 1), np.log((1 - confidence) / len(CHARSET)), np.float32)
    for t, k in enumerate(path):
        out[t, k] = np.log(confidence)
    return out


class FakeEngineInfer:
    """Returns ``texts[plane]`` with a confidence per call; records the batch sizes."""

    def __init__(self, plan: list[dict[int, tuple[str, float]]]) -> None:
        self.plan = plan
        self.batches: list[int] = []

    def __call__(self, x: np.ndarray) -> np.ndarray:
        step = self.plan[min(len(self.batches), len(self.plan) - 1)]
        self.batches.append(len(x))
        planes = np.rint(x[:, 1, 0, 0] * 2).astype(int)
        return np.stack([logits_for(*step[int(p)]) for p in planes])


class TestWindow:
    def test_centre_maps_to_the_middle_of_the_input(self) -> None:
        card = np.zeros((638, 1011), np.uint8)
        card[170:178, 686:694] = 255
        w = window(card, (690.0, 174.0), (360.0, 166.0))
        assert w.shape == (INPUT_H, INPUT_W)
        ys, xs = np.nonzero(w > 128)
        assert abs(xs.mean() - INPUT_W / 2) < 2 and abs(ys.mean() - INPUT_H / 2) < 2

    def test_scale_zooms_out(self) -> None:
        card = np.zeros((638, 1011), np.uint8)
        card[150:200, 650:730] = 255
        small = window(card, (690.0, 174.0), (360.0, 166.0), 1.0)
        large = window(card, (690.0, 174.0), (360.0, 166.0), 1.2)
        assert (large > 128).sum() < (small > 128).sum()

    def test_normalise_is_zero_mean_unit_variance(self) -> None:
        x = normalise(np.random.default_rng(0).integers(0, 255, (2, 8, 8), dtype=np.uint8))
        assert abs(float(x[0].mean())) < 1e-4 and abs(float(x[0].std()) - 1) < 0.01


class TestDecode:
    def test_ctc_collapse_and_confidence(self) -> None:
        [(text, conf)] = decode(logits_for("1380/04/01", 0.9)[None])
        assert text == "1380/04/01" and conf == pytest.approx(0.9, abs=0.01)

    def test_blank_only_is_empty(self) -> None:
        [(text, conf)] = decode(logits_for("")[None])
        assert text == "" and conf == 0.0

    def test_repeated_digits_need_a_blank_between(self) -> None:
        [(text, _)] = decode(logits_for("1122")[None])
        assert text == "1122"

    def test_well_formed(self) -> None:
        assert well_formed("national_id", "0123456789")
        assert not well_formed("national_id", "123456789")
        assert well_formed("date", "1380/04/01") and not well_formed("date", "1380/4/1")


class TestRegistration:
    def test_nominal_has_no_offset(self) -> None:
        assert centres(SPEC, NOMINAL) == {
            "national_id": (690.0, 174.0),
            "birth_date": (700.0, 357.0),
        }

    def test_shift_moves_every_field_and_scale_moves_fields_relative_to_the_anchor(self) -> None:
        c = centres(SPEC, (1.0, 30, -40))
        assert c["national_id"] == (650.0, 204.0) and c["birth_date"] == (660.0, 387.0)
        zoomed = centres(SPEC, (1.25, 0, 0))  # content drawn at 80%: the date line is closer
        assert zoomed["national_id"] == (690.0, 174.0)
        assert zoomed["birth_date"][1] == pytest.approx(174 + 183 * 0.8)

    def test_grid_contains_the_nominal_hypothesis(self) -> None:
        assert NOMINAL in GRID and len(GRID) == 63


class TestEngine:
    CARD = np.full((638, 1011), 200, np.uint8)

    def test_confident_nominal_reading_costs_one_pass(self) -> None:
        infer = FakeEngineInfer([{0: ("0123456789", 0.99), 1: ("1380/04/01", 0.98)}])
        out = CrnnEngine(infer).read(self.CARD, SPEC)
        assert infer.batches == [2]
        assert out["national_id"].text == "0123456789" and out["birth_date"].text == "1380/04/01"

    def test_unconfident_nominal_triggers_the_grid_and_the_best_hypothesis_wins(self) -> None:
        infer = FakeEngineInfer(
            [
                {0: ("0123456789", 0.5), 1: ("1380/04/01", 0.5)},
                {0: ("9999999999", 0.99), 1: ("1400/01/01", 0.99)},
            ]
        )
        out = CrnnEngine(infer).read(self.CARD, SPEC)
        assert infer.batches == [2, len(GRID) * 2]
        assert out["national_id"].text == "9999999999"

    def test_malformed_text_scores_zero_confidence(self) -> None:
        infer = FakeEngineInfer([{0: ("12345", 0.99), 1: ("1380/04/01", 0.99)}])
        out = CrnnEngine(infer).read(self.CARD, SPEC)
        assert out["national_id"].confidence == 0.0 and out["birth_date"].confidence > 0.9

    def test_nominal_is_kept_when_the_grid_is_no_better(self) -> None:
        bad = {0: ("0123456789", 0.9), 1: ("12", 0.1)}
        infer = FakeEngineInfer([bad])
        out = CrnnEngine(infer).read(self.CARD, SPEC)
        assert out["national_id"].hypothesis == NOMINAL

    def test_bgr_input_is_accepted(self) -> None:
        infer = FakeEngineInfer([{0: ("0123456789", 0.99), 1: ("1380/04/01", 0.99)}])
        bgr = np.dstack([self.CARD] * 3)
        assert CrnnEngine(infer).read(bgr, SPEC)["national_id"].confidence > 0.9

    def test_missing_model_file(self, tmp_path: Path) -> None:
        pytest.importorskip("onnxruntime")
        with pytest.raises(FileNotFoundError):
            CrnnEngine.from_onnx(tmp_path / "nope.onnx")


class _NoOcr:
    def recognize(self, *_a: object, **_k: object) -> OcrText:  # pragma: no cover - must not run
        raise AssertionError("the classic OCR path must not run for a confident line reading")


class TestReadFieldIntegration:
    CARD = np.full((638, 1011, 3), 200, np.uint8)
    ID = FieldSpec("national_id", Box(560, 146, 260, 56), "national_id")
    DATE = FieldSpec("birth_date", Box(580, 330, 240, 54), "date")

    def _read(self, spec: FieldSpec, text: str, confidence: float):  # type: ignore[no-untyped-def]
        return read_field(
            _NoOcr(),  # type: ignore[arg-type]
            self.CARD,
            spec,
            (1011, 638),
            review_confidence=0.8,
            crnn=CrnnReading(text, confidence, NOMINAL),
            crnn_min_confidence=0.9,
        )

    def test_confident_valid_id_skips_the_classic_path(self) -> None:
        res = self._read(self.ID, "0013542419", 0.99)  # a checksum-valid id
        assert res.value == "0013542419" and res.valid and not res.needs_review

    def test_confident_valid_date_is_normalised_with_gregorian(self) -> None:
        res = self._read(self.DATE, "1380/04/01", 0.97)
        assert res.value == "1380/04/01" and res.valid and res.gregorian

    def test_low_confidence_falls_back_and_is_flagged_for_review(self) -> None:
        from tests.conftest import FakeEngine

        res = read_field(
            FakeEngine("1380/04/01", 0.99),
            self.CARD,
            self.DATE,
            (1011, 638),
            review_confidence=0.5,
            crnn=CrnnReading("1380/04/01", 0.5, NOMINAL),
            crnn_min_confidence=0.9,
        )
        assert res.value == "1380/04/01" and res.needs_review
        assert "line_reader_unconfirmed" in res.issues

    def test_without_a_line_reader_nothing_changes(self) -> None:
        from tests.conftest import FakeEngine

        res = read_field(
            FakeEngine("1380/04/01", 0.99), self.CARD, self.DATE, (1011, 638), review_confidence=0.5
        )
        assert "line_reader_unconfirmed" not in res.issues


def test_layout_crnn_section_is_parsed() -> None:
    raw = json.loads(Path("calibration/kaggle/new-front-kaggle.json").read_text())
    layout = parse_layout(raw)
    assert layout.crnn is not None and set(layout.crnn.fields) == {
        "national_id",
        "birth_date",
        "expiry_date",
    }
    raw["crnn"]["fields"]["nope"] = {"centre": [1, 1], "plane": 0, "kind": "date"}
    with pytest.raises(Exception, match="unknown field"):
        parse_layout(raw)


def test_window_box_is_clipped_to_the_image() -> None:
    from idcard_ocr.ocr.crnn import window_box

    assert window_box((690.0, 174.0), (360.0, 166.0), (638, 1011)) == (510, 91, 870, 257)
    assert window_box((100.0, 20.0), (360.0, 166.0), (638, 1011)) == (0, 0, 280, 103)
