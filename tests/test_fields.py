from __future__ import annotations

import numpy as np

from idcard_ocr.fields import crop_roi, preprocess_variants, read_field
from idcard_ocr.layouts import Box, FieldSpec
from idcard_ocr.ocr.base import OcrText
from idcard_ocr.ocr.digits import DigitEngine


class Scripted:
    """An engine that replays scripted outputs, recording the languages it was asked for."""

    name = "scripted"

    def __init__(self, *outputs: OcrText) -> None:
        self.outputs = list(outputs)
        self.langs: list[str | None] = []

    def recognize(self, image, *, psm=7, whitelist=None, lang=None) -> OcrText:  # type: ignore[no-untyped-def]
        self.langs.append(lang)
        return self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]

    def check_ready(self) -> None:
        return None


CARD = np.full((638, 1011, 3), 235, np.uint8)
SIZE = (1011, 638)


def _spec(kind: str, regex: str | None = None) -> FieldSpec:
    return FieldSpec("f", Box(300, 100, 400, 60), kind, regex=regex)


def _read(engine, kind: str, *, digits: DigitEngine | None = None, regex=None, **kw):  # type: ignore[no-untyped-def]
    return read_field(
        engine, CARD, _spec(kind, regex), SIZE, review_confidence=0.6, digit_engine=digits, **kw
    )


def test_national_id_falls_back_to_tesseract_and_validates() -> None:
    res = _read(Scripted(OcrText("۰۴۹۹۳۷۰۸۹۹", 0.9)), "national_id")
    assert res.valid and res.value == "0499370899" and not res.needs_review
    assert res.confidence > 0.9


def test_national_id_checksum_failure_is_flagged() -> None:
    res = _read(Scripted(OcrText("0499370898", 0.9)), "national_id")
    assert not res.valid and res.needs_review and "checksum_failed" in res.issues
    assert res.confidence <= 0.6 * 0.9 + 1e-9  # a failing checksum caps confidence


def test_national_id_without_text() -> None:
    res = _read(Scripted(OcrText("", 0.0)), "national_id")
    assert res.value == "" and "missing" in res.issues and res.needs_review


def test_date_fallback_and_gregorian() -> None:
    res = _read(Scripted(OcrText("۱۳۷۰/۰۵/۱۲", 0.8)), "date")
    assert res.valid and res.value == "1370/05/12" and res.gregorian == "1991-08-03"


def test_invalid_date_is_flagged() -> None:
    res = _read(Scripted(OcrText("1370/13/45", 0.9)), "date")
    assert not res.valid and res.needs_review and "invalid_date" in res.issues


def test_serial_uses_latin_language_and_pattern() -> None:
    engine = Scripted(OcrText("ab 12345678", 0.9))
    res = _read(engine, "serial", regex=r"^[A-Z0-9]{8,12}$", latin_lang="eng")
    assert res.value == "AB12345678" and res.valid
    assert set(engine.langs) == {"eng"}
    bad = _read(Scripted(OcrText("A1", 0.9)), "serial", regex=r"^[A-Z0-9]{8,12}$")
    assert not bad.valid and "pattern_mismatch" in bad.issues


def test_names_are_cleaned_and_use_alt_language() -> None:
    engine = Scripted(OcrText("عليرضا 12", 0.5), OcrText("علیرضا", 0.7), OcrText("علیرضا", 0.95))
    res = _read(engine, "name", alt_lang="ara")
    assert res.value == "علیرضا" and res.valid
    assert "ara" in engine.langs
    assert res.needs_review and "free_text_unverified" in res.issues  # names are never trusted


def test_names_can_be_trusted_when_configured() -> None:
    res = _read(Scripted(OcrText("علیرضا", 0.95)), "name", trust_names=True)
    assert not res.needs_review


def test_low_confidence_triggers_review() -> None:
    res = _read(Scripted(OcrText("علیرضا", 0.2)), "name", trust_names=True)
    assert res.needs_review


def test_empty_crop() -> None:
    spec = FieldSpec("f", Box(1000, 630, 10, 8), "name")
    tiny = np.zeros((10, 10, 3), np.uint8)
    res = read_field(Scripted(OcrText("x", 1)), tiny, spec, SIZE, review_confidence=0.6)
    assert res.needs_review


def test_preprocess_variants_shapes() -> None:
    variants = preprocess_variants(CARD[100:160, 300:700])
    assert len(variants) == 3
    assert all(v.ndim == 2 and v.dtype == np.uint8 for v in variants)
    assert preprocess_variants(np.zeros((0, 0, 3), np.uint8)) == []


def test_crop_roi_scales_with_card_size() -> None:
    small = np.zeros((319, 505, 3), np.uint8)
    crop = crop_roi(small, Box(300, 100, 400, 60), SIZE)
    assert crop.shape[:2] == (30, 200)


class TestLocateOffset:
    """The text block is found by anchoring on the national id line (ten digits, self-checking)."""

    def _setup(self, flat_cards, registry):  # type: ignore[no-untyped-def]
        card, values = flat_cards["new-front"]
        spec = registry.get("new-front").fields["national_id"]
        return card, values, spec, registry.get("new-front").size

    def test_well_registered_card_needs_no_shift(self, flat_cards, registry) -> None:  # type: ignore[no-untyped-def]
        from idcard_ocr.fields import locate_offset

        card, _, spec, size = self._setup(flat_cards, registry)
        assert locate_offset(DigitEngine(), card, spec, size) == (0, 0)

    def test_recovers_a_vertical_shift(self, flat_cards, registry) -> None:  # type: ignore[no-untyped-def]
        import cv2

        from idcard_ocr.fields import locate_offset

        card, _, spec, size = self._setup(flat_cards, registry)
        # Content moved 24 px down, so the box has to move 24 px down too.
        moved = cv2.warpAffine(
            card, np.float32([[1, 0, 0], [0, 1, 24]]), size, borderMode=cv2.BORDER_REPLICATE
        )
        found = locate_offset(DigitEngine(), moved, spec, size)
        assert found is not None
        assert abs(found[1] - 24) <= 12 and abs(found[0]) <= 40

    def test_returns_none_when_no_line_reads_as_ten_digits(self, registry) -> None:  # type: ignore[no-untyped-def]
        from idcard_ocr.fields import locate_offset

        spec = registry.get("new-front").fields["national_id"]
        assert locate_offset(DigitEngine(), CARD, spec, SIZE) is None

    def test_read_field_applies_the_offset(self) -> None:
        engine = Scripted(OcrText("1370/05/12", 0.9))
        spec = FieldSpec("birth", Box(500, 300, 200, 50), "date")
        # The shifted box would leave the card, so the crop is clamped rather than failing.
        res = read_field(engine, CARD, spec, SIZE, review_confidence=0.6, offset=(400, 400))
        assert isinstance(res.value, str)
