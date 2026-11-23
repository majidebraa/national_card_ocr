"""Per-field OCR: crop, preprocess into a few variants, read, validate, pick the best candidate.

The key production idea: for fields with a verifiable structure (national id checksum, calendar
dates, serial patterns) the *validator chooses between OCR attempts*. Variants are tried in
order and the loop stops at the first valid result, so the average cost stays low on CPU while
hard crops still get several chances.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

from idcard_ocr.layouts import Box, FieldSpec
from idcard_ocr.models import FieldResult
from idcard_ocr.ocr.base import OcrEngine
from idcard_ocr.ocr.digits import DigitEngine, decode_national_id
from idcard_ocr.validation.dates import parse_jalali
from idcard_ocr.validation.national_id import normalize_national_id
from idcard_ocr.validation.text import clean_name, to_ascii_digits

TARGET_HEIGHT = 84
PAD = 14
_SERIAL_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def crop_roi(card: np.ndarray, box: Box, size: tuple[int, int]) -> np.ndarray:
    h, w = card.shape[:2]
    b = box.scaled(w / size[0], h / size[1])
    x0, y0 = max(0, b.x), max(0, b.y)
    return card[y0 : min(h, b.y + b.h), x0 : min(w, b.x + b.w)]


def _ensure_dark_on_light(binary: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(binary) if float(binary.mean()) < 127 else binary


def preprocess_variants(crop_bgr: np.ndarray) -> list[np.ndarray]:
    """Return three single-channel variants of a field crop, ready for OCR.

    1. flat-field + Otsu: the background (paper, guilloche, watermark) is estimated with a
       morphological closing and divided out, so only high-contrast print survives;
    2. CLAHE + Otsu;
    3. contrast-stretched grey, leaving binarisation to the OCR engine.
    """
    if crop_bgr.size == 0:
        return []
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    scale = TARGET_HEIGHT / max(1, gray.shape[0])
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=5, searchWindowSize=15)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    ink = cv2.normalize(cv2.subtract(background, gray), None, 0, 255, cv2.NORM_MINMAX)
    _, flat = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    flat = cv2.bitwise_not(flat)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    out = []
    for img in (flat, _ensure_dark_on_light(otsu), stretched):
        out.append(cv2.copyMakeBorder(img, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=255))
    return out


@dataclass(slots=True)
class _Candidate:
    value: str
    raw: str
    confidence: float
    valid: bool
    issues: list[str]
    gregorian: str | None = None


def _score(c: _Candidate) -> tuple[int, float]:
    return (1 if c.valid else 0, c.confidence)


def _read_national_id(
    engine: OcrEngine, digits: DigitEngine | None, variants: list[np.ndarray]
) -> _Candidate:
    best: _Candidate | None = None
    repaired: list[_Candidate] = []

    if digits is not None:
        for img in variants:
            reading = digits.read(img)
            res = normalize_national_id(reading.text)
            if res.valid:
                conf = min(1.0, 0.6 + 0.4 * reading.confidence)
                return _Candidate(res.value, reading.text, conf, True, list(res.issues))
            decoded = decode_national_id(reading.probs) if reading.probs.shape[0] == 10 else None
            if decoded is not None:
                value, mean_prob, changes = decoded
                repaired.append(
                    _Candidate(
                        value,
                        reading.text,
                        min(0.7, 0.35 + 0.35 * mean_prob),
                        True,
                        ["checksum_repaired"] if changes else [],
                    )
                )
            cand = _Candidate(
                res.value, reading.text, 0.6 * reading.confidence, False, list(res.issues)
            )
            if best is None or _score(cand) > _score(best):
                best = cand
        if repaired:
            # Agreement between independently preprocessed variants is the best tie-breaker.
            votes: dict[str, list[_Candidate]] = {}
            for cand in repaired:
                votes.setdefault(cand.value, []).append(cand)
            winner = max(votes.values(), key=lambda cs: (len(cs), max(c.confidence for c in cs)))
            top = max(winner, key=lambda c: c.confidence)
            if len(votes) > 1:
                top.issues.append("variants_disagree")
            return top

    for img in variants:
        text = engine.recognize(img, psm=7)
        res = normalize_national_id(text.text)
        # A passing checksum is strong evidence (1 in 11 random strings pass), so it lifts the
        # confidence; a failing one caps it.
        conf = 0.6 + 0.4 * text.confidence if res.valid else 0.6 * text.confidence
        cand = _Candidate(res.value, text.text, min(1.0, conf), res.valid, list(res.issues))
        if best is None or _score(cand) > _score(best):
            best = cand
        if cand.valid:
            break
    return best or _Candidate("", "", 0.0, False, ["no_text"])


def _read_date(
    engine: OcrEngine, digits: DigitEngine | None, variants: list[np.ndarray]
) -> _Candidate:
    best: _Candidate | None = None
    attempts: list[tuple[bool, np.ndarray]] = []
    if digits is not None:
        attempts += [(True, v) for v in variants]
    attempts += [(False, v) for v in variants]
    for use_digits, img in attempts:
        text = digits.read_date(img) if use_digits and digits else engine.recognize(img, psm=7)
        res = parse_jalali(text.text)
        conf = 0.55 + 0.45 * text.confidence if res.valid else 0.5 * text.confidence
        cand = _Candidate(
            res.value, text.text, min(1.0, conf), res.valid, list(res.issues), res.gregorian
        )
        if best is None or _score(cand) > _score(best):
            best = cand
        if cand.valid:
            break
    return best or _Candidate("", "", 0.0, False, ["no_text"])


def _read_name(engine: OcrEngine, variants: list[np.ndarray], alt_lang: str) -> _Candidate:
    """Try a few (variant, language) pairs and keep the most confident valid reading.

    Measured on synthetic cards, taking the highest-confidence of several attempts beat any
    single variant (80% vs 68% exact). Tesseract's Persian model is the weakest link of the
    pipeline, so names are always flagged for review when confidence is low.
    """
    attempts: list[tuple[np.ndarray, str | None]] = []
    if len(variants) >= 3:
        attempts = [(variants[2], None), (variants[1], None)]
        if alt_lang:
            attempts.append((variants[2], alt_lang))
    else:
        attempts = [(v, None) for v in variants]
    best: _Candidate | None = None
    for img, lang in attempts:
        text = engine.recognize(img, psm=7, lang=lang)
        value, issues = clean_name(text.text)
        valid = bool(value) and "too_short" not in issues
        cand = _Candidate(value, text.text, text.confidence, valid, issues)
        if best is None or _score(cand) > _score(best):
            best = cand
        if valid and not issues and text.confidence >= 0.9:
            break
    return best or _Candidate("", "", 0.0, False, ["no_text"])


def _read_serial(
    engine: OcrEngine, variants: list[np.ndarray], spec: FieldSpec, latin_lang: str
) -> _Candidate:
    pattern = re.compile(spec.regex) if spec.regex else None
    best: _Candidate | None = None
    for img in variants:
        text = engine.recognize(img, psm=7, whitelist=_SERIAL_WHITELIST, lang=latin_lang)
        value = re.sub(r"[^A-Z0-9]", "", to_ascii_digits(text.text).upper())
        valid = bool(value) and (pattern.fullmatch(value) is not None if pattern else True)
        issues = [] if valid else ["pattern_mismatch"]
        conf = 0.5 + 0.5 * text.confidence if valid else 0.5 * text.confidence
        cand = _Candidate(value, text.text, min(1.0, conf), valid, issues)
        if best is None or _score(cand) > _score(best):
            best = cand
        if valid:
            break
    return best or _Candidate("", "", 0.0, False, ["no_text"])


def read_field(
    engine: OcrEngine,
    card: np.ndarray,
    spec: FieldSpec,
    layout_size: tuple[int, int],
    *,
    review_confidence: float,
    latin_lang: str = "eng",
    alt_lang: str = "",
    digit_engine: DigitEngine | None = None,
    trust_names: bool = False,
) -> FieldResult:
    crop = crop_roi(card, spec.roi, layout_size)
    variants = preprocess_variants(crop)
    if not variants:
        return FieldResult(
            value="", raw="", confidence=0.0, valid=False, needs_review=True, issues=["empty_crop"]
        )
    if spec.kind == "national_id":
        cand = _read_national_id(engine, digit_engine, variants)
    elif spec.kind == "date":
        cand = _read_date(engine, digit_engine, variants)
    elif spec.kind == "serial":
        cand = _read_serial(engine, variants, spec, latin_lang)
    else:
        cand = _read_name(engine, variants, alt_lang)
    issues = list(cand.issues)
    if not cand.value:
        issues.append("missing")
    unverified_text = spec.kind == "name" and not trust_names
    if unverified_text:
        issues.append("free_text_unverified")
    needs_review = (
        (not cand.valid)
        or cand.confidence < review_confidence
        or "checksum_repaired" in issues
        or unverified_text
    )
    return FieldResult(
        value=cand.value,
        raw=cand.raw,
        confidence=round(float(cand.confidence), 4),
        valid=cand.valid,
        needs_review=needs_review,
        issues=issues,
        gregorian=cand.gregorian,
    )
