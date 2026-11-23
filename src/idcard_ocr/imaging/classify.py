"""Side / version / orientation classification of a rectified card.

Each layout declares a background colour probe and marker regions (photo, barcode, chip).
Every (layout, 0/180 degree rotation) hypothesis is scored from simple image statistics:

* palette similarity of the probe region (separates card versions),
* marker agreement: expected markers must be present, markers of the *other* side absent.

No training data is needed, and callers that already know the side/version can skip this
step with hints (see :class:`idcard_ocr.pipeline.Pipeline`).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from idcard_ocr.errors import ClassificationError
from idcard_ocr.layouts import Box, Layout, LayoutRegistry

MIN_SCORE = 0.45
MIN_MARGIN = 0.04


@dataclass(frozen=True, slots=True)
class Classification:
    layout: Layout
    rotated_180: bool
    confidence: float
    scores: dict[str, float]


def _crop(img: np.ndarray, box: Box, size: tuple[int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    sx, sy = w / size[0], h / size[1]
    b = box.scaled(sx, sy)
    return img[b.y : b.y + b.h, b.x : b.x + b.w]


def _clip01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def barcode_score(gray: np.ndarray) -> float:
    """High for regions dominated by thin vertical bars."""
    if gray.size == 0:
        return 0.0
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)).mean()
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)).mean()
    ratio = gx / (gy + 1e-3)
    contrast = float(gray.std())
    return _clip01((ratio - 2.0) / 6.0) * _clip01((contrast - 35.0) / 40.0)


def photo_score(bgr: np.ndarray) -> float:
    """High for a region with portrait-like tonal variation that is not barcode-like."""
    if bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    tone = _clip01((float(gray.std()) - 22.0) / 35.0)
    colour = _clip01((float(hsv[:, :, 1].mean()) - 25.0) / 70.0)
    return (0.6 * tone + 0.4 * colour) * (1.0 - barcode_score(gray))


def chip_score(bgr: np.ndarray) -> float:
    """High for a gold-ish metallic patch (smart-card chip)."""
    if bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    gold = (hue >= 12) & (hue <= 32) & (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 110)
    return _clip01(float(gold.mean()) * 2.0)


def _marker_score(name: str, bgr: np.ndarray) -> float:
    if name == "photo":
        return photo_score(bgr)
    if name == "barcode":
        return barcode_score(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    if name == "chip":
        return chip_score(bgr)
    return 0.0


def _palette_similarity(card_bgr: np.ndarray, layout: Layout) -> float:
    probe = _crop(card_bgr, layout.bg_probe, layout.size)
    if probe.size == 0:
        return 0.0
    lab = cv2.cvtColor(probe, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    med = np.median(lab, axis=0)
    r, g, b = layout.palette_rgb
    ref = cv2.cvtColor(np.array([[[b, g, r]]], dtype=np.uint8), cv2.COLOR_BGR2LAB)[0, 0]
    dist = float(np.linalg.norm(med - ref.astype(np.float32)))
    return float(np.exp(-dist / 30.0))


def score_hypothesis(card_bgr: np.ndarray, layout: Layout, registry: LayoutRegistry) -> float:
    palette = _palette_similarity(card_bgr, layout)
    present = [
        _marker_score(name, _crop(card_bgr, box, layout.size))
        for name, box in layout.markers.items()
    ]
    positive = float(np.mean(present)) if present else 0.5

    # Markers that belong to *other* sides must be absent at their own positions.
    negatives: list[float] = []
    for other in registry.all():
        if other.side == layout.side:
            continue
        for name, box in other.markers.items():
            if name in layout.markers and layout.markers[name] == box:
                continue
            negatives.append(_marker_score(name, _crop(card_bgr, box, other.size)))
    negative = float(np.mean(negatives)) if negatives else 0.0
    marker = _clip01(positive - 0.5 * negative)
    return 0.4 * palette + 0.6 * marker


def classify(
    card_bgr: np.ndarray,
    registry: LayoutRegistry,
    *,
    side: str | None = None,
    version: str | None = None,
) -> Classification:
    """Pick the best (layout, rotation) hypothesis or raise :class:`ClassificationError`.

    ``side`` / ``version`` hints restrict the candidates; rotation is always inferred.
    """
    flipped = cv2.rotate(card_bgr, cv2.ROTATE_180)
    scored: list[tuple[float, Layout, bool]] = []
    scores: dict[str, float] = {}
    candidates = [
        layout
        for layout in registry.all()
        if (side is None or layout.side == side) and (version is None or layout.version == version)
    ]
    if not candidates:
        raise ClassificationError("No layout matches the given side/version hints.")
    for layout in candidates:
        for rot, img in ((False, card_bgr), (True, flipped)):
            s = score_hypothesis(img, layout, registry)
            scored.append((s, layout, rot))
            scores[f"{layout.id}{'@180' if rot else ''}"] = round(s, 3)
    scored.sort(key=lambda t: t[0], reverse=True)
    best, second = scored[0], scored[1]
    margin = best[0] - second[0]
    if best[0] < MIN_SCORE or margin < MIN_MARGIN:
        raise ClassificationError(
            "Could not tell which side or version of the card this is. "
            "Provide the `side` and `version` hints or retake the photo.",
            details={"best": round(best[0], 3), "margin": round(margin, 3)},
        )
    confidence = _clip01(best[0] * (0.5 + min(1.0, margin / 0.2) * 0.5))
    return Classification(best[1], best[2], confidence, scores)
