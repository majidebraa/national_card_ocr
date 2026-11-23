"""Image-quality gates evaluated on the rectified card."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from idcard_ocr.config import Settings
from idcard_ocr.layouts import Box


@dataclass(frozen=True, slots=True)
class QualityReport:
    sharpness: float
    brightness: float
    glare_ratio: float  # share of field regions that are blown out by over-exposure
    source_width_px: float
    issues: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, float | list[str] | bool]:
        return {
            "sharpness": round(self.sharpness, 1),
            "brightness": round(self.brightness, 1),
            "glare_ratio": round(self.glare_ratio, 4),
            "source_width_px": round(self.source_width_px, 1),
            "issues": list(self.issues),
            "ok": self.ok,
        }


# Stable issue codes that a client can translate into user-facing messages.
ISSUE_BLURRY = "blurry"
ISSUE_DARK = "too_dark"
ISSUE_BRIGHT = "too_bright"
ISSUE_GLARE = "glare"
ISSUE_LOW_RES = "low_resolution"


def assess_quality(
    card_bgr: np.ndarray, source_width_px: float, settings: Settings
) -> QualityReport:
    """Whole-card checks that do not need the layout (sharpness, exposure, resolution)."""
    gray = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)
    # Variance of the Laplacian on a fixed-size card makes the score comparable across inputs.
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    issues: list[str] = []
    if sharpness < settings.min_sharpness:
        issues.append(ISSUE_BLURRY)
    if brightness < settings.min_brightness:
        issues.append(ISSUE_DARK)
    if brightness > settings.max_brightness:
        issues.append(ISSUE_BRIGHT)
    if source_width_px < settings.min_source_width_px:
        issues.append(ISSUE_LOW_RES)
    return QualityReport(sharpness, brightness, 0.0, source_width_px, tuple(issues))


def blown_out_ratio(card_bgr: np.ndarray, boxes: Iterable[Box], size: tuple[int, int]) -> float:
    """Share of the given regions that are washed out (glare): bright and without contrast.

    A plain white label area is not glare; a region is "blown out" only when it is almost
    entirely near-white *and* has no tonal variation, i.e. the printed text is gone.
    """
    h, w = card_bgr.shape[:2]
    hsv = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)
    total = blown = 0
    for box in boxes:
        b = box.scaled(w / size[0], h / size[1])
        region = (slice(max(0, b.y), b.y + b.h), slice(max(0, b.x), b.x + b.w))
        v, g = hsv[region][:, :, 2], gray[region]
        if v.size == 0:
            continue
        total += 1
        if float((v > 245).mean()) > 0.85 and float(g.std()) < 8.0:
            blown += 1
    return blown / total if total else 0.0


def with_glare(report: QualityReport, ratio: float, settings: Settings) -> QualityReport:
    issues = report.issues
    if ratio > settings.max_glare_ratio and ISSUE_GLARE not in issues:
        issues = (*issues, ISSUE_GLARE)
    return replace(report, glare_ratio=ratio, issues=issues)
