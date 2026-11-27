"""Face-photo extraction from a rectified card.

The portrait is kept for later identity verification (face match against a selfie), so it is
stored next to the card image. Detection uses OpenCV's YuNet (ONNX, MIT, 232 KB, ~40 ms on one
CPU core, no GPU). The search is limited to the layout's ``photo`` marker, shifted by the located
text block, so a second face (ghost image, hologram) elsewhere on the card is ignored.

If no face is found the crop falls back to the layout's photo box and ``face_found`` is False:
callers must treat that as "needs review", never as a verified portrait.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import cv2
import numpy as np

from idcard_ocr.layouts import Box

MODEL_NAME = "yunet.onnx"
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
SCORE_THRESHOLD = 0.6
MIN_FACE_FRACTION = 0.25  # face width relative to the photo box width
PAD_X = 0.45  # margin around the face box, relative to its size
PAD_TOP = 0.55
PAD_BOTTOM = 0.45
SEARCH_GROW = 0.35  # the search region is the photo box grown by this fraction per side


@dataclass(frozen=True, slots=True)
class Portrait:
    box: tuple[int, int, int, int]  # x, y, w, h on the rectified card
    face_found: bool
    confidence: float
    face_box: tuple[int, int, int, int] | None
    image: np.ndarray


def default_model_path() -> Path:
    return Path(str(resources.files("idcard_ocr") / "assets" / MODEL_NAME))


def _clip(box: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0 = max(0, box[0]), max(0, box[1])
    x1, y1 = min(w, box[0] + box[2]), min(h, box[1] + box[3])
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


class PortraitExtractor:
    """Thread-safe (the YuNet object is stateful, so calls are serialised by a lock)."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        path = Path(model_path) if model_path else default_model_path()
        if not path.is_file():
            raise FileNotFoundError(f"Face model not found: {path}")
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("OpenCV >= 4.5.4 is required for YuNet face detection.")
        self._path = str(path)
        self._lock = threading.Lock()
        self._det = cv2.FaceDetectorYN.create(self._path, "", (320, 320), SCORE_THRESHOLD, 0.3, 10)

    def extract(self, card: np.ndarray, photo: Box, offset: tuple[int, int] = (0, 0)) -> Portrait:
        ch, cw = card.shape[:2]
        px, py = photo.x + offset[0], photo.y + offset[1]
        gx, gy = round(photo.w * SEARCH_GROW), round(photo.h * SEARCH_GROW)
        sx, sy, sw, sh = _clip((px - gx, py - gy, photo.w + 2 * gx, photo.h + 2 * gy), cw, ch)
        region = np.ascontiguousarray(card[sy : sy + sh, sx : sx + sw])

        with self._lock:
            self._det.setInputSize((region.shape[1], region.shape[0]))
            _, faces = self._det.detect(region)

        best: tuple[float, np.ndarray] | None = None
        for f in faces if faces is not None else []:
            fw = float(f[2])
            if fw < MIN_FACE_FRACTION * photo.w:
                continue
            score = float(f[14])
            if best is None or score * fw > best[0] * float(best[1][2]):
                best = (score, f)

        if best is None:
            box = _clip((px, py, photo.w, photo.h), cw, ch)
            x, y, w, h = box
            return Portrait(box, False, 0.0, None, card[y : y + h, x : x + w].copy())

        score, f = best
        fx, fy, fw, fh = (float(v) for v in f[:4])
        fx, fy = fx + sx, fy + sy
        face = _clip((round(fx), round(fy), round(fw), round(fh)), cw, ch)
        box = _clip(
            (
                round(fx - PAD_X * fw),
                round(fy - PAD_TOP * fh),
                round(fw * (1 + 2 * PAD_X)),
                round(fh * (1 + PAD_TOP + PAD_BOTTOM)),
            ),
            cw,
            ch,
        )
        x, y, w, h = box
        return Portrait(box, True, round(score, 4), face, card[y : y + h, x : x + w].copy())
