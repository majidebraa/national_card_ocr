"""Line-level digit recogniser (CRNN + CTC) with joint card registration.

Why this exists
---------------
Segmenting a digit line into glyphs and classifying each glyph (``digits.py``) caps out around 89%
per glyph on hand-held photos, i.e. ~31% for a ten-digit id. A convolutional-recurrent network
trained with CTC (Shi et al., IEEE TPAMI 2017; Graves et al., ICML 2006) reads the whole line, needs
no segmentation, and is trained from the *line-level* labels a KYC database already holds.

Two ideas make it work on imprecisely rectified photos:

* **Weakly supervised alignment.** Training windows are placed where a first model already reads the
  label correctly (label-verified pseudo-alignment), so mis-registered cards never teach noise.
* **Joint registration by recognition.** The card's residual shift/scale is found by maximising
  the summed CTC confidence of *all* fields read from windows derived from one hypothesis. Dates
  can then no longer be read from the neighbouring line, and the winning hypothesis is validated
  by independent readings. CTC confidence separates right from wrong almost perfectly (AUC ~1.0), so
  "no confident reading" is a reliable signal to fall back or send the card to review.

Inference uses ONNX Runtime (``pip install idcard-ocr[crnn]``); training needs PyTorch and lives in
``scripts/train_crnn.py``. The model file is deliberately not part of the package: it is trained on
data whose licence the project cannot redistribute.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from idcard_ocr.layouts import CrnnSpec

CHARSET = "0123456789/"
BLANK = len(CHARSET)
INPUT_H, INPUT_W = 112, 240
DATE_RE = re.compile(r"^\d{4}/\d{2}/\d{2}$")

#: (scale, dy, dx) hypotheses searched when the nominal one is not confident.
SCALES = (0.85, 1.0, 1.2)
DYS = (-90, -60, -30, 0, 30, 60, 90)
DXS = (-40, 0, 40)
GRID: tuple[tuple[float, int, int], ...] = tuple(
    (s, dy, dx) for s in SCALES for dy in DYS for dx in DXS
)
NOMINAL = (1.0, 0, 0)
EARLY_EXIT = 0.95  # the nominal hypothesis is final when every field is at least this confident

Infer = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class CrnnReading:
    text: str
    confidence: float  # mean top-1 probability over the emitted characters; 0 if malformed
    hypothesis: tuple[float, int, int]


def window(
    card: np.ndarray,
    centre: tuple[float, float],
    size: tuple[float, float],
    scale: float = 1.0,
) -> np.ndarray:
    """Resample the window of ``size * scale`` card pixels at ``centre`` to the network input."""
    w, h = size[0] * scale, size[1] * scale
    sx, sy = INPUT_W / w, INPUT_H / h
    m = np.array(
        [[sx, 0.0, INPUT_W / 2 - centre[0] * sx], [0.0, sy, INPUT_H / 2 - centre[1] * sy]],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        card,
        m,
        (INPUT_W, INPUT_H),
        flags=cv2.INTER_AREA if scale > 1 else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def normalise(windows: np.ndarray) -> np.ndarray:
    x = windows.astype(np.float32) / 255.0
    mean = x.mean(axis=(1, 2), keepdims=True)
    std = x.std(axis=(1, 2), keepdims=True) + 1e-3
    return np.asarray((x - mean) / std, dtype=np.float32)


def decode(logits: np.ndarray) -> list[tuple[str, float]]:
    """Greedy CTC decoding; returns (text, mean top-1 probability of emitted characters)."""
    z = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    best = p.argmax(axis=-1)
    top = p.max(axis=-1)
    out: list[tuple[str, float]] = []
    for seq, conf in zip(best, top, strict=True):
        chars: list[str] = []
        probs: list[float] = []
        prev = -1
        for k, c in zip(seq.tolist(), conf.tolist(), strict=True):
            if k != prev and k != BLANK:
                chars.append(CHARSET[k])
                probs.append(c)
            prev = k
        out.append(("".join(chars), float(np.mean(probs)) if probs else 0.0))
    return out


def well_formed(kind: str, text: str) -> bool:
    if kind == "national_id":
        return len(text) == 10 and text.isdigit()
    if kind == "date":
        return bool(DATE_RE.match(text))
    return False


def centres(spec: CrnnSpec, hyp: tuple[float, int, int]) -> dict[str, tuple[float, float]]:
    """Window centres of every field under one registration hypothesis.

    ``scale`` is the window zoom (card content is drawn at ``1/scale`` of nominal size), so the
    offset of a field from the anchor shrinks or grows with it.
    """
    scale, dy, dx = hyp
    k = 1.0 / scale
    ax, ay = spec.anchor
    return {
        name: (
            cx + dx + (cx - ax) * (k - 1.0),
            cy + dy + (cy - ay) * (k - 1.0),
        )
        for name, (cx, cy, _plane, _kind) in spec.fields.items()
    }


class CrnnEngine:
    def __init__(self, infer: Infer) -> None:
        self._infer = infer

    @classmethod
    def from_onnx(cls, path: str | Path, threads: int = 1) -> CrnnEngine:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "onnxruntime is required for the CRNN recogniser: pip install 'idcard-ocr[crnn]'"
            ) from exc
        if not Path(path).is_file():
            raise FileNotFoundError(f"CRNN model not found: {path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, threads)
        session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])

        def infer(x: np.ndarray) -> np.ndarray:
            return np.asarray(session.run(None, {"x": x})[0])

        return cls(infer)

    def _evaluate(
        self, card: np.ndarray, spec: CrnnSpec, hyps: list[tuple[float, int, int]]
    ) -> dict[tuple[float, int, int], dict[str, CrnnReading]]:
        wins: list[np.ndarray] = []
        planes: list[int] = []
        keys: list[tuple[tuple[float, int, int], str]] = []
        for hyp in hyps:
            for name, centre in centres(spec, hyp).items():
                wins.append(window(card, centre, spec.window, hyp[0]))
                planes.append(spec.fields[name][2])
                keys.append((hyp, name))
        x = normalise(np.stack(wins))[:, None]
        plane = (np.asarray(planes, dtype=np.float32) / 2.0).reshape(-1, 1, 1, 1)
        x = np.concatenate([x, np.broadcast_to(plane, x.shape)], axis=1).astype(np.float32)
        decoded = decode(self._infer(x))
        result: dict[tuple[float, int, int], dict[str, CrnnReading]] = {}
        for (hyp, name), (text, conf) in zip(keys, decoded, strict=True):
            ok = well_formed(spec.fields[name][3], text)
            result.setdefault(hyp, {})[name] = CrnnReading(text, conf if ok else 0.0, hyp)
        return result

    def read(self, card: np.ndarray, spec: CrnnSpec) -> dict[str, CrnnReading]:
        """Read all fields of ``spec`` from a rectified, oriented card (BGR or gray)."""
        gray = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY) if card.ndim == 3 else card
        nominal = self._evaluate(gray, spec, [NOMINAL])[NOMINAL]
        if min(r.confidence for r in nominal.values()) >= EARLY_EXIT:
            return nominal
        scored = self._evaluate(gray, spec, list(GRID))
        best = max(scored.values(), key=lambda rs: sum(r.confidence for r in rs.values()))
        if sum(r.confidence for r in nominal.values()) >= sum(r.confidence for r in best.values()):
            return nominal
        return best
