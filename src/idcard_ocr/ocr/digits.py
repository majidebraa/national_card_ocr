"""Lightweight digit recognizer for Persian digits and the date separator.

Why this exists: general-purpose OCR models (Tesseract ``fas`` fast included) are weak on Persian
numerals, in particular on the dot-shaped zero, and the national id is the most valuable field
on the card. Digits are a closed 11-class problem, so a classical pipeline is both more accurate
and far cheaper on CPU:

    binarise -> connected components -> glyph features (HOG on a line-normalised canvas)
             -> k-NN over a small model trained on synthetic glyph strings

The classifier also returns per-glyph probability estimates, which enables checksum-guided
decoding of the national id (see :func:`decode_national_id`).

Honest limits: the shipped model is trained on the fonts available at training time
(``idcard-ocr train-digits --fonts ...`` retrains with yours) and it has **not** been validated
on real cards. Use the benchmark with your own consented samples before relying on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from idcard_ocr.errors import OcrUnavailableError
from idcard_ocr.ocr.base import OcrText

CLASSES = "0123456789/"
GLYPH = 32
K = 7
MODEL_PATH = Path(__file__).with_name("digit_model.npz")

_HOG = cv2.HOGDescriptor((GLYPH, GLYPH), (16, 16), (8, 8), (8, 8), 9)


@dataclass(frozen=True, slots=True)
class Component:
    x: int
    y: int
    w: int
    h: int
    mask: np.ndarray  # full-image uint8 mask (0/255) containing only this component

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


def binarize(gray: np.ndarray) -> np.ndarray:
    """Ink = 255. Handles dark-on-light and light-on-dark input."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if float(ink.mean()) > 127:  # more ink than paper: the polarity was inverted
        ink = cv2.bitwise_not(ink)
    return ink


MAX_SKEW_DEG = 10.0


def _sharpness(profile: np.ndarray) -> float:
    return float(np.square(profile.astype(np.float64)).sum())


def deskew(ink: np.ndarray) -> np.ndarray:
    """Straighten a line of digits by the rotation that sharpens the row projection.

    Hand-held photos leave a residual rotation after the card is rectified from imperfect corners,
    which makes the HOG features of one digit differ wildly. Only the ink mask is warped. A shear
    (slant) search was tried and rejected: the slash in dates is itself slanted.
    """
    h, w = ink.shape[:2]
    if h < 8 or w < 8 or not ink.any():
        return ink
    scale = min(1.0, 96.0 / h)
    small = cv2.resize(
        ink, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA
    )
    sh, sw = small.shape[:2]
    centre = (sw / 2.0, sh / 2.0)

    def rotated(angle: float) -> np.ndarray:
        m = cv2.getRotationMatrix2D(centre, angle, 1.0)
        return cv2.warpAffine(small, m, (sw, sh), flags=cv2.INTER_LINEAR)

    angles = np.arange(-MAX_SKEW_DEG, MAX_SKEW_DEG + 0.1, 1.0)
    best_angle = float(max(angles, key=lambda a: _sharpness(rotated(a).sum(axis=1))))
    if abs(best_angle) < 1.0:
        return ink
    m_rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), best_angle, 1.0)
    out = cv2.warpAffine(ink, m_rot, (w, h), flags=cv2.INTER_LINEAR)
    _, out = cv2.threshold(out, 127, 255, cv2.THRESH_BINARY)
    return out


def _make_component(mask: np.ndarray) -> Component | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return Component(
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
        mask,
    )


def _split_wide(comp: Component, ref_w: float, line_h: float) -> list[Component]:
    """Split a blob of touching glyphs.

    The number of pieces comes from the blob width over the typical single-digit width; each
    cut is placed at the thinnest column of the ink profile near its expected position.
    """
    if comp.w <= 1.15 * line_h:  # a single digit is always narrower than it is tall
        return [comp]
    pieces = max(2, round(comp.w / max(ref_w, 1.0)))
    sub = comp.mask[comp.y : comp.y + comp.h, comp.x : comp.x + comp.w]
    profile = (sub > 0).sum(axis=0).astype(np.float32)
    cuts: list[int] = []
    half = max(2, int(0.3 * comp.w / pieces))
    for i in range(1, pieces):
        centre = round(comp.w * i / pieces)
        lo, hi = max(2, centre - half), min(comp.w - 2, centre + half)
        if hi <= lo:
            return [comp]
        cuts.append(lo + int(np.argmin(profile[lo:hi])))
    out: list[Component] = []
    for lo, hi in pairwise([0, *cuts, comp.w]):
        if hi - lo < 3:
            continue
        m = np.zeros_like(comp.mask)
        m[comp.y : comp.y + comp.h, comp.x + lo : comp.x + hi] = sub[:, lo:hi]
        piece = _make_component(m)
        if piece is not None and piece.w * piece.h >= 12:
            out.append(piece)
    return out or [comp]


def segment(ink: np.ndarray) -> list[Component]:
    """Glyph components in left-to-right order (Persian numerals are written left to right)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    comps: list[Component] = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < 12:
            continue
        mask = np.where(labels == i, 255, 0).astype(np.uint8)
        comps.append(Component(x, y, w, h, mask))
    if not comps:
        return []
    max_h = max(c.h for c in comps)
    tall = [c for c in comps if c.h >= 0.45 * max_h]
    top = float(np.median([c.y for c in tall]))
    bottom = float(np.median([c.y + c.h for c in tall]))
    line_h = max(1.0, bottom - top)

    kept: list[Component] = [c for c in comps if c.h >= 0.45 * max_h]
    small = [c for c in comps if c.h < 0.45 * max_h]

    # Some fonts draw a digit as a main stroke plus a detached fragment that touches or almost
    # touches it. Such fragments belong to that digit; fold them in before judging dots.
    margin = 0.08 * line_h
    leftover: list[Component] = []
    for c in small:
        host = None
        for i, t in enumerate(kept):
            gap = max(t.x - (c.x + c.w), c.x - (t.x + t.w), 0)
            inside_y = c.y >= t.y - 0.1 * line_h and c.y + c.h <= t.y + t.h + 0.1 * line_h
            if gap <= margin and inside_y:
                host = i
                break
        if host is None:
            leftover.append(c)
            continue
        merged = _make_component(cv2.bitwise_or(kept[host].mask, c.mask))
        if merged is not None:
            kept[host] = merged

    tall_spans = [(c.x, c.x + c.w) for c in kept]
    for c in leftover:
        # A candidate zero (a dot): small, roundish, in the vertical middle of the text band and
        # in a slot of its own. Ticks and marks sitting above or beside a digit are dropped.
        round_ok = 0.5 <= c.w / max(c.h, 1) <= 2.0
        mid = top + 0.40 * line_h <= c.cy <= top + 0.80 * line_h
        big_enough = min(c.w, c.h) >= 0.10 * line_h
        own_slot = not any(lo - margin <= c.cx <= hi + margin for lo, hi in tall_spans)
        if round_ok and mid and big_enough and own_slot:
            kept.append(c)
    kept.sort(key=lambda c: c.cx)

    singles = [c.w for c in kept if c.h >= 0.45 * max_h and 0.4 * line_h <= c.w <= 1.15 * line_h]
    median_w = float(np.median(singles)) if singles else 0.75 * line_h
    out: list[Component] = []
    for c in kept:
        out.extend(_split_wide(c, median_w, line_h) if c.h >= 0.45 * max_h else [c])
    out.sort(key=lambda c: c.cx)
    return out


def line_box(comps: list[Component]) -> tuple[float, float]:
    """(top, height) of the text line, from the tall components."""
    max_h = max(c.h for c in comps)
    tall = [c for c in comps if c.h >= 0.45 * max_h]
    top = float(np.median([c.y for c in tall]))
    bottom = float(np.median([c.y + c.h for c in tall]))
    return top, max(1.0, bottom - top)


def glyph_features(comp: Component, top: float, line_h: float) -> np.ndarray:
    """HOG features on a canvas normalised to the line height, so size and height are encoded."""
    side = 1.25 * line_h
    scale = GLYPH / side
    tx = -(comp.cx - side / 2.0) * scale
    ty = -(top - 0.125 * line_h) * scale
    m = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)
    src = cv2.GaussianBlur(comp.mask, (0, 0), max(0.5, 0.5 / scale))
    canvas = cv2.warpAffine(src, m, (GLYPH, GLYPH), flags=cv2.INTER_AREA, borderValue=0)
    feat = _HOG.compute(canvas)
    return np.asarray(feat, dtype=np.float32).reshape(-1)


def glyph_matrix(gray: np.ndarray) -> tuple[list[Component], np.ndarray]:
    comps = segment(deskew(binarize(gray)))
    if not comps:
        return [], np.zeros((0, 324), dtype=np.float32)
    top, line_h = line_box(comps)
    return comps, np.stack([glyph_features(c, top, line_h) for c in comps])


class DigitModel:
    """k-NN over quantised HOG features; small, deterministic and dependency-free."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        if features.ndim != 2 or features.shape[0] != labels.shape[0] or features.shape[0] < K:
            raise ValueError("invalid digit model arrays")
        self._knn = cv2.ml.KNearest_create()
        self._knn.setDefaultK(K)
        self._knn.setIsClassifier(True)
        self._knn.train(features.astype(np.float32), cv2.ml.ROW_SAMPLE, labels.astype(np.float32))

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> DigitModel:
        p = Path(path)
        if not p.is_file():
            raise OcrUnavailableError(
                "Digit model not found. Run `idcard-ocr train-digits` to create it."
            )
        with np.load(p) as data:
            return cls(data["x"].astype(np.float32) / 255.0, data["y"])

    @staticmethod
    def save(path: str | Path, features: np.ndarray, labels: np.ndarray) -> None:
        q = np.clip(np.rint(features * 255.0), 0, 255).astype(np.uint8)
        np.savez_compressed(path, x=q, y=labels.astype(np.uint8))

    def predict_proba(self, feats: np.ndarray) -> np.ndarray:
        """Return an (n, 11) matrix of neighbour-vote probabilities with light smoothing."""
        if feats.shape[0] == 0:
            return np.zeros((0, len(CLASSES)), dtype=np.float32)
        _, _, neighbours, dist = self._knn.findNearest(feats.astype(np.float32), K)
        probs = np.zeros((feats.shape[0], len(CLASSES)), dtype=np.float32)
        weights = 1.0 / (dist + 0.05)
        for i in range(feats.shape[0]):
            for lab, w in zip(neighbours[i], weights[i], strict=True):
                probs[i, int(lab)] += w
        probs /= probs.sum(axis=1, keepdims=True)
        return (probs * 0.97 + 0.03 / len(CLASSES)).astype(np.float32)


@dataclass(frozen=True, slots=True)
class DigitReading:
    text: str
    probs: np.ndarray  # (n, 11)
    confidence: float


class DigitEngine:
    """An :class:`OcrEngine` for digit-and-slash text lines."""

    name = "digits-knn"

    def __init__(self, model: DigitModel | None = None) -> None:
        self._model = model

    @property
    def model(self) -> DigitModel:
        if self._model is None:
            self._model = DigitModel.load()
        return self._model

    def check_ready(self) -> None:
        _ = self.model

    def read(self, image: np.ndarray) -> DigitReading:
        comps, feats = glyph_matrix(image)
        if not comps:
            return DigitReading("", np.zeros((0, len(CLASSES)), dtype=np.float32), 0.0)
        probs = self.model.predict_proba(feats)
        idx = probs.argmax(axis=1)
        text = "".join(CLASSES[i] for i in idx)
        top = probs.max(axis=1)
        conf = float(0.5 * top.mean() + 0.5 * top.min())
        return DigitReading(text, probs, conf)

    def read_date(self, image: np.ndarray) -> OcrText:
        """Read ``yyyy/mm/dd``: separators are dropped and the 8 digits re-formatted.

        Returns an empty text unless exactly eight digit glyphs remain, so the caller can fall
        back to another engine instead of guessing.
        """
        reading = self.read(image)
        keep = [i for i, ch in enumerate(reading.text) if ch != "/"]
        if len(keep) != 8:
            return OcrText("", 0.0)
        digits = "".join(reading.text[i] for i in keep)
        top = reading.probs[keep].max(axis=1)
        conf = float(0.5 * top.mean() + 0.5 * top.min())
        return OcrText(f"{digits[:4]}/{digits[4:6]}/{digits[6:]}", conf)

    def recognize(
        self,
        image: np.ndarray,
        *,
        psm: int = 7,
        whitelist: str | None = None,
        lang: str | None = None,
    ) -> OcrText:
        reading = self.read(image)
        return OcrText(reading.text, reading.confidence)


def decode_national_id(
    probs: np.ndarray, *, max_changes: int = 1, change_below: float = 0.9
) -> tuple[str, float, int] | None:
    """Most likely checksum-valid id for 10 glyph probability rows, or ``None``.

    The checksum is linear modulo 11, so a dynamic program over (position, partial sum) finds
    the best valid id exactly. To stay conservative, at most ``max_changes`` positions may
    differ from the classifier's own arg-max, and only positions where the arg-max probability
    is below ``change_below``. Returns ``(id, mean_prob, number_of_changes)``.
    """
    if probs.shape[0] != 10:
        return None
    p = probs[:, :10].astype(np.float64)
    p = p / p.sum(axis=1, keepdims=True)
    argmax = p.argmax(axis=1)
    logp = np.log(p + 1e-9)

    # DP state: (partial weighted sum mod 11, changes used) -> (score, digits)
    State = tuple[float, tuple[int, ...]]
    layer: dict[tuple[int, int], State] = {(0, 0): (0.0, ())}
    for pos in range(9):
        weight = 10 - pos
        nxt: dict[tuple[int, int], State] = {}
        for (s, ch), (score, digits) in layer.items():
            for d in range(10):
                changed = int(d != argmax[pos])
                if changed and (ch + 1 > max_changes or p[pos, argmax[pos]] >= change_below):
                    continue
                key = ((s + d * weight) % 11, ch + changed)
                cand = (score + float(logp[pos, d]), (*digits, d))
                if key not in nxt or cand[0] > nxt[key][0]:
                    nxt[key] = cand
        layer = nxt

    best: tuple[float, str, int] | None = None
    for (s, ch), (score, digits) in layer.items():
        r = s % 11
        check = r if r < 2 else 11 - r
        if check > 9:  # cannot happen (r<2 or 11-r in 2..9), kept for safety
            continue
        changed_last = int(check != argmax[9])
        if changed_last and (ch + 1 > max_changes or p[9, argmax[9]] >= change_below):
            continue
        total = score + float(logp[9, check])
        value = "".join(map(str, digits)) + str(check)
        if len(set(value)) == 1:
            continue
        if best is None or total > best[0]:
            best = (total, value, ch + changed_last)
    if best is None:
        return None
    total, value, changes = best
    mean_prob = float(math.exp(total / 10.0))
    return value, mean_prob, changes
