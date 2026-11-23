"""Turn a flat card render into a plausible phone photo (perspective, light, blur, noise)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class Difficulty:
    name: str
    scale: tuple[float, float]  # card width as a fraction of the frame width
    rotation_deg: float
    perspective: float  # corner jitter as a fraction of the card width
    blur_sigma: tuple[float, float]
    noise_sigma: float
    jpeg_quality: tuple[int, int]
    glare_prob: float
    light_gain: tuple[float, float]


DIFFICULTIES: dict[str, Difficulty] = {
    "easy": Difficulty(
        "easy", (0.70, 0.82), 4.0, 0.015, (0.0, 0.5), 1.5, (88, 95), 0.0, (0.95, 1.05)
    ),
    "medium": Difficulty(
        "medium", (0.55, 0.75), 14.0, 0.045, (0.3, 1.1), 3.5, (72, 88), 0.15, (0.80, 1.10)
    ),
    "hard": Difficulty(
        "hard", (0.45, 0.70), 28.0, 0.085, (0.6, 1.8), 6.0, (55, 75), 0.40, (0.65, 1.15)
    ),
}


@dataclass(frozen=True, slots=True)
class Scene:
    image: np.ndarray  # BGR uint8
    quad: np.ndarray  # 4x2 float32, flat-card corners (tl, tr, br, bl) in image pixels
    rotation: int  # 0 or 180: how the flat card was turned before composition


def _background(rng: random.Random, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    palette = [
        (40, 55, 75),
        (70, 90, 110),
        (30, 30, 34),
        (95, 120, 140),
        (60, 75, 60),
        (150, 150, 155),
    ]
    c1 = np.array(rng.choice(palette), dtype=np.float32)
    c2 = np.array(rng.choice(palette), dtype=np.float32)
    xs = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    ys = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    t = np.clip(0.6 * xs + 0.4 * ys, 0, 1)
    bg = c1 * (1 - t) + c2 * t
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    texture = cv2.GaussianBlur(np_rng.normal(0, 9, (h, w)).astype(np.float32), (0, 0), 3)[
        :, :, None
    ]
    return np.clip(bg + texture, 0, 255).astype(np.uint8)


def _random_quad(
    rng: random.Random, frame: tuple[int, int], card: tuple[int, int], diff: Difficulty
) -> np.ndarray:
    fw, fh = frame
    cw = rng.uniform(*diff.scale) * fw
    ch = cw * card[1] / card[0]
    if ch > 0.88 * fh:
        ch = 0.88 * fh
        cw = ch * card[0] / card[1]
    corners = np.array(
        [[-cw / 2, -ch / 2], [cw / 2, -ch / 2], [cw / 2, ch / 2], [-cw / 2, ch / 2]],
        dtype=np.float32,
    )
    angle = math.radians(rng.uniform(-diff.rotation_deg, diff.rotation_deg))
    rot = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32
    )
    corners = corners @ rot.T
    jitter = diff.perspective * cw
    corners += np.array(
        [[rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)] for _ in range(4)],
        dtype=np.float32,
    )
    margin = 0.04
    lo = corners.min(axis=0)
    hi = corners.max(axis=0)
    span = hi - lo
    cx = (
        rng.uniform(margin * fw + span[0] / 2, (1 - margin) * fw - span[0] / 2)
        if span[0] < (1 - 2 * margin) * fw
        else fw / 2
    )
    cy = (
        rng.uniform(margin * fh + span[1] / 2, (1 - margin) * fh - span[1] / 2)
        if span[1] < (1 - 2 * margin) * fh
        else fh / 2
    )
    corners += np.array([cx, cy], dtype=np.float32) - (lo + hi) / 2
    return corners.astype(np.float32)


def compose_scene(
    card_bgr: np.ndarray,
    rng: random.Random,
    difficulty: str = "medium",
    *,
    frame: tuple[int, int] = (1600, 1200),
    rotate_180: bool = False,
) -> Scene:
    diff = DIFFICULTIES[difficulty]
    flat = cv2.rotate(card_bgr, cv2.ROTATE_180) if rotate_180 else card_bgr
    ch, cw = flat.shape[:2]
    fw, fh = frame
    canvas = _background(rng, frame).astype(np.float32)
    quad = _random_quad(rng, frame, (cw, ch), diff)
    src = np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, quad)

    mask = cv2.warpPerspective(
        np.full((ch, cw), 255, np.uint8), matrix, frame, flags=cv2.INTER_LINEAR
    )
    warped = cv2.warpPerspective(flat, matrix, frame, flags=cv2.INTER_CUBIC)

    # Soft drop shadow, offset down-right.
    shadow = (
        cv2.GaussianBlur(
            cv2.warpAffine(mask, np.float32([[1, 0, 10], [0, 1, 14]]), frame), (0, 0), 14
        ).astype(np.float32)
        / 255.0
    )
    canvas *= 1.0 - 0.45 * shadow[:, :, None]

    alpha = (cv2.GaussianBlur(mask, (0, 0), 0.8).astype(np.float32) / 255.0)[:, :, None]
    img = canvas * (1 - alpha) + warped.astype(np.float32) * alpha

    # Uneven lighting: a linear gain gradient across the frame.
    g0, g1 = rng.uniform(*diff.light_gain), rng.uniform(*diff.light_gain)
    gain = np.linspace(g0, g1, fw, dtype=np.float32)[None, :, None]
    img *= gain

    if rng.random() < diff.glare_prob:
        gx, gy = (
            rng.randint(int(0.3 * fw), int(0.7 * fw)),
            rng.randint(int(0.3 * fh), int(0.7 * fh)),
        )
        glare = np.zeros((fh, fw), np.float32)
        cv2.ellipse(
            glare,
            (gx, gy),
            (rng.randint(40, 110), rng.randint(25, 70)),
            rng.randint(0, 180),
            0,
            360,
            1.0,
            -1,
        )
        glare = cv2.GaussianBlur(glare, (0, 0), 30) * rng.uniform(0.5, 0.9)
        img = img * (1 - glare[:, :, None]) + 255.0 * glare[:, :, None]

    lo, hi = diff.blur_sigma
    sigma = rng.uniform(lo, hi)
    if sigma > 0.05:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    np_rng = np.random.default_rng(rng.randint(0, 2**31 - 1))
    img += np_rng.normal(0, diff.noise_sigma, img.shape).astype(np.float32)
    out = np.clip(img, 0, 255).astype(np.uint8)

    quality = rng.randint(*diff.jpeg_quality)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return Scene(out, quad, 180 if rotate_180 else 0)
