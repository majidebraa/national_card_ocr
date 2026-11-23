"""Train the digit model from synthetic glyph strings rendered with local fonts.

Training data goes through the *same* preprocessing, segmentation and feature code as inference,
and a sample is kept only when segmentation yields exactly one component per character, so the
labels are trustworthy without any manual annotation.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from idcard_ocr.fields import preprocess_variants
from idcard_ocr.ocr.digits import CLASSES, MODEL_PATH, DigitModel, glyph_matrix
from idcard_ocr.synth.render import to_persian_digits

DEFAULT_FONT_GLOBS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf",
)
# The synthetic benchmark renders with DejaVu Sans Bold. Keeping it out of training makes the
# benchmark a test on an unseen font instead of a memorisation check.
HELD_OUT = ("DejaVuSans-Bold.ttf", "DejaVuSansCondensed-Bold.ttf")
HELD_OUT_PATHS = tuple(f"/usr/share/fonts/truetype/dejavu/{name}" for name in HELD_OUT)

ROI = (430, 60)


@dataclass(frozen=True, slots=True)
class TrainStats:
    samples: int
    per_class: dict[str, int]
    fonts: list[str]
    attempted: int
    kept_strings: int


def _random_text(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.55:
        return "".join(rng.choice("0123456789") for _ in range(10))
    y = rng.randint(1330, 1415)
    return f"{y}/{rng.randint(1, 12):02d}/{rng.randint(1, 30):02d}"


def _background(rng: random.Random) -> np.ndarray:
    w, h = ROI
    base = np.array(
        rng.choice([(222, 230, 196), (205, 225, 240), (235, 235, 235), (240, 232, 215)]),
        dtype=np.float32,
    )
    grad = np.linspace(rng.uniform(0.94, 1.04), rng.uniform(0.9, 1.0), h, dtype=np.float32)
    img = (base[None, None, :] * grad[:, None, None]).repeat(w, axis=1)
    img = np.clip(img, 0, 255).astype(np.uint8)
    for _ in range(rng.randint(0, 4)):  # faint guilloche-like lines
        cx = rng.randint(0, w)
        r = rng.randint(40, 200)
        shade = int(max(0, base.min() - rng.randint(8, 28)))
        cv2.ellipse(
            img,
            (cx, rng.randint(-30, 30)),
            (r, r),
            0,
            20,
            160,
            (shade, shade, shade),
            1,
            cv2.LINE_AA,
        )
    return img


def render_sample(text: str, font_path: str, rng: random.Random) -> np.ndarray:
    persian = to_persian_digits(text)
    img = Image.fromarray(_background(rng)[:, :, ::-1])
    draw = ImageDraw.Draw(img)
    size = rng.randint(34, 50)
    while size > 16:
        font = ImageFont.truetype(font_path, size, layout_engine=ImageFont.Layout.RAQM)
        left, top, right, bottom = draw.textbbox((0, 0), persian, font=font)
        if right - left <= ROI[0] - 24 and bottom - top <= ROI[1] - 8:
            break
        size -= 2
    ink = rng.randint(0, 70)
    x = ROI[0] - rng.randint(4, 14) - (rng.randint(0, 30) if rng.random() < 0.3 else 0)
    draw.text(
        (x, ROI[1] // 2 + rng.randint(-2, 2)), persian, font=font, fill=(ink, ink, ink), anchor="rm"
    )
    arr = np.asarray(img)[:, :, ::-1].copy()

    # Photographic degradations.
    h, w = arr.shape[:2]
    ang = rng.uniform(-1.8, 1.8)
    jit = rng.uniform(0, 0.02) * w
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = src + np.float32([[rng.uniform(-jit, jit), rng.uniform(-jit, jit)] for _ in range(4)])
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    arr = cv2.warpAffine(arr, rot, (w, h), borderMode=cv2.BORDER_REPLICATE)
    arr = cv2.warpPerspective(
        arr, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE
    )
    sigma = rng.uniform(0.0, 1.5)
    if sigma > 0.1:
        arr = cv2.GaussianBlur(arr, (0, 0), sigma)
    gain = rng.uniform(0.85, 1.1)
    noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, rng.uniform(0, 6), arr.shape)
    arr = np.clip(arr.astype(np.float32) * gain + noise, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(55, 95)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else arr


def discover_fonts(
    extra_dir: str | Path | None = None, *, include_held_out: bool = False
) -> list[str]:
    paths = [p for p in DEFAULT_FONT_GLOBS if Path(p).is_file()]
    if extra_dir:
        paths += [str(p) for p in sorted(Path(extra_dir).glob("*.[to]tf"))]
    if include_held_out:
        paths += [p for p in HELD_OUT_PATHS if Path(p).is_file()]
    else:
        paths = [p for p in paths if not p.endswith(HELD_OUT)]
    return paths


def train(
    out_path: str | Path = MODEL_PATH,
    *,
    fonts: list[str] | None = None,
    per_class: int = 450,
    seed: int = 7,
    max_attempts: int = 6000,
) -> TrainStats:
    fonts = fonts or discover_fonts()
    if not fonts:
        raise FileNotFoundError("No usable fonts found; pass --fonts-dir with Persian-digit fonts.")
    rng = random.Random(seed)
    feats: list[np.ndarray] = []
    labels: list[int] = []
    counts: Counter[int] = Counter()
    attempted = kept = 0
    while attempted < max_attempts and any(counts[i] < per_class for i in range(len(CLASSES))):
        attempted += 1
        text = _random_text(rng)
        img = render_sample(text, rng.choice(fonts), rng)
        variant = rng.choice(preprocess_variants(img))
        comps, matrix = glyph_matrix(variant)
        if len(comps) != len(text):
            continue
        kept += 1
        for ch, row in zip(text, matrix, strict=True):
            cls = CLASSES.index(ch)
            if counts[cls] >= per_class:
                continue
            counts[cls] += 1
            feats.append(row)
            labels.append(cls)
    if not feats:
        raise RuntimeError("Segmentation never matched the text; cannot train.")
    x = np.stack(feats)
    y = np.array(labels, dtype=np.uint8)
    DigitModel.save(out_path, x, y)
    return TrainStats(
        len(labels),
        {CLASSES[i]: counts[i] for i in range(len(CLASSES))},
        [Path(f).name for f in fonts],
        attempted,
        kept,
    )
