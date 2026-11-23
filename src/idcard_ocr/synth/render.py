"""Render a flat specimen card (front or back, old or new) from a layout and an identity."""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from idcard_ocr.layouts import Layout
from idcard_ocr.synth.identity import Identity

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
)
_BOLD_CANDIDATES = tuple(
    p.replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf") for p in _FONT_CANDIDATES
)


def find_font(bold: bool = False) -> str:
    import os

    override = os.environ.get("IDCARD_SYNTH_FONT")
    if override and Path(override).is_file():
        return override
    for path in _BOLD_CANDIDATES if bold else _FONT_CANDIDATES:
        if Path(path).is_file():
            return path
    raise FileNotFoundError(
        "No font with Persian glyphs found. Set IDCARD_SYNTH_FONT to a .ttf such as "
        "Vazirmatn, Tahoma or DejaVu Sans."
    )


@lru_cache(maxsize=32)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)


def to_persian_digits(text: str) -> str:
    return text.translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))


def _fit(
    draw: ImageDraw.ImageDraw,
    text: str,
    path: str,
    box_w: int,
    box_h: int,
    start: int = 44,
    **kw: str,
) -> ImageFont.FreeTypeFont:
    size = start
    while size > 12:
        font = _font(path, size)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font, **kw)
        if right - left <= box_w - 8 and bottom - top <= box_h - 4:
            return font
        size -= 2
    return _font(path, 12)


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def _background(layout: Layout, rng: random.Random) -> Image.Image:
    w, h = layout.size
    base = layout.palette_rgb
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    top = _lerp(base, (255, 255, 255), 0.25)
    bot = _lerp(base, (0, 0, 0), 0.04)
    for y in range(h):
        arr[y, :, :] = _lerp(top, bot, y / h)
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    # Fine guilloche-like arcs, kept away from the bottom colour probe strip.
    ink = _lerp(base, (0, 0, 0), 0.10)
    py = layout.bg_probe.y
    for i in range(14):
        cx = rng.randint(0, w)
        r = 90 + i * 22
        d.arc((cx - r, -r - 40, cx + r, r - 40), 20, 160, fill=ink, width=1)
    d.rectangle((0, 0, w, 90), fill=_lerp(base, (0, 0, 0), 0.30))
    d.rectangle((0, py - 6, w, py + layout.bg_probe.h + 6), fill=_lerp(base, (255, 255, 255), 0.0))
    return img


def _draw_photo(img: Image.Image, box: tuple[int, int, int, int], rng: random.Random) -> None:
    """Abstract placeholder portrait (no real face): drawn on a tile, then pasted clipped."""
    x, y, w, h = box
    tile = Image.new(
        "RGB", (w, h), (rng.randint(150, 190), rng.randint(165, 195), rng.randint(190, 225))
    )
    d = ImageDraw.Draw(tile)
    skin = (rng.randint(190, 235), rng.randint(150, 185), rng.randint(120, 160))
    hair = (rng.randint(20, 60), rng.randint(15, 45), rng.randint(10, 35))
    cx = w // 2
    d.ellipse((cx - 55, 45, cx + 55, 175), fill=skin)  # head
    d.pieslice((cx - 62, 32, cx + 62, 150), 180, 360, fill=hair)  # hair
    d.ellipse((cx - 105, 185, cx + 105, h + 120), fill=(70, 80, 100))  # shoulders
    d.rectangle((0, 0, w - 1, h - 1), outline=(60, 60, 60), width=2)
    img.paste(tile, (x, y))


def _draw_chip(img: Image.Image, box: tuple[int, int, int, int]) -> None:
    x, y, w, h = box
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(
        (x, y, x + w, y + h), radius=12, fill=(212, 175, 55), outline=(150, 120, 30)
    )
    for i in range(1, 3):
        d.line((x, y + i * h // 3, x + w, y + i * h // 3), fill=(150, 120, 30), width=2)
    d.line((x + w // 2, y, x + w // 2, y + h), fill=(150, 120, 30), width=2)


def _draw_barcode(img: Image.Image, box: tuple[int, int, int, int], rng: random.Random) -> None:
    x, y, w, h = box
    d = ImageDraw.Draw(img)
    d.rectangle((x, y, x + w, y + h), fill=(255, 255, 255))
    cx = x + 10
    while cx < x + w - 14:
        bar = rng.choice((2, 3, 4, 6))
        d.rectangle((cx, y + 8, cx + bar - 1, y + h - 8), fill=(0, 0, 0))
        cx += bar + rng.choice((2, 3, 4))


def render_card(layout: Layout, identity: Identity, rng: random.Random) -> np.ndarray:
    """Return a flat card as a BGR uint8 array of ``layout.size``."""
    path = find_font()
    bold = find_font(bold=True)
    img = _background(layout, rng)
    d = ImageDraw.Draw(img)
    w, _ = layout.size

    title = "کارت ملی هوشمند (نمونه)" if layout.version == "new" else "کارت ملی (نمونه)"
    title_font = _font(bold, 38)
    d.text(
        (w - 30, 45),
        title,
        font=title_font,
        fill=(255, 255, 255),
        anchor="rm",
        direction="rtl",
        language="fa",
    )
    d.text((30, 45), "SPECIMEN", font=_font(bold, 34), fill=(255, 220, 220), anchor="lm")

    if "photo" in layout.markers:
        _draw_photo(img, layout.markers["photo"].as_tuple(), rng)
    if "chip" in layout.markers:
        _draw_chip(img, layout.markers["chip"].as_tuple())
    if "barcode" in layout.markers:
        _draw_barcode(img, layout.markers["barcode"].as_tuple(), rng)

    values = identity.values()
    for name, spec in layout.fields.items():
        if spec.label_roi is not None:
            lx, ly, lw, lh = spec.label_roi.as_tuple()
            d.text(
                (lx + lw, ly + lh // 2),
                spec.label + ":",
                font=_font(path, 30),
                fill=(40, 40, 40),
                anchor="rm",
                direction="rtl",
                language="fa",
            )
        x, y, rw, rh = spec.roi.as_tuple()
        raw = values[name]
        if spec.kind == "serial":
            text, kw = raw, {}
            font = _fit(d, text, bold, rw, rh)
            d.text((x + rw - 6, y + rh // 2), text, font=font, fill=(10, 10, 10), anchor="rm")
        else:
            text = to_persian_digits(raw) if spec.kind in ("national_id", "date") else raw
            kw = {"direction": "rtl", "language": "fa"}
            font = _fit(d, text, bold, rw, rh, **kw)
            d.text((x + rw - 6, y + rh // 2), text, font=font, fill=(10, 10, 10), anchor="rm", **kw)

    # Large translucent watermark: these images are test fixtures, never documents.
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.text(
        (img.size[0] // 2, img.size[1] // 2 + 20),
        "SPECIMEN",
        font=_font(bold, 120),
        fill=(200, 30, 30, 28),
        anchor="mm",
    )
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return np.ascontiguousarray(np.asarray(img)[:, :, ::-1])
