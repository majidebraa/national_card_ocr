"""Safe image decoding: format allow-list, pixel cap, EXIF orientation."""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from idcard_ocr.errors import ImageTooLargeError, InvalidImageError

ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
MIN_SIDE_PX = 200


def decode_image(data: bytes, *, max_pixels: int) -> np.ndarray:
    """Decode bytes to a contiguous BGR uint8 array.

    The header is inspected before any pixel data is decoded, so oversized or unsupported
    files are rejected without allocating their pixel buffers.
    """
    if not data:
        raise InvalidImageError("Empty upload.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            width, height = probe.size
            if fmt not in ALLOWED_FORMATS:
                raise InvalidImageError(
                    "Unsupported image format; use JPEG, PNG or WEBP.",
                    details={"format": fmt},
                )
            if width * height > max_pixels:
                raise ImageTooLargeError(
                    "Image has too many pixels.",
                    details={"max_pixels": max_pixels},
                )
            if min(width, height) < MIN_SIDE_PX:
                raise InvalidImageError("Image is too small to contain a readable card.")
            probe.load()
            oriented = ImageOps.exif_transpose(probe)
            rgb = oriented.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("The upload is not a valid image.") from exc
    return np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])


def encode_jpeg(image_bgr: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:  # pragma: no cover - cv2 only fails on pathological input
        raise InvalidImageError("Could not encode image.")
    return bytes(buf.tobytes())
